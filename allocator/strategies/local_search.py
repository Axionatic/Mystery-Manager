"""
Local-search allocation strategy.

Bootstrap from discard-worst (greedy draft + penalty-delta trim), then
iteratively relocate/swap items between boxes to minimise the composite
penalty matching compare.py's scoring (value sweet-spot, weighted dupes,
4D diversity, fairness via stddev).

Uses incremental objective evaluation: only the 2 affected boxes are
recomputed per candidate move, not all boxes. When run via compare.py
--all-strategies, pre-computed discard-worst allocations are passed in
to avoid redundant work.
"""

import logging
import math

from allocator.config import (
    DIVERSITY_PENALTY_MULTIPLIER,
    DIVERSITY_WEIGHTS,
    DUPE_PENALTY_FLOOR,
    DUPE_PENALTY_MULTIPLIER,
    FAIRNESS_PENALTY_MULTIPLIER,
    LOCAL_SEARCH_MAX_ITERATIONS,
)
from allocator.models import AllocationResult
from allocator.strategies._helpers import (
    compute_available_tags,
    has_hard_fungible_conflict,
    would_exceed_ceiling,
)
from allocator.strategies._scoring import value_penalty

logger = logging.getLogger(__name__)

MAX_ITERATIONS = LOCAL_SEARCH_MAX_ITERATIONS


class _BoxState:
    """Cached per-box metrics for incremental objective computation."""

    __slots__ = ("value_pct", "diversity_score", "weighted_dupe_penalty")

    def __init__(self):
        self.value_pct = 0.0
        self.diversity_score = 0.0
        self.weighted_dupe_penalty = 0.0


class _ObjectiveCache:
    """
    Maintains per-box cached state so the composite objective can be
    recomputed in O(n_boxes) after updating only the affected boxes in O(1).
    """

    def __init__(self, result: AllocationResult, available_tags: dict[str, set[str]]):
        self.result = result
        self.available_tags = available_tags
        # Pre-compute available tag counts (constant for the run)
        self._avail_counts = {
            dim: len(tags) for dim, tags in available_tags.items()
        }
        n = len(result.boxes)
        self.states = [_BoxState() for _ in range(n)]
        for i in range(n):
            self._recompute(i)

    def _recompute(self, box_idx: int) -> None:
        """Recompute cached state for a single box."""
        box = self.result.boxes[box_idx]
        items = self.result.items
        state = self.states[box_idx]

        # Value as % of target
        value = 0
        for item_id, qty in box.allocations.items():
            if item_id in items:
                value += items[item_id].price * qty
        state.value_pct = value / box.target_value * 100 if box.target_value > 0 else 0.0

        # Diversity score using effective species (1/HHI), inlined for speed
        tag_counts: dict[str, dict[str, int]] = {
            "sub_category": {}, "usage": {}, "colour": {}, "shape": {},
        }
        _dim_attrs = (
            ("sub_category", "sub_category"),
            ("usage", "usage_type"),
            ("colour", "colour"),
            ("shape", "shape"),
        )
        for item_id, qty in box.allocations.items():
            if qty > 0 and item_id in items:
                item = items[item_id]
                for dim, attr in _dim_attrs:
                    tag = getattr(item, attr, "")
                    if tag:
                        tc = tag_counts[dim]
                        tc[tag] = tc.get(tag, 0) + qty

        score = 0.0
        ac = self._avail_counts
        for dim, weight in DIVERSITY_WEIGHTS.items():
            n_avail = ac.get(dim, 0)
            dc = tag_counts[dim]
            if n_avail > 0 and dc:
                total = sum(dc.values())
                hhi = sum((q / total) ** 2 for q in dc.values())
                eff = 1.0 / hhi
                score += weight * min(eff / n_avail, 1.0)
            elif n_avail == 0:
                score += weight
        state.diversity_score = score

        # Weighted dupe penalty (track count + degree per fungible group)
        group_counts: dict[str, tuple[int, float]] = {}
        for item_id, qty in box.allocations.items():
            if qty > 0 and item_id in items:
                item = items[item_id]
                if item.fungible_group:
                    if item.fungible_group in group_counts:
                        prev_count, degree = group_counts[item.fungible_group]
                        group_counts[item.fungible_group] = (prev_count + 1, degree)
                    else:
                        group_counts[item.fungible_group] = (1, item.fungible_degree)
        state.weighted_dupe_penalty = sum(
            max(0, count - 1) * max(degree - DUPE_PENALTY_FLOOR, 0.0)
            for count, degree in group_counts.values()
        )

    def save(self, *box_indices: int) -> list[tuple[float, float, float]]:
        """Save state of specified boxes for cheap restore on revert."""
        return [
            (self.states[i].value_pct, self.states[i].diversity_score,
             self.states[i].weighted_dupe_penalty)
            for i in box_indices
        ]

    def restore(self, box_indices: tuple[int, ...], saved: list[tuple[float, float, float]]) -> None:
        """Restore previously saved state (O(1) instead of recompute)."""
        for idx, (vp, ds, wdp) in zip(box_indices, saved):
            s = self.states[idx]
            s.value_pct = vp
            s.diversity_score = ds
            s.weighted_dupe_penalty = wdp

    def recompute(self, *box_indices: int) -> None:
        """Recompute only the specified boxes."""
        for i in box_indices:
            self._recompute(i)

    def objective(self) -> float:
        """
        Compute composite objective from cached per-box state.

        Matches compare.py's composite scoring:
        avg(value_penalty + dupe_penalty + diversity_penalty) + fairness_penalty
        """
        states = self.states
        n = len(states)
        if n == 0:
            return 0.0

        # Per-box penalties (averaged)
        total_box_pen = 0.0
        for s in states:
            total_box_pen += (
                value_penalty(s.value_pct)
                + s.weighted_dupe_penalty * DUPE_PENALTY_MULTIPLIER
                + (1.0 - s.diversity_score) * DIVERSITY_PENALTY_MULTIPLIER
            )
        avg_box_pen = total_box_pen / n

        # Fairness: stddev of value_pct
        mean_vp = sum(s.value_pct for s in states) / n
        variance = sum((s.value_pct - mean_vp) ** 2 for s in states) / n
        fair_pen = math.sqrt(variance) * FAIRNESS_PENALTY_MULTIPLIER

        return avg_box_pen + fair_pen


def run(result: AllocationResult) -> None:
    """Local search: bootstrap from deal-topup then improve via moves."""
    # Bootstrap from discard-worst (better starting point than deal-topup).
    # Skip if boxes already have allocations (pre-filled by allocate()).
    if not any(box.allocations for box in result.boxes):
        from allocator.strategies.discard_worst import run as bootstrap_run
        bootstrap_run(result)

    if len(result.boxes) < 2:
        return

    available_tags = compute_available_tags(result)
    cache = _ObjectiveCache(result, available_tags)
    best_obj = cache.objective()
    logger.info(f"Local search starting objective: {best_obj:.4f}")

    improved = True
    iteration = 0

    while improved and iteration < MAX_ITERATIONS:
        improved = False
        iteration += 1

        # Try relocations
        for i, box_from in enumerate(result.boxes):
            for item_id, qty in list(box_from.allocations.items()):
                current_qty = box_from.allocations.get(item_id, 0)
                if current_qty <= 0 or item_id not in result.items:
                    continue

                item = result.items[item_id]

                for j, box_to in enumerate(result.boxes):
                    if i == j:
                        continue

                    # Check constraints for receiving box
                    if box_to.is_excluded(item):
                        continue
                    if has_hard_fungible_conflict(item, box_to, result):
                        continue
                    if would_exceed_ceiling(box_to, item, 1, result):
                        continue

                    # Try relocate: move 1 unit from box_from to box_to
                    cur = box_from.allocations.get(item_id, 0)
                    if cur <= 0:
                        break
                    saved = cache.save(i, j)
                    box_from.allocations[item_id] = cur - 1
                    if box_from.allocations[item_id] == 0:
                        del box_from.allocations[item_id]
                    box_to.allocations[item_id] = box_to.allocations.get(item_id, 0) + 1

                    cache.recompute(i, j)
                    new_obj = cache.objective()
                    if new_obj < best_obj:
                        best_obj = new_obj
                        improved = True
                    else:
                        # Revert
                        box_to.allocations[item_id] -= 1
                        if box_to.allocations[item_id] == 0:
                            del box_to.allocations[item_id]
                        box_from.allocations[item_id] = box_from.allocations.get(item_id, 0) + 1
                        cache.restore((i, j), saved)

                if improved:
                    break
            if improved:
                break

        if improved:
            continue

        # Try swaps
        for i, box_a in enumerate(result.boxes):
            for item_a_id, qty_a in list(box_a.allocations.items()):
                cur_a = box_a.allocations.get(item_a_id, 0)
                if cur_a <= 0 or item_a_id not in result.items:
                    continue
                item_a = result.items[item_a_id]

                for j, box_b in enumerate(result.boxes):
                    if j <= i:
                        continue

                    for item_b_id, qty_b in list(box_b.allocations.items()):
                        cur_b = box_b.allocations.get(item_b_id, 0)
                        if cur_b <= 0 or item_b_id not in result.items:
                            continue
                        if item_a_id == item_b_id:
                            continue
                        # Re-check box_a still has item_a
                        if box_a.allocations.get(item_a_id, 0) <= 0:
                            break

                        item_b = result.items[item_b_id]

                        # Check constraints
                        if box_b.is_excluded(item_a) or box_a.is_excluded(item_b):
                            continue

                        saved = cache.save(i, j)

                        # Temporarily remove both, check fungible conflicts
                        box_a.allocations[item_a_id] = box_a.allocations.get(item_a_id, 0) - 1
                        if box_a.allocations[item_a_id] == 0:
                            del box_a.allocations[item_a_id]
                        box_b.allocations[item_b_id] = box_b.allocations.get(item_b_id, 0) - 1
                        if box_b.allocations[item_b_id] == 0:
                            del box_b.allocations[item_b_id]

                        ok_a = not has_hard_fungible_conflict(item_b, box_a, result)
                        ok_b = not has_hard_fungible_conflict(item_a, box_b, result)

                        # Check ceiling
                        ok_a = ok_a and not would_exceed_ceiling(box_a, item_b, 1, result)
                        ok_b = ok_b and not would_exceed_ceiling(box_b, item_a, 1, result)

                        if ok_a and ok_b:
                            # Apply swap
                            box_a.allocations[item_b_id] = box_a.allocations.get(item_b_id, 0) + 1
                            box_b.allocations[item_a_id] = box_b.allocations.get(item_a_id, 0) + 1

                            cache.recompute(i, j)
                            new_obj = cache.objective()
                            if new_obj < best_obj:
                                best_obj = new_obj
                                improved = True
                            else:
                                # Revert swap
                                box_a.allocations[item_b_id] -= 1
                                if box_a.allocations[item_b_id] == 0:
                                    del box_a.allocations[item_b_id]
                                box_b.allocations[item_a_id] -= 1
                                if box_b.allocations[item_a_id] == 0:
                                    del box_b.allocations[item_a_id]
                                # Restore originals
                                box_a.allocations[item_a_id] = box_a.allocations.get(item_a_id, 0) + 1
                                box_b.allocations[item_b_id] = box_b.allocations.get(item_b_id, 0) + 1
                                cache.restore((i, j), saved)
                        else:
                            # Revert removal
                            box_a.allocations[item_a_id] = box_a.allocations.get(item_a_id, 0) + 1
                            box_b.allocations[item_b_id] = box_b.allocations.get(item_b_id, 0) + 1
                            cache.restore((i, j), saved)

                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break

    logger.info(
        f"Local search complete: {iteration} iterations, "
        f"final objective: {best_obj:.4f}"
    )
