"""
Shared penalty functions matching compare.py's composite scoring.

These let strategies optimise the same objective they're measured on.
All constants are imported from config.py (single source of truth).
"""

from allocator.config import (
    DIVERSITY_PENALTY_MULTIPLIER,
    DUPE_PENALTY_FLOOR,
    DUPE_PENALTY_MULTIPLIER,
    FAIRNESS_PENALTY_MULTIPLIER,
    VALUE_FAR_PENALTY_RATE,
    VALUE_HEAVY_PENALTY_THRESHOLD,
    VALUE_NEAR_PENALTY_RATE,
    VALUE_OVER_HARD_THRESHOLD,
    VALUE_OVER_MODERATE_RATE,
    VALUE_OVER_SOFT_THRESHOLD,
    VALUE_SWEET_SPOT_HIGH,
    VALUE_SWEET_SPOT_LOW,
)
from allocator.models import AllocationResult, MysteryBox
from allocator.strategies._helpers import compute_diversity_score

# Pre-computed base penalties (constant across all calls)
_NEAR_BASE = (VALUE_SWEET_SPOT_LOW - VALUE_HEAVY_PENALTY_THRESHOLD) * VALUE_NEAR_PENALTY_RATE  # 6.0
_OVER_SOFT_BASE = (VALUE_OVER_SOFT_THRESHOLD - VALUE_SWEET_SPOT_HIGH) * VALUE_NEAR_PENALTY_RATE  # 4.5
_OVER_HARD_BASE = _OVER_SOFT_BASE + (VALUE_OVER_HARD_THRESHOLD - VALUE_OVER_SOFT_THRESHOLD) * VALUE_OVER_MODERATE_RATE  # 34.5


def value_penalty(vp: float) -> float:
    """
    Piecewise-linear value penalty for a box at *vp* % of target.

    Matches compare.py:423-443 exactly.
    """
    if VALUE_SWEET_SPOT_LOW <= vp <= VALUE_SWEET_SPOT_HIGH:
        return 0.0
    if VALUE_HEAVY_PENALTY_THRESHOLD <= vp < VALUE_SWEET_SPOT_LOW:
        return (VALUE_SWEET_SPOT_LOW - vp) * VALUE_NEAR_PENALTY_RATE
    if VALUE_SWEET_SPOT_HIGH < vp <= VALUE_OVER_SOFT_THRESHOLD:
        return (vp - VALUE_SWEET_SPOT_HIGH) * VALUE_NEAR_PENALTY_RATE
    if vp < VALUE_HEAVY_PENALTY_THRESHOLD:
        return _NEAR_BASE + (VALUE_HEAVY_PENALTY_THRESHOLD - vp) * VALUE_FAR_PENALTY_RATE
    if vp <= VALUE_OVER_HARD_THRESHOLD:
        return _OVER_SOFT_BASE + (vp - VALUE_OVER_SOFT_THRESHOLD) * VALUE_OVER_MODERATE_RATE
    # vp > 130
    return _OVER_HARD_BASE + (vp - VALUE_OVER_HARD_THRESHOLD) * VALUE_FAR_PENALTY_RATE


def weighted_dupe_penalty_for_box(box: MysteryBox, result: AllocationResult) -> float:
    """
    Sum of per-group weighted dupe penalties for a box.

    For each fungible group present: dupes * max(degree - DUPE_PENALTY_FLOOR, 0).
    Returns the raw weighted penalty (caller multiplies by DUPE_PENALTY_MULTIPLIER).
    Matches compare.py:360-373.
    """
    group_counts: dict[str, tuple[int, float]] = {}
    for item_id, qty in box.allocations.items():
        if qty > 0 and item_id in result.items:
            item = result.items[item_id]
            if item.fungible_group:
                if item.fungible_group in group_counts:
                    prev_count, degree = group_counts[item.fungible_group]
                    group_counts[item.fungible_group] = (prev_count + 1, degree)
                else:
                    group_counts[item.fungible_group] = (1, item.fungible_degree)
    penalty = 0.0
    for count, degree in group_counts.values():
        dupes = max(0, count - 1)
        penalty += dupes * max(degree - DUPE_PENALTY_FLOOR, 0.0)
    return penalty


def box_penalty(
    box: MysteryBox,
    result: AllocationResult,
    available_tags: dict[str, set[str]],
) -> float:
    """
    Total penalty for a single box (value + dupes + diversity).

    This is the per-box contribution to the composite score (before fairness).
    """
    value = result.box_value(box)
    vp = value / box.target_value * 100 if box.target_value > 0 else 0.0

    val_pen = value_penalty(vp)
    dupe_pen = weighted_dupe_penalty_for_box(box, result) * DUPE_PENALTY_MULTIPLIER
    div_score = compute_diversity_score(box, result, available_tags)
    div_pen = (1.0 - div_score) * DIVERSITY_PENALTY_MULTIPLIER

    return val_pen + dupe_pen + div_pen


def total_penalty(result: AllocationResult, available_tags: dict[str, set[str]]) -> float:
    """
    Full composite penalty across all boxes (lower = better).

    avg(box_penalties) + stddev(value_pct) * FAIRNESS_PENALTY_MULTIPLIER
    Preference violations are omitted (already enforced by is_excluded()).
    """
    n = len(result.boxes)
    if n == 0:
        return 0.0

    box_pens = []
    value_pcts = []
    for box in result.boxes:
        box_pens.append(box_penalty(box, result, available_tags))
        value = result.box_value(box)
        value_pcts.append(value / box.target_value * 100 if box.target_value > 0 else 0.0)

    avg_pen = sum(box_pens) / n

    mean_vp = sum(value_pcts) / n
    std_vp = (sum((vp - mean_vp) ** 2 for vp in value_pcts) / n) ** 0.5
    fair_pen = std_vp * FAIRNESS_PENALTY_MULTIPLIER

    return avg_pen + fair_pen
