"""
Shared penalty functions matching compare.py's composite scoring.

These let strategies optimise the same objective they're measured on.
All constants are imported from config.py (single source of truth).
"""

from allocator.config import (
    BOX_TIERS,
    DIVERSITY_PENALTY_MULTIPLIER,
    DUPE_PENALTY_FLOOR,
    DUPE_PENALTY_MULTIPLIER,
    FAIRNESS_PENALTY_MULTIPLIER,
    VALUE_PENALTY_EXPONENT,
    VALUE_SWEET_FROM,
    VALUE_SWEET_TO,
)
from allocator.models import AllocationResult, MysteryBox
from allocator.strategies._helpers import compute_diversity_score


def value_penalty(vp: float) -> float:
    """
    Power-function value penalty for a box at *vp* % of box price.

    Symmetric: penalty = distance_from_sweet_spot ** exponent.
    Small deviations are gentle, large deviations are harsh.
    """
    if VALUE_SWEET_FROM <= vp <= VALUE_SWEET_TO:
        return 0.0
    if vp < VALUE_SWEET_FROM:
        x = VALUE_SWEET_FROM - vp
    else:
        x = vp - VALUE_SWEET_TO
    return x ** VALUE_PENALTY_EXPONENT


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
    box_price = BOX_TIERS[box.tier]["price"]
    vp = value / box_price * 100 if box_price > 0 else 0.0

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
        box_price = BOX_TIERS[box.tier]["price"]
        value_pcts.append(value / box_price * 100 if box_price > 0 else 0.0)

    avg_pen = sum(box_pens) / n

    mean_vp = sum(value_pcts) / n
    std_vp = (sum((vp - mean_vp) ** 2 for vp in value_pcts) / n) ** 0.5
    fair_pen = std_vp * FAIRNESS_PENALTY_MULTIPLIER

    return avg_pen + fair_pen
