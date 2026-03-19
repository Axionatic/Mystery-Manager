"""
Shared penalty functions matching compare.py's composite scoring.

These let strategies optimise the same objective they're measured on.
All constants are imported from config.py (single source of truth).
"""

from allocator.config import (
    BOX_TIERS,
    DESIRABILITY_PENALTY_MULTIPLIER,
    DIVERSITY_PENALTY_MULTIPLIER,
    FAIRNESS_PENALTY_MULTIPLIER,
    GROUP_QTY_ALLOWANCE_BASE,
    GROUP_QTY_EXPONENT,
    GROUP_QTY_MULTIPLIER,
    GROUP_QTY_TIER_RATIO,
    VALUE_PENALTY_EXPONENT,
    VALUE_SWEET_FROM,
    VALUE_SWEET_TO,
)
from allocator.models import AllocationResult, MysteryBox
from allocator.strategies._helpers import compute_diversity_score


def _p(params: dict | None, key: str, default):
    """Look up a param from override dict, falling back to default."""
    if params and key in params:
        return params[key]
    return default


def value_penalty(vp: float, params: dict | None = None) -> float:
    """
    Power-function value penalty for a box at *vp* % of box price.

    Symmetric: penalty = distance_from_sweet_spot ** exponent.
    Small deviations are gentle, large deviations are harsh.
    """
    sweet_from = _p(params, "value_sweet_from", VALUE_SWEET_FROM)
    sweet_to = _p(params, "value_sweet_to", VALUE_SWEET_TO)
    exponent = _p(params, "value_penalty_exponent", VALUE_PENALTY_EXPONENT)

    if sweet_from <= vp <= sweet_to:
        return 0.0
    if vp < sweet_from:
        x = sweet_from - vp
    else:
        x = vp - sweet_to
    return x ** exponent


def group_qty_penalty_for_box(
    box: MysteryBox, result: AllocationResult, params: dict | None = None,
) -> float:
    """
    Sum of per-group qty-excess penalties for a box.

    For each fungible group (and singleton items without a group):
      penalty = max(0, total_group_qty - allowance) ^ exponent * degree
    Allowance = base * tier_ratio[box.tier].

    Returns the raw penalty (caller multiplies by GROUP_QTY_MULTIPLIER).
    """
    base = _p(params, "group_qty_allowance_base", GROUP_QTY_ALLOWANCE_BASE)
    tier_ratio = _p(params, "group_qty_tier_ratio", GROUP_QTY_TIER_RATIO)
    exponent = _p(params, "group_qty_exponent", GROUP_QTY_EXPONENT)

    allowance = base * tier_ratio.get(box.tier, 1.0)

    # Build {group_key: (total_qty, degree)}
    groups: dict[str, tuple[int, float]] = {}
    for item_id, qty in box.allocations.items():
        if qty > 0 and item_id in result.items:
            item = result.items[item_id]
            if item.fungible_group:
                key = item.fungible_group
                degree = item.fungible_degree
            else:
                key = f"__item_{item_id}"
                degree = 1.0
            if key in groups:
                prev_qty, prev_degree = groups[key]
                groups[key] = (prev_qty + qty, prev_degree)
            else:
                groups[key] = (qty, degree)

    penalty = 0.0
    for total_qty, degree in groups.values():
        excess = max(0, total_qty - allowance)
        if excess > 0:
            penalty += (excess ** exponent) * degree
    return penalty


def box_penalty(
    box: MysteryBox,
    result: AllocationResult,
    available_tags: dict[str, set[str]],
    params: dict | None = None,
) -> float:
    """
    Total penalty for a single box (value + group-qty + diversity + desirability).

    This is the per-box contribution to the composite score (before fairness).
    """
    gq_mult = _p(params, "group_qty_multiplier", GROUP_QTY_MULTIPLIER)
    div_mult = _p(params, "diversity_penalty_multiplier", DIVERSITY_PENALTY_MULTIPLIER)
    desir_mult = _p(params, "desirability_penalty_multiplier", DESIRABILITY_PENALTY_MULTIPLIER)

    value = result.box_value(box)
    box_price = BOX_TIERS[box.tier]["price"]
    vp = value / box_price * 100 if box_price > 0 else 0.0

    val_pen = value_penalty(vp, params)
    gq_pen = group_qty_penalty_for_box(box, result, params) * gq_mult
    div_score = compute_diversity_score(box, result, available_tags)
    div_pen = (1.0 - div_score) * div_mult

    # Desirability penalty
    from allocator.desirability import compute_box_desirability
    desir_score = compute_box_desirability(box.allocations, result.items)
    desir_pen = (1.0 - desir_score) * desir_mult

    return val_pen + gq_pen + div_pen + desir_pen


def total_penalty(
    result: AllocationResult,
    available_tags: dict[str, set[str]],
    params: dict | None = None,
) -> float:
    """
    Full composite penalty across all boxes (lower = better).

    avg(box_penalties) + stddev(value_pct) * FAIRNESS_PENALTY_MULTIPLIER
    Preference violations are omitted (already enforced by is_excluded()).
    """
    fair_mult = _p(params, "fairness_penalty_multiplier", FAIRNESS_PENALTY_MULTIPLIER)
    n = len(result.boxes)
    if n == 0:
        return 0.0

    box_pens = []
    value_pcts = []
    for box in result.boxes:
        box_pens.append(box_penalty(box, result, available_tags, params))
        value = result.box_value(box)
        box_price = BOX_TIERS[box.tier]["price"]
        value_pcts.append(value / box_price * 100 if box_price > 0 else 0.0)

    avg_pen = sum(box_pens) / n

    mean_vp = sum(value_pcts) / n
    std_vp = (sum((vp - mean_vp) ** 2 for vp in value_pcts) / n) ** 0.5
    fair_pen = std_vp * fair_mult

    return avg_pen + fair_pen
