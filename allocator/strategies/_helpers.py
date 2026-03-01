"""
Shared helper functions for allocation strategies.

Common constraint checks, mutation helpers, and box analysis utilities
used across multiple strategies.
"""

from allocator.config import DIVERSITY_WEIGHTS, VALUE_CEILING_PCT
from allocator.models import AllocationResult, Item, MysteryBox


def would_exceed_ceiling(
    box: MysteryBox, item: Item, qty: int, result: AllocationResult
) -> bool:
    """Check if adding qty of item would push box value above the ceiling."""
    current = result.box_value(box)
    new_value = current + item.price * qty
    return new_value > VALUE_CEILING_PCT * box.target_value


def box_fungible_groups(box: MysteryBox, result: AllocationResult) -> set[str]:
    """Return the set of fungible groups already present in a box."""
    groups: set[str] = set()
    for alloc_id, alloc_qty in box.allocations.items():
        if alloc_qty > 0 and alloc_id in result.items:
            g = result.items[alloc_id].fungible_group
            if g:
                groups.add(g)
    return groups


def has_hard_fungible_conflict(
    item: Item, box: MysteryBox, result: AllocationResult
) -> bool:
    """Check if item has a hard fungible conflict (degree >= 1.0) with box contents."""
    if not item.fungible_group:
        return False
    for alloc_id, alloc_qty in box.allocations.items():
        if alloc_qty > 0 and alloc_id in result.items:
            other = result.items[alloc_id]
            if (
                other.fungible_group == item.fungible_group
                and item.fungible_degree >= 1.0
            ):
                return True
    return False


def can_assign(
    item: Item, qty: int, box: MysteryBox, result: AllocationResult
) -> bool:
    """
    Composite check: can we assign qty of item to box?

    Checks exclusion, overage, ceiling, and hard fungible conflict.
    """
    if box.is_excluded(item):
        return False
    if result.remaining_overage(item.id) < qty:
        return False
    if would_exceed_ceiling(box, item, qty, result):
        return False
    if has_hard_fungible_conflict(item, box, result):
        return False
    return True


def box_deficit(box: MysteryBox, result: AllocationResult) -> int:
    """Return target - current value (positive means under-target)."""
    return box.target_value - result.box_value(box)


def assign_item(item_id: int, qty: int, box: MysteryBox) -> None:
    """Add qty of item to box allocations."""
    box.allocations[item_id] = box.allocations.get(item_id, 0) + qty


def remove_item(item_id: int, qty: int, box: MysteryBox) -> None:
    """Remove qty of item from box allocations. Clamps to 0."""
    current = box.allocations.get(item_id, 0)
    new_qty = max(0, current - qty)
    if new_qty == 0:
        box.allocations.pop(item_id, None)
    else:
        box.allocations[item_id] = new_qty


def compute_available_tags(result: AllocationResult) -> dict[str, set[str]]:
    """Compute the set of distinct tags available across all items with overage."""
    tags: dict[str, set[str]] = {
        "sub_category": set(),
        "usage": set(),
        "colour": set(),
        "shape": set(),
    }
    for item in result.items.values():
        if item.sub_category:
            tags["sub_category"].add(item.sub_category)
        if item.usage_type:
            tags["usage"].add(item.usage_type)
        if item.colour:
            tags["colour"].add(item.colour)
        if item.shape:
            tags["shape"].add(item.shape)
    return tags


def _box_tag_counts(box: MysteryBox, result: AllocationResult) -> dict[str, dict[str, int]]:
    """Compute qty-weighted tag counts per dimension for a box's allocations."""
    counts: dict[str, dict[str, int]] = {
        "sub_category": {},
        "usage": {},
        "colour": {},
        "shape": {},
    }
    dim_attrs = {
        "sub_category": "sub_category",
        "usage": "usage_type",
        "colour": "colour",
        "shape": "shape",
    }
    for alloc_id, alloc_qty in box.allocations.items():
        if alloc_qty > 0 and alloc_id in result.items:
            item = result.items[alloc_id]
            for dim, attr in dim_attrs.items():
                tag = getattr(item, attr, "")
                if tag:
                    counts[dim][tag] = counts[dim].get(tag, 0) + alloc_qty
    return counts


def _effective_species(tag_counts: dict[str, int]) -> float:
    """Compute effective number of species (1/HHI) from tag→qty counts."""
    total = sum(tag_counts.values())
    if total == 0:
        return 0.0
    hhi = sum((q / total) ** 2 for q in tag_counts.values())
    return 1.0 / hhi


def compute_diversity_score(
    box: MysteryBox,
    result: AllocationResult,
    available_tags: dict[str, set[str]] | None = None,
) -> float:
    """
    Compute quantity-weighted diversity score for a box (0.0 to 1.0).

    Uses effective number of species (1/HHI) per dimension, normalised by
    the number of available tags. Identical to binary coverage when all items
    have qty=1; progressively penalises concentration.
    """
    if available_tags is None:
        available_tags = compute_available_tags(result)

    tc = _box_tag_counts(box, result)

    score = 0.0
    for dim, weight in DIVERSITY_WEIGHTS.items():
        n_available = len(available_tags.get(dim, set()))
        dim_counts = tc.get(dim, {})
        if n_available > 0 and dim_counts:
            eff = _effective_species(dim_counts)
            score += weight * min(eff / n_available, 1.0)
        elif n_available == 0:
            score += weight  # no variety available = full marks
        # else: dim_counts empty, 0 contribution

    return score
