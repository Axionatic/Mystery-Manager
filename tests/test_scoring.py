"""Tests for allocator/strategies/_scoring.py — value_penalty, dupe penalty, box/total penalty."""

import pytest

from allocator.strategies._scoring import (
    box_penalty,
    total_penalty,
    value_penalty,
    weighted_dupe_penalty_for_box,
)
from allocator.strategies._helpers import compute_available_tags


# ── value_penalty ───────────────────────────────────────────────────────────


class TestValuePenalty:
    def test_sweet_spot_zero(self):
        """Values within [114, 117] should have zero penalty."""
        assert value_penalty(114.0) == 0.0
        assert value_penalty(115.5) == 0.0
        assert value_penalty(117.0) == 0.0

    def test_below_sweet_spot(self):
        """Below 114 → penalty increases."""
        p = value_penalty(110.0)
        assert p > 0.0
        # 114 - 110 = 4, 4^1.25 ≈ 5.66
        assert abs(p - 4.0 ** 1.25) < 0.01

    def test_above_sweet_spot(self):
        """Above 117 → penalty increases."""
        p = value_penalty(120.0)
        assert p > 0.0
        # 120 - 117 = 3, 3^1.25 ≈ 3.95
        assert abs(p - 3.0 ** 1.25) < 0.01

    def test_symmetry(self):
        """Equal distance from sweet spot edges → equal penalty."""
        assert abs(value_penalty(109.0) - value_penalty(122.0)) < 0.01

    def test_monotonically_increasing_below(self):
        """Further below sweet spot → higher penalty."""
        assert value_penalty(100.0) > value_penalty(110.0) > value_penalty(113.0) > 0.0

    def test_monotonically_increasing_above(self):
        """Further above sweet spot → higher penalty."""
        assert value_penalty(130.0) > value_penalty(125.0) > value_penalty(118.0) > 0.0

    def test_zero_value(self):
        """0% value → large penalty."""
        p = value_penalty(0.0)
        assert p > 100.0  # 114^1.25 is very large

    def test_boundary_just_outside(self):
        """Just outside sweet spot → small nonzero penalty."""
        assert value_penalty(113.9) > 0.0
        assert value_penalty(117.1) > 0.0
        assert value_penalty(113.9) < 1.0  # 0.1^1.25 is tiny


# ── weighted_dupe_penalty_for_box ───────────────────────────────────────────


class TestWeightedDupePenalty:
    def test_no_dupes(self, make_item, make_box, make_result):
        item1 = make_item(id=1, fungible_group="apple", fungible_degree=0.7)
        item2 = make_item(id=2, fungible_group="banana", fungible_degree=1.0)
        box = make_box(allocations={1: 1, 2: 1})
        result = make_result(items=[item1, item2], boxes=[box])
        assert weighted_dupe_penalty_for_box(box, result) == 0.0

    def test_single_dupe(self, make_item, make_box, make_result):
        item1 = make_item(id=1, fungible_group="apple", fungible_degree=0.7)
        item2 = make_item(id=2, fungible_group="apple", fungible_degree=0.7)
        box = make_box(allocations={1: 1, 2: 1})
        result = make_result(items=[item1, item2], boxes=[box])
        # 1 dupe * max(0.7 - 0.5, 0) = 1 * 0.2 = 0.2
        assert abs(weighted_dupe_penalty_for_box(box, result) - 0.2) < 0.01

    def test_degree_below_floor(self, make_item, make_box, make_result):
        """Degree < DUPE_PENALTY_FLOOR (0.5) → zero penalty even with dupes."""
        item1 = make_item(id=1, fungible_group="herb", fungible_degree=0.3)
        item2 = make_item(id=2, fungible_group="herb", fungible_degree=0.3)
        box = make_box(allocations={1: 1, 2: 1})
        result = make_result(items=[item1, item2], boxes=[box])
        assert weighted_dupe_penalty_for_box(box, result) == 0.0

    def test_no_fungible_items(self, make_item, make_box, make_result):
        item = make_item(id=1, fungible_group=None)
        box = make_box(allocations={1: 2})
        result = make_result(items=[item], boxes=[box])
        assert weighted_dupe_penalty_for_box(box, result) == 0.0

    def test_multi_group_dupes(self, make_item, make_box, make_result):
        items = [
            make_item(id=1, fungible_group="apple", fungible_degree=0.7),
            make_item(id=2, fungible_group="apple", fungible_degree=0.7),
            make_item(id=3, fungible_group="tomato", fungible_degree=1.0),
            make_item(id=4, fungible_group="tomato", fungible_degree=1.0),
        ]
        box = make_box(allocations={1: 1, 2: 1, 3: 1, 4: 1})
        result = make_result(items=items, boxes=[box])
        # apple: 1 dupe * max(0.7-0.5,0)=0.2, tomato: 1 dupe * max(1.0-0.5,0)=0.5
        assert abs(weighted_dupe_penalty_for_box(box, result) - 0.7) < 0.01

    def test_empty_box(self, make_box, make_result, make_item):
        box = make_box()
        result = make_result(items=[make_item()], boxes=[box])
        assert weighted_dupe_penalty_for_box(box, result) == 0.0


# ── box_penalty ─────────────────────────────────────────────────────────────


class TestBoxPenalty:
    def test_perfect_box_low_penalty(self, make_item, make_box, make_result):
        """Box at sweet spot with good diversity → low penalty."""
        # Create a box with value exactly at 115% of small box price
        # Small price = 2000, target = 2300
        # 115% of 2000 = 2300
        items = [
            make_item(id=1, price=800, sub_category="tropical", usage_type="snacking",
                      colour="yellow", shape="long"),
            make_item(id=2, price=750, sub_category="root_veg", usage_type="cooking",
                      colour="orange", shape="long"),
            make_item(id=3, price=750, sub_category="pome_fruit", usage_type="snacking",
                      colour="red", shape="round"),
        ]
        box = make_box(allocations={1: 1, 2: 1, 3: 1})
        result = make_result(items=items, boxes=[box])
        tags = compute_available_tags(result)
        pen = box_penalty(box, result, tags)
        # Should be relatively low but not zero (diversity won't be perfect)
        assert pen < 10.0

    def test_empty_box_has_penalty(self, make_box, make_result, make_item):
        items = [make_item(id=1)]
        box = make_box()
        result = make_result(items=items, boxes=[box])
        tags = compute_available_tags(result)
        pen = box_penalty(box, result, tags)
        # 0% value → high value penalty, no diversity → high diversity penalty
        assert pen > 50.0


# ── total_penalty ───────────────────────────────────────────────────────────


class TestTotalPenalty:
    def test_zero_boxes(self, make_result):
        result = make_result(boxes=[])
        tags = {"sub_category": set(), "usage": set(), "colour": set(), "shape": set()}
        assert total_penalty(result, tags) == 0.0

    def test_uniform_values_no_fairness_penalty(self, make_item, make_box, make_result):
        """Two identical boxes → fairness penalty = 0 (std dev = 0)."""
        items = [make_item(id=1, price=500), make_item(id=2, price=500)]
        box1 = make_box(name="a@test", allocations={1: 1})
        box2 = make_box(name="b@test", allocations={2: 1})
        result = make_result(items=items, boxes=[box1, box2])
        tags = compute_available_tags(result)
        pen = total_penalty(result, tags)
        # Fairness component should be 0 since both boxes have same value
        # Total penalty = avg(box_penalties) + 0
        bp1 = box_penalty(box1, result, tags)
        bp2 = box_penalty(box2, result, tags)
        avg_bp = (bp1 + bp2) / 2
        assert abs(pen - avg_bp) < 0.01

    def test_unequal_values_has_fairness_penalty(self, make_item, make_box, make_result):
        """Two boxes with different values → positive fairness penalty."""
        items = [make_item(id=1, price=1000), make_item(id=2, price=100)]
        box1 = make_box(name="a@test", allocations={1: 1})
        box2 = make_box(name="b@test", allocations={2: 1})
        result = make_result(items=items, boxes=[box1, box2])
        tags = compute_available_tags(result)

        pen_with_fairness = total_penalty(result, tags)
        bp1 = box_penalty(box1, result, tags)
        bp2 = box_penalty(box2, result, tags)
        avg_bp = (bp1 + bp2) / 2
        assert pen_with_fairness > avg_bp  # fairness penalty adds to total

    def test_single_box_no_fairness(self, make_item, make_box, make_result):
        """Single box → std dev = 0 → no fairness penalty."""
        item = make_item(id=1, price=500)
        box = make_box(allocations={1: 1})
        result = make_result(items=[item], boxes=[box])
        tags = compute_available_tags(result)
        pen = total_penalty(result, tags)
        bp = box_penalty(box, result, tags)
        assert abs(pen - bp) < 0.01
