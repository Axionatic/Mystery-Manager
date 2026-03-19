"""Tests for allocator/strategies/_scoring.py — value_penalty, group-qty penalty, box/total penalty."""

import pytest

from allocator.strategies._scoring import (
    box_penalty,
    group_qty_penalty_for_box,
    total_penalty,
    value_penalty,
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


# ── group_qty_penalty_for_box ─────────────────────────────────────────────


class TestGroupQtyPenalty:
    def test_below_allowance_zero_penalty(self, make_item, make_box, make_result):
        """Qty within allowance → zero penalty."""
        # allowance = 2 * 1.0 = 2 for small tier
        item = make_item(id=1, fungible_group="apple", fungible_degree=0.7)
        box = make_box(tier="small", allocations={1: 2})
        result = make_result(items=[item], boxes=[box])
        assert group_qty_penalty_for_box(box, result) == 0.0

    def test_at_allowance_zero_penalty(self, make_item, make_box, make_result):
        """Exactly at allowance boundary → zero penalty."""
        item1 = make_item(id=1, fungible_group="apple", fungible_degree=0.7)
        item2 = make_item(id=2, fungible_group="apple", fungible_degree=0.7)
        box = make_box(tier="small", allocations={1: 1, 2: 1})
        result = make_result(items=[item1, item2], boxes=[box])
        assert group_qty_penalty_for_box(box, result) == 0.0

    def test_above_allowance_penalised(self, make_item, make_box, make_result):
        """Qty > allowance → positive penalty."""
        item = make_item(id=1, fungible_group="apple", fungible_degree=0.7)
        box = make_box(tier="small", allocations={1: 4})
        result = make_result(items=[item], boxes=[box])
        pen = group_qty_penalty_for_box(box, result)
        # excess = 4 - 2 = 2, penalty = 2^1.5 * 0.7 ≈ 1.98
        assert pen > 0.0
        assert abs(pen - (2 ** 1.5) * 0.7) < 0.01

    def test_singleton_penalised(self, make_item, make_box, make_result):
        """Item without fungible_group, qty > allowance → penalised as singleton."""
        item = make_item(id=1, fungible_group=None)
        box = make_box(tier="small", allocations={1: 4})
        result = make_result(items=[item], boxes=[box])
        pen = group_qty_penalty_for_box(box, result)
        # excess = 4 - 2 = 2, penalty = 2^1.5 * 1.0 (singleton degree=1.0) ≈ 2.83
        assert pen > 0.0

    def test_degree_scales_penalty(self, make_item, make_box, make_result):
        """Higher degree → higher penalty for same qty."""
        item_low = make_item(id=1, fungible_group="apple", fungible_degree=0.3)
        item_high = make_item(id=2, fungible_group="banana", fungible_degree=1.0)
        box_low = make_box(name="low@test", tier="small", allocations={1: 4})
        box_high = make_box(name="high@test", tier="small", allocations={2: 4})
        result = make_result(items=[item_low, item_high], boxes=[box_low, box_high])
        pen_low = group_qty_penalty_for_box(box_low, result)
        pen_high = group_qty_penalty_for_box(box_high, result)
        assert pen_high > pen_low

    def test_tier_scaling(self, make_item, make_box, make_result):
        """Larger tiers get higher allowance → less penalty for same qty."""
        item_s = make_item(id=1, fungible_group="apple", fungible_degree=0.7)
        item_l = make_item(id=2, fungible_group="apple", fungible_degree=0.7)
        box_small = make_box(name="s@test", tier="small", allocations={1: 4})
        box_large = make_box(name="l@test", tier="large", allocations={2: 4})
        result = make_result(items=[item_s, item_l], boxes=[box_small, box_large])
        # small allowance=2, large allowance=2*2.0=4
        pen_small = group_qty_penalty_for_box(box_small, result)
        pen_large = group_qty_penalty_for_box(box_large, result)
        assert pen_small > pen_large
        assert pen_large == 0.0  # 4 == allowance of 4

    def test_multiple_groups_summed(self, make_item, make_box, make_result):
        """Penalties from multiple groups add together."""
        items = [
            make_item(id=1, fungible_group="apple", fungible_degree=0.7),
            make_item(id=2, fungible_group="banana", fungible_degree=1.0),
        ]
        box = make_box(tier="small", allocations={1: 4, 2: 4})
        result = make_result(items=items, boxes=[box])
        pen = group_qty_penalty_for_box(box, result)
        # Each group: excess=2, penalty=2^1.5 * degree
        apple_pen = (2 ** 1.5) * 0.7
        banana_pen = (2 ** 1.5) * 1.0
        assert abs(pen - (apple_pen + banana_pen)) < 0.01


# ── box_penalty ─────────────────────────────────────────────────────────────


class TestBoxPenalty:
    def test_perfect_box_low_penalty(self, make_item, make_box, make_result):
        """Box at sweet spot with good diversity → low penalty."""
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
        # Should be relatively low but not zero (diversity + desirability won't be perfect)
        assert pen < 15.0

    def test_empty_box_has_penalty(self, make_box, make_result, make_item):
        items = [make_item(id=1)]
        box = make_box()
        result = make_result(items=items, boxes=[box])
        tags = compute_available_tags(result)
        pen = box_penalty(box, result, tags)
        # 0% value → high value penalty, no diversity → high diversity penalty
        assert pen > 50.0

    def test_includes_desirability(self, make_item, make_box, make_result):
        """box_penalty should include a desirability component."""
        item = make_item(id=1, price=2300)  # 115% of small box price
        box = make_box(allocations={1: 1})
        result = make_result(items=[item], boxes=[box])
        tags = compute_available_tags(result)
        pen = box_penalty(box, result, tags)
        # Should include desirability penalty (unknown item = 0.5, so (1-0.5)*5.0 = 2.5)
        assert pen > 0.0


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


# ── params override ─────────────────────────────────────────────────────────


class TestParamsOverride:
    def test_value_penalty_params_override(self):
        """Params dict should override config defaults."""
        # Default sweet spot is 114-117
        assert value_penalty(115.0) == 0.0
        # Override to make 115 outside sweet spot
        params = {"value_sweet_from": 116, "value_sweet_to": 118}
        pen = value_penalty(115.0, params=params)
        assert pen > 0.0

    def test_group_qty_params_override(self, make_item, make_box, make_result):
        """Params dict should override group-qty config."""
        item = make_item(id=1, fungible_group="apple", fungible_degree=0.7)
        box = make_box(tier="small", allocations={1: 4})
        result = make_result(items=[item], boxes=[box])

        # Default allowance base=2, exponent=1.5
        pen_default = group_qty_penalty_for_box(box, result)
        assert pen_default > 0.0

        # Override with higher allowance → less penalty
        params = {"group_qty_allowance_base": 4, "group_qty_tier_ratio": {"small": 1.0}}
        pen_override = group_qty_penalty_for_box(box, result, params=params)
        assert pen_override == 0.0  # 4 == allowance of 4
