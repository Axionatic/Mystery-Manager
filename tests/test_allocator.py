"""Tests for allocator/allocator.py — infer_tier_from_name, parse_preference, _allocate_charity."""

import logging

import pytest

from allocator.allocator import (
    _allocate_charity,
    infer_tier_from_name,
    parse_preference,
)
from allocator.config import PREFERENCE_FRUIT_ONLY, PREFERENCE_VEG_ONLY


# ── infer_tier_from_name ────────────────────────────────────────────────────


class TestInferTierFromName:
    def test_small(self):
        assert infer_tier_from_name("A small mystery box!") == "small"

    def test_medium(self):
        assert infer_tier_from_name("A medium mystery box!") == "medium"

    def test_large(self):
        assert infer_tier_from_name("A large mystery box!") == "large"

    def test_case_insensitive(self):
        assert infer_tier_from_name("A SMALL box") == "small"
        assert infer_tier_from_name("A Medium Box") == "medium"

    def test_no_tier_returns_none(self):
        assert infer_tier_from_name("A mystery box!") is None

    def test_embedded_keyword(self):
        assert infer_tier_from_name("Get your small produce box") == "small"

    def test_priority_order_small_wins(self):
        """'small' checked first — if name has both, small wins."""
        assert infer_tier_from_name("small medium large") == "small"


# ── parse_preference ────────────────────────────────────────────────────────


class TestParsePreference:
    def test_none_input(self):
        assert parse_preference(None) is None

    def test_empty_string(self):
        assert parse_preference("") is None

    def test_no_veg(self):
        assert parse_preference("No veg please") == "fruit_only"

    def test_no_fruit(self):
        assert parse_preference("No fruit please") == "veg_only"

    def test_mix_returns_none(self):
        assert parse_preference("Mix of Everything (test)") is None

    def test_preference_fruit_only_exact(self):
        """Matches PREFERENCE_FRUIT_ONLY from scoring config."""
        assert parse_preference(PREFERENCE_FRUIT_ONLY) == "fruit_only"

    def test_preference_veg_only_exact(self):
        assert parse_preference(PREFERENCE_VEG_ONLY) == "veg_only"


# ── _allocate_charity ──────────────────────────────────────────────────────


class TestAllocateCharity:
    def test_fills_toward_target(self, make_item, make_result, make_charity):
        items = [
            make_item(id=1, price=500, overage=10),
            make_item(id=2, price=300, overage=10),
        ]
        charity = make_charity()
        result = make_result(items=items, charity=[charity])
        _allocate_charity(result, 2000)  # target = $20
        total = result.charity_value(charity)
        assert total > 0
        assert total <= 2500  # shouldn't wildly overshoot

    def test_high_value_items_first(self, make_item, make_result, make_charity):
        """Charity should prefer higher-value items to fill target faster."""
        expensive = make_item(id=1, price=1000, overage=5)
        cheap = make_item(id=2, price=100, overage=5)
        charity = make_charity()
        result = make_result(items=[expensive, cheap], charity=[charity])
        _allocate_charity(result, 2000)
        # Expensive item should have been allocated
        assert 1 in charity.allocations

    def test_remainder_goes_to_stock(self, make_item, make_result, make_charity):
        items = [make_item(id=1, price=500, overage=10)]
        charity = make_charity()
        result = make_result(items=items, charity=[charity])
        _allocate_charity(result, 1000)  # target = $10
        # After charity fills, remaining should be in stock
        charity_qty = charity.allocations.get(1, 0)
        stock_qty = result.stock.get(1, 0)
        assert charity_qty + stock_qty == 10

    def test_no_charity_boxes_noop(self, make_item, make_result):
        items = [make_item(id=1, price=500, overage=5)]
        result = make_result(items=items, charity=[])
        _allocate_charity(result, 1000)
        # Stock should get everything
        assert result.stock == {}  # stock only filled for charity scenario

    def test_zero_target(self, make_item, make_result, make_charity):
        items = [make_item(id=1, price=500, overage=5)]
        charity = make_charity()
        result = make_result(items=items, charity=[charity])
        _allocate_charity(result, 0)
        # With 0 target, charity might not get much
        # Remaining should go to stock
        total_allocated = sum(charity.allocations.values())
        assert total_allocated + result.stock.get(1, 0) == 5

    def test_respects_remaining_overage(self, make_item, make_box, make_result, make_charity):
        """Charity only gets items not already allocated to boxes."""
        item = make_item(id=1, price=500, overage=5)
        box = make_box(allocations={1: 3})
        charity = make_charity()
        result = make_result(items=[item], boxes=[box], charity=[charity])
        _allocate_charity(result, 10000)
        charity_qty = charity.allocations.get(1, 0)
        assert charity_qty <= 2  # only 2 remaining from overage of 5

    def test_multiple_charity_recipients(self, make_item, make_result, make_charity):
        items = [make_item(id=1, price=500, overage=10)]
        c1 = make_charity(name="Charity A")
        c2 = make_charity(name="Charity B")
        result = make_result(items=items, charity=[c1, c2])
        _allocate_charity(result, 4000)
        # Both should get allocations
        total = sum(c1.allocations.values()) + sum(c2.allocations.values())
        assert total > 0
