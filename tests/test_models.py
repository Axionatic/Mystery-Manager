"""Tests for allocator/models.py — Item, ExclusionRule, MysteryBox, AllocationResult."""

import pytest

from allocator.config import CATEGORY_FRUIT, CATEGORY_VEGETABLES
from allocator.models import AllocationResult, CharityBox, ExclusionRule, Item, MysteryBox


# ── ExclusionRule.matches() ─────────────────────────────────────────────────


class TestExclusionRuleMatches:
    def test_substring_match(self, make_item):
        rule = ExclusionRule(pattern="apple", source="note")
        item = make_item(name="Apples - Royal Gala")
        assert rule.matches(item)

    def test_case_insensitive(self, make_item):
        rule = ExclusionRule(pattern="APPLE", source="note")
        item = make_item(name="apples - gala")
        assert rule.matches(item)

    def test_no_match(self, make_item):
        rule = ExclusionRule(pattern="banana", source="note")
        item = make_item(name="Apples - Royal Gala")
        assert not rule.matches(item)

    def test_empty_pattern_matches_everything(self, make_item):
        rule = ExclusionRule(pattern="", source="note")
        item = make_item(name="Anything")
        assert rule.matches(item)

    def test_pineapple_matches_apple_known_limitation(self, make_item):
        """Known limitation: 'apple' pattern matches 'Pineapple' (substring)."""
        rule = ExclusionRule(pattern="apple", source="note")
        item = make_item(name="Pineapple")
        # This is a known limitation — substring matching is too broad
        assert rule.matches(item)

    def test_exact_name_match(self, make_item):
        rule = ExclusionRule(pattern="Broccoli", source="manual")
        item = make_item(name="Broccoli")
        assert rule.matches(item)

    def test_partial_word_match(self, make_item):
        rule = ExclusionRule(pattern="bro", source="note")
        item = make_item(name="Broccoli")
        assert rule.matches(item)

    def test_source_field_preserved(self):
        rule = ExclusionRule(pattern="x", source="preference")
        assert rule.source == "preference"


# ── MysteryBox.is_excluded() ────────────────────────────────────────────────


class TestMysteryBoxIsExcluded:
    def test_no_exclusions(self, make_item, make_box):
        box = make_box()
        item = make_item(category_id=10)
        assert not box.is_excluded(item)

    def test_manual_exclusion(self, make_item, make_box):
        rule = ExclusionRule(pattern="banana", source="note")
        box = make_box(exclusions=[rule])
        item = make_item(name="Bananas - Cavendish")
        assert box.is_excluded(item)

    def test_fruit_only_excludes_veg(self, make_item, make_box):
        box = make_box(preference="fruit_only")
        veg = make_item(category_id=CATEGORY_VEGETABLES)
        assert box.is_excluded(veg)

    def test_fruit_only_allows_fruit(self, make_item, make_box):
        box = make_box(preference="fruit_only")
        fruit = make_item(category_id=CATEGORY_FRUIT)
        assert not box.is_excluded(fruit)

    def test_veg_only_excludes_fruit(self, make_item, make_box):
        box = make_box(preference="veg_only")
        fruit = make_item(category_id=CATEGORY_FRUIT)
        assert box.is_excluded(fruit)

    def test_veg_only_allows_veg(self, make_item, make_box):
        box = make_box(preference="veg_only")
        veg = make_item(category_id=CATEGORY_VEGETABLES)
        assert not box.is_excluded(veg)

    def test_multiple_rules_any_matches(self, make_item, make_box):
        rules = [
            ExclusionRule(pattern="kiwi", source="note"),
            ExclusionRule(pattern="melon", source="note"),
        ]
        box = make_box(exclusions=rules)
        assert box.is_excluded(make_item(name="Kiwifruit"))
        assert not box.is_excluded(make_item(name="Bananas"))

    def test_preference_and_rule_combined(self, make_item, make_box):
        """Preference + rule both apply: fruit_only box with banana exclusion."""
        rule = ExclusionRule(pattern="banana", source="note")
        box = make_box(preference="fruit_only", exclusions=[rule])
        # Veg excluded by preference
        assert box.is_excluded(make_item(name="Carrots", category_id=CATEGORY_VEGETABLES))
        # Banana excluded by rule
        assert box.is_excluded(make_item(name="Bananas - Cavendish", category_id=CATEGORY_FRUIT))
        # Other fruit OK
        assert not box.is_excluded(make_item(name="Apples - Gala", category_id=CATEGORY_FRUIT))


# ── AllocationResult.box_value() ────────────────────────────────────────────


class TestAllocationResultBoxValue:
    def test_empty_box(self, make_box, make_result, make_item):
        box = make_box()
        result = make_result(items=[make_item()], boxes=[box])
        assert result.box_value(box) == 0

    def test_single_item(self, make_box, make_result, make_item):
        item = make_item(id=1, price=500)
        box = make_box(allocations={1: 2})
        result = make_result(items=[item], boxes=[box])
        assert result.box_value(box) == 1000

    def test_multiple_items(self, make_box, make_result, make_item):
        items = [
            make_item(id=1, price=500),
            make_item(id=2, price=300),
        ]
        box = make_box(allocations={1: 1, 2: 3})
        result = make_result(items=items, boxes=[box])
        assert result.box_value(box) == 500 + 900

    def test_missing_item_ignored(self, make_box, make_result, make_item):
        """If box references an item_id not in items dict, it's silently skipped."""
        item = make_item(id=1, price=500)
        box = make_box(allocations={1: 1, 999: 2})
        result = make_result(items=[item], boxes=[box])
        assert result.box_value(box) == 500


# ── AllocationResult.total_allocated_qty() ──────────────────────────────────


class TestTotalAllocatedQty:
    def test_empty_result(self, make_result, make_item):
        item = make_item(id=1)
        result = make_result(items=[item])
        assert result.total_allocated_qty(1) == 0

    def test_across_boxes_and_charity(self, make_box, make_result, make_item, make_charity):
        item = make_item(id=1, overage=10)
        box1 = make_box(name="a@test", allocations={1: 2})
        box2 = make_box(name="b@test", allocations={1: 3})
        charity = make_charity(allocations={1: 1})
        result = make_result(items=[item], boxes=[box1, box2], charity=[charity])
        assert result.total_allocated_qty(1) == 6

    def test_includes_stock(self, make_result, make_item):
        item = make_item(id=1, overage=10)
        result = make_result(items=[item], stock={1: 4})
        assert result.total_allocated_qty(1) == 4


# ── AllocationResult.remaining_overage() ────────────────────────────────────


class TestRemainingOverage:
    def test_full_overage_when_unallocated(self, make_result, make_item):
        item = make_item(id=1, overage=5)
        result = make_result(items=[item])
        assert result.remaining_overage(1) == 5

    def test_partial_allocation(self, make_box, make_result, make_item):
        item = make_item(id=1, overage=5)
        box = make_box(allocations={1: 3})
        result = make_result(items=[item], boxes=[box])
        assert result.remaining_overage(1) == 2

    def test_fully_allocated(self, make_box, make_result, make_item):
        item = make_item(id=1, overage=5)
        box = make_box(allocations={1: 5})
        result = make_result(items=[item], boxes=[box])
        assert result.remaining_overage(1) == 0

    def test_over_allocated_goes_negative(self, make_box, make_result, make_item):
        item = make_item(id=1, overage=2)
        box = make_box(allocations={1: 5})
        result = make_result(items=[item], boxes=[box])
        assert result.remaining_overage(1) == -3

    def test_unknown_item_returns_zero(self, make_result, make_item):
        result = make_result(items=[make_item(id=1)])
        assert result.remaining_overage(999) == 0


# ── AllocationResult.charity_value() ────────────────────────────────────────


class TestCharityValue:
    def test_empty_charity(self, make_result, make_item, make_charity):
        charity = make_charity()
        result = make_result(items=[make_item(id=1, price=500)], charity=[charity])
        assert result.charity_value(charity) == 0

    def test_charity_with_allocations(self, make_result, make_item, make_charity):
        item = make_item(id=1, price=500)
        charity = make_charity(allocations={1: 3})
        result = make_result(items=[item], charity=[charity])
        assert result.charity_value(charity) == 1500


# ── Item.value property ────────────────────────────────────────────────────


class TestItemValue:
    def test_value_equals_price(self, make_item):
        item = make_item(price=750)
        assert item.value == 750
