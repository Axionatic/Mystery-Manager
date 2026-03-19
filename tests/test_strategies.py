"""Tests for allocator/strategies/ — registry and cross-strategy invariants."""

import pytest

from allocator.config import CATEGORY_FRUIT, CATEGORY_VEGETABLES
from allocator.strategies import DEFAULT_STRATEGY, get_strategy, list_strategies
from allocator.strategies._helpers import has_hard_fungible_conflict


# ── Strategy registry ──────────────────────────────────────────────────────


class TestStrategyRegistry:
    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError, match="Unknown strategy"):
            get_strategy("nonexistent-strategy")

    def test_all_registered_strategies_loadable(self):
        for name in list_strategies():
            if name == "ilp-optimal":
                # ILP requires PuLP — skip if not installed
                try:
                    fn = get_strategy(name)
                    assert callable(fn)
                except ImportError:
                    pytest.skip("PuLP not installed")
            else:
                fn = get_strategy(name)
                assert callable(fn)

    def test_default_strategy_exists(self):
        assert DEFAULT_STRATEGY in list_strategies()

    def test_list_strategies_returns_list(self):
        strategies = list_strategies()
        assert isinstance(strategies, list)
        assert len(strategies) >= 5


# ── Cross-strategy invariants ──────────────────────────────────────────────


_STRATEGIES_TO_TEST = [
    "deal-topup",
    "greedy-best-fit",
    "round-robin",
    "minmax-deficit",
    "discard-worst",
    "local-search",
]


@pytest.mark.parametrize("strategy_name", _STRATEGIES_TO_TEST)
class TestStrategyInvariants:
    def test_no_over_allocation(self, strategy_name, two_box_result):
        """No item should be allocated more than its overage."""
        strategy = get_strategy(strategy_name)
        strategy(two_box_result)
        for item_id, item in two_box_result.items.items():
            total = sum(
                box.allocations.get(item_id, 0)
                for box in two_box_result.boxes
            )
            assert total <= item.overage, (
                f"{strategy_name}: item {item_id} ({item.name}) allocated {total} > overage {item.overage}"
            )

    def test_no_negative_allocations(self, strategy_name, two_box_result):
        """No allocation should be negative."""
        strategy = get_strategy(strategy_name)
        strategy(two_box_result)
        for box in two_box_result.boxes:
            for item_id, qty in box.allocations.items():
                assert qty >= 0, (
                    f"{strategy_name}: box {box.name} has negative qty for item {item_id}"
                )

    def test_boxes_get_allocations(self, strategy_name, two_box_result):
        """Each box should receive at least one item."""
        strategy = get_strategy(strategy_name)
        strategy(two_box_result)
        for box in two_box_result.boxes:
            total = sum(box.allocations.values())
            assert total > 0, (
                f"{strategy_name}: box {box.name} received no allocations"
            )

    def test_no_hard_fungible_conflicts(self, strategy_name, two_box_result):
        """No group should exceed 2x allowance in any box."""
        from allocator.config import GROUP_QTY_ALLOWANCE_BASE, GROUP_QTY_TIER_RATIO
        strategy = get_strategy(strategy_name)
        strategy(two_box_result)
        for box in two_box_result.boxes:
            allowance = GROUP_QTY_ALLOWANCE_BASE * GROUP_QTY_TIER_RATIO.get(box.tier, 1.0)
            group_totals: dict[str, int] = {}
            for item_id, qty in box.allocations.items():
                if qty > 0 and item_id in two_box_result.items:
                    item = two_box_result.items[item_id]
                    key = item.fungible_group or f"__item_{item_id}"
                    group_totals[key] = group_totals.get(key, 0) + qty
            for group_key, total_qty in group_totals.items():
                assert total_qty <= 2 * allowance, (
                    f"{strategy_name}: box {box.name} has group {group_key} "
                    f"with qty {total_qty} > 2x allowance {2 * allowance}"
                )

    def test_no_excluded_items(self, strategy_name, make_item, make_box, make_result, make_charity):
        """Strategies should respect exclusion rules."""
        items = [
            make_item(id=1, name="Apples - Gala", price=400, overage=5, category_id=CATEGORY_FRUIT,
                      sub_category="pome_fruit", usage_type="snacking", colour="red", shape="round"),
            make_item(id=2, name="Broccoli", price=500, overage=5, category_id=CATEGORY_VEGETABLES,
                      sub_category="brassica", usage_type="cooking", colour="green", shape="chunky"),
            make_item(id=3, name="Carrots", price=350, overage=5, category_id=CATEGORY_VEGETABLES,
                      sub_category="root_veg", usage_type="cooking", colour="orange", shape="long"),
        ]
        from allocator.models import ExclusionRule
        rule = ExclusionRule(pattern="broccoli", source="note")
        box1 = make_box(name="a@test", exclusions=[rule])
        box2 = make_box(name="b@test")
        charity = make_charity()
        result = make_result(items=items, boxes=[box1, box2], charity=[charity])

        strategy = get_strategy(strategy_name)
        strategy(result)

        # Box1 should not have broccoli (id=2)
        assert box1.allocations.get(2, 0) == 0, (
            f"{strategy_name}: box with broccoli exclusion got broccoli allocated"
        )
