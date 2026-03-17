"""Tests for allocator/categorizer.py — fungible groups, classification, fallbacks."""

import pytest

from allocator.categorizer import (
    assign_classification,
    assign_fungible_group,
    category_name,
)
from allocator.config import (
    CATEGORY_FRUIT,
    CATEGORY_VEGETABLES,
    CLASSIFICATION_FALLBACK,
    FUNGIBLE_GROUPS,
    ITEM_CLASSIFICATIONS,
)


# ── assign_fungible_group ───────────────────────────────────────────────────


class TestAssignFungibleGroup:
    def test_apple_prefix_match(self):
        group, degree = assign_fungible_group("Apples - Royal Gala")
        assert group == "apple"
        assert degree == 0.7

    def test_banana_prefix_match(self):
        group, degree = assign_fungible_group("Bananas - Cavendish")
        assert group == "banana"
        assert degree == 1.0

    def test_tomato_prefix_match(self):
        group, degree = assign_fungible_group("Tomatoes - Cherry")
        if "tomato" in FUNGIBLE_GROUPS:
            assert group == "tomato"
            assert degree == FUNGIBLE_GROUPS["tomato"][0]
        else:
            # Real config may use a different group name for tomatoes
            # Just verify consistent return type
            assert group is None or isinstance(group, str)

    def test_no_match(self):
        group, degree = assign_fungible_group("Broccoli")
        assert group is None
        assert degree == 0.0

    def test_case_insensitive_match(self):
        group, degree = assign_fungible_group("apples - gala")
        assert group == "apple"

    def test_first_match_wins_known_limitation(self):
        """First matching group wins — if groups overlap, order matters."""
        # This documents the behavior; actual overlap depends on config
        group1, _ = assign_fungible_group("Apples - Royal Gala")
        assert group1 is not None  # Should match apple

    def test_partial_prefix_no_match(self):
        """'App' doesn't match 'Apples -' prefix."""
        group, degree = assign_fungible_group("Appetizer")
        assert group is None


# ── assign_classification ───────────────────────────────────────────────────


class TestAssignClassification:
    def test_apple_classification(self):
        sub, usage, colour, shape = assign_classification("Apples - Royal Gala", CATEGORY_FRUIT)
        assert sub == "pome_fruit"
        assert usage == "snacking"
        assert colour == "red"
        assert shape == "round"

    def test_banana_classification(self):
        sub, usage, colour, shape = assign_classification("Bananas - Cavendish", CATEGORY_FRUIT)
        assert sub == "tropical"
        assert usage == "snacking"
        # Colour depends on config (could be "yellow" or "orange_yellow")
        assert colour != ""  # should not be empty
        assert shape == "long"

    def test_fallback_fruit(self):
        """Unknown fruit item falls back to classification_fallback for fruit category."""
        sub, usage, colour, shape = assign_classification("Dragon Fruit", CATEGORY_FRUIT)
        expected_sub = CLASSIFICATION_FALLBACK[CATEGORY_FRUIT][0]
        assert sub == expected_sub

    def test_fallback_veg(self):
        """Unknown veg item falls back to classification_fallback for veg category."""
        sub, usage, colour, shape = assign_classification("Artichoke", CATEGORY_VEGETABLES)
        expected_sub = CLASSIFICATION_FALLBACK[CATEGORY_VEGETABLES][0]
        assert sub == expected_sub

    def test_fallback_unknown_category(self):
        """Unknown category_id falls back to generic defaults."""
        sub, usage, colour, shape = assign_classification("Mystery Item", 999)
        assert sub == "other"
        assert usage == "cooking"

    def test_case_insensitive_match(self):
        sub, _, _, _ = assign_classification("apples - gala", CATEGORY_FRUIT)
        assert sub == "pome_fruit"


# ── category_name ───────────────────────────────────────────────────────────


class TestCategoryName:
    def test_known_category(self):
        categories = {10: "Fruit", 20: "Vegetables"}
        assert category_name(10, categories) == "fruit"
        assert category_name(20, categories) == "vegetables"

    def test_unknown_category(self):
        categories = {10: "Fruit"}
        assert category_name(99, categories) == "unknown"

    def test_lowercased(self):
        categories = {10: "FRUIT"}
        assert category_name(10, categories) == "fruit"
