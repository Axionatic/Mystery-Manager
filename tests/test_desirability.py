"""Tests for allocator/desirability.py — loading, shrinkage, normalisation, lookup."""

import pytest
from pathlib import Path

from allocator.desirability import (
    _load,
    _reset_cache,
    compute_box_desirability,
    get_item_desirability,
)

_FIXTURE_CSV = Path(__file__).parent / "fixtures" / "desirability_items.csv"


@pytest.fixture(autouse=True)
def _clear_cache():
    """Reset module cache before each test."""
    _reset_cache()
    yield
    _reset_cache()


class TestLoadCSV:
    def test_parses_fixture(self):
        scores = _load(_FIXTURE_CSV)
        assert len(scores) == 7
        assert "Apples - Fuji" in scores
        assert "Ginger" in scores

    def test_all_values_in_range(self):
        scores = _load(_FIXTURE_CSV)
        for name, score in scores.items():
            assert 0.0 <= score <= 1.0, f"{name} score {score} out of [0, 1]"


class TestShrinkage:
    def test_high_n_preserves_residual(self):
        """Items with many appearances should stay close to their raw residual direction."""
        scores = _load(_FIXTURE_CSV, prior=1)
        # Apples (n=10, residual=0.35) should be higher than Ginger (n=8, residual=-0.15)
        assert scores["Apples - Fuji"] > scores["Ginger"]

    def test_low_n_pulled_toward_mean(self):
        """With high prior, low-n items converge more toward the global mean."""
        # Use raw adjusted values (before normalisation) to check shrinkage
        import csv
        rows = []
        with open(_FIXTURE_CSV) as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append((row["name"], int(row["n_appearances"]), float(row["residual"])))
        total_n = sum(n for _, n, _ in rows)
        global_mean = sum(n * r for _, n, r in rows) / total_n

        # High prior should shrink more
        prior_low, prior_high = 1, 100
        adj_low = {name: (n * r + prior_low * global_mean) / (n + prior_low) for name, n, r in rows}
        adj_high = {name: (n * r + prior_high * global_mean) / (n + prior_high) for name, n, r in rows}

        spread_low = max(adj_low.values()) - min(adj_low.values())
        spread_high = max(adj_high.values()) - min(adj_high.values())
        assert spread_high < spread_low


class TestNormalisationBounds:
    def test_min_is_zero(self):
        scores = _load(_FIXTURE_CSV)
        assert min(scores.values()) == pytest.approx(0.0)

    def test_max_is_one(self):
        scores = _load(_FIXTURE_CSV)
        assert max(scores.values()) == pytest.approx(1.0)


class TestUnknownItemNeutral:
    def test_returns_half(self):
        score = get_item_desirability("Nonexistent Item 12345", csv_path=_FIXTURE_CSV)
        assert score == 0.5


class TestBoxDesirability:
    def test_single_item_box(self):
        lookup = {1: {"name": "Apples - Fuji"}}
        score = compute_box_desirability({1: 1}, lookup, csv_path=_FIXTURE_CSV)
        expected = get_item_desirability("Apples - Fuji", csv_path=_FIXTURE_CSV)
        assert abs(score - expected) < 0.001

    def test_qty_weighted_mean(self):
        lookup = {
            1: {"name": "Apples - Fuji"},
            2: {"name": "Ginger"},
        }
        score = compute_box_desirability({1: 3, 2: 1}, lookup, csv_path=_FIXTURE_CSV)
        apple = get_item_desirability("Apples - Fuji", csv_path=_FIXTURE_CSV)
        ginger = get_item_desirability("Ginger", csv_path=_FIXTURE_CSV)
        expected = (apple * 3 + ginger * 1) / 4
        assert abs(score - expected) < 0.001

    def test_empty_box_neutral(self):
        score = compute_box_desirability({}, {}, csv_path=_FIXTURE_CSV)
        assert score == 0.5
