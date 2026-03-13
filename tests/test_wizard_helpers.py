"""Unit tests for WizardState helper functions.

Run with: python3 tests/test_wizard_helpers.py
"""
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
from allocator.screens.wizard_state import (
    WizardState,
    detect_offer_id,
    discover_xlsx_files,
)


def test_detect_offer_id_standard():
    result = detect_offer_id(Path("offer_108_shopping_list.xlsx"))
    assert result == 108, f"Expected 108, got {result}"


def test_detect_offer_id_single_digit():
    result = detect_offer_id(Path("offer_5_shopping_list.xlsx"))
    assert result == 5, f"Expected 5, got {result}"


def test_detect_offer_id_non_matching_name():
    result = detect_offer_id(Path("my_file.xlsx"))
    assert result is None, f"Expected None, got {result}"


def test_detect_offer_id_non_numeric():
    result = detect_offer_id(Path("offer_abc_shopping_list.xlsx"))
    assert result is None, f"Expected None, got {result}"


def test_discover_xlsx_files_order():
    """discover_xlsx_files returns files sorted newest mtime first."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        older = root / "older.xlsx"
        newer = root / "newer.xlsx"
        older.touch()
        newer.touch()
        # Set older file's mtime to 10 seconds in the past
        old_time = time.time() - 10
        os.utime(older, (old_time, old_time))
        # newer.xlsx keeps the current mtime (more recent)
        result = discover_xlsx_files(root)
        assert len(result) == 2, f"Expected 2 files, got {len(result)}"
        assert result[0].name == "newer.xlsx", f"Expected newer.xlsx first, got {result[0].name}"
        assert result[1].name == "older.xlsx", f"Expected older.xlsx second, got {result[1].name}"


def test_discover_xlsx_files_only_xlsx():
    """discover_xlsx_files returns only .xlsx files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "data.xlsx").touch()
        (root / "notes.txt").touch()
        (root / "other.csv").touch()
        result = discover_xlsx_files(root)
        assert len(result) == 1, f"Expected 1 file, got {len(result)}"
        assert result[0].name == "data.xlsx"


def test_discover_xlsx_files_empty():
    """discover_xlsx_files returns empty list when no xlsx files exist."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        result = discover_xlsx_files(root)
        assert result == [], f"Expected empty list, got {result}"


def test_wizard_state_defaults():
    """WizardState default values match spec."""
    state = WizardState()
    assert state.strategy == "ilp-optimal", f"Expected 'ilp-optimal', got {state.strategy!r}"
    assert state.xlsx_path is None, f"Expected None, got {state.xlsx_path!r}"
    assert state.offer_id is None, f"Expected None, got {state.offer_id!r}"
    assert state.boxes == [], f"Expected [], got {state.boxes!r}"
    assert state.result is None, f"Expected None, got {state.result!r}"


def run_tests():
    test_detect_offer_id_standard()
    test_detect_offer_id_single_digit()
    test_detect_offer_id_non_matching_name()
    test_detect_offer_id_non_numeric()
    test_discover_xlsx_files_order()
    test_discover_xlsx_files_only_xlsx()
    test_discover_xlsx_files_empty()
    test_wizard_state_defaults()
    print("All tests passed.")


if __name__ == "__main__":
    run_tests()
