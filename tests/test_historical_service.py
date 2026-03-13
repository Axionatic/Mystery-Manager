"""Smoke tests for HistoricalService and HistoricalDataScreen (Plan 02 gaps).

Verifies behavioral contracts required by HIST-03:
  - validate_all() accepts progress_callback and cancel_check parameters
  - _run_validation() worker method exists on HistoricalDataScreen
  - action_toggle_filter() keybinding method exists on HistoricalDataScreen
  - _show_check_detail() drill-down method exists on HistoricalDataScreen

Run with: python3 tests/test_historical_service.py
"""
import inspect
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from allocator.services.historical_service import HistoricalService
from allocator.screens.historical_data import HistoricalDataScreen


# ---------------------------------------------------------------------------
# 5-02-01: validate_all() signature + _run_validation() worker
# ---------------------------------------------------------------------------

def test_validate_all_accepts_progress_callback():
    """validate_all() must accept a progress_callback keyword argument (HIST-03)."""
    sig = inspect.signature(HistoricalService.validate_all)
    assert "progress_callback" in sig.parameters, (
        f"validate_all() missing 'progress_callback' parameter. "
        f"Got parameters: {list(sig.parameters)}"
    )


def test_validate_all_accepts_cancel_check():
    """validate_all() must accept a cancel_check keyword argument (HIST-03)."""
    sig = inspect.signature(HistoricalService.validate_all)
    assert "cancel_check" in sig.parameters, (
        f"validate_all() missing 'cancel_check' parameter. "
        f"Got parameters: {list(sig.parameters)}"
    )


def test_validate_all_callable():
    """validate_all is a callable method on HistoricalService (HIST-03)."""
    assert callable(HistoricalService.validate_all), (
        "HistoricalService.validate_all must be callable"
    )


def test_run_validation_worker_exists():
    """_run_validation() worker method must exist on HistoricalDataScreen (HIST-03)."""
    assert hasattr(HistoricalDataScreen, "_run_validation"), (
        "HistoricalDataScreen must have a '_run_validation' method"
    )
    assert callable(getattr(HistoricalDataScreen, "_run_validation")), (
        "HistoricalDataScreen._run_validation must be callable"
    )


# ---------------------------------------------------------------------------
# 5-02-02: action_toggle_filter() keybinding method
# ---------------------------------------------------------------------------

def test_action_toggle_filter_exists():
    """action_toggle_filter() keybinding method must exist on HistoricalDataScreen (HIST-03)."""
    assert hasattr(HistoricalDataScreen, "action_toggle_filter"), (
        "HistoricalDataScreen must have an 'action_toggle_filter' method"
    )
    assert callable(getattr(HistoricalDataScreen, "action_toggle_filter")), (
        "HistoricalDataScreen.action_toggle_filter must be callable"
    )


# ---------------------------------------------------------------------------
# 5-02-03: _show_check_detail() drill-down method
# ---------------------------------------------------------------------------

def test_show_check_detail_exists():
    """_show_check_detail() drill-down method must exist on HistoricalDataScreen (HIST-03)."""
    assert hasattr(HistoricalDataScreen, "_show_check_detail"), (
        "HistoricalDataScreen must have a '_show_check_detail' method"
    )
    assert callable(getattr(HistoricalDataScreen, "_show_check_detail")), (
        "HistoricalDataScreen._show_check_detail must be callable"
    )


# ---------------------------------------------------------------------------
# Bonus: validate_all() can be called with no-op callbacks (no DB needed)
# ---------------------------------------------------------------------------

def test_validate_all_callable_with_noop_callbacks_no_db():
    """validate_all() accepts callback signatures without crashing on import/wiring.

    Does not actually run DB queries -- just verifies the method can be
    called with None callbacks and the cancel_check fires immediately to
    return an empty list without touching the DB.
    """
    hs = HistoricalService()
    # Cancel immediately on first offer -- verify we get back an empty list
    # without needing a live DB connection.
    results = hs.validate_all(
        progress_callback=None,
        cancel_check=lambda: True,  # cancel before first offer
    )
    assert isinstance(results, list), (
        f"validate_all() must return a list; got {type(results)}"
    )
    # With immediate cancel, should return [] (loop breaks before appending)
    assert results == [], (
        f"validate_all() with immediate cancel_check should return []; got {results}"
    )


def run_tests():
    test_validate_all_accepts_progress_callback()
    test_validate_all_accepts_cancel_check()
    test_validate_all_callable()
    test_run_validation_worker_exists()
    test_action_toggle_filter_exists()
    test_show_check_detail_exists()
    test_validate_all_callable_with_noop_callbacks_no_db()
    print("All tests passed.")


if __name__ == "__main__":
    run_tests()
