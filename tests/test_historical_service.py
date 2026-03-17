"""Tests for HistoricalService and HistoricalDataScreen contracts (migrated to pytest).

Verifies behavioral contracts: validate_all() signature, screen methods.
"""

import inspect

from allocator.screens.historical_data import HistoricalDataScreen
from allocator.services.historical_service import HistoricalService


class TestValidateAllSignature:
    def test_accepts_progress_callback(self):
        sig = inspect.signature(HistoricalService.validate_all)
        assert "progress_callback" in sig.parameters

    def test_accepts_cancel_check(self):
        sig = inspect.signature(HistoricalService.validate_all)
        assert "cancel_check" in sig.parameters

    def test_callable(self):
        assert callable(HistoricalService.validate_all)


class TestHistoricalDataScreenMethods:
    def test_run_validation_exists(self):
        assert hasattr(HistoricalDataScreen, "_run_validation")
        assert callable(getattr(HistoricalDataScreen, "_run_validation"))

    def test_action_toggle_filter_exists(self):
        assert hasattr(HistoricalDataScreen, "action_toggle_filter")
        assert callable(getattr(HistoricalDataScreen, "action_toggle_filter"))

    def test_show_check_detail_exists(self):
        assert hasattr(HistoricalDataScreen, "_show_check_detail")
        assert callable(getattr(HistoricalDataScreen, "_show_check_detail"))


class TestValidateAllBehavior:
    def test_immediate_cancel_returns_empty_list(self):
        hs = HistoricalService()
        results = hs.validate_all(
            progress_callback=None,
            cancel_check=lambda: True,
        )
        assert isinstance(results, list)
        assert results == []
