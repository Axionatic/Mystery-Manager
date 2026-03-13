"""Clean history service -- thin wrapper for TUI use.

Provides CleanHistoryService with run_standard_clean(), run_llm_clean(),
available_llm_methods(), and parse_offer_range() for the TUI screen.

All methods are synchronous. Callers must run them in a thread worker
(@work(thread=True)) to avoid blocking the Textual event loop.

All heavy imports are lazy (inside method bodies).
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class CleanHistoryService:
    """Blocking clean-history operations for the TUI."""

    @staticmethod
    def run_standard_clean(
        include_older: bool = True,
        only_offers: set[int] | None = None,
        progress_callback=None,
        cancel_check=None,
    ) -> dict:
        """Run standard XLSX → CSV cleaning.

        Returns summary dict from clean_all().
        """
        from allocator.clean_history import clean_all

        return clean_all(
            include_older=include_older,
            only_offers=only_offers,
            progress_callback=progress_callback,
            cancel_check=cancel_check,
        )

    @staticmethod
    def run_llm_clean(
        methods: list[str] | None = None,
        force: bool = False,
        only_offers: set[int] | None = None,
        progress_callback=None,
        cancel_check=None,
    ) -> dict:
        """Run LLM extraction on Tier C/D workbooks.

        Returns summary dict from run_llm_extraction().
        """
        from allocator.clean_history import run_llm_extraction

        return run_llm_extraction(
            offer_ids=only_offers,
            force=force,
            methods=methods,
            progress_callback=progress_callback,
            cancel_check=cancel_check,
        )

    @staticmethod
    def available_llm_methods() -> list[str]:
        """Return list of available LLM extraction method names."""
        from allocator.benchmark_extraction import ALL_STRATEGIES

        return list(ALL_STRATEGIES)

    @staticmethod
    def parse_offer_range(spec: str) -> set[int]:
        """Parse a comma-separated offer range spec like '55-63' or '45,50,55-63'.

        Raises ValueError on invalid input.
        """
        result: set[int] = set()
        for part in spec.split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                lo, hi = part.split("-", 1)
                lo_int, hi_int = int(lo.strip()), int(hi.strip())
                if lo_int > hi_int:
                    raise ValueError(f"Invalid range: {part}")
                result.update(range(lo_int, hi_int + 1))
            else:
                result.add(int(part))
        return result
