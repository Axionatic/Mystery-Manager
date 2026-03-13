"""Allocation workflow service -- wraps build_boxes_from_db() and allocate()."""
from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class AllocationService:
    """Provides blocking allocation operations for the TUI wizard.

    All methods are synchronous. Callers must run them in a thread
    worker (@work(thread=True)) to avoid blocking the Textual event loop.
    """

    def load_boxes(self, offer_id: int) -> list:
        """Load mystery boxes for offer_id from DB.

        Returns list[MysteryBox]. Raises on DB error (caller should catch).
        """
        from allocator.allocator import build_boxes_from_db
        logger.debug("Loading boxes for offer %d", offer_id)
        return build_boxes_from_db(offer_id)

    def run_allocation(
        self,
        offer_id: int,
        xlsx_path: Path,
        boxes: list,
        strategy: str,
    ) -> object:
        """Run the full allocation pipeline.

        Returns AllocationResult. Raises on allocation error (caller catches).
        Uses exit_on_error=False in the worker -- exceptions propagate as
        WorkerState.ERROR, not app crashes.
        """
        from allocator.allocator import allocate
        logger.debug(
            "Running allocation: offer=%d strategy=%s xlsx=%s",
            offer_id, strategy, xlsx_path
        )
        return allocate(
            offer_id=offer_id,
            xlsx_path=xlsx_path,
            boxes=boxes,
            strategy=strategy,
        )
