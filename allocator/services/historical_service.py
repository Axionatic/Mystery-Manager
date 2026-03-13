"""Historical data service -- filesystem scanning for historical offer data.

Provides HistoricalService with discover_offers() and validate_all() for TUI use.

All methods are synchronous. Callers must run them in a thread worker
(@work(thread=True)) to avoid blocking the Textual event loop.

No DB imports at module level -- discover_offers() is pure filesystem I/O.
validate_all() lazily imports scripts.validate_cleaned inside the method body.
"""
from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class HistoricalService:
    """Provides blocking historical data operations for the TUI.

    All methods are synchronous. Callers must run them in a thread
    worker (@work(thread=True)) to avoid blocking the Textual event loop.

    No DB imports -- discover_offers() is pure filesystem I/O.
    """

    @staticmethod
    def _offer_tier(offer_id: int) -> str:
        """Determine data tier from offer ID using CLAUDE.md boundaries.

        CLAUDE.md defines: A=64+, B=55-63, C=45-54, D=22-44.
        Note: validate_cleaned.py uses >=49 for C boundary, but CLAUDE.md is canonical.
        """
        if offer_id >= 64:
            return "A"
        if offer_id >= 55:
            return "B"
        if offer_id >= 45:
            return "C"
        return "D"

    def discover_offers(self) -> list[dict]:
        """Scan filesystem and return all historical offers.

        Scans cleaned/ for mystery CSVs and historical/ + historical/older/
        for XLSX files. Returns a union of all offer IDs found, sorted by
        tier then offer ID descending (Tier A newest-first, then B, C, D).

        Each dict has:
            offer_id: int
            tier: str          -- "A" | "B" | "C" | "D"
            has_mystery_csv: bool
            box_count: int | None  -- from mystery CSV header; None if no CSV
            xlsx_path: str | None  -- path to XLSX if found
            validation_status: str | None  -- always None at discovery time

        Returns:
            List of offer dicts sorted by (tier_order, -offer_id).
        """
        # Project root: service is at allocator/services/historical_service.py
        # so 3 levels up: allocator/services/ -> allocator/ -> project root
        project_root = Path(__file__).resolve().parent.parent.parent

        cleaned_dir = project_root / "cleaned"
        historical_dir = project_root / "historical"
        historical_older_dir = project_root / "historical" / "older"

        # --- Step 1: scan cleaned/ for offer_*_mystery.csv ---
        mystery_data: dict[int, dict] = {}  # offer_id -> {has_mystery_csv, box_count}

        if cleaned_dir.is_dir():
            for csv_path in sorted(cleaned_dir.glob("offer_*_mystery.csv")):
                # Extract offer ID from filename: offer_{N}_mystery.csv
                name = csv_path.stem  # e.g. "offer_106_mystery"
                parts = name.split("_")
                if len(parts) < 3:
                    continue
                try:
                    offer_id = int(parts[1])
                except ValueError:
                    continue

                # Read first line to count boxes (cols minus the 'id' column)
                box_count: int | None = None
                try:
                    with open(csv_path) as fh:
                        first_line = fh.readline().strip()
                    if first_line:
                        cols = first_line.split(",")
                        # Subtract 1 for the 'id' column
                        box_count = max(0, len(cols) - 1)
                except OSError as exc:
                    logger.warning("Could not read %s: %s", csv_path, exc)

                mystery_data[offer_id] = {
                    "has_mystery_csv": True,
                    "box_count": box_count,
                }

        # --- Step 2: scan historical/ and historical/older/ for XLSX ---
        xlsx_data: dict[int, str] = {}  # offer_id -> path string

        for xlsx_dir in (historical_dir, historical_older_dir):
            if not xlsx_dir.is_dir():
                continue
            for xlsx_path in sorted(xlsx_dir.glob("offer_*_shopping_list.xlsx")):
                name = xlsx_path.stem  # e.g. "offer_106_shopping_list"
                parts = name.split("_")
                if len(parts) < 3:
                    continue
                try:
                    offer_id = int(parts[1])
                except ValueError:
                    continue
                # Use relative path string for display
                try:
                    rel = str(xlsx_path.relative_to(project_root))
                except ValueError:
                    rel = str(xlsx_path)
                xlsx_data[offer_id] = rel

        # --- Step 3: union all offer IDs ---
        all_offer_ids = set(mystery_data.keys()) | set(xlsx_data.keys())

        tier_order = {"A": 0, "B": 1, "C": 2, "D": 3}

        offers: list[dict] = []
        for offer_id in all_offer_ids:
            tier = self._offer_tier(offer_id)
            md = mystery_data.get(offer_id, {})
            offers.append({
                "offer_id": offer_id,
                "tier": tier,
                "has_mystery_csv": md.get("has_mystery_csv", False),
                "box_count": md.get("box_count", None),
                "xlsx_path": xlsx_data.get(offer_id, None),
                "validation_status": None,
            })

        # Sort: tier A newest-first, then B, C, D each newest-first within tier
        offers.sort(key=lambda o: (tier_order[o["tier"]], -o["offer_id"]))

        logger.info("discover_offers: found %d offers", len(offers))
        return offers

    def validate_all(
        self,
        progress_callback=None,  # Callable[[int, int, int], None] | None
        cancel_check=None,       # Callable[[], bool] | None
    ) -> list:
        """Run validate_offer() for all offers that have mystery CSVs.

        progress_callback(offer_id, completed, total) called before each offer.
        cancel_check() returns True to stop early (returns partial results).

        Returns list[OfferReport].

        All imports are lazy to avoid DB connections at service import time.
        """
        import sys
        from pathlib import Path

        # Add project root to sys.path so scripts/ is importable
        project_root = str(Path(__file__).resolve().parent.parent.parent)
        if project_root not in sys.path:
            sys.path.insert(0, project_root)

        from scripts.validate_cleaned import validate_offer, _discover_offer_ids

        offer_ids = _discover_offer_ids(include_tier_a=True)  # all 73 mystery CSVs
        total = len(offer_ids)
        reports = []

        for completed, offer_id in enumerate(offer_ids):
            if cancel_check is not None and cancel_check():
                break
            if progress_callback is not None:
                progress_callback(offer_id, completed, total)
            report = validate_offer(offer_id, use_db=True)
            reports.append(report)

        return reports

    @staticmethod
    def dashboard_summary(offers: list[dict]) -> dict:
        """Compute dashboard summary counts from discovered offers.

        Pure function -- no I/O. Takes the output of discover_offers()
        and returns per-tier totals and cleaned counts.

        Args:
            offers: List of offer dicts from discover_offers().

        Returns:
            {
                "total": int,
                "cleaned_total": int,
                "by_tier": {
                    "A": {"total": int, "cleaned": int},
                    "B": {"total": int, "cleaned": int},
                    "C": {"total": int, "cleaned": int},
                    "D": {"total": int, "cleaned": int},
                }
            }
        """
        by_tier: dict[str, dict[str, int]] = {
            "A": {"total": 0, "cleaned": 0},
            "B": {"total": 0, "cleaned": 0},
            "C": {"total": 0, "cleaned": 0},
            "D": {"total": 0, "cleaned": 0},
        }

        for offer in offers:
            tier = offer.get("tier", "D")
            if tier not in by_tier:
                continue
            by_tier[tier]["total"] += 1
            if offer.get("has_mystery_csv"):
                by_tier[tier]["cleaned"] += 1

        total = sum(t["total"] for t in by_tier.values())
        cleaned_total = sum(t["cleaned"] for t in by_tier.values())

        return {
            "total": total,
            "cleaned_total": cleaned_total,
            "by_tier": by_tier,
        }
