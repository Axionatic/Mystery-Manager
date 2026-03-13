"""Shared state contract for the weekly allocation wizard.

WizardState is passed forward between screens. Helper functions
detect_offer_id and discover_xlsx_files are pure (no DB/IO side effects)
and are unit-tested in tests/test_wizard_helpers.py.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


_OFFER_RE = re.compile(r"offer_(\d+)_shopping_list\.xlsx")


def detect_offer_id(path: Path) -> int | None:
    """Extract offer ID integer from an offer_NNN_shopping_list.xlsx filename.

    Returns the integer ID if the filename matches the canonical pattern,
    or None if the name does not match or the ID part is not numeric.

    Examples:
        detect_offer_id(Path("offer_108_shopping_list.xlsx")) == 108
        detect_offer_id(Path("my_file.xlsx")) is None
    """
    m = _OFFER_RE.match(path.name)
    if m is None:
        return None
    return int(m.group(1))


def discover_xlsx_files(root: Path) -> list[Path]:
    """Return all .xlsx files under root sorted newest-first by mtime.

    Only searches directly in root (non-recursive). Returns an empty
    list if no .xlsx files exist.
    """
    return sorted(
        root.glob("*.xlsx"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )


@dataclass
class WizardState:
    """Mutable state container passed between wizard screens.

    Starts with safe defaults. Screens mutate this object in place
    as the user progresses through the wizard steps.

    Attributes:
        xlsx_path: Path to the shopping list XLSX selected by the user.
        offer_id: Integer offer ID, extracted from xlsx_path or entered manually.
        boxes: List of MysteryBox objects loaded from the DB.
        strategy: Allocation strategy name (default: ilp-optimal).
        result: AllocationResult returned by AllocationService.run_allocation().
    """
    xlsx_path: Path | None = None
    offer_id: int | None = None
    boxes: list = field(default_factory=list)
    strategy: str = "ilp-optimal"
    result: object | None = None
