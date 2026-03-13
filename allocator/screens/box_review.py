"""Box review screen and supporting modals for the weekly allocation wizard.

BoxReviewScreen -- shows a DataTable of all auto-detected boxes, with
actions to edit, remove, and add boxes before confirming and advancing
to strategy selection.

BoxEditScreen -- full-screen form for editing a single box's tier,
merged flag, and manual exclusion rules.

ConfirmRemoveBoxModal -- confirmation modal before removing a box.

AddBoxModal -- form for adding an off-system/standalone box.

All heavy allocator imports (models, services, config) are lazy: done
inside method bodies or worker functions so that importing this module
never triggers module-level DB or SSH side effects.
"""
from __future__ import annotations

import copy
import re as _re


_EXCL_VIEW_WIDTH = 50   # visible chars in the Exclusions column
_EXCL_SCROLL_STEP = 8  # chars shifted per left/right keypress

_PREF_LABELS: dict[str | None, str] = {
    "fruit_only": "Fruit only",
    "veg_only":   "Veg only",
    None:         "—",
}


def _sort_key(cell) -> tuple:
    """Universal DataTable sort key: numeric if cell looks numeric, else string."""
    plain = cell.plain if hasattr(cell, "plain") else str(cell)
    stripped = _re.sub(r"[^\d.-]", "", plain)
    try:
        return (0, float(stripped), "")
    except ValueError:
        return (1, 0.0, plain.lower())


def _excl_display(full: str, scroll: int) -> str:
    """Return the visible slice of *full* at *scroll* offset, with '…' indicators."""
    if not full:
        return "—"
    if len(full) <= _EXCL_VIEW_WIDTH:
        return full
    end = min(scroll + _EXCL_VIEW_WIDTH, len(full))
    chunk = full[scroll:end]
    if scroll > 0:
        chunk = "…" + chunk[1:]
    if end < len(full):
        chunk = chunk[:-1] + "…"
    return chunk

from textual import work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.screen import ModalScreen, Screen
from textual.widgets import (
    Button,
    Checkbox,
    DataTable,
    Footer,
    Header,
    Input,
    Label,
    LoadingIndicator,
    Select,
    TextArea,
)
from textual.worker import Worker, WorkerState

from allocator.screens.help_overlay import HelpMixin
from allocator.screens.wizard_state import WizardState


class BoxReviewScreen(HelpMixin, Screen):
    """Wizard step 3: review and edit the detected mystery boxes."""

    BINDINGS = [
        Binding("escape", "app.pop_screen", "Back"),
        Binding("e", "edit_box", "Edit box"),
        Binding("a", "add_box", "Add box"),
        Binding("d", "remove_box", "Remove box"),
        Binding("left",  "scroll_excl_left",  show=False, priority=True),
        Binding("right", "scroll_excl_right", show=False, priority=True),
        Binding("enter", "confirm_boxes", "Confirm boxes ->", priority=True),
        Binding("question_mark", "help", "Help", key_display="?"),
    ]

    HELP_TITLE = "Review Boxes"
    HELP_TEXT = (
        "This screen shows all mystery boxes auto-detected from the DB for this offer.\n\n"
        "Columns:\n"
        "  Email / Name — customer email or standalone box name (prefixed with ?).\n"
        "  Tier — box size tier (small, medium, large). Affects target value.\n"
        "  Merged — whether items ship with the customer's regular order (Yes/No).\n"
        "  Preference — fruit-only, veg-only, or no restriction.\n"
        "  Exclusions — items manually excluded from this box (scroll with Left/Right).\n\n"
        "Actions:\n"
        "  E — edit the highlighted box (tier, merged, preference, exclusions).\n"
        "  A — add an off-system / standalone box.\n"
        "  D — remove the highlighted box from this allocation.\n"
        "  Enter — confirm all boxes and proceed to strategy selection.\n"
        "  Click a column header to sort by that column."
    )

    def __init__(self, state: WizardState) -> None:
        super().__init__()
        self._state = state
        self._sort_column = None
        self._sort_reverse: bool = False
        self._excl_full: dict[str, str] = {}       # row key → full comma-sep exclusions
        self._excl_scroll: int = 0                  # char offset for highlighted row
        self._highlighted_row_key: str | None = None

    def on_show(self) -> None:
        self.app.sub_title = "Weekly Allocation — Review Boxes"

    def compose(self) -> ComposeResult:
        yield Header()
        yield Label("Loading boxes from DB...", id="load-label")
        yield LoadingIndicator(id="load-indicator")
        table = DataTable(id="box-table", cursor_type="row")
        table.disabled = True  # disabled until load completes
        yield table
        yield Label("", id="status-label")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#box-table", DataTable)
        table.add_column("Email / Name", width=35, key="name")
        table.add_column("Tier",         width=8,  key="tier")
        table.add_column("Merged",       width=7,  key="merged")
        table.add_column("Preference",   width=14, key="pref")
        table.add_column("Exclusions",             key="excl")  # auto-width fills remainder
        self._load_boxes()

    @work(thread=True, exit_on_error=False)
    def _load_boxes(self) -> list:
        from allocator.services.allocation_service import AllocationService
        return AllocationService().load_boxes(self._state.offer_id)

    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        if event.state == WorkerState.SUCCESS:
            boxes = event.worker.result
            if not self._state.boxes:
                # Only populate from DB on first visit — preserve user edits on return
                self._state.boxes = boxes
            self._populate_table()
            self.query_one("#load-indicator").display = False
            self.query_one("#load-label").update(
                f"{len(self._state.boxes)} box(es) detected"
            )
            self.query_one("#box-table", DataTable).disabled = False
        elif event.state == WorkerState.ERROR:
            err = str(event.worker.error)
            self.query_one("#load-indicator").display = False
            self.query_one("#load-label").update(
                f"Failed to load boxes: {err}"
            )
            self.app.notify(f"DB error: {err}", severity="error")

    def _populate_table(self) -> None:
        table = self.query_one("#box-table", DataTable)
        table.clear()
        self._excl_full.clear()
        self._excl_scroll = 0
        self._highlighted_row_key = None
        for i, box in enumerate(self._state.boxes):
            full_excl = ", ".join(
                r.pattern for r in box.exclusions if r.source != "preference"
            )
            self._excl_full[str(i)] = full_excl
            table.add_row(
                box.name[:35],
                box.tier,
                "Yes" if box.merged else "No",
                _PREF_LABELS.get(box.preference, box.preference or "—"),
                _excl_display(full_excl, 0),
                key=str(i),
            )

    def on_data_table_header_selected(self, event: DataTable.HeaderSelected) -> None:
        if self._sort_column == event.column_key:
            self._sort_reverse = not self._sort_reverse
        else:
            self._sort_column = event.column_key
            self._sort_reverse = False
        event.data_table.sort(event.column_key, key=_sort_key, reverse=self._sort_reverse)

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        new_key = event.row_key.value
        if self._highlighted_row_key is not None and self._highlighted_row_key != new_key:
            # Reset scroll on previously highlighted row
            table = self.query_one("#box-table", DataTable)
            prev_full = self._excl_full.get(self._highlighted_row_key, "")
            table.update_cell(self._highlighted_row_key, "excl",
                              _excl_display(prev_full, 0), update_width=False)
        self._highlighted_row_key = new_key
        self._excl_scroll = 0

    def _update_excl_cell(self) -> None:
        if self._highlighted_row_key is None:
            return
        full = self._excl_full.get(self._highlighted_row_key, "")
        self.query_one("#box-table", DataTable).update_cell(
            self._highlighted_row_key, "excl",
            _excl_display(full, self._excl_scroll), update_width=False,
        )

    def action_scroll_excl_left(self) -> None:
        self._excl_scroll = max(0, self._excl_scroll - _EXCL_SCROLL_STEP)
        self._update_excl_cell()

    def action_scroll_excl_right(self) -> None:
        full = self._excl_full.get(self._highlighted_row_key or "", "")
        max_scroll = max(0, len(full) - _EXCL_VIEW_WIDTH)
        self._excl_scroll = min(max_scroll, self._excl_scroll + _EXCL_SCROLL_STEP)
        self._update_excl_cell()

    def _edit_box_at(self, idx: int) -> None:
        if idx < 0 or idx >= len(self._state.boxes):
            return
        box_copy = copy.copy(self._state.boxes[idx])
        box_copy.exclusions = list(self._state.boxes[idx].exclusions)

        def _on_edit_done(result) -> None:
            if result is not None:
                self._state.boxes[idx] = result
                self._populate_table()

        self.app.push_screen(BoxEditScreen(box_copy), _on_edit_done)

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        idx = int(event.row_key.value) if event.row_key.value is not None else event.cursor_row
        self._edit_box_at(idx)

    def action_edit_box(self) -> None:
        from textual.coordinate import Coordinate
        table = self.query_one("#box-table", DataTable)
        row_key = table.coordinate_to_cell_key(Coordinate(table.cursor_row, 0)).row_key
        idx = int(row_key.value) if row_key.value is not None else table.cursor_row
        self._edit_box_at(idx)

    def action_add_box(self) -> None:
        def _on_add_done(result) -> None:
            if result is not None:
                self._state.boxes.append(result)
                self._populate_table()

        self.app.push_screen(AddBoxModal(), _on_add_done)

    def action_remove_box(self) -> None:
        from textual.coordinate import Coordinate
        table = self.query_one("#box-table", DataTable)
        row_key = table.coordinate_to_cell_key(Coordinate(table.cursor_row, 0)).row_key
        idx = int(row_key.value) if row_key.value is not None else table.cursor_row
        if idx < 0 or idx >= len(self._state.boxes):
            return
        box = self._state.boxes[idx]

        def _on_confirm(result) -> None:
            if result:
                self._state.boxes.pop(idx)
                self._populate_table()

        self.app.push_screen(ConfirmRemoveBoxModal(box.name), _on_confirm)

    def action_confirm_boxes(self) -> None:
        if not self._state.boxes:
            self.app.notify(
                "No boxes to allocate — load boxes first", severity="warning"
            )
            return
        # Strategy selection comes next (CONTEXT.md: Boxes -> Strategy -> Confirm -> Run)
        from allocator.screens.early_steps import StrategyStep
        self.app.push_screen(StrategyStep(self._state))


class BoxEditScreen(HelpMixin, Screen):
    """Full-screen form for editing a single box's configuration."""

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
        Binding("s", "save", "Save"),
        Binding("question_mark", "help", "Help", key_display="?", priority=True),
    ]

    HELP_TITLE = "Edit Box"
    HELP_TEXT = (
        "Edit a single box's configuration before allocation runs.\n\n"
        "Fields:\n"
        "  Tier — change the box size tier (small/medium/large). "
        "This affects the target value for this box.\n"
        "  Merged — whether items ship with the customer's regular order. "
        "Not shown for standalone boxes (? prefix).\n"
        "  Preference — fruit-only, veg-only, or no preference. "
        "Items violating a preference incur a scoring penalty.\n"
        "  Exclusions — one exclusion pattern per line. Any item whose name contains "
        "the text (case-insensitive) will be excluded from this box.\n\n"
        "Press S to save changes and return to box review. "
        "Press Escape to cancel without saving."
    )

    def __init__(self, box) -> None:  # box: MysteryBox (already a copy)
        super().__init__()
        self._box = box

    def on_show(self) -> None:
        self.app.sub_title = f"Edit Box — {self._box.name[:30]}"

    def compose(self) -> ComposeResult:
        from allocator.config import BOX_TIERS
        tier_options = [(t, t) for t in BOX_TIERS.keys()]
        pref_options = [
            ("No preference", "none"),
            ("Fruit only", "fruit_only"),
            ("Veg only", "veg_only"),
        ]
        pref_value = self._box.preference or "none"
        is_standalone = self._box.name.startswith("?")
        yield Header()
        yield Label(f"Editing: {self._box.name}", id="edit-title")
        yield Label("Tier:")
        yield Select(options=tier_options, value=self._box.tier, allow_blank=False, id="tier-select")
        if not is_standalone:
            yield Label("Merged with regular order:")
            yield Checkbox("Merged", value=self._box.merged, id="merged-check")
        yield Label("Preference:")
        yield Select(options=pref_options, value=pref_value, allow_blank=False, id="pref-select")
        yield Label("Exclusions (one per line, substring match):")
        excl_text = "\n".join(
            r.pattern for r in self._box.exclusions if r.source == "manual"
        )
        yield TextArea(
            excl_text,
            id="exclusions-area",
            placeholder="e.g.\nbanana\napple\n\nAny item whose name contains the text will be excluded.",
        )
        yield Label("", id="edit-status")
        yield Footer()

    def action_save(self) -> None:
        from allocator.config import BOX_TIERS
        tier_val = self.query_one("#tier-select", Select).value
        if tier_val and tier_val in BOX_TIERS:
            self._box.tier = tier_val
            # Leave target_value unchanged — allocate() uses tier name for strategy
            # scoring, and the original target_value from build_boxes_from_db() remains
            # valid.
        if not self._box.name.startswith("?"):
            self._box.merged = self.query_one("#merged-check", Checkbox).value
        pref_val = self.query_one("#pref-select", Select).value
        self._box.preference = None if pref_val == "none" else pref_val
        from allocator.models import ExclusionRule
        raw = self.query_one("#exclusions-area", TextArea).text.strip()
        non_manual = [r for r in self._box.exclusions if r.source != "manual"]
        seen: set[str] = set()
        manual = []
        for line in raw.splitlines():
            pattern = line.strip().title()
            if pattern and pattern not in seen:
                seen.add(pattern)
                manual.append(ExclusionRule(pattern=pattern, source="manual"))
        self._box.exclusions = non_manual + manual
        self.dismiss(self._box)

    def action_cancel(self) -> None:
        excl = self.query_one("#exclusions-area", TextArea)
        if self.focused is excl:
            excl.blur()
        else:
            self.dismiss(None)


class ConfirmRemoveBoxModal(ModalScreen):
    """Confirmation modal before removing a box from the allocation."""

    BINDINGS = [
        Binding("y", "confirm_remove", "Yes, remove"),
        Binding("escape", "app.pop_screen", "Cancel"),
    ]

    def __init__(self, box_name: str) -> None:
        super().__init__()
        self._box_name = box_name

    def compose(self) -> ComposeResult:
        yield Label(
            f"Remove '{self._box_name[:40]}'? Press Y to confirm, Escape to cancel.",
            id="confirm-label",
        )

    def action_confirm_remove(self) -> None:
        self.dismiss(True)


class AddBoxModal(ModalScreen):
    """Simple form for adding an off-system/standalone box."""

    BINDINGS = [Binding("escape", "app.pop_screen", "Cancel")]

    def compose(self) -> ComposeResult:
        from allocator.config import BOX_TIERS
        tier_options = [(t, t) for t in BOX_TIERS.keys()]
        yield Label("Add off-system box")
        yield Label("Name (will appear as '?Name' in output):")
        yield Input(placeholder="Display name", id="box-name-input")
        yield Label("Tier:")
        yield Select(options=tier_options, id="add-tier-select")
        yield Button("Add", id="add-btn")
        yield Label("", id="add-error")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "add-btn":
            self._submit()

    def _submit(self) -> None:
        import os
        from allocator.config import BOX_TIERS
        name = self.query_one("#box-name-input", Input).value.strip()
        if not name:
            self.query_one("#add-error", Label).update("Name is required")
            return
        tier = self.query_one("#add-tier-select", Select).value
        if not tier or tier not in BOX_TIERS:
            self.query_one("#add-error", Label).update("Select a tier")
            return
        from allocator.models import MysteryBox
        tier_cfg = BOX_TIERS[tier]
        target_pct = int(os.environ.get("BOX_TARGET_PCT", "115")) / 100
        box = MysteryBox(
            name=f"?{name}",
            tier=tier,
            merged=False,
            target_value=int(tier_cfg["price"] * target_pct),
        )
        self.dismiss(box)
