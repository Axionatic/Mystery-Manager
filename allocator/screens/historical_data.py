"""Historical Data screen -- two-panel offer browser with dashboard summary and
validation runner.

Left panel: DataTable showing all historical offers (offer ID, tier, box count,
CSV status, validation status).  After validation, the Validated column updates
with colour-coded PASS/WARN/FAIL.  Press F to filter to FAIL/WARN only.

Right panel:
  - dashboard state: summary (per-tier cleaned offer counts) + Validate All button
    + offer detail section for the selected row.
  - running state: progress counter with elapsed timer.
  - results state: summary line + per-check detail table for the offer selected
    in the left panel.

Implements HIST-01 (offer browser), HIST-02 (dashboard summary), and
HIST-03 (validation runner).
"""
from __future__ import annotations

import logging
import time

from textual import work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.screen import Screen
from textual.widgets import Button, DataTable, Footer, Header, Label
from textual.containers import Horizontal, Vertical
from textual.worker import Worker, WorkerState, get_current_worker

from allocator.screens.help_overlay import HelpMixin

logger = logging.getLogger(__name__)


def _severity_colour(status: str) -> str:
    """Return a Rich colour name for a validation status string."""
    if status == "PASS":
        return "green"
    if status == "WARN":
        return "yellow"
    if status == "FAIL":
        return "red"
    return "white"


class HistoricalDataScreen(HelpMixin, Screen):
    """Two-panel historical offer browser with dashboard summary and validation runner.

    Left panel: DataTable listing all historical offers with tier, box count,
    CSV status and validation status. Populated by a background worker.  After
    validation, Validated column updates; F key toggles FAIL/WARN filter.

    Right panel: Three states (dashboard / running / results).
      - dashboard: Per-tier cleaned offer counts, Validate All button, and offer
        detail on row selection.
      - running: Progress counter (offer ID, N/total) with elapsed timer.
      - results: Summary line + per-check detail table for the selected offer.
    """

    BINDINGS = [
        Binding("escape", "escape_or_back", "Back"),
        Binding("question_mark", "help", "Help", key_display="?"),
        Binding("f", "toggle_filter", "Toggle Filter", show=False),
    ]

    HELP_TITLE = "Historical Data"
    HELP_TEXT = (
        "The Historical Data screen lets you browse all 75 historical offers "
        "across Tiers A-D (offers 22-108).\n\n"
        "Left panel: Offer browser\n"
        "  Lists all offers with tier, box count, and CSV cleaning status.\n"
        "  Offers are sorted: Tier A newest-first, then B, C, D.\n"
        "  Select a row to see its details in the right panel.\n\n"
        "Right panel: Dashboard / validation detail\n"
        "  Shows per-tier totals and cleaned CSV counts.\n"
        "  Click 'Validate All' to run structural checks on all cleaned CSVs.\n"
        "  Requires a DB connection.\n\n"
        "After validation:\n"
        "  The left panel's Validated column updates with PASS/WARN/FAIL.\n"
        "  Select any offer to see its full check results in the right panel.\n"
        "  Press F to filter the offer list to FAIL/WARN only.\n\n"
        "Escape during validation cancels the run and returns to the dashboard.\n"
        "Escape from dashboard returns to the main menu."
    )

    def __init__(self) -> None:
        super().__init__()
        self._offers: list[dict] = []
        self._dashboard_state: str = "dashboard"  # dashboard | running | results
        self._validation_reports: list = []  # list[OfferReport]
        self._filter_failures_only: bool = False
        self._t0: float = 0.0
        self._timer = None

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="hist-layout"):
            yield DataTable(id="offer-table", classes="loading")
            with Vertical(id="dashboard-panel"):
                # -- dashboard state (initially visible) --
                yield Label("Loading offers...", id="dash-summary")
                yield Button("Validate All", id="validate-btn")
                yield Label("", id="detail-panel")

                # -- running state (hidden) --
                yield Label("Preparing validation...", id="running-label", classes="hidden")
                yield Label("0s elapsed", id="running-elapsed", classes="hidden")

                # -- results state (hidden) --
                yield Label("", id="results-summary", classes="hidden")
                detail_table = DataTable(id="detail-table", classes="hidden")
                detail_table.cursor_type = "row"
                yield detail_table
        yield Footer()

    def on_mount(self) -> None:
        """Set up table columns and launch the background worker."""
        offer_table = self.query_one("#offer-table", DataTable)
        offer_table.cursor_type = "row"
        for label in ("Offer", "Tier", "Boxes", "CSV", "Validated"):
            offer_table.add_column(label, key=label)
        offer_table.disabled = True  # Prevent focus-stealing while loading

        dt = self.query_one("#detail-table", DataTable)
        dt.add_columns("Check", "Severity", "Message")
        dt.disabled = True

        self._load_offers()

    def on_show(self) -> None:
        self.app.sub_title = "Historical Data"

    # -----------------------------------------------------------------------
    # State management
    # -----------------------------------------------------------------------

    def _show_dashboard_state(self, new_state: str) -> None:
        """Switch the right panel between dashboard / running / results states."""
        self._dashboard_state = new_state

        # dashboard-only widgets
        for wid in ("#dash-summary", "#validate-btn"):
            self.query_one(wid).set_class(new_state != "dashboard", "hidden")

        # detail-panel: visible in dashboard AND results states
        self.query_one("#detail-panel").set_class(new_state == "running", "hidden")

        # running state widgets
        for wid in ("#running-label", "#running-elapsed"):
            self.query_one(wid).set_class(new_state != "running", "hidden")

        # results state widgets
        self.query_one("#results-summary").set_class(new_state != "results", "hidden")

        # detail-table: visible in results state (after row selection populates it)
        if new_state != "results":
            self.query_one("#detail-table").add_class("hidden")

        # Disable hidden DataTable to prevent focus-stealing (Textual 6.5 issue)
        self.query_one("#detail-table", DataTable).disabled = (new_state != "results")

    # -----------------------------------------------------------------------
    # Background worker: load offers
    # -----------------------------------------------------------------------

    @work(thread=True, exit_on_error=False, name="load-offers")
    def _load_offers(self) -> None:
        """Load historical offers in a background thread."""
        from allocator.services.historical_service import HistoricalService

        offers = HistoricalService().discover_offers()
        self.app.call_from_thread(self._populate_offer_table, offers)

    # -----------------------------------------------------------------------
    # Background worker: validation
    # -----------------------------------------------------------------------

    @work(thread=True, exit_on_error=False, name="validation")
    def _run_validation(self) -> list:
        """Run validate_all() in a background thread."""
        from allocator.services.historical_service import HistoricalService

        worker = get_current_worker()

        def progress_cb(offer_id: int, completed: int, total: int) -> None:
            self.app.call_from_thread(
                self._update_validation_progress, offer_id, completed, total
            )

        def cancel_check() -> bool:
            return worker.is_cancelled

        return HistoricalService().validate_all(
            progress_callback=progress_cb,
            cancel_check=cancel_check,
        )

    # -----------------------------------------------------------------------
    # Timer tick (updates elapsed label during validation)
    # -----------------------------------------------------------------------

    def _tick(self) -> None:
        """Called every second by set_interval during validation."""
        elapsed = int(time.monotonic() - self._t0)
        self.query_one("#running-elapsed", Label).update(f"{elapsed}s elapsed")

    # -----------------------------------------------------------------------
    # Thread-safe progress callback
    # -----------------------------------------------------------------------

    def _update_validation_progress(
        self, offer_id: int, completed: int, total: int
    ) -> None:
        """Update the running-label widget. Called on main thread via call_from_thread."""
        self.query_one("#running-label", Label).update(
            f"Validating offer {offer_id} ({completed + 1}/{total})..."
        )

    # -----------------------------------------------------------------------
    # Button handler
    # -----------------------------------------------------------------------

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id != "validate-btn":
            return

        if not self.app.db_connected:
            self.query_one("#dash-summary", Label).update(
                "[red]Validation requires a DB connection.[/red]\n"
                "Check the connection status on the main menu."
            )
            return

        self._show_dashboard_state("running")
        self._t0 = time.monotonic()
        self._timer = self.set_interval(1.0, self._tick)
        self._run_validation()

    # -----------------------------------------------------------------------
    # Table population (called from main thread via call_from_thread)
    # -----------------------------------------------------------------------

    def _populate_offer_table(self, offers: list[dict]) -> None:
        """Populate the offer DataTable with discovered offers."""
        self._offers = offers

        offer_table = self.query_one("#offer-table", DataTable)
        offer_table.clear()

        for offer in offers:
            offer_id = offer["offer_id"]
            tier = offer["tier"]
            box_count = str(offer["box_count"]) if offer["box_count"] is not None else "--"
            has_csv = "yes" if offer["has_mystery_csv"] else "no"
            validated = offer["validation_status"] if offer["validation_status"] else "--"

            offer_table.add_row(
                str(offer_id),
                tier,
                box_count,
                has_csv,
                validated,
                key=str(offer_id),
            )

        # Re-enable: allow interaction now data is loaded
        offer_table.disabled = False

        # Update dashboard summary
        self._refresh_dashboard_summary()

    def _refresh_dashboard_summary(self) -> None:
        """Update the dashboard summary label with per-tier counts."""
        from allocator.services.historical_service import HistoricalService

        summary = HistoricalService.dashboard_summary(self._offers)

        total = summary["total"]
        cleaned_total = summary["cleaned_total"]
        by_tier = summary["by_tier"]

        tier_parts = []
        for tier in ("A", "B", "C", "D"):
            t = by_tier[tier]
            tier_parts.append(f"Tier {tier}: {t['cleaned']}/{t['total']}")

        text = (
            f"Historical Offers: {total} total, {cleaned_total} cleaned\n"
            + "  ".join(tier_parts)
        )

        self.query_one("#dash-summary", Label).update(text)

    # -----------------------------------------------------------------------
    # Results display
    # -----------------------------------------------------------------------

    def _show_results(self) -> None:
        """Transition to results state: update offer table statuses and show detail."""
        self._show_dashboard_state("results")
        self._refresh_offer_table_statuses()

        # Aggregate summary in #results-summary
        reports = self._validation_reports
        fails = sum(1 for r in reports if r.status == "FAIL")
        warns = sum(1 for r in reports if r.status == "WARN")
        passes = sum(1 for r in reports if r.status == "PASS")
        n = len(reports)
        parts = []
        if fails:
            parts.append(f"[red]{fails} FAIL[/red]")
        if warns:
            parts.append(f"[yellow]{warns} WARN[/yellow]")
        if passes:
            parts.append(f"[green]{passes} PASS[/green]")
        self.query_one("#results-summary", Label).update(
            f"Validation complete: {n} offers  |  {'  '.join(parts)}"
        )

        # Auto-show detail for the currently selected offer row
        offer_table = self.query_one("#offer-table", DataTable)
        if offer_table.cursor_row is not None and offer_table.row_count > 0:
            try:
                key = list(offer_table.rows.keys())[offer_table.cursor_row]
                offer_id = int(str(key.value))
                self._show_check_detail(offer_id)
            except (IndexError, ValueError, TypeError):
                pass

    def _refresh_offer_table_statuses(self) -> None:
        """Update the Validated column in the offer browser after a validation run."""
        status_map = {r.offer_id: r.status for r in self._validation_reports}
        offer_table = self.query_one("#offer-table", DataTable)

        for offer_id, status in status_map.items():
            colour = _severity_colour(status)
            coloured_status = f"[{colour}]{status}[/{colour}]"
            try:
                offer_table.update_cell(str(offer_id), "Validated", coloured_status)
            except Exception:
                # Row not in table (e.g. offer_id not in offer browser); skip
                pass

        # Also update in-memory _offers list so detail panel reflects status
        for offer in self._offers:
            if offer["offer_id"] in status_map:
                offer["validation_status"] = status_map[offer["offer_id"]]

    def _show_check_detail(self, offer_id: int) -> None:
        """Populate the right panel with rich detail for the given offer.

        Uses #detail-panel for the offer header/metadata and #detail-table
        for the full per-check breakdown.  Both are visible in results state.
        """
        report = next(
            (r for r in self._validation_reports if r.offer_id == offer_id), None
        )
        offer = next((o for o in self._offers if o["offer_id"] == offer_id), None)

        # -- Header / metadata in #detail-panel --
        lines: list[str] = []
        if report:
            colour = _severity_colour(report.status)
            lines.append(
                f"[bold]Offer {offer_id}[/bold]  Tier {report.tier}  "
                f"[{colour}]{report.status}[/{colour}]"
            )
        else:
            lines.append(f"[bold]Offer {offer_id}[/bold]  (no validation data)")

        if offer:
            if offer["has_mystery_csv"]:
                box_str = f"  ({offer['box_count']} boxes)" if offer["box_count"] is not None else ""
                lines.append(f"  CSV: cleaned/offer_{offer_id}_mystery.csv{box_str}")
            else:
                lines.append("  CSV: not cleaned")
            if offer["xlsx_path"]:
                lines.append(f"  XLSX: {offer['xlsx_path']}")

        if report:
            by_sev: dict[str, int] = {}
            for c in report.checks:
                s = c.severity.value
                by_sev[s] = by_sev.get(s, 0) + 1
            parts = []
            for sev in ("PASS", "INFO", "WARN", "FAIL", "SKIP"):
                n = by_sev.get(sev, 0)
                if n:
                    c = _severity_colour(sev)
                    parts.append(f"[{c}]{n} {sev}[/{c}]")
            lines.append(f"  Checks: {len(report.checks)} total  |  {'  '.join(parts)}")

        self.query_one("#detail-panel", Label).update("\n".join(lines))

        # -- Check table in #detail-table --
        if report is None:
            return

        dt = self.query_one("#detail-table", DataTable)
        dt.clear()

        for check in report.checks:
            colour = _severity_colour(check.severity.value)
            dt.add_row(
                check.name,
                f"[{colour}]{check.severity.value}[/{colour}]",
                check.message,
            )

        dt.remove_class("hidden")
        dt.disabled = False

    # -----------------------------------------------------------------------
    # Row selection handler
    # -----------------------------------------------------------------------

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """Show detail for the selected row in the offer table."""
        if event.data_table.id == "offer-table":
            self._on_offer_row_selected(event)

    def _on_offer_row_selected(self, event: DataTable.RowSelected) -> None:
        """Show detail for the selected offer.

        In results state, delegates to _show_check_detail for the full
        right-panel view.  In dashboard state, shows basic metadata.
        """
        offer_id = int(str(event.row_key.value))

        if self._dashboard_state == "results":
            self._show_check_detail(offer_id)
            return

        # Dashboard state: simple metadata in #detail-panel
        offer = next((o for o in self._offers if o["offer_id"] == offer_id), None)
        if offer is None:
            return

        lines: list[str] = [
            f"Offer {offer_id}  Tier {offer['tier']}",
        ]

        if offer["has_mystery_csv"]:
            box_str = f"  ({offer['box_count']} boxes)" if offer["box_count"] is not None else ""
            lines.append(f"CSV: cleaned/offer_{offer_id}_mystery.csv{box_str}")
        else:
            lines.append("CSV: not cleaned")

        if offer["xlsx_path"]:
            lines.append(f"XLSX: {offer['xlsx_path']}")

        validated = offer["validation_status"] if offer["validation_status"] else "--"
        lines.append(f"Validation: {validated}")

        self.query_one("#detail-panel", Label).update("\n".join(lines))

    # -----------------------------------------------------------------------
    # Worker state changed
    # -----------------------------------------------------------------------

    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        """Handle load-offers and validation worker state changes."""
        if event.worker.name == "load-offers":
            if event.state == WorkerState.ERROR:
                err = str(event.worker.error).strip() if event.worker.error else "Unknown error"
                logger.error("load-offers worker failed: %s", err)
                self.query_one("#dash-summary", Label).update(
                    f"Error loading offers: {err}"
                )

        elif event.worker.name == "validation":
            if event.state in (WorkerState.SUCCESS, WorkerState.ERROR, WorkerState.CANCELLED):
                # Stop the timer
                if self._timer is not None:
                    self._timer.stop()
                    self._timer = None

            if event.state == WorkerState.SUCCESS:
                self._validation_reports = event.worker.result or []
                self._show_results()

            elif event.state == WorkerState.ERROR:
                err = str(event.worker.error).strip() if event.worker.error else "Unknown error"
                logger.error("validation worker failed: %s", err)
                self._show_dashboard_state("dashboard")
                self.query_one("#dash-summary", Label).update(
                    f"[red]Validation error:[/red] {err}"
                )

            elif event.state == WorkerState.CANCELLED:
                # Show partial results if any were collected, else back to dashboard
                partial = getattr(event.worker, "result", None) or []
                if partial:
                    self._validation_reports = partial
                    self._show_results()
                else:
                    self._show_dashboard_state("dashboard")

    # -----------------------------------------------------------------------
    # Actions
    # -----------------------------------------------------------------------

    def action_escape_or_back(self) -> None:
        """Cancel validation if running, otherwise pop back to menu."""
        if self._dashboard_state == "running":
            for w in self.workers:
                w.cancel()
            # Don't pop screen -- stay on dashboard after cancel
        else:
            self.app.pop_screen()

    def action_toggle_filter(self) -> None:
        """Toggle filter: show all offers or only FAIL/WARN in the offer table."""
        if self._dashboard_state != "results":
            return
        self._filter_failures_only = not self._filter_failures_only
        status_map = {r.offer_id: r.status for r in self._validation_reports}

        if self._filter_failures_only:
            filtered = [o for o in self._offers
                        if status_map.get(o["offer_id"]) in ("FAIL", "WARN")]
        else:
            filtered = self._offers

        # Re-populate the offer table with filtered list
        offer_table = self.query_one("#offer-table", DataTable)
        offer_table.clear()
        for offer in filtered:
            offer_id = offer["offer_id"]
            tier = offer["tier"]
            box_count = str(offer["box_count"]) if offer["box_count"] is not None else "--"
            has_csv = "yes" if offer["has_mystery_csv"] else "no"
            status = status_map.get(offer_id)
            if status:
                colour = _severity_colour(status)
                validated = f"[{colour}]{status}[/{colour}]"
            else:
                validated = offer["validation_status"] if offer["validation_status"] else "--"
            offer_table.add_row(
                str(offer_id), tier, box_count, has_csv, validated,
                key=str(offer_id),
            )

        filter_note = " (FAIL/WARN only)" if self._filter_failures_only else ""
        reports = self._validation_reports
        fails = sum(1 for r in reports if r.status == "FAIL")
        warns = sum(1 for r in reports if r.status == "WARN")
        passes = sum(1 for r in reports if r.status == "PASS")
        n = len(reports)
        self.query_one("#results-summary", Label).update(
            f"Validation complete: {n} offers -- "
            f"{fails} FAIL, {warns} WARN, {passes} PASS{filter_note}"
        )
