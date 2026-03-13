"""Glossary screen for Mystery Manager.

GlossaryScreen -- a two-panel term browser. The left panel shows a ListView
of domain terms; the right panel shows the full definition for the highlighted
term. Navigation is keyboard-driven: arrow keys move through the list and
immediately update the definition panel (no Enter required).

Accessible from:
  - Main menu "4. Help" (wired in app.py)
  - Any help overlay via the G shortcut (wired in help_overlay.py)
"""
from __future__ import annotations

from textual.binding import Binding
from textual.containers import Horizontal
from textual.screen import Screen
from textual.widgets import Footer, Header, Label, ListItem, ListView, Static

from allocator.config import BOX_TIERS, VALUE_SWEET_FROM, VALUE_SWEET_TO
from allocator.screens.help_overlay import HelpMixin

# ---------------------------------------------------------------------------
# Glossary data (built at import time from config)
# ---------------------------------------------------------------------------

_sweet = f"{VALUE_SWEET_FROM}-{VALUE_SWEET_TO}%"
_example_price = f"${BOX_TIERS['small']['price']/100:.0f}"

_GLOSSARY: list[tuple[str, str]] = [
    (
        "Composite Score",
        "Overall quality rating for a mystery box allocation. Calculated as 100 "
        "minus the sum of all penalties (value, duplication, diversity, fairness, "
        "preference). Higher is better: 95+ is excellent, 80-90 is good, below 80 "
        "needs review.",
    ),
    (
        "Diversity Score",
        "Measures how well a box covers different types of produce. Scored across "
        "four dimensions: sub-category (e.g. citrus, leafy greens), usage (e.g. "
        "salad, cooking), colour, and shape. Penalty = (1 - score) x 10. A box "
        "with all apples scores poorly; a box with varied fruit and veg scores well.",
    ),
    (
        "Dupe Penalty",
        "Penalty applied when the same or very similar items appear in one box. "
        "Weighted by the degree of duplication above a floor threshold. For example, "
        "two types of apple in one box incurs a dupe penalty because they belong to "
        "the same fungible group.",
    ),
    (
        "Fungible Group",
        "A set of items that count as interchangeable varieties of the same produce. "
        "For example, Granny Smith, Royal Gala, and Fuji apples are all in the "
        "'apple' fungible group. The allocator places at most one item from each "
        "fungible group per box to ensure variety.",
    ),
    (
        "Sweet Spot",
        f"The ideal value percentage range for a mystery box, currently set to "
        f"{_sweet} of the box price. Boxes in the sweet spot receive zero value "
        f"penalty. Outside this range, the value penalty increases as a power "
        f"function of the distance.",
    ),
    (
        "Value %",
        f"The total wholesale value of items in a box, expressed as a percentage of "
        f"the box's retail price. For example, a {_example_price} box containing "
        f"120% worth of produce. The target sweet spot is {_sweet}.",
    ),
]


# ---------------------------------------------------------------------------
# Screen
# ---------------------------------------------------------------------------


class GlossaryScreen(HelpMixin, Screen):
    """Two-panel domain term browser.

    Left panel: ListView of all glossary terms (alphabetical order).
    Right panel: Static widget showing the full definition for the
    currently highlighted term. Definition updates on every arrow-key
    move (ListView.Highlighted), not on Enter/Selected.
    """

    BINDINGS = [
        Binding("escape", "app.pop_screen", "Back"),
        Binding("question_mark", "help", "Help", key_display="?"),
    ]

    HELP_TITLE = "Glossary"
    HELP_TEXT = (
        "Browse definitions of domain terms used throughout Mystery Manager. "
        "Use arrow keys to highlight a term and see its full definition on the right."
    )

    def compose(self):
        yield Header()
        with Horizontal(id="glossary-layout"):
            yield ListView(
                *[
                    ListItem(Label(term), id=f"term-{i}")
                    for i, (term, _) in enumerate(_GLOSSARY)
                ],
                id="glossary-list",
            )
            yield Static("Select a term to see its definition.", id="glossary-detail")
        yield Footer()

    def on_show(self) -> None:
        self.app.sub_title = "Help -- Glossary"

    def on_list_view_highlighted(self, event: ListView.Highlighted) -> None:
        """Update detail panel when a term is highlighted (arrow-key navigation)."""
        if event.item is None:
            return
        item_id = event.item.id  # e.g. "term-0"
        if item_id is None:
            return
        try:
            index = int(item_id.split("-", 1)[1])
        except (IndexError, ValueError):
            return
        if 0 <= index < len(_GLOSSARY):
            _term, definition = _GLOSSARY[index]
            self.query_one("#glossary-detail", Static).update(definition)
