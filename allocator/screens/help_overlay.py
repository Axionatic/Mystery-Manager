"""Help overlay modal and HelpMixin for context-sensitive help in every screen.

HelpMixin -- a mixin class that provides action_help() to any Screen subclass.
Each screen defines HELP_TEXT (and optionally HELP_TITLE) as class constants.
The mixin reads these and pushes HelpOverlayScreen when the user presses ?.

HelpOverlayScreen -- a ModalScreen that displays screen-specific help text,
with a G shortcut to open the Glossary (Plan 02) and Escape to close.

Note: The G shortcut for the Glossary will raise ImportError until Plan 02
creates allocator/screens/glossary.py. This is intentional -- the binding is
registered now so it appears in the hint, and will work once glossary.py ships.
"""
from __future__ import annotations

from textual.binding import Binding
from textual.screen import ModalScreen


class HelpMixin:
    """Mixin that provides action_help() to any Screen subclass.

    Usage::

        class MyScreen(HelpMixin, Screen):
            HELP_TITLE = "My Screen"
            HELP_TEXT = "This screen does X. Press ..."
            BINDINGS = [
                ...,
                Binding("question_mark", "help", "Help", key_display="?"),
            ]

    The mixin does NOT define BINDINGS itself so that each screen controls its
    own Footer display and can set priority=True when needed (e.g. screens with
    Input or TextArea widgets).
    """

    HELP_TITLE: str = ""
    HELP_TEXT: str = "No help text defined for this screen."

    def action_help(self) -> None:
        """Push the help overlay for this screen."""
        # Lazy import to avoid circular import at module load time.
        from allocator.screens.help_overlay import HelpOverlayScreen

        title = self.HELP_TITLE or self.__class__.__name__
        self.app.push_screen(HelpOverlayScreen(title, self.HELP_TEXT))


class HelpOverlayScreen(ModalScreen):
    """Context-sensitive help modal.

    Displays the title and help text for the calling screen, plus a hint
    to open the Glossary (G) or close (Escape).
    """

    BINDINGS = [
        Binding("escape", "app.pop_screen", "Close"),
        Binding("g", "open_glossary", "Glossary"),
    ]

    def __init__(self, title: str, help_text: str) -> None:
        super().__init__()
        self._title = title
        self._help_text = help_text

    def compose(self):
        from textual.widgets import Label

        yield Label(f"Help: {self._title}", id="help-title")
        yield Label("---" * 20, id="help-rule")
        yield Label(self._help_text, id="help-body")
        yield Label("", id="help-spacer")
        yield Label("Press G to open Glossary  |  Escape to close", id="help-hint")

    def action_open_glossary(self) -> None:
        """Open the Glossary screen (available once Plan 02 ships glossary.py)."""
        from allocator.screens.glossary import GlossaryScreen  # noqa: PLC0415
        self.app.push_screen(GlossaryScreen())
