"""
Box name parser for historical mystery box column headers.

Handles the full variety of naming conventions across historical offers:
  - "?Md Name", "?Sm Name", "?Lg Name"
  - "(?) Lg Name", "(?) Md Name"
  - "Sm Name", "Md Unsold #1", "Lg CCI"
  - "Small - Name" (Size - Name pattern)
  - "M Box 1", "L Box 2" (single-letter size codes)
  - "SM 1", "Med1", "Lge CCI", "Sml 2"
  - "Small1", "Small2", "Med 1"
  - "2nd Md", "4th Medium", "Sm 4Sale"
  - "Md+", "Sm+" (+ suffix as noise)
  - "Box 26: Lge CCI", "Box 27: Md Name #1"
"""

import re

from allocator.config import (
    BOX_SIZE_OVERRIDES,
    DONATION_IDENTIFIERS,
    STAFF_IDENTIFIERS,
    STANDALONE_NAME_TO_EMAIL,
)

# Size keyword → normalized tier
_SIZE_MAP = {
    "sm": "small", "sml": "small", "small": "small",
    "md": "medium", "med": "medium", "medium": "medium",
    "lg": "large", "lge": "large", "large": "large",
    # Single-letter codes used in offer 24
    "m": "medium", "l": "large", "s": "small",
}

# Pattern: optional "?" or "(?) " prefix
_QUESTION_PREFIX_RE = re.compile(r"^\(?\s*\?\s*\)?\s*")

# Pattern: "Box NN: rest" prefix (offer 34 transposed trello)
_BOX_NUM_PREFIX_RE = re.compile(r"^Box\s+\d+:\s*", re.IGNORECASE)

# Pattern: ordinal prefix like "2nd ", "4th ", "1st ", "3rd "
_ORDINAL_PREFIX_RE = re.compile(r"^\d+(?:st|nd|rd|th)\s+", re.IGNORECASE)

# Pattern: "Size - Name" (offer 26)
_SIZE_DASH_NAME_RE = re.compile(
    r"^(Small|Sml|Sm|Medium|Med|Md|Large|Lge|Lg)\s*-\s*(.+)$", re.IGNORECASE
)

# Pattern: "M Box N" / "L Box N" (offer 24)
_LETTER_BOX_RE = re.compile(r"^([MSLA-Z])\s+Box\s+(\d+)$", re.IGNORECASE)

# Size keywords as prefix or suffix (with optional number/noise after)

# Order matters: longer prefixes first to avoid partial matches (Sml before Sm, Lge before Lg)
_SIZE_PREFIX_RE = re.compile(
    r"^(Small|Sml|Sm|Medium|Med|Md|Large|Lge|Lg)\+?\s*(.*?)$", re.IGNORECASE
)
_SIZE_SUFFIX_RE = re.compile(
    r"^(.*?)\s+(Small|Sml|Sm|Medium|Med|Md|Large|Lge|Lg)\+?\s*$", re.IGNORECASE
)

# Identifiers that make size classification unreliable
_UNRELIABLE_SIZE_NAMES = DONATION_IDENTIFIERS | STAFF_IDENTIFIERS


def parse_box_name(header: str) -> tuple[str, str | None]:
    """
    Parse a box column header into (cleaned_name, size_tier).

    size_tier is "small", "medium", "large", or None if undetermined.
    cleaned_name has ?, size prefix/suffix, and noise stripped.
    """
    if not header:
        return ("", None)

    name = str(header).strip()
    if not name:
        return ("", None)

    # Check explicit overrides first
    if name in BOX_SIZE_OVERRIDES:
        return (name, BOX_SIZE_OVERRIDES[name])

    # Strip "Box NN: " prefix (offer 34 transposed)
    name = _BOX_NUM_PREFIX_RE.sub("", name).strip()

    # Strip "?" / "(?) " prefix
    name = _QUESTION_PREFIX_RE.sub("", name).strip()

    # Try "Size - Name" pattern first (offer 26: "Small - Mark")
    m = _SIZE_DASH_NAME_RE.match(name)
    if m:
        size_word = m.group(1).lower()
        remainder = m.group(2).strip()
        tier = _SIZE_MAP.get(size_word)
        clean = remainder if remainder else name
        if clean in _UNRELIABLE_SIZE_NAMES:
            return (clean, None)
        return (clean, tier)

    # Try "M Box N" / "L Box N" pattern (offer 24)
    m = _LETTER_BOX_RE.match(name)
    if m:
        letter = m.group(1).upper()
        num = m.group(2)
        tier = _SIZE_MAP.get(letter.lower())
        clean = f"{letter} Box {num}"
        return (clean, tier)

    # Strip ordinal prefix: "2nd Md", "4th Medium"
    stripped = _ORDINAL_PREFIX_RE.sub("", name).strip()
    if stripped != name:
        # Re-parse the stripped version but preserve ordinal in clean name
        _, tier = parse_box_name(stripped)
        return (name, tier)

    # Try size as prefix: "Sm Name", "Md Name", "Lg CCI", "Med1"
    m = _SIZE_PREFIX_RE.match(name)
    if m:
        size_word = m.group(1).lower()
        remainder = m.group(2).strip()
        tier = _SIZE_MAP.get(size_word)
        # If remainder is just a number or empty, use original as clean name
        clean = remainder if remainder and not remainder.isdigit() else name
        # Strip trailing + noise
        clean = clean.rstrip("+").strip()
        if clean in _UNRELIABLE_SIZE_NAMES:
            return (clean, None)
        return (clean, tier)

    # Try size as suffix: "Name Sm", "Name medium"
    m = _SIZE_SUFFIX_RE.match(name)
    if m:
        remainder = m.group(1).strip()
        size_word = m.group(2).lower()
        tier = _SIZE_MAP.get(size_word)
        clean = remainder if remainder else name
        if clean in _UNRELIABLE_SIZE_NAMES:
            return (clean, None)
        return (clean, tier)

    # No size indicator found
    # Check for common size assumptions
    h = name.lower()
    if "market" in h or "mystery" in h or "4-sale" in h or "4sale" in h:
        return (name, "small")

    return (name, None)


def classify_box(header: str) -> tuple[str, str | None, str]:
    """
    Full box classification: (cleaned_name, size_tier, box_type).

    box_type is one of: "merged", "standalone", "charity", "donation",
    "staff", "skip".
    """
    if not header:
        return ("", None, "skip")

    raw = str(header).strip()
    if not raw:
        return ("", None, "skip")

    cleaned, size_tier = parse_box_name(raw)

    # Build set of forms to check against identifiers:
    # raw header, cleaned name, and header with just ? stripped
    q_stripped = _QUESTION_PREFIX_RE.sub("", raw).strip()
    forms = {raw, cleaned, q_stripped}

    # Check STANDALONE_NAME_TO_EMAIL first (these are merged boxes under aliases)
    if cleaned in STANDALONE_NAME_TO_EMAIL:
        return (cleaned, size_tier, "merged")

    # Email → merged
    if "@" in raw:
        if raw in DONATION_IDENTIFIERS:
            return (cleaned, None, "donation")
        if raw in STAFF_IDENTIFIERS:
            return (cleaned, None, "staff")
        return (cleaned, size_tier, "merged")

    # Check all forms against donation/staff identifiers
    # (headers may include ? prefix and/or size prefix before the identifier)
    for form in forms:
        if form in DONATION_IDENTIFIERS:
            return (cleaned, None, "donation")

    for form in forms:
        if form in STAFF_IDENTIFIERS:
            return (cleaned, None, "staff")

    # Check common CCI patterns
    if cleaned.upper() == "CCI" or "CCI" in cleaned.upper().split():
        return (cleaned, size_tier, "donation")

    # Everything else is a standalone customer box
    return (cleaned, size_tier, "standalone")
