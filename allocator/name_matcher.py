"""
LLM-based item name → DB ID matching for Tier C offers (no ID column).

Uses exact matching first, then Claude CLI (Haiku) for fuzzy matching.
Results are cached in mappings/offer_{N}_name_map.json.
"""

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from allocator.claude_cli import call_claude_cli
from allocator.db import fetch_offer_parts_by_name

logger = logging.getLogger(__name__)

MAPPINGS_DIR = Path(__file__).parent.parent / "mappings"

# Non-F&V supplier prefixes to exclude from matching
_EXCLUDE_PREFIXES = (
    "Baker Boys", "Basketcase", "Honey", "Pollen", "Eddies",
    "Summer Snow", "A large mystery", "A medium mystery", "A small mystery",
)


def _cache_path(offer_id: int) -> Path:
    return MAPPINGS_DIR / f"offer_{offer_id}_name_map.json"


def _load_cache(offer_id: int) -> dict[str, dict]:
    """Load cached name mappings. Returns {xlsx_name: {id, db_name, method}}."""
    path = _cache_path(offer_id)
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def _save_cache(offer_id: int, mappings: dict[str, dict]):
    """Save name mappings to cache."""
    MAPPINGS_DIR.mkdir(parents=True, exist_ok=True)
    path = _cache_path(offer_id)
    with open(path, "w") as f:
        json.dump(mappings, f, indent=2)


def _should_exclude(name: str) -> bool:
    """Check if an item name is non-F&V and should be excluded."""
    for prefix in _EXCLUDE_PREFIXES:
        if name.startswith(prefix):
            return True
    return False


def _exact_match(xlsx_name: str, db_parts: dict[str, dict]) -> dict | None:
    """Try exact or prefix match against DB names."""
    # Exact match
    if xlsx_name in db_parts:
        return {"id": db_parts[xlsx_name]["id"], "db_name": xlsx_name, "method": "exact"}

    # XLSX names are often truncated — try prefix match
    xlsx_clean = xlsx_name.rstrip(".")
    matches = []
    for db_name, info in db_parts.items():
        if db_name.startswith(xlsx_clean) or xlsx_clean.startswith(db_name):
            matches.append((db_name, info))

    if len(matches) == 1:
        db_name, info = matches[0]
        return {"id": info["id"], "db_name": db_name, "method": "prefix"}

    return None


def _llm_match(xlsx_name: str, db_names: list[str]) -> dict | None:
    """Use Claude CLI to fuzzy-match an XLSX item name to DB names."""
    prompt = f"""Match the item name from a spreadsheet to the closest item in the database list.

Spreadsheet item: "{xlsx_name}"

Database items:
{chr(10).join(f'- {n}' for n in db_names)}

Reply with ONLY the exact database item name that best matches, or "NONE" if no good match exists.
Do not include quotes or explanation."""

    result = call_claude_cli(prompt, timeout=30, model="haiku")
    if result is None or result.upper() == "NONE":
        return None

    # Verify the result is actually in our DB list
    result = result.strip().strip('"').strip("'")
    for db_name in db_names:
        if result == db_name:
            return {"db_name": db_name, "method": "llm"}

    # Try case-insensitive match
    result_lower = result.lower()
    for db_name in db_names:
        if db_name.lower() == result_lower:
            return {"db_name": db_name, "method": "llm"}

    logger.warning(f"LLM returned '{result}' which doesn't match any DB name")
    return None


def match_items(
    offer_id: int,
    xlsx_names: list[str],
    price_column: dict[str, float] | None = None,
    force: bool = False,
) -> dict[str, dict]:
    """
    Match XLSX item names to DB offer_part IDs.

    Args:
        offer_id: The offer ID to match against
        xlsx_names: List of item names from the XLSX file
        price_column: Optional {name: price_dollars} for cross-validation
        force: If True, re-run LLM matching even if cache exists

    Returns:
        {xlsx_name: {id, db_name, method, validated?}} for matched items.
        Unmatched items are omitted.
    """
    db_parts = fetch_offer_parts_by_name(offer_id)
    db_names = list(db_parts.keys())

    # Load existing cache
    cache = {} if force else _load_cache(offer_id)

    # Filter out non-F&V items and already-cached items
    to_match = []
    for name in xlsx_names:
        if _should_exclude(name):
            continue
        if name in cache and not force:
            continue
        to_match.append(name)

    # Phase 1: exact/prefix matching
    llm_needed = []
    for name in to_match:
        result = _exact_match(name, db_parts)
        if result:
            cache[name] = result
        else:
            llm_needed.append(name)

    # Phase 2: LLM matching (parallel)
    if llm_needed:
        logger.info(f"Offer {offer_id}: {len(llm_needed)} items need LLM matching")

        def _match_one(xlsx_name: str) -> tuple[str, dict | None]:
            result = _llm_match(xlsx_name, db_names)
            if result:
                result["id"] = db_parts[result["db_name"]]["id"]
            return xlsx_name, result

        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {executor.submit(_match_one, n): n for n in llm_needed}
            for future in as_completed(futures):
                xlsx_name, result = future.result()
                if result:
                    cache[xlsx_name] = result
                else:
                    logger.warning(f"Offer {offer_id}: no match for '{xlsx_name}'")

    # Cross-validate with price column if available
    if price_column:
        for xlsx_name, mapping in cache.items():
            if xlsx_name not in price_column:
                continue
            xlsx_price_cents = round(price_column[xlsx_name] * 100)
            db_name = mapping.get("db_name", "")
            if db_name in db_parts:
                db_price = db_parts[db_name]["price"]
                # Allow 10% tolerance for rounding
                if abs(xlsx_price_cents - db_price) <= max(db_price * 0.1, 5):
                    mapping["price_validated"] = True
                else:
                    mapping["price_validated"] = False
                    mapping["xlsx_price_cents"] = xlsx_price_cents
                    mapping["db_price_cents"] = db_price

    # Save cache
    _save_cache(offer_id, cache)

    return cache
