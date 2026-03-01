"""
Optional LLM review pass using Claude CLI.

Two uses:
A. Note parsing (pre-allocation): Parse free-text buyer notes into exclusion rules
B. Allocation review (post-allocation): Check quality and suggest swaps
"""

import json
import logging
import subprocess

from allocator.models import AllocationResult, ExclusionRule, MysteryBox

logger = logging.getLogger(__name__)


def call_claude_cli(
    prompt: str,
    timeout: int = 120,
    model: str = "sonnet",
    output_format: str = "text",
) -> str | None:
    """Call Claude CLI with the given prompt."""
    try:
        cmd = ["claude", "-p", "--model", model]
        if output_format != "text":
            cmd.extend(["--output-format", output_format])

        result = subprocess.run(
            cmd,
            input=prompt,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        if result.returncode != 0:
            logger.warning(f"Claude CLI error: {result.stderr}")
            return None

        return result.stdout.strip()

    except subprocess.TimeoutExpired:
        logger.warning("Claude CLI timed out")
        return None
    except FileNotFoundError:
        logger.error("Claude CLI not found")
        return None
    except Exception as e:
        logger.warning(f"Claude CLI failed: {e}")
        return None


def parse_buyer_notes(boxes: list[MysteryBox]) -> list[tuple[int, list[ExclusionRule]]]:
    """
    Use LLM to parse free-text buyer notes into exclusion rules.

    Returns list of (box_index, [ExclusionRule]) tuples for boxes that
    have parseable notes.
    """
    notes_to_parse = []
    for i, box in enumerate(boxes):
        if box.notes and box.notes.strip() and box.notes.strip() != "-":
            notes_to_parse.append((i, box.name, box.notes.strip()))

    if not notes_to_parse:
        return []

    # Build prompt
    notes_text = "\n".join(
        f"Box {i} ({name}): \"{note}\""
        for i, name, note in notes_to_parse
    )

    prompt = f"""Parse these mystery box buyer notes into item exclusion rules.
Each note may contain requests like "no mushrooms", "no capsicum", etc.
Only extract concrete item exclusion requests. Ignore general comments like
"thank you", "can't wait", etc.

Notes:
{notes_text}

Respond with ONLY a JSON array. Each element should be:
{{"box_index": N, "exclusions": ["pattern1", "pattern2"]}}

If a note has no exclusion requests, omit it from the array.
Return an empty array [] if no notes contain exclusion requests.
"""

    response = call_claude_cli(prompt, timeout=60)
    if not response:
        return []

    try:
        # Extract JSON from response (may have markdown fences)
        json_str = response
        if "```" in json_str:
            start = json_str.find("[")
            end = json_str.rfind("]") + 1
            json_str = json_str[start:end]

        parsed = json.loads(json_str)
        results = []
        for entry in parsed:
            idx = entry["box_index"]
            rules = [
                ExclusionRule(pattern=p, source="note")
                for p in entry.get("exclusions", [])
            ]
            if rules:
                results.append((idx, rules))
        return results
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        logger.warning(f"Failed to parse LLM response: {e}")
        return []


def review_allocation(result: AllocationResult) -> str | None:
    """
    Use LLM to review the completed allocation for quality issues.

    Returns review text with suggestions, or None.
    """
    # Build allocation summary for review
    lines = [f"Offer {result.offer_id} allocation review:"]

    for box in result.boxes:
        value = result.box_value(box)
        items_desc = []
        for item_id, qty in sorted(box.allocations.items()):
            if qty > 0 and item_id in result.items:
                item = result.items[item_id]
                items_desc.append(f"  {item.name} x{qty} (${item.price*qty/100:.2f})")

        pref = f" [{box.preference}]" if box.preference else ""
        lines.append(
            f"\n{box.name} ({box.tier}{pref}) - ${value/100:.2f} target ${box.target_value/100:.2f}:"
        )
        lines.extend(items_desc)

    for charity in result.charity:
        value = result.charity_value(charity)
        items_desc = []
        for item_id, qty in sorted(charity.allocations.items()):
            if qty > 0 and item_id in result.items:
                item = result.items[item_id]
                items_desc.append(f"  {item.name} x{qty}")
        lines.append(f"\n{charity.name} (charity) - ${value/100:.2f} target ${charity.target_value/100:.2f}:")
        lines.extend(items_desc)

    allocation_text = "\n".join(lines)

    prompt = f"""Review this mystery box allocation for quality issues.

{allocation_text}

Check for:
1. Boxes with too many of the same type (e.g. 5 types of apple)
2. Boxes with poor variety (all fruit, no veg, or vice versa)
3. Boxes significantly over or under target value
4. Any other quality concerns

If the allocation looks reasonable, say "Allocation looks good."
If there are issues, suggest specific swaps (e.g. "Move 2x Tomatoes from Box A to Box B, replace with Capsicum").

Keep response concise - max 10 lines.
"""

    return call_claude_cli(prompt, timeout=90)
