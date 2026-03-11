#!/usr/bin/env python3
"""
Per-offer historical value analysis.

Computes per-offer, per-size-tier average box values as % of box price.
Output enables per-offer sweet spot overrides for training data.

Usage:
    python3 scripts/analyze_offer_values.py
    python3 scripts/analyze_offer_values.py --only-offers 64-106
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import logging
logging.getLogger("paramiko").setLevel(logging.WARNING)
logging.getLogger("allocator.categorizer").setLevel(logging.ERROR)
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

from allocator.box_parser import classify_box, infer_box_tier
from allocator.config import BOX_TIERS, DONATION_IDENTIFIERS
from compare import (
    build_item_lookup,
    compute_box_metrics,
    load_historical_csv,
    load_summary,
    read_xlsx_pack_overrides,
)

CLEANED_DIR = Path(__file__).parent.parent / "cleaned"
DIAGNOSTICS_DIR = Path(__file__).parent.parent / "diagnostics"

# ANSI
_BOLD = "\033[1m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_DIM = "\033[90m"
_RESET = "\033[0m"


def offer_tier(offer_id: int) -> str:
    if offer_id >= 64:
        return "A"
    if offer_id >= 55:
        return "B"
    if offer_id >= 49:
        return "C"
    return "D"


def discover_offer_ids() -> list[int]:
    ids = set()
    for f in CLEANED_DIR.glob("offer_*_mystery.csv"):
        try:
            num = int(f.stem.split("_")[1])
            ids.add(num)
        except (ValueError, IndexError):
            pass
    return sorted(ids)


def _parse_only_offers(raw: str) -> set[int]:
    ids = set()
    for part in raw.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            ids.update(range(int(lo), int(hi) + 1))
        else:
            ids.add(int(part))
    return ids


def _should_exclude_box(box_name: str, offer_meta: dict) -> bool:
    donation_names = set(offer_meta.get("donation_names", []))
    charity_names = set(offer_meta.get("charity_names", []))
    if box_name in (donation_names | charity_names | DONATION_IDENTIFIERS):
        return True
    skip = {"Unallocated", "unallocated", "Stock", "stock", "Buffer", "buffer",
            "Volunteers", "volunteers", "Overs", "overs"}
    if box_name in skip:
        return True
    _, _, box_type = classify_box(box_name)
    if box_type in ("donation", "charity", "staff", "skip"):
        return True
    return False


def analyze_offer(offer_id: int, summary: dict) -> dict | None:
    """Compute per-box value_pct for an offer, grouped by size tier."""
    box_names, hist_allocs = load_historical_csv(offer_id)
    if not box_names:
        return None

    try:
        pack_overrides = read_xlsx_pack_overrides(offer_id)
    except Exception:
        pack_overrides = {}

    try:
        item_lookup = build_item_lookup(offer_id, price_overrides=pack_overrides or None)
    except Exception:
        return None

    if not item_lookup:
        return None

    offer_meta = summary.get("offers", {}).get(str(offer_id), {})
    box_names = [bn for bn in box_names if not _should_exclude_box(bn, offer_meta)]

    tier_values: dict[str, list[float]] = defaultdict(list)

    for bn in box_names:
        tier = infer_box_tier(offer_id, bn, summary)
        if tier is None:
            continue
        box_allocs = {}
        for item_id, per_box in hist_allocs.items():
            qty = per_box.get(bn, 0)
            if qty > 0:
                box_allocs[item_id] = qty

        m = compute_box_metrics(bn, box_allocs, item_lookup, tier)
        if m is None:
            continue

        tier_values[tier].append(m["value_pct"])

    if not tier_values:
        return None

    result = {"data_tier": offer_tier(offer_id)}
    for tier, vals in tier_values.items():
        sv = sorted(vals)
        n = len(sv)
        result[tier] = {
            "mean_pct": round(sum(sv) / n, 1),
            "median_pct": round(sv[n // 2], 1),
            "min_pct": round(sv[0], 1),
            "max_pct": round(sv[-1], 1),
            "count": n,
        }

    return result


def main():
    parser = argparse.ArgumentParser(description="Per-offer historical value analysis")
    parser.add_argument("--only-offers", type=str, default=None,
                        help="Comma-separated offer IDs/ranges (e.g. '64-106')")
    args = parser.parse_args()

    all_ids = discover_offer_ids()
    if args.only_offers:
        requested = _parse_only_offers(args.only_offers)
        offer_ids = sorted(set(all_ids) & requested)
    else:
        offer_ids = all_ids

    if not offer_ids:
        print("No matching offers found.")
        sys.exit(1)

    summary = load_summary()
    DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"  Analyzing {len(offer_ids)} offers...")

    offers_data = {}
    # Aggregate per-tier across all offers
    all_tier_values: dict[str, list[float]] = defaultdict(list)
    failed = 0

    for i, oid in enumerate(offer_ids, 1):
        sys.stdout.write(f"\r  Processing offer {oid} [{i}/{len(offer_ids)}]")
        sys.stdout.flush()
        result = analyze_offer(oid, summary)
        if result is None:
            failed += 1
            continue
        offers_data[str(oid)] = result
        for tier in ("small", "medium", "large"):
            if tier in result:
                all_tier_values[tier].extend(
                    [result[tier]["mean_pct"]] * result[tier]["count"]
                )

    sys.stdout.write("\r" + " " * 60 + "\r")
    sys.stdout.flush()

    print(f"  Processed {len(offers_data)} offers ({failed} failed)\n")

    # Print summary table
    print(f"  {_BOLD}Per-Offer Value Targets{_RESET}")
    print(f"    {'Offer':>5} {'Tier':>4} {'Sm Mean':>8} {'Sm Med':>7} {'Sm #':>4} "
          f"{'Md Mean':>8} {'Md Med':>7} {'Md #':>4} "
          f"{'Lg Mean':>8} {'Lg Med':>7} {'Lg #':>4}")
    print(f"    {'─' * 85}")

    for oid_str in sorted(offers_data, key=int):
        d = offers_data[oid_str]
        cols = [f"{int(oid_str):>5}", f"{d['data_tier']:>4}"]
        for tier in ("small", "medium", "large"):
            if tier in d:
                t = d[tier]
                cols.append(f"{t['mean_pct']:>7.1f}%")
                cols.append(f"{t['median_pct']:>6.1f}%")
                cols.append(f"{t['count']:>4}")
            else:
                cols.extend(["     n/a", "    n/a", "   -"])
        print(f"    {' '.join(cols)}")

    # Global tier summary
    print(f"\n  {_BOLD}Global Size Tier Summary{_RESET}")
    print(f"    {'Tier':<8} {'Count':>6} {'Mean':>8} {'Median':>8} {'Std':>7} {'Min':>7} {'Max':>7}")
    print(f"    {'─' * 55}")
    for tier in ("small", "medium", "large"):
        vals = all_tier_values.get(tier, [])
        if not vals:
            continue
        n = len(vals)
        sv = sorted(vals)
        mean = sum(sv) / n
        std = (sum((v - mean) ** 2 for v in sv) / n) ** 0.5
        print(f"    {tier:<8} {n:>6} {mean:>7.1f}% {sv[n//2]:>7.1f}% "
              f"{std:>6.1f}% {sv[0]:>6.1f}% {sv[-1]:>6.1f}%")

    print()

    # Write JSON output
    out = {"offers": offers_data}
    out_path = DIAGNOSTICS_DIR / "offer_value_targets.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  Written to {out_path}")


if __name__ == "__main__":
    main()
