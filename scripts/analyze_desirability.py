#!/usr/bin/env python3
"""
Item desirability analysis from historical mystery box packing decisions.

Mines ~700+ manually-packed boxes across 42+ offers to quantify per-item
desirability from human packing decisions. Computes allocation rates, runs
exploratory statistics (distribution, correlations, category breakdowns),
and fits a simple OLS regression to identify items that are more or less
desirable than their features predict.

Usage:
    python3 scripts/analyze_desirability.py                      # full analysis, Tier A
    python3 scripts/analyze_desirability.py --only-offers 64-106 # explicit range
    python3 scripts/analyze_desirability.py --csv --no-plots     # export data, skip visuals
    python3 scripts/analyze_desirability.py --min-appearances 5  # stricter filtering
"""

import argparse
import csv
import logging
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Suppress noise from allocator library and paramiko
logging.getLogger("paramiko").setLevel(logging.WARNING)
logging.getLogger("allocator.categorizer").setLevel(logging.ERROR)
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

import numpy as np
from scipy import stats as sp_stats

from allocator.box_parser import classify_box
from allocator.config import CATEGORY_FRUIT, CATEGORY_VEGETABLES, DONATION_IDENTIFIERS
from allocator.db import fetch_offer_items
from allocator.excel_io import read_overage_from_xlsx
from compare import (
    _find_xlsx_path,
    build_item_lookup,
    load_historical_csv,
    load_summary,
    read_xlsx_pack_overrides,
)

CLEANED_DIR = Path(__file__).parent.parent / "cleaned"
DIAGNOSTICS_DIR = Path(__file__).parent.parent / "diagnostics"

# ANSI colours
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_RED = "\033[31m"
_CYAN = "\033[36m"
_DIM = "\033[90m"
_BOLD = "\033[1m"
_RESET = "\033[0m"


# ---------------------------------------------------------------------------
# Offer discovery
# ---------------------------------------------------------------------------

def _offer_tier(offer_id: int) -> str:
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
            ids.add(int(f.stem.split("_")[1]))
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


# ---------------------------------------------------------------------------
# Box exclusion (matches compare.py / diagnose_scoring.py)
# ---------------------------------------------------------------------------

_SKIP_BOX_NAMES = {"Unallocated", "unallocated", "Stock", "stock", "Buffer", "buffer",
                    "Volunteers", "volunteers", "Overs", "overs"}


def _should_exclude_box(box_name: str, offer_meta: dict) -> bool:
    donation_names = set(offer_meta.get("donation_names", []))
    charity_names = set(offer_meta.get("charity_names", []))
    if box_name in (donation_names | charity_names | DONATION_IDENTIFIERS):
        return True
    if box_name in _SKIP_BOX_NAMES:
        return True
    _, _, box_type = classify_box(box_name)
    if box_type in ("donation", "charity", "staff", "skip"):
        return True
    return False


# ---------------------------------------------------------------------------
# Extended item lookup (adds size from DB)
# ---------------------------------------------------------------------------

def build_extended_item_lookup(offer_id: int) -> dict[int, dict]:
    """Build item lookup with size field merged from fetch_offer_items."""
    try:
        pack_overrides = read_xlsx_pack_overrides(offer_id)
    except Exception:
        pack_overrides = {}

    lookup = build_item_lookup(offer_id, price_overrides=pack_overrides or None)

    # Merge size from DB
    try:
        db_items = fetch_offer_items(offer_id)
        size_map = {r["item_id"]: r.get("size") for r in db_items}
        for iid, info in lookup.items():
            info["size"] = size_map.get(iid)
    except Exception:
        for info in lookup.values():
            info["size"] = None

    return lookup


# ---------------------------------------------------------------------------
# Per-offer data collection
# ---------------------------------------------------------------------------

def collect_offer_desirability(offer_id: int, summary: dict) -> dict | None:
    """
    Collect per-item desirability metrics for a single offer.

    Returns dict with:
        items: list of per-item dicts (id, name, allocation_rate, inclusion_rate, etc.)
        n_boxes: number of mystery boxes (excluding charity/donation)
        n_items_with_overage: count of items available for allocation
    """
    box_names, hist_allocs = load_historical_csv(offer_id)
    if not box_names:
        return None

    # Get overage from XLSX
    xlsx_path = _find_xlsx_path(offer_id)
    if xlsx_path is None:
        return None
    try:
        overage = read_overage_from_xlsx(xlsx_path)
    except Exception:
        return None
    if not overage:
        return None

    # Build item lookup with size
    lookup = build_extended_item_lookup(offer_id)
    if not lookup:
        return None

    offer_meta = summary.get("offers", {}).get(str(offer_id), {})

    # Filter to mystery boxes only
    mystery_boxes = [bn for bn in box_names if not _should_exclude_box(bn, offer_meta)]
    if not mystery_boxes:
        return None
    n_boxes = len(mystery_boxes)

    # Compute per-item metrics for items with overage > 0
    items = []
    for item_id, overage_qty in overage.items():
        if overage_qty <= 0:
            continue
        if item_id not in lookup:
            continue

        info = lookup[item_id]
        allocs = hist_allocs.get(item_id, {})

        # Total qty allocated to mystery boxes
        total_mystery_qty = sum(allocs.get(bn, 0) for bn in mystery_boxes)
        # Number of boxes containing this item
        boxes_with_item = sum(1 for bn in mystery_boxes if allocs.get(bn, 0) > 0)

        allocation_rate = total_mystery_qty / overage_qty
        inclusion_rate = boxes_with_item / n_boxes
        avg_qty = total_mystery_qty / boxes_with_item if boxes_with_item > 0 else 0

        is_fruit = info["category_id"] == CATEGORY_FRUIT
        is_veg = info["category_id"] == CATEGORY_VEGETABLES

        items.append({
            "item_id": item_id,
            "offer_id": offer_id,
            "name": info["name"],
            "price": info["price"],
            "category_id": info["category_id"],
            "is_fruit": is_fruit,
            "size": info.get("size"),
            "fungible_group": info.get("fungible_group"),
            "allocation_rate": allocation_rate,
            "inclusion_rate": inclusion_rate,
            "avg_qty_when_included": avg_qty,
            "total_mystery_qty": total_mystery_qty,
            "overage_qty": overage_qty,
            "boxes_with_item": boxes_with_item,
        })

    if not items:
        return None

    return {
        "items": items,
        "n_boxes": n_boxes,
        "n_items_with_overage": len(items),
    }


# ---------------------------------------------------------------------------
# Cross-offer aggregation
# ---------------------------------------------------------------------------

def aggregate_by_item_name(offers_data: list[dict]) -> list[dict]:
    """
    Aggregate item metrics across offers, grouped by item name.

    Returns list of dicts sorted by mean allocation_rate descending.
    """
    by_name = defaultdict(list)
    for od in offers_data:
        for item in od["items"]:
            by_name[item["name"]].append(item)

    aggs = []
    for name, records in by_name.items():
        alloc_rates = [r["allocation_rate"] for r in records]
        incl_rates = [r["inclusion_rate"] for r in records]
        prices = [r["price"] for r in records]
        sizes = [r["size"] for r in records if r["size"] is not None]

        # Use most common category across appearances
        is_fruit = sum(1 for r in records if r["is_fruit"]) > len(records) / 2

        mean_alloc = np.mean(alloc_rates)
        std_alloc = np.std(alloc_rates, ddof=1) if len(alloc_rates) > 1 else 0
        cv = std_alloc / mean_alloc if mean_alloc > 0 else float("inf")

        aggs.append({
            "name": name,
            "n_appearances": len(records),
            "mean_allocation_rate": float(mean_alloc),
            "std_allocation_rate": float(std_alloc),
            "cv_allocation_rate": float(cv),
            "mean_inclusion_rate": float(np.mean(incl_rates)),
            "mean_price": float(np.mean(prices)),
            "is_fruit": is_fruit,
            "mean_size": float(np.mean(sizes)) if sizes else None,
            "fungible_group": records[0]["fungible_group"],
            "offer_ids": sorted(set(r["offer_id"] for r in records)),
        })

    aggs.sort(key=lambda x: x["mean_allocation_rate"], reverse=True)
    return aggs


# ---------------------------------------------------------------------------
# Layer 1: Exploratory statistics
# ---------------------------------------------------------------------------

def compute_distribution_stats(item_aggs: list[dict]) -> dict:
    """Compute distribution statistics on mean allocation rates."""
    rates = np.array([a["mean_allocation_rate"] for a in item_aggs])

    # Bimodality coefficient: BC = (skewness^2 + 1) / kurtosis_excess + 3
    # BC > 0.555 suggests bimodality
    skew = float(sp_stats.skew(rates))
    kurt = float(sp_stats.kurtosis(rates))  # excess kurtosis
    bc = (skew ** 2 + 1) / (kurt + 3) if (kurt + 3) != 0 else 0

    return {
        "n_items": len(rates),
        "mean": float(np.mean(rates)),
        "median": float(np.median(rates)),
        "std": float(np.std(rates, ddof=1)),
        "min": float(np.min(rates)),
        "max": float(np.max(rates)),
        "p25": float(np.percentile(rates, 25)),
        "p75": float(np.percentile(rates, 75)),
        "skewness": skew,
        "kurtosis": kurt,
        "bimodality_coefficient": float(bc),
        "n_zero_alloc": int(np.sum(rates == 0)),
        "n_full_alloc": int(np.sum(rates >= 1.0)),
    }


def compute_correlations(item_aggs: list[dict]) -> dict:
    """Compute correlations between features and allocation metrics."""
    rates = np.array([a["mean_allocation_rate"] for a in item_aggs])
    prices = np.array([a["mean_price"] for a in item_aggs])
    is_fruit = np.array([1.0 if a["is_fruit"] else 0.0 for a in item_aggs])

    results = {}

    # Price vs allocation rate
    r, p = sp_stats.pearsonr(prices, rates)
    results["price_vs_alloc"] = {"r": float(r), "p": float(p)}

    # Fruit vs allocation rate (point-biserial = Pearson with binary var)
    r, p = sp_stats.pearsonr(is_fruit, rates)
    results["fruit_vs_alloc"] = {"r": float(r), "p": float(p)}

    # Size vs allocation rate (items with size data)
    sized = [(a["mean_size"], a["mean_allocation_rate"])
             for a in item_aggs if a["mean_size"] is not None]
    if len(sized) >= 3:
        sizes, sized_rates = zip(*sized)
        r, p = sp_stats.pearsonr(sizes, sized_rates)
        results["size_vs_alloc"] = {"r": float(r), "p": float(p), "n": len(sized)}
    else:
        results["size_vs_alloc"] = {"r": None, "p": None, "n": len(sized)}

    # Category breakdown: fruit vs veg t-test
    fruit_rates = [a["mean_allocation_rate"] for a in item_aggs if a["is_fruit"]]
    veg_rates = [a["mean_allocation_rate"] for a in item_aggs if not a["is_fruit"]]
    if fruit_rates and veg_rates:
        t, p = sp_stats.ttest_ind(fruit_rates, veg_rates)
        results["fruit_veg_ttest"] = {
            "t": float(t), "p": float(p),
            "fruit_mean": float(np.mean(fruit_rates)),
            "veg_mean": float(np.mean(veg_rates)),
            "n_fruit": len(fruit_rates),
            "n_veg": len(veg_rates),
        }

    return results


# ---------------------------------------------------------------------------
# Layer 2: OLS regression
# ---------------------------------------------------------------------------

def fit_item_level_ols(item_aggs: list[dict]) -> dict:
    """
    OLS: allocation_rate ~ price + is_fruit + size + n_appearances.

    Uses numpy.linalg.lstsq. Returns coefficients, R-squared, t-stats,
    and residuals for each item.
    """
    # Build feature matrix — only items with size data
    rows = [a for a in item_aggs if a["mean_size"] is not None]
    if len(rows) < 5:
        # Fall back to model without size
        rows = item_aggs
        has_size = False
    else:
        has_size = True

    n = len(rows)
    y = np.array([r["mean_allocation_rate"] for r in rows])

    if has_size:
        feature_names = ["intercept", "price", "is_fruit", "size", "n_appearances"]
        X = np.column_stack([
            np.ones(n),
            np.array([r["mean_price"] for r in rows]),
            np.array([1.0 if r["is_fruit"] else 0.0 for r in rows]),
            np.array([r["mean_size"] for r in rows]),
            np.array([float(r["n_appearances"]) for r in rows]),
        ])
    else:
        feature_names = ["intercept", "price", "is_fruit", "n_appearances"]
        X = np.column_stack([
            np.ones(n),
            np.array([r["mean_price"] for r in rows]),
            np.array([1.0 if r["is_fruit"] else 0.0 for r in rows]),
            np.array([float(r["n_appearances"]) for r in rows]),
        ])

    # Solve
    beta, residuals_ss, rank, sv = np.linalg.lstsq(X, y, rcond=None)

    y_hat = X @ beta
    resid = y - y_hat
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    # Manual t-stats: beta_i / se(beta_i)
    k = X.shape[1]
    mse = ss_res / (n - k) if n > k else ss_res
    try:
        cov = mse * np.linalg.inv(X.T @ X)
        se = np.sqrt(np.diag(cov))
        t_stats = beta / se
        p_values = [float(2 * sp_stats.t.sf(abs(t), df=n - k)) for t in t_stats]
    except np.linalg.LinAlgError:
        se = np.full(k, np.nan)
        t_stats = np.full(k, np.nan)
        p_values = [np.nan] * k

    # Attach residuals to items
    residual_items = []
    for i, row in enumerate(rows):
        residual_items.append({
            "name": row["name"],
            "mean_allocation_rate": row["mean_allocation_rate"],
            "predicted": float(y_hat[i]),
            "residual": float(resid[i]),
            "n_appearances": row["n_appearances"],
        })
    residual_items.sort(key=lambda x: x["residual"], reverse=True)

    return {
        "n": n,
        "has_size": has_size,
        "feature_names": feature_names,
        "coefficients": {fn: float(b) for fn, b in zip(feature_names, beta)},
        "se": {fn: float(s) for fn, s in zip(feature_names, se)},
        "t_stats": {fn: float(t) for fn, t in zip(feature_names, t_stats)},
        "p_values": {fn: p for fn, p in zip(feature_names, p_values)},
        "r_squared": float(r_squared),
        "residual_items": residual_items,
    }


# ---------------------------------------------------------------------------
# Output: console
# ---------------------------------------------------------------------------

def _sig_stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def print_summary(
    dist: dict,
    corr: dict,
    ols: dict,
    item_aggs: list[dict],
    min_appearances: int,
) -> None:
    filtered = [a for a in item_aggs if a["n_appearances"] >= min_appearances]

    # --- Distribution ---
    print(f"\n{_BOLD}{'='*70}")
    print(f"  ITEM DESIRABILITY ANALYSIS")
    print(f"{'='*70}{_RESET}\n")

    print(f"{_BOLD}Distribution of mean allocation rates{_RESET}")
    print(f"  Items (unique names): {dist['n_items']}")
    print(f"  Mean:   {dist['mean']:.3f}   Median: {dist['median']:.3f}   Std: {dist['std']:.3f}")
    print(f"  Range:  [{dist['min']:.3f}, {dist['max']:.3f}]   IQR: [{dist['p25']:.3f}, {dist['p75']:.3f}]")
    print(f"  Skewness: {dist['skewness']:.3f}   Kurtosis: {dist['kurtosis']:.3f}")

    bc = dist["bimodality_coefficient"]
    bc_color = _GREEN if bc > 0.555 else _YELLOW
    bc_label = "bimodal" if bc > 0.555 else "unimodal"
    print(f"  Bimodality coefficient: {bc_color}{bc:.3f}{_RESET} ({bc_label}, threshold 0.555)")
    print(f"  Zero-allocation items: {dist['n_zero_alloc']}   Full-allocation (>=100%): {dist['n_full_alloc']}")

    # --- Top / Bottom items ---
    print(f"\n{_BOLD}Top 10 most desirable items{_RESET} (min {min_appearances} appearances)")
    print(f"  {'Item':<40} {'Rate':>6} {'Incl%':>6} {'Price':>7} {'Cat':>5} {'N':>3}")
    print(f"  {'-'*40} {'-'*6} {'-'*6} {'-'*7} {'-'*5} {'-'*3}")
    for a in filtered[:10]:
        cat = "fruit" if a["is_fruit"] else "veg"
        color = _GREEN
        print(f"  {color}{a['name']:<40}{_RESET} {a['mean_allocation_rate']:>6.3f} "
              f"{a['mean_inclusion_rate']:>5.1%} ${a['mean_price']/100:>6.2f} "
              f"{cat:>5} {a['n_appearances']:>3}")

    print(f"\n{_BOLD}Top 10 least desirable items{_RESET} (min {min_appearances} appearances)")
    print(f"  {'Item':<40} {'Rate':>6} {'Incl%':>6} {'Price':>7} {'Cat':>5} {'N':>3}")
    print(f"  {'-'*40} {'-'*6} {'-'*6} {'-'*7} {'-'*5} {'-'*3}")
    bottom = sorted(filtered, key=lambda x: x["mean_allocation_rate"])
    for a in bottom[:10]:
        cat = "fruit" if a["is_fruit"] else "veg"
        color = _RED
        print(f"  {color}{a['name']:<40}{_RESET} {a['mean_allocation_rate']:>6.3f} "
              f"{a['mean_inclusion_rate']:>5.1%} ${a['mean_price']/100:>6.2f} "
              f"{cat:>5} {a['n_appearances']:>3}")

    # --- Consistency ---
    print(f"\n{_BOLD}Consistency (coefficient of variation){_RESET} (min {min_appearances} appearances)")
    consistent = sorted(filtered, key=lambda x: x["cv_allocation_rate"])
    inconsistent = sorted(filtered, key=lambda x: x["cv_allocation_rate"], reverse=True)
    print(f"  Most consistent (low CV):")
    for a in consistent[:5]:
        print(f"    {a['name']:<40} CV={a['cv_allocation_rate']:.3f}  rate={a['mean_allocation_rate']:.3f}")
    print(f"  Most variable (high CV):")
    for a in inconsistent[:5]:
        print(f"    {a['name']:<40} CV={a['cv_allocation_rate']:.3f}  rate={a['mean_allocation_rate']:.3f}")

    # --- Correlations ---
    print(f"\n{_BOLD}Correlations{_RESET}")
    for label, key in [
        ("Price vs allocation rate", "price_vs_alloc"),
        ("Fruit vs allocation rate", "fruit_vs_alloc"),
        ("Size vs allocation rate", "size_vs_alloc"),
    ]:
        c = corr.get(key, {})
        r = c.get("r")
        p = c.get("p")
        if r is None:
            print(f"  {label:<35} insufficient data")
        else:
            stars = _sig_stars(p)
            n_note = f" (n={c['n']})" if "n" in c else ""
            print(f"  {label:<35} r={r:+.3f}  p={p:.4f}{stars}{n_note}")

    # Fruit vs veg t-test
    fvt = corr.get("fruit_veg_ttest")
    if fvt:
        stars = _sig_stars(fvt["p"])
        print(f"\n  {_BOLD}Fruit vs Veg category breakdown:{_RESET}")
        print(f"    Fruit mean: {fvt['fruit_mean']:.3f} (n={fvt['n_fruit']})")
        print(f"    Veg mean:   {fvt['veg_mean']:.3f} (n={fvt['n_veg']})")
        print(f"    t={fvt['t']:.3f}  p={fvt['p']:.4f}{stars}")

    # --- OLS regression ---
    print(f"\n{_BOLD}OLS Regression: allocation_rate ~ features{_RESET}")
    print(f"  n={ols['n']}   R²={ols['r_squared']:.4f}")
    size_note = " (with size)" if ols["has_size"] else " (no size — insufficient data)"
    print(f"  Model{size_note}")
    print(f"  {'Feature':<20} {'Coeff':>10} {'SE':>10} {'t':>8} {'p':>8}")
    print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*8} {'-'*8}")
    for fn in ols["feature_names"]:
        coeff = ols["coefficients"][fn]
        se = ols["se"][fn]
        t = ols["t_stats"][fn]
        p = ols["p_values"][fn]
        stars = _sig_stars(p) if not np.isnan(p) else ""
        print(f"  {fn:<20} {coeff:>10.5f} {se:>10.5f} {t:>8.3f} {p:>8.4f}{stars}")

    r2 = ols["r_squared"]
    if r2 < 0.2:
        r2_interp = "Features explain little — desirability is mostly item-specific"
    elif r2 < 0.5:
        r2_interp = "Features explain some variance — mixed signal"
    else:
        r2_interp = "Features explain substantial variance — desirability is feature-predictable"
    print(f"\n  {_DIM}Interpretation: {r2_interp}{_RESET}")

    # --- Residuals (unexpected desirability) ---
    print(f"\n{_BOLD}Most unexpectedly desirable{_RESET} (large positive residual)")
    residuals = ols["residual_items"]
    # Filter to items with enough appearances
    res_filtered = [r for r in residuals
                    if any(a["name"] == r["name"] and a["n_appearances"] >= min_appearances
                           for a in item_aggs)]
    top_res = res_filtered[:10]
    print(f"  {'Item':<40} {'Actual':>7} {'Pred':>7} {'Resid':>7}")
    print(f"  {'-'*40} {'-'*7} {'-'*7} {'-'*7}")
    for r in top_res:
        color = _GREEN
        print(f"  {color}{r['name']:<40}{_RESET} {r['mean_allocation_rate']:>7.3f} "
              f"{r['predicted']:>7.3f} {r['residual']:>+7.3f}")

    print(f"\n{_BOLD}Most unexpectedly undesirable{_RESET} (large negative residual)")
    bottom_res = list(reversed(res_filtered))[:10]
    print(f"  {'Item':<40} {'Actual':>7} {'Pred':>7} {'Resid':>7}")
    print(f"  {'-'*40} {'-'*7} {'-'*7} {'-'*7}")
    for r in bottom_res:
        color = _RED
        print(f"  {color}{r['name']:<40}{_RESET} {r['mean_allocation_rate']:>7.3f} "
              f"{r['predicted']:>7.3f} {r['residual']:>+7.3f}")

    # --- Key questions summary ---
    print(f"\n{_BOLD}{'='*70}")
    print(f"  KEY FINDINGS")
    print(f"{'='*70}{_RESET}\n")

    bc_answer = f"{'Yes' if bc > 0.555 else 'No'} (BC={bc:.3f})"
    print(f"  1. Bimodal staple/niche split?  {bc_answer}")
    print(f"  2. Feature-predictable?          R²={r2:.3f} — {'no, item-specific' if r2 < 0.2 else 'partially' if r2 < 0.5 else 'yes'}")

    size_corr = corr.get("size_vs_alloc", {})
    if size_corr.get("r") is not None:
        print(f"  3. Size correlation?             r={size_corr['r']:+.3f}, p={size_corr['p']:.4f}")
    else:
        print(f"  3. Size correlation?             insufficient data")

    price_corr = corr["price_vs_alloc"]
    print(f"  4. Price as proxy?               r={price_corr['r']:+.3f}, p={price_corr['p']:.4f}")
    print(f"  5. Residual items?               see tables above")
    print(f"  6. Binary or continuous?          {'bimodal → binary likely' if bc > 0.555 else 'continuous → use continuous score'}")
    print()


# ---------------------------------------------------------------------------
# Output: CSV
# ---------------------------------------------------------------------------

def write_csv_output(item_aggs: list[dict], ols: dict) -> Path:
    DIAGNOSTICS_DIR.mkdir(exist_ok=True)
    path = DIAGNOSTICS_DIR / "desirability_items.csv"

    # Build residual lookup
    resid_map = {r["name"]: r for r in ols["residual_items"]}

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "name", "n_appearances", "mean_allocation_rate", "std_allocation_rate",
            "cv_allocation_rate", "mean_inclusion_rate", "mean_price_cents",
            "is_fruit", "mean_size", "fungible_group",
            "predicted_rate", "residual",
        ])
        for a in item_aggs:
            res = resid_map.get(a["name"], {})
            writer.writerow([
                a["name"],
                a["n_appearances"],
                f"{a['mean_allocation_rate']:.4f}",
                f"{a['std_allocation_rate']:.4f}",
                f"{a['cv_allocation_rate']:.4f}",
                f"{a['mean_inclusion_rate']:.4f}",
                f"{a['mean_price']:.0f}",
                1 if a["is_fruit"] else 0,
                f"{a['mean_size']:.1f}" if a["mean_size"] is not None else "",
                a["fungible_group"] or "",
                f"{res['predicted']:.4f}" if "predicted" in res else "",
                f"{res['residual']:.4f}" if "residual" in res else "",
            ])

    return path


# ---------------------------------------------------------------------------
# Output: plots
# ---------------------------------------------------------------------------

def generate_plots(item_aggs: list[dict], ols: dict) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"{_YELLOW}matplotlib not available — skipping plots{_RESET}")
        return

    DIAGNOSTICS_DIR.mkdir(exist_ok=True)

    rates = [a["mean_allocation_rate"] for a in item_aggs]

    # 1. Histogram of allocation rates
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(rates, bins=30, edgecolor="black", alpha=0.7)
    ax.set_xlabel("Mean Allocation Rate")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Item Allocation Rates")
    ax.axvline(np.median(rates), color="red", linestyle="--", label=f"Median={np.median(rates):.3f}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(DIAGNOSTICS_DIR / "desirability_distribution.png", dpi=150)
    plt.close(fig)

    # 2. Price vs allocation rate scatter
    prices = [a["mean_price"] / 100 for a in item_aggs]
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["tab:orange" if a["is_fruit"] else "tab:green" for a in item_aggs]
    ax.scatter(prices, rates, c=colors, alpha=0.6, s=30)
    ax.set_xlabel("Mean Price ($)")
    ax.set_ylabel("Mean Allocation Rate")
    ax.set_title("Price vs Allocation Rate (orange=fruit, green=veg)")
    fig.tight_layout()
    fig.savefig(DIAGNOSTICS_DIR / "desirability_price_scatter.png", dpi=150)
    plt.close(fig)

    # 3. Actual vs predicted
    residuals = ols["residual_items"]
    actual = [r["mean_allocation_rate"] for r in residuals]
    predicted = [r["predicted"] for r in residuals]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(predicted, actual, alpha=0.5, s=20)
    lims = [min(min(predicted), min(actual)), max(max(predicted), max(actual))]
    ax.plot(lims, lims, "--", color="gray", alpha=0.5)
    ax.set_xlabel("Predicted Allocation Rate")
    ax.set_ylabel("Actual Allocation Rate")
    ax.set_title(f"OLS Fit (R²={ols['r_squared']:.3f})")
    fig.tight_layout()
    fig.savefig(DIAGNOSTICS_DIR / "desirability_ols_fit.png", dpi=150)
    plt.close(fig)

    print(f"{_DIM}Plots saved to {DIAGNOSTICS_DIR}/{_RESET}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Item desirability analysis from historical packing")
    parser.add_argument("--only-offers", type=str, default=None,
                        help="Comma-separated offer IDs or ranges (e.g. 64-106)")
    parser.add_argument("--min-appearances", type=int, default=3,
                        help="Minimum offer appearances to include in rankings (default: 3)")
    parser.add_argument("--csv", action="store_true", help="Write CSV to diagnostics/")
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation")
    args = parser.parse_args()

    # Discover offers
    all_ids = discover_offer_ids()
    if args.only_offers:
        wanted = _parse_only_offers(args.only_offers)
        offer_ids = sorted(set(all_ids) & wanted)
    else:
        offer_ids = [oid for oid in all_ids if _offer_tier(oid) == "A"]

    print(f"{_BOLD}Analyzing {len(offer_ids)} offers{_RESET} "
          f"({offer_ids[0]}–{offer_ids[-1]})")

    summary = load_summary()

    # Collect per-offer data
    offers_data = []
    skipped = 0
    for oid in offer_ids:
        result = collect_offer_desirability(oid, summary)
        if result is None:
            skipped += 1
            continue
        offers_data.append(result)
        n_items = result["n_items_with_overage"]
        n_boxes = result["n_boxes"]
        print(f"  {_DIM}Offer {oid}: {n_items} items, {n_boxes} boxes{_RESET}")

    if not offers_data:
        print(f"{_RED}No offers processed successfully.{_RESET}")
        sys.exit(1)

    total_items = sum(d["n_items_with_overage"] for d in offers_data)
    total_boxes = sum(d["n_boxes"] for d in offers_data)
    print(f"\nProcessed {len(offers_data)} offers ({skipped} skipped): "
          f"{total_items} item-offer observations, {total_boxes} mystery boxes")

    # Aggregate
    item_aggs = aggregate_by_item_name(offers_data)
    print(f"Unique items: {len(item_aggs)}")

    # Layer 1: Exploratory stats
    dist = compute_distribution_stats(item_aggs)
    corr = compute_correlations(item_aggs)

    # Layer 2: OLS regression
    ols = fit_item_level_ols(item_aggs)

    # Output
    print_summary(dist, corr, ols, item_aggs, args.min_appearances)

    if args.csv:
        csv_path = write_csv_output(item_aggs, ols)
        print(f"CSV written to {csv_path}")

    if not args.no_plots:
        generate_plots(item_aggs, ols)


if __name__ == "__main__":
    main()
