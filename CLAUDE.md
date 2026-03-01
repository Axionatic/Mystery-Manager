# Mystery Manager

We run a fruit & veggie box business. Customers place orders, we buy bulk boxes from farmers, then split & pack produce. This project is a python script to allocate fruit & veggie items from bulk purchase overage to "mystery" boxes bought by customers each week. Mystery boxes are better value than buying piecemeal, but without the latter's flexibility.

## Running

```bash
# Full run (TUI + LLM review):
python3 run.py <offer_id> <shopping_list.xlsx>

# Quick run:
python3 run.py <offer_id> <shopping_list.xlsx> --no-tui --no-llm

# Verbose (see deal/topup phase logs):
python3 run.py <offer_id> <shopping_list.xlsx> --no-tui --no-llm -v

# Use a specific allocation strategy:
python3 run.py <offer_id> <shopping_list.xlsx> --no-tui --no-llm --algorithm deal-topup

# Validate algorithm against historical manual allocations (42 Tier A offers):
python3 compare.py
python3 compare.py --algorithm deal-topup

# Specific offers or ranges:
python3 compare.py --only-offers 55-63          # Tier B
python3 compare.py --only-offers 75-86,88-104   # original 29

# Run all strategies and print a leaderboard:
python3 compare.py --all-strategies

# Clean historical XLSX → CSVs (includes older/ by default):
python3 allocator/clean_history.py
python3 allocator/clean_history.py --no-older    # historical/ only

# Validate XLSX prices against DB:
python3 validate_prices.py --offers 55,60,74,90

# Standardize XLSX filenames (dry-run, then --apply):
python3 standardize_filenames.py
python3 standardize_filenames.py --apply
```

There are no tests. `compare.py` is the validation tool — it compares algorithm output against cleaned historical CSVs and prints per-box and aggregate metrics with a composite score. Default run uses Tier A offers only; use `--only-offers` for others.

## Architecture

Pluggable allocation framework with shared infrastructure and swappable strategies.

**Data flow:** XLSX overage + DB items/buyers → `AllocationResult` → strategy fills boxes → CCI allocation → tab-delimited output

### Pipeline (in `allocator/allocator.py`)

```
allocate()
  ├── shared: build_items, build_boxes, create AllocationResult
  ├── optional: apply bootstrap_allocations (pre-fill boxes from prior run)
  ├── STRATEGY(result)          ← pluggable (fills box.allocations in place)
  ├── shared: _allocate_cci()
  └── shared: remaining → stock
```

### Strategies (in `allocator/strategies/`)

A strategy is a callable `(AllocationResult) -> None`. Strategies are registered in `allocator/strategies/__init__.py` and lazy-loaded to avoid circular imports.

**`ilp-optimal`** — ILP via PuLP/CBC. Minimises a composite penalty matching compare.py exactly: convex piecewise-linear value penalty (6 epigraph constraints, no binary vars), full 4D diversity coverage, soft weighted dupe penalty, and MAD fairness proxy (≈stddev). Falls back to deal-topup if PuLP is missing or solver fails.

**`local-search`** — Bootstraps from discard-worst, then iteratively relocates/swaps items between boxes to minimise the same composite penalty. Incremental evaluation: only the 2 affected boxes are recomputed per move. When run via `compare.py --all-strategies`, pre-computed discard-worst allocations are passed in to avoid redundant work.

**`discard-worst`** — Subtractive. (1) **Greedy draft seed**: boxes take turns picking the item that most improves their diversity/dupe profile (ceiling ignored). (2) **Penalty-delta trim**: remove items whose removal most reduces the box's composite penalty — first to ceiling, then toward the 114-117% sweet spot. A soft sole-provider diversity guard makes items harder to remove when they're the only source of a diversity tag in the box. Stops trimming when no removal improves the penalty.

**`deal-topup`** (default) — Three-phase "Deal + Top-up":
1. **Deal round** — deal qty=1 of each item to every eligible box (card-dealing style). Non-fungible items first, then one per fungible group. Stops dealing to a box once it hits target value.
2. **Top-up round** — fill under-target boxes: (a) add new items at qty=1, (b) bump cheap items to qty=2, (c) last resort bump any item +1. Hard ceiling at 130% of target.

**`greedy-best-fit`**, **`round-robin`**, **`minmax-deficit`** — Simpler additive strategies with varying heuristics. See their module docstrings for details.

CCI allocation (remaining overage to charity toward computed target, then stock) is shared infrastructure, not part of any strategy.

To add a new strategy: create `allocator/strategies/my_strat.py` with a `run(result)` function, then register it in `_REGISTRY` in `allocator/strategies/__init__.py`. Select it with `--algorithm my-strat`.

### Key modules

- **`strategies/`** — pluggable allocation strategies. `__init__.py` has the registry; `deal_topup.py` is the default strategy. `_scoring.py` provides shared penalty functions (`value_penalty`, `box_penalty`, `total_penalty`) matching compare.py's composite scoring exactly, so strategies optimise the same objective they're measured on. `_helpers.py` has shared constraint checks and diversity scoring.
- **`models.py`** — `Item`, `MysteryBox`, `CharityBox`, `AllocationResult`, `ExclusionRule`. All prices in cents.
- **`config.py`** — tier definitions, scoring weights, fungible groups, item classifications (`ITEM_CLASSIFICATIONS`), diversity dimension weights (`DIVERSITY_WEIGHTS`), composite scoring constants (penalty rates/multipliers), identifier sets (`DONATION_IDENTIFIERS`, `STAFF_IDENTIFIERS`, `SKIP_COLUMN_IDENTIFIERS`, `STANDALONE_NAME_TO_EMAIL`).
- **`scorer.py`** — deal-topup specific scoring. `prioritize_items_for_deal()` sorts items for deal phase; `score_topup_candidate()` scores top-up additions with hard constraints and soft scoring. (Strategy-level composite scoring lives in `strategies/_scoring.py`.)
- **`db.py`** — SSH tunnel (via paramiko) to Laravel MySQL DB. Singleton `TunnelManager` with reference counting. All query functions are `@functools.cache`-decorated for within-run deduplication. `fetch_offer_parts_by_name()` for name-based matching.
- **`excel_io.py`** — reads `ID` + `Overage` columns from XLSX; writes tab-delimited output matching `parseMysteryBoxInput()` format.
- **`categorizer.py`** — assigns fungible groups and diversity classifications (sub-category, usage, colour, shape) by item name prefix matching.
- **`tui.py`** — Rich interactive UI for reviewing/editing boxes before allocation.
- **`llm_review.py`** — optional Claude CLI integration for note parsing and post-allocation review.
- **`clean_history.py`** — multi-tier historical data processing. Handles 57 offers across Tiers A–C from `historical/` and `historical/older/`. Discovers files, selects sheets, detects transposed layouts, classifies columns via `box_parser.py`. Tier C uses `name_matcher.py` for LLM-based item matching.
- **`box_parser.py`** — parses box column headers across all historical naming conventions (`?Sm Name`, `(?) Lg Name`, `Size - Name`, `M Box N`, `Lge CCI`, etc.) into `(cleaned_name, size_tier, box_type)`.
- **`name_matcher.py`** — LLM-based item name → DB ID matching for Tier C offers (no ID column). Exact/prefix match first, then Claude CLI (Haiku) for fuzzy matching. Cached in `mappings/`.
- **`claude_cli.py`** — subprocess wrapper for `claude -p` CLI calls.

## Database gotchas

- **Always filter soft deletes**: `buys`, `buy_parts`, and `offer_parts` all have `deleted_at`. Every query joining these tables must include `deleted_at IS NULL`.
- **User names are encrypted** (Laravel encryption). Use `email` as the identifier, never `first_name`/`surname`.
- **Prices are in cents** (integer) everywhere — DB, models, config, output.
- SSH key path and connection config in `.env` (set `SSH_ENABLED=true` to use tunnel).
- **Local dev DB**: `.env` points at the local MySQL mirror by default. Sync with the app's pull-db script weekly.

## Conventions

- Fungible groups prevent putting multiple varieties of the same item (e.g. 3 apple types) in one box. The deal phase skips boxes that already have the group; the top-up scorer returns `-inf` for fungible dupes.
- Merged boxes (emails) get added to the customer's existing order. Standalone boxes (`?Name` prefix in output) ship separately.
- `BOX_TIERS` target values are `BOX_TARGET_PCT`% of price (configured in `.env`).
- Category IDs: fruit=2, vegetables=3, giving=6, mystery_boxes=8.

## Strategy Leaderboard

Composite scores across 42 Tier A offers (2026-02-27). Update when algorithms change by running `python3 compare.py --all-strategies`.

```
Rank  Strategy            Score   Value  Dupes  Diver   Fair   Pref
1.    ilp-optimal          97.8    -0.1   -0.0   -1.6   -0.5   -0.0
2.    local-search         87.0    -5.9   -2.2   -2.5   -2.5   -0.0
3.    discard-worst        82.2    -7.8   -3.5   -2.6   -3.8   -0.0
4.    minmax-deficit       60.8   -32.9   -3.6   -2.6   -0.2   -0.0
5.    greedy-best-fit      60.2   -33.0   -4.7   -1.7   -0.4   -0.0
6.    deal-topup           59.4   -32.2   -0.5   -3.3   -4.5   -0.0
7.    manual               58.9   -30.9   -3.9   -2.4   -4.0   -0.0
8.    round-robin          51.6   -40.3   -2.3   -4.3   -1.4   -0.0
```

Score = 100 minus penalties. Value penalty: 114-117% sweet spot, heavy penalty <110% or >120%. Dupes: weighted by `max(degree - DUPE_PENALTY_FLOOR, 0)` (see `config.py`). Diver(sity): coverage of sub-categories, usages, colours, shapes across available items; penalty = (1 - score) * 8.0. Fair: std dev of value%. Pref: preference violations.

## Historical data tiers

| Tier | Offers | Count | Has IDs? | Source dir | Notes |
|------|--------|-------|----------|------------|-------|
| A | 64–106 | 42 | Yes | `historical/` | Full algorithm comparison |
| B | 55–63 | 9 | Yes | `historical/older/` | All standalone boxes |
| C | 49–54 | 6 | No (names) | `historical/older/` | LLM name matching; 58–100% match quality |
| D | 22–48 | 16 | — | `historical/older/` | No DB data (soft-deleted); cannot process |

Offers 45–48 and 22–44 have all `offer_parts` soft-deleted in DB. Tier C LLM matching requires running `clean_history.py` outside Claude Code (nested session restriction). Cached matches in `mappings/`.

## Reference

- `parseMysteryBoxInput()`: in the Laravel app's `OfferAdminTools` Livewire component
- `downloadShoppingList()`: same component
- Historical data: 74 XLSX files across `historical/` (42) and `historical/older/` (32); 57 produce cleaned CSVs
