# Mystery Manager

We run a fruit & veggie box business. Customers place orders, we buy bulk boxes from farmers, then split & pack produce. This project is a python script to allocate fruit & veggie items from bulk purchase overage to "mystery" boxes bought by customers each week. Mystery boxes are better value than buying piecemeal, but without the latter's flexibility.

## Running

### Primary tools (root)

```bash
python3 run.py <offer_id> <shopping_list.xlsx>                    # full run (TUI + LLM review)
python3 run.py <offer_id> <xlsx> --no-tui --no-llm               # quick run
python3 run.py <offer_id> <xlsx> --no-tui --no-llm -v            # verbose (deal/topup logs)
python3 run.py <offer_id> <xlsx> --no-tui --no-llm --algorithm deal-topup

python3 compare.py                                                # validate vs 42 Tier A offers
python3 compare.py --algorithm deal-topup                         # specific strategy
python3 compare.py --only-offers 55-63                            # Tier B
python3 compare.py --all-strategies                               # full leaderboard
python3 compare.py --detail                                       # per-offer breakdown + detailed JSON
python3 compare.py --csv                                          # write per-box metrics CSV to output/
```

### Library tools (allocator/)

```bash
python3 allocator/clean_history.py                                # clean historical XLSX → CSVs
python3 allocator/clean_history.py --no-older                     # historical/ only
python3 allocator/clean_history.py --llm-extract                  # LLM extraction (run outside Claude Code)
python3 allocator/clean_history.py --llm-extract --llm-method sonnet-low
python3 allocator/fill_workbook.py 106 offer_106_shopping_list.xlsx   # write strategy sheets into XLSX
python3 allocator/benchmark_extraction.py 5                       # benchmark LLM extraction (outside Claude Code)
```

### Utility scripts (scripts/)

```bash
python3 scripts/score_offer.py 106 offer_106_shopping_list.xlsx   # per-offer strategy leaderboard
python3 scripts/diagnose_scoring.py --no-plots                    # penalty breakdown diagnostics
python3 scripts/validate_cleaned.py                               # structural + DB checks on cleaned CSVs
python3 scripts/validate_cleaned.py --no-db                       # offline structural checks only
python3 scripts/validate_cleaned.py --only-offers 22-48           # Tier D only
python3 scripts/validate_prices.py --offers 55,60,74,90           # XLSX vs DB price validation
python3 scripts/standardize_filenames.py                          # dry-run filename normalization
python3 scripts/standardize_filenames.py --apply                  # apply renames
python3 scripts/compare_llm_outputs.py                            # side-by-side LLM extraction comparison
python3 scripts/analyze_offer_values.py                           # per-offer value targets by size tier
python3 scripts/analyze_offer_values.py --only-offers 64-106      # Tier A only
```

There are no tests. `compare.py` is the validation tool — it compares algorithm output against cleaned historical CSVs and prints per-box and aggregate metrics with a composite score. Default run uses Tier A offers only; use `--only-offers` for others.

## Architecture

Pluggable allocation framework with shared infrastructure and swappable strategies.

**Data flow:** XLSX overage + DB items/buyers → `AllocationResult` → strategy fills boxes → charity allocation → tab-delimited output

### Pipeline (in `allocator/allocator.py`)

```
allocate()
  ├── shared: build_items, build_boxes, create AllocationResult
  ├── optional: apply bootstrap_allocations (pre-fill boxes from prior run)
  ├── STRATEGY(result)          ← pluggable (fills box.allocations in place)
  ├── shared: _allocate_charity()
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

Charity allocation (remaining overage to charity toward computed target, then stock) is shared infrastructure, not part of any strategy.

To add a new strategy: create `allocator/strategies/my_strat.py` with a `run(result)` function, then register it in `_REGISTRY` in `allocator/strategies/__init__.py`. Select it with `--algorithm my-strat`.

### Key modules (`allocator/`)

- **`strategies/`** — pluggable allocation strategies. `__init__.py` has the registry; `deal_topup.py` is the default strategy. `_scoring.py` provides shared penalty functions (`value_penalty`, `box_penalty`, `total_penalty`) matching compare.py's composite scoring exactly, so strategies optimise the same objective they're measured on. `_helpers.py` has shared constraint checks and diversity scoring.
- **`models.py`** — `Item`, `MysteryBox`, `CharityBox`, `AllocationResult`, `ExclusionRule`. All prices in cents.
- **`config.py`** — tier definitions, scoring weights, fungible groups, item classifications (`ITEM_CLASSIFICATIONS`), diversity dimension weights (`DIVERSITY_WEIGHTS`), composite scoring constants (`VALUE_SWEET_FROM`, `VALUE_SWEET_TO`, `VALUE_PENALTY_EXPONENT`, penalty multipliers), identifier sets (`DONATION_IDENTIFIERS`, `STAFF_IDENTIFIERS`, `SKIP_COLUMN_IDENTIFIERS`, `STANDALONE_NAME_TO_EMAIL`).
- **`scorer.py`** — deal-topup specific scoring. `prioritize_items_for_deal()` sorts items for deal phase; `score_topup_candidate()` scores top-up additions with hard constraints and soft scoring. (Strategy-level composite scoring lives in `strategies/_scoring.py`.)
- **`db.py`** — SSH tunnel (via paramiko) to Laravel MySQL DB. Singleton `TunnelManager` with reference counting. Supports `DB_SOCKET` env var for Unix socket connections (overrides host/port). All query functions are `@functools.cache`-decorated for within-run deduplication. `fetch_offer_parts_by_name()` for name-based matching.
- **`excel_io.py`** — reads `ID` + `Overage` columns from XLSX; writes tab-delimited output matching `parseMysteryBoxInput()` format.
- **`categorizer.py`** — assigns fungible groups and diversity classifications (sub-category, usage, colour, shape) by item name prefix matching.
- **`tui.py`** — Rich interactive UI for reviewing/editing boxes before allocation.
- **`llm_review.py`** — optional Claude CLI integration for note parsing and post-allocation review.
- **`clean_history.py`** — multi-tier historical data processing. Handles 57 offers across Tiers A–C from `historical/` and `historical/older/`. Discovers files, selects sheets, detects transposed layouts, classifies columns via `box_parser.py`. Tier C uses `name_matcher.py` for LLM-based item matching. `--llm-extract` flag runs extraction for non-standard Tier C/D workbooks via selectable strategy (`--llm-method`, default `haiku-whole`); reuses `benchmark_extraction.STRATEGY_RUNNERS`. Output per method to `cleaned_llm/{method}/`; cache per (offer, method) at `mappings/offer_N_llm_extraction_{method}.json`, with fallback to `benchmark_results/offer_N_{method}.json`.
- **`fill_workbook.py`** — runs all strategies against an offer and writes result sheets into the XLSX. Also imported by `tui.py` for the TUI's fill-workbook command.
- **`benchmark_extraction.py`** — benchmarks LLM extraction strategies for non-standard historical workbooks. Must be run outside Claude Code.
- **`sheet_analyzer.py`** — LLM-based workbook analysis for non-standard historical offers. Sends full workbook content to Sonnet with a Tier A example, gets back structured per-box allocation data. Cached in `mappings/offer_N_llm_extraction.json`.
- **`box_parser.py`** — parses box column headers across all historical naming conventions (`?Sm Name`, `(?) Lg Name`, `Size - Name`, `M Box N`, `Lge Charity`, etc.) into `(cleaned_name, size_tier, box_type)`.
- **`name_matcher.py`** — LLM-based item name → DB ID matching for Tier C offers (no ID column). Exact/prefix match first, then Claude CLI (Haiku) for fuzzy matching. Cached in `mappings/`.
- **`claude_cli.py`** — subprocess wrapper for `claude -p` CLI calls.

### Utility scripts (`scripts/`)

- **`score_offer.py`** — runs all strategies against a single offer, prints per-box metrics and a leaderboard.
- **`diagnose_scoring.py`** — penalty breakdowns, pricing anomaly detection, and visualisations across all historical tiers.
- **`validate_cleaned.py`** — structural integrity, DB consistency, and cross-file checks on cleaned CSVs.
- **`validate_prices.py`** — SUMPRODUCT validation comparing XLSX prices against DB `offer_parts.price`.
- **`standardize_filenames.py`** — renames historical XLSX files to canonical `offer_{N}_shopping_list.xlsx` format.
- **`compare_llm_outputs.py`** — side-by-side comparison of LLM extraction methods with Jaccard similarity and optional Claude investigation.
- **`analyze_offer_values.py`** — per-offer, per-size-tier average box values. Writes `diagnostics/offer_value_targets.json` for Phase 3 training data.

## Database gotchas

- **Always filter soft deletes**: `buys`, `buy_parts`, and `offer_parts` all have `deleted_at`. Every query joining these tables must include `deleted_at IS NULL` — **except** `fetch_offer_parts_by_name(include_deleted=True)` which is used for historical name matching on older offers where parts are soft-deleted but prices are still valid.
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

Composite scores across 42 Tier A offers (2026-03-11). Update when algorithms change by running `python3 compare.py --all-strategies`.

```
Rank  Strategy            Score   Value  Dupes  Diver   Fair   Pref
1.    ilp-optimal          95.3    -0.2   -0.0   -3.8   -0.7   -0.0
2.    local-search         88.7    -2.0   -2.3   -5.6   -1.5   -0.0
3.    round-robin          86.9    -3.6   -2.0   -5.8   -1.7   -0.0
4.    discard-worst        86.4    -2.2   -3.4   -5.6   -2.4   -0.0
5.    deal-topup           81.8    -9.2   -0.4   -5.5   -3.0   -0.0
6.    minmax-deficit       68.9   -23.5   -2.6   -4.7   -0.2   -0.0
7.    greedy-best-fit      68.8   -23.5   -3.6   -3.9   -0.2   -0.0
8.    manual               65.7   -19.3   -3.9   -4.8   -6.3   -0.0
```

Score = 100 minus penalties. Value penalty uses a symmetric power function: `penalty = distance^1.25` where distance is pp from the 114-117% sweet spot (configurable via `VALUE_SWEET_FROM`, `VALUE_SWEET_TO`, `VALUE_PENALTY_EXPONENT`). Dupes: weighted by `max(degree - DUPE_PENALTY_FLOOR, 0)` (see `config.py`). Diver(sity): coverage of sub-categories, usages, colours, shapes across available items; penalty = (1 - score) * 10.0. Fair: std dev of value%. Pref: preference violations.

## Historical data tiers

| Tier | Offers | Count | Has IDs? | Source dir | Notes |
|------|--------|-------|----------|------------|-------|
| A | 64–106 | 42 | Yes | `historical/` | Full algorithm comparison |
| B | 55–63 | 9 | Yes | `historical/older/` | All standalone boxes |
| C | 45–54 | 10 | No (names) | `historical/older/` | Programmatic extraction validated; name matching via cached LLM maps in `mappings/` |
| D | 22–44 | 12 | — | `historical/older/` | `offer_parts` soft-deleted but prices still valid; uses `include_deleted=True` for name matching |

Offers 45–48 and 22–44 have all `offer_parts` soft-deleted in DB, but price data is still usable for historical name matching. Name matcher falls back to cached mappings when DB is unavailable. Cached matches in `mappings/`.

## Reference

- `parseMysteryBoxInput()`: in the Laravel app's `OfferAdminTools` Livewire component
- `downloadShoppingList()`: same component
- Historical data: 74 XLSX files across `historical/` (42) and `historical/older/` (32); 57 produce cleaned CSVs
