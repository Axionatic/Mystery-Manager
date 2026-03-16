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

### Tests

```bash
python3 -m pytest                                                 # run full suite (253 tests)
python3 -m pytest tests/test_strategies.py -v                     # single module, verbose
python3 -m pytest -k "test_value_penalty"                         # run by name pattern
```

Tests use synthetic fixtures (no DB required). See `tests/conftest.py` for factory fixtures and config bootstrap.

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

`compare.py` is the primary validation tool — it compares algorithm output against cleaned historical CSVs and prints per-box and aggregate metrics with a composite score. Default run uses Tier A offers only; use `--only-offers` for others.

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

**`ilp-optimal`** — ILP-based multi-objective optimiser via PuLP/CBC. Minimises a composite penalty across value, diversity, dupes, and fairness. Falls back to deal-topup if PuLP is missing or solver fails.

**`local-search`** — Bootstraps from discard-worst, then iteratively relocates and swaps items between boxes to minimise composite penalty.

**`discard-worst`** — Subtractive. Seeds all boxes via greedy draft, then trims items whose removal most reduces penalty.

**`deal-topup`** (default) — Additive. Deals items round-robin to all eligible boxes, then tops up under-target boxes.

**`greedy-best-fit`**, **`round-robin`**, **`minmax-deficit`** — Simpler additive strategies. See module docstrings.

Charity allocation (remaining overage to charity toward computed target, then stock) is shared infrastructure, not part of any strategy.

To add a new strategy: create `allocator/strategies/my_strat.py` with a `run(result)` function, then register it in `_REGISTRY` in `allocator/strategies/__init__.py`. Select it with `--algorithm my-strat`.

### Key modules (`allocator/`)

- **`strategies/`** — pluggable allocation strategies. `__init__.py` has the registry; `deal_topup.py` is the default strategy. `_scoring.py` provides shared penalty functions used by strategies and compare.py. `_helpers.py` has shared constraint checks and diversity scoring.
- **`models.py`** — `Item`, `MysteryBox`, `CharityBox`, `AllocationResult`, `ExclusionRule`. All prices in cents.
- **`config.py`** — tier definitions (from `.env`), identifier sets (from `identifiers.json`), scoring/classification config (from `scoring_config.json`). Key exports: `BOX_TIERS`, `FUNGIBLE_GROUPS`, `ITEM_CLASSIFICATIONS`, `DIVERSITY_WEIGHTS`, composite scoring constants.
- **`scorer.py`** — deal-topup specific scoring. `prioritize_items_for_deal()` sorts items for deal phase; `score_topup_candidate()` scores top-up additions with hard constraints and soft scoring.
- **`db.py`** — SSH tunnel (via paramiko) to MySQL DB. Singleton `TunnelManager` with reference counting. Supports `DB_SOCKET` env var for Unix socket connections (overrides host/port). All query functions are `@functools.cache`-decorated for within-run deduplication. SQL loaded from `queries.json` (gitignored).
- **`excel_io.py`** — reads `ID` + `Overage` columns from XLSX; writes tab-delimited output for import.
- **`categorizer.py`** — assigns fungible groups and diversity classifications (sub-category, usage, colour, shape) by item name prefix matching.
- **`app.py`** — Textual TUI application. Main entry point when `run.py` is called without `--no-tui`. Implements a 5-section main menu with DB status badge and section screens.
- **`screens/`** — TUI screen modules: wizard (early steps, box review, progress/results), strategy comparison, historical data, clean history, glossary, and help overlay.
- **`services/`** — service layer for the TUI: allocation, comparison, historical data, clean history, and DB connectivity services.
- **`tui.py`** — legacy Rich interactive UI (pre-Textual). Still importable but superseded by `app.py`.
- **`llm_review.py`** — optional Claude CLI integration for note parsing and post-allocation review.
- **`clean_history.py`** — multi-tier historical data processing. Handles 57 offers across Tiers A–C from `historical/` and `historical/older/`. Discovers files, selects sheets, detects transposed layouts, classifies columns via `box_parser.py`. Tier C uses `name_matcher.py` for LLM-based item matching. `--llm-extract` flag runs extraction for non-standard Tier C/D workbooks via selectable strategy (`--llm-method`, default `haiku-whole`); reuses `benchmark_extraction.STRATEGY_RUNNERS`. Output per method to `cleaned_llm/{method}/`; cache per (offer, method) at `mappings/offer_N_llm_extraction_{method}.json`, with fallback to `benchmark_results/offer_N_{method}.json`.
- **`fill_workbook.py`** — runs all strategies against an offer and writes result sheets into the XLSX. Also imported by the TUI for the fill-workbook command.
- **`benchmark_extraction.py`** — benchmarks LLM extraction strategies for non-standard historical workbooks. Must be run outside Claude Code.
- **`sheet_analyzer.py`** — LLM-based workbook analysis for non-standard historical offers. Sends full workbook content to Sonnet with a Tier A example, gets back structured per-box allocation data. Cached in `mappings/offer_N_llm_extraction.json`.
- **`box_parser.py`** — parses box column headers across all historical naming conventions (`?Sm Name`, `(?) Lg Name`, `Size - Name`, `M Box N`, `Lge Charity`, etc.) into `(cleaned_name, size_tier, box_type)`.
- **`name_matcher.py`** — LLM-based item name → DB ID matching for Tier C offers (no ID column). Exact/prefix match first, then Claude CLI (Haiku) for fuzzy matching. Cached in `mappings/`.
- **`claude_cli.py`** — subprocess wrapper for `claude -p` CLI calls.

### Tests (`tests/`)

253 tests across 13 modules covering models, config, categorizer, scoring, strategies, allocator pipeline, box parser, excel I/O, wizard helpers, and historical service. Uses synthetic fixtures — no DB or network required.

- **`conftest.py`** — test config bootstrap (sets env vars before allocator import), factory fixtures for Item/MysteryBox/CharityBox/AllocationResult.
- **`tests/fixtures/`** — synthetic `identifiers.json` and `scoring_config.json` for CI portability.

### Utility scripts (`scripts/`)

- **`score_offer.py`** — runs all strategies against a single offer, prints per-box metrics and a leaderboard.
- **`diagnose_scoring.py`** — penalty breakdowns, pricing anomaly detection, and visualisations across all historical tiers.
- **`validate_cleaned.py`** — structural integrity, DB consistency, and cross-file checks on cleaned CSVs. (underlying library for `HistoricalService` — validate_cleaned logic is now accessible via the TUI Historical Data screen via `python3 run.py` → Historical Data → Validate All. Keep as standalone tool for direct CLI use.)
- **`validate_prices.py`** — SUMPRODUCT validation comparing XLSX prices against DB prices.
- **`standardize_filenames.py`** — renames historical XLSX files to canonical `offer_{N}_shopping_list.xlsx` format. (dev tool — infrequent use, kept as standalone)
- **`compare_llm_outputs.py`** — side-by-side comparison of LLM extraction methods with Jaccard similarity and optional Claude investigation. (dev tool — infrequent use, kept as standalone)
- **`analyze_offer_values.py`** — per-offer, per-size-tier average box values. Writes `diagnostics/offer_value_targets.json` for training data.

## Database gotchas

- **Always filter soft deletes**: the relevant tables have a soft-delete column. Every query joining these tables must filter for non-deleted records — **except** `fetch_offer_parts_by_name(include_deleted=True)` which is used for historical name matching on older offers where parts are soft-deleted but prices are still valid.
- **User names are encrypted**. Use `email` as the identifier, never name fields.
- **Prices are in cents** (integer) everywhere — DB, models, config, output.
- SSH key path and connection config in `.env` (set `SSH_ENABLED=true` to use tunnel).
- **SQL queries** are loaded from `queries.json` (gitignored). See `queries.json.example` for the expected structure and column aliases.

## Conventions

- Fungible groups prevent putting multiple varieties of the same item (e.g. 3 apple types) in one box. The deal phase skips boxes that already have the group; the top-up scorer returns `-inf` for fungible dupes.
- Merged boxes (emails) get added to the customer's existing order. Standalone boxes (`?Name` prefix in output) ship separately.
- `BOX_TIERS` target values are `BOX_TARGET_PCT`% of price (configured in `.env`).
- Category IDs are configured in `scoring_config.json`.

## Strategy Leaderboard

Composite scores across 42 Tier A offers (2026-03-11). Update when algorithms change by running `python3 compare.py --all-strategies`.

Rank order: ilp-optimal > local-search > round-robin > discard-worst > deal-topup > minmax-deficit > greedy-best-fit > manual.

All automated strategies outperform manual packing. Score = 100 minus penalties across value accuracy, duplicate avoidance, diversity coverage, cross-box fairness, and preference compliance.

## Historical data tiers

| Tier | Offers | Count | Has IDs? | Source dir | Notes |
|------|--------|-------|----------|------------|-------|
| A | 64–106 | 42 | Yes | `historical/` | Full algorithm comparison |
| B | 55–63 | 9 | Yes | `historical/older/` | All standalone boxes |
| C | 45–54 | 10 | No (names) | `historical/older/` | Programmatic extraction validated; name matching via cached LLM maps in `mappings/` |
| D | 22–44 | 12 | — | `historical/older/` | Items soft-deleted but prices still valid; uses `include_deleted=True` for name matching |

Offers 45–48 and 22–44 have all items soft-deleted in DB, but price data is still usable for historical name matching. Name matcher falls back to cached mappings when DB is unavailable. Cached matches in `mappings/`.

## Reference

- Historical data: 74 XLSX files across `historical/` (42) and `historical/older/` (32); 57 produce cleaned CSVs
