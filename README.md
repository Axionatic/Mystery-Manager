# Mystery Manager

Allocates bulk produce overage into customer "mystery boxes" for a fruit & veggie box business. Reads weekly shopping-list spreadsheets and item/buyer data from the Laravel DB, runs a pluggable allocation strategy (ILP, local search, or greedy heuristics), then outputs tab-delimited box assignments for import back into the app.

## Quick start

```bash
python3 run.py 106 offer_106_shopping_list.xlsx                   # full run (TUI + LLM review)
python3 run.py 106 offer_106_shopping_list.xlsx --no-tui --no-llm # quick run
python3 compare.py                                                # validate against 42 historical offers
python3 compare.py --all-strategies                               # strategy leaderboard
```

## Project structure

```
Mystery-Manager/
├── run.py                  # Weekly allocation entry point
├── compare.py              # Algorithm validation against historical data
├── allocator/              # Core library
│   ├── allocator.py        #   Pipeline: XLSX + DB → strategy → output
│   ├── strategies/         #   Pluggable strategies (ILP, local-search, etc.)
│   ├── models.py           #   Item, MysteryBox, CharityBox, AllocationResult
│   ├── config.py           #   Tiers, weights, scoring constants, identifiers
│   ├── db.py               #   SSH tunnel + MySQL queries
│   ├── excel_io.py         #   XLSX reader + tab-delimited writer
│   ├── tui.py              #   Rich interactive review UI
│   ├── clean_history.py    #   Historical XLSX → CSV pipeline
│   ├── fill_workbook.py    #   Write strategy results into XLSX
│   └── benchmark_extraction.py  # LLM extraction benchmarks
├── scripts/                # Maintenance utilities
│   ├── score_offer.py      #   Per-offer strategy leaderboard
│   ├── diagnose_scoring.py #   Penalty breakdown diagnostics
│   ├── validate_cleaned.py #   Structural + DB checks on cleaned CSVs
│   ├── validate_prices.py  #   XLSX vs DB price validation
│   ├── standardize_filenames.py  # Canonical XLSX filenames
│   └── compare_llm_outputs.py   # Side-by-side LLM extraction comparison
├── docs/                   # Design docs (gitignored)
├── historical/             # Source XLSX files (gitignored)
├── cleaned/                # Processed CSVs (gitignored)
├── mappings/               # Cached LLM name maps (gitignored)
├── CLAUDE.md               # Full architecture and conventions
└── requirements.txt
```

## Gitignored items a new user needs to provide

**Secrets / config**
- `.env` / `.env.local` — DB credentials, SSH tunnel config, pricing params (box target %, penalty weights)
- `identifiers.json` — customer/donor/staff email lists (see `identifiers.json.example`)

**Business data**
- `historical/` and `cleaned/` — historical shopping lists and processed CSVs used by `compare.py`
- `offer_*.xlsx` — weekly input files (overage spreadsheets)
- `mappings/` — cached item-name-to-DB-ID mappings for older offers without ID columns

See `CLAUDE.md` for full architecture, commands, and conventions.
