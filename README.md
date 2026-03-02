# Mystery Manager

Allocates bulk produce overage into customer "mystery boxes" for a fruit & veggie box business. Reads weekly shopping-list spreadsheets and item/buyer data from the Laravel DB, runs a pluggable allocation strategy (ILP, local search, or greedy heuristics), then outputs tab-delimited box assignments for import back into the app.

See `CLAUDE.md` for full architecture, commands, and conventions.

## Gitignored items a new user needs to provide

**Secrets / config**
- `.env` / `.env.local` — DB credentials, SSH tunnel config, pricing params (box target %, penalty weights)
- `identifiers.json` — customer/donor/staff email lists (see `identifiers.json.example`)

**Business data**
- `historical/` and `cleaned/` — historical shopping lists and processed CSVs used by `compare.py`
- `offer_*.xlsx` — weekly input files (overage spreadsheets)
- `mappings/` — cached item-name-to-DB-ID mappings for older offers without ID columns

**Run outputs**
- `output/` — `compare.py` run outputs
- `OUTPUT_REFERENCE.md` — reference output for manual comparison
