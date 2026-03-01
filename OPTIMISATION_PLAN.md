# Scoring Optimisation Plan

## Goal

Calibrate the composite scoring function so that manually-packed boxes — our ground truth for "good" — score **>= 85**. Currently they average ~50, while degenerate solutions (e.g. 45 apples in a fruit-only box) can score ~99. The scoring function should reward what humans intuitively consider a good box and penalise gaming.

## The Problem in Detail

Our composite score is `100 - penalties` across five dimensions:

| Dimension | Weight/Multiplier | What it measures |
|-----------|-------------------|------------------|
| **Value** | Piecewise-linear, up to 5.0/pp | How close to 114-117% of target price |
| **Dupes** | `degree * 8.0` per dupe | Fungible group collisions (e.g. 3 apple varieties) |
| **Diversity** | `(1 - score) * 10.0` | Coverage of sub-category, usage, colour, shape |
| **Fairness** | `stddev * 0.5` | Consistency across boxes |
| **Preference** | `20.0` per violation | Fruit-only/veg-only compliance |

**Known issues:**
1. **Value sweet spot is too narrow.** 114-117% is a 3pp window. Manual boxes routinely land at 108-125% and are perfectly fine. The steep penalties outside the window dominate the score.
2. **Diversity scoring can be gamed.** ILP can achieve near-perfect diversity by concentrating on items that happen to cover many tags, regardless of whether the box makes culinary sense.
3. **No penalty for item count / concentration.** A box with 1 item at qty=45 scores well on value and diversity (if the item has tags). There's no "a good box has 8-12 distinct items" constraint in scoring.
4. **Multipliers are hand-tuned guesses.** The 8.0/10.0/0.5 weights were chosen by feel, not calibrated against data.
5. **Diversity dimensions may be wrong.** The 50/20/15/15 split across sub_category/usage/colour/shape is arbitrary. Colour and shape may not matter much to customers.

## Phase 1: Diagnostic Analysis

**Requires:** Historical data only (no DB needed)

Build a diagnostic script that processes all manual boxes and answers:

1. **Penalty breakdown**: Where do manual boxes lose their ~50 points? Which penalty dimension is the biggest offender?
2. **Distribution analysis**: Are scores clustered (all ~50) or bimodal (some great, some terrible)? Histogram + percentiles.
3. **Per-dimension deep dive**:
   - Value: what's the actual distribution of value% in manual boxes? Is the sweet spot in the wrong place?
   - Diversity: what diversity scores do manual boxes achieve? Are they genuinely low, or is the scoring too harsh?
   - Dupes: how often do manual boxes contain fungible dupes? Are we penalising things humans consider fine?
4. **Correlation analysis**: Which penalty dimensions correlate most with "surprisingly low" scores?
5. **Tier breakdown**: Do small/medium/large boxes behave differently?

**Output:** Visualisations (matplotlib/seaborn), summary statistics, and a clear diagnosis of which penalties need structural changes vs parameter tuning.

## Phase 2: Structural Scoring Fixes

**Requires:** Phase 1 findings

Based on diagnostics, likely structural changes include:

- **Widen the value sweet spot** — probably 110-120% or even 108-122%, with gentler slopes.
- **Add item count/concentration penalty** — penalise boxes below minimum unique items or with any single item exceeding X% of total value/quantity.
- **Rethink diversity metric** — effective species (1/HHI) may not match human intuition. Consider:
  - Minimum category coverage thresholds instead of continuous scoring.
  - Weight sub_category much more heavily (or exclusively).
  - Cap diversity reward so it can't compensate for a terrible box.
- **Add new penalty terms** if diagnostics reveal gaps:
  - Fruit/veg balance (for mixed boxes).
  - Maximum single-item quantity.
  - "Boring box" detector (too many similar items even if technically non-fungible).

## Phase 3: Parameter Optimisation (Bayesian/Optuna)

**Requires:** Phase 2 structure finalised, 700+ manual boxes

Use [Optuna](https://optuna.org/) (TPE sampler) to tune all scoring parameters simultaneously:

### Search space (illustrative)
```python
# Value penalty shape
sweet_spot_low = trial.suggest_float("sweet_low", 108, 116)
sweet_spot_high = trial.suggest_float("sweet_high", 116, 125)
near_penalty_rate = trial.suggest_float("near_rate", 0.5, 3.0)
far_penalty_rate = trial.suggest_float("far_rate", 2.0, 10.0)

# Multipliers
dupe_multiplier = trial.suggest_float("dupe_mult", 2.0, 20.0)
diversity_multiplier = trial.suggest_float("div_mult", 2.0, 20.0)
fairness_multiplier = trial.suggest_float("fair_mult", 0.1, 2.0)

# Diversity dimension weights (normalised)
w_subcat = trial.suggest_float("w_subcat", 0.2, 0.8)
w_usage = trial.suggest_float("w_usage", 0.05, 0.4)
# ... etc
```

### Objective function

The tricky part: what should Optuna maximise? Options:

1. **Mean score of manual boxes** — simple, but doesn't penalise gaming.
2. **Mean score of manual boxes MINUS max achievable by ILP** — directly penalises gameable scoring functions, but expensive (needs ILP solve per trial).
3. **Mean manual score, subject to constraints** — e.g. "no single-item box can score > 50" as a hard constraint. Cheaper than option 2.
4. **Rank correlation** — score should rank manual boxes higher than random/degenerate ones. Generate synthetic bad boxes as negative examples.

Recommendation: **Option 3 or 4**. Generate a set of known-bad boxes (synthetic degenerates like 45 apples, random allocations, etc.) and optimise for the gap between manual and bad box scores.

### Validation

- **Train/test split**: 80/20 on historical boxes, stratified by tier.
- **Cross-validation**: K-fold (k=5) given the moderate sample size.
- **Overfit check**: Compare train vs test scores. With ~20 parameters and 600+ training boxes, overfitting risk is low.

## Phase 4: ML Scoring Model (Optional/Future)

**Requires:** Phase 3 as baseline, full 700+ box corpus

If the hand-crafted function plateaus, consider learning the scoring function directly:

### Approach A: Gradient-boosted ranker
- Features: value%, unique item count, diversity metrics, dupe counts, category coverage, price variance, etc.
- Labels: manual = good (1), synthetic bad = bad (0), or pairwise ranking.
- Models: LightGBM or XGBoost ranker. 800 samples is comfortable for this.

### Approach B: Learn penalty weights via regression
- Use manual box features as input, target score ~90 as output.
- Linear regression on the penalty terms themselves — essentially learns the multipliers.
- More interpretable than a black-box model.

### Approach C: Bradley-Terry preference model
- Generate pairs of boxes, label which is better (manual > random, manual > degenerate).
- Fit a preference model that implicitly learns the scoring function.
- Elegant but harder to turn into a deployable scoring function.

### Sample size assessment
- 700-800 boxes with ~15 features = very comfortable for tree models and linear models.
- Not enough for deep learning (nor would it be warranted here).
- K-fold cross-validation essential at this scale.

## Phase 5: Validation & Deployment

- Re-run `compare.py --all-strategies` with new scoring to verify leaderboard makes sense.
- Manual spot-check: do high-scoring boxes actually look good? Do low-scoring ones look bad?
- Run ILP with new scoring on a few offers — verify it can't game the new function.
- Update `_scoring.py` and `config.py` with final parameters.
- Update CLAUDE.md leaderboard.

## Open Questions

1. **What's the right target score for manual boxes?** >= 85 is aspirational. Maybe 80 is more realistic if manual packing is genuinely suboptimal in some dimensions.
2. **Should we weight recent offers more?** Packing quality/approach may have evolved over 30 weeks.
3. **Do we need per-tier scoring parameters?** Small boxes have fewer items and less room for diversity — should their diversity penalty be softer?
4. **How do we handle the charity/CCI allocation?** Currently excluded from scoring. Should it stay that way?
