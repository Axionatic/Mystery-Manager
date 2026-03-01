"""Configuration for mystery box allocation."""

import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Detect pack items (e.g. "Avocado - Hass (3 value pack)", "Lemons - 5 pack")
# and return the pack size so we can divide the DB price to get per-unit price.
_PACK_RE = re.compile(r"(\d+)\s*(?:value\s*)?pack", re.IGNORECASE)


def detect_pack_size(name: str) -> int:
    """Return the pack size from an item name, or 1 if not a pack."""
    m = _PACK_RE.search(name)
    return int(m.group(1)) if m else 1


# Box size tiers (all values in cents, loaded from .env)
def _load_box_tiers() -> dict:
    """Build BOX_TIERS from environment variables."""
    missing = [v for v in ("BOX_PRICE_SMALL", "BOX_PRICE_MEDIUM", "BOX_PRICE_LARGE")
               if not os.environ.get(v)]
    if missing:
        raise EnvironmentError(
            f"Missing required pricing env vars: {', '.join(missing)}. "
            "Copy .env.example to .env and fill in the real values."
        )
    target_pct = int(os.environ.get("BOX_TARGET_PCT", "115")) / 100
    tiers = {}
    for size, env_key in [("small", "BOX_PRICE_SMALL"),
                          ("medium", "BOX_PRICE_MEDIUM"),
                          ("large", "BOX_PRICE_LARGE")]:
        price = int(os.environ[env_key])
        tiers[size] = {"price": price, "target_value": round(price * target_pct)}
    return tiers

BOX_TIERS = _load_box_tiers()

# Hard ceiling: never exceed this % of target value
VALUE_CEILING_PCT = 1.30

# Items at or below this price (cents) are candidates for qty bump in top-up
CHEAP_ITEM_THRESHOLD = 500

# Target unique-item counts per tier (min, max)
TARGET_ITEM_COUNTS = {
    "small": (8, 12),
    "medium": (14, 20),
    "large": (20, 28),
}

# Scoring weights for top-up phase
SCORING_WEIGHTS = {
    "new_item_bonus": 5.0,
    "fungible_spread": 3.0,
    "diversity": 2.0,
    "value_progress": 1.0,
}

# Groups with degree >= this threshold get slot-based dealing (qty > 1)
SLOT_DEGREE_THRESHOLD = 0.7

# Dupe penalty floor: penalty multiplier = max(degree - DUPE_PENALTY_FLOOR, 0)
# Groups with degree <= this get zero dupe penalty in scoring.
DUPE_PENALTY_FLOOR = 0.5

# Max qty per fungible group slot per box tier
MAX_SLOT_QTY = {
    "small": 4,
    "medium": 6,
    "large": 8,
}

# Fungible groups: (degree, [prefixes])
# degree 1.0 = same product, different size/pack — hard block, dealt as slot
# degree 0.9 = near-identical — dealt as slot
# degree 0.7 = same type, different variety — dealt as slot
# degree 0.5 = related items — dealt individually at qty=1
# degree 0.3 = similar culinary role — dealt individually at qty=1
FUNGIBLE_GROUPS = {
    "banana":        (1.0, ["Bananas -"]),
    "strawberry":    (1.0, ["Strawberries -"]),
    "field_tomato":  (1.0, ["Truss Tomato", "Tomatoes - Roma", "Tomatoes - Field"]),
    "cherry_tomato": (0.9, ["Cherry Tomato", "Grape Tomato", "Tomatoes - Mini Roma"]),
    "apple":         (0.7, ["Apples -"]),
    "pear":          (0.7, ["Pears -"]),
    "cucumber":      (0.7, ["Cucumber -", "Continental Cucumber"]),
    "potato":        (0.7, ["Potatoes -"]),
    "grape":         (0.7, ["Grapes -"]),
    "orange":        (0.7, ["Oranges"]),
    "lemon":         (0.7, ["Lemons"]),
    "lime":          (0.7, ["Limes"]),
    "mandarin":      (0.7, ["Mandarins -"]),
    "chilli":        (0.7, ["Chillies -"]),
    "lettuce":       (0.7, ["Iceberg", "Baby Cos", "Cos Lettuce"]),
    "onion":         (0.5, ["Onion -"]),
    "stone_fruit":   (0.5, ["Peaches", "Nectarines"]),
    "berry":         (0.3, ["Raspberries", "Blackberries", "Blueberries"]),
    "leafy_green":   (0.3, ["Roquette", "Baby Spinach", "Salad"]),
    "snap_pea":      (0.3, ["Snowpeas", "Sugar snaps"]),
}

# ---------------------------------------------------------------------------
# PII-bearing identifiers (loaded from gitignored identifiers.json)
# ---------------------------------------------------------------------------

def _load_identifiers() -> dict:
    """Load identifier sets from identifiers.json at project root."""
    path = Path(__file__).resolve().parent.parent / "identifiers.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Copy identifiers.json.example to identifiers.json "
            "and fill in the real values."
        )
    with open(path) as f:
        return json.load(f)

_IDENTIFIERS = _load_identifiers()

# Non-customer boxes: donations and charity recipients (exclude from scoring)
DONATION_IDENTIFIERS = set(_IDENTIFIERS["donation_identifiers"])

# Backwards compat alias
CHARITY_IDENTIFIERS = DONATION_IDENTIFIERS

# Staff who self-pick from overs (not a standard mystery box, exclude from
# historical training data and algorithm auto-detection)
STAFF_IDENTIFIERS = set(_IDENTIFIERS["staff_identifiers"])

# Pre-account standalone names → email mappings for merged-box treatment
STANDALONE_NAME_TO_EMAIL = _IDENTIFIERS["standalone_name_to_email"]

# Explicit size overrides for non-standard box names
BOX_SIZE_OVERRIDES = _IDENTIFIERS.get("box_size_overrides", {})

# Special columns to skip in historical data
STOCK_IDENTIFIERS = {"Stock"}
BUFFER_IDENTIFIERS = {"Buffer", "Volunteers"}
SUM_IDENTIFIERS = {"SUM", "Sum", "sum"}

# Additional skip-column patterns for older historical data
SKIP_COLUMN_IDENTIFIERS = {
    "Ov - Mys", "Ov-Mys", "Ov - mys", "Over Mys", "Overs", "Sum Mys",
    "Md Actual", "Magda Actual",
    "Total",
    "Price Ea", "JS Price Ea", "JS Cost Ea", "Cost Ea", "RRP Ea",
    "Qty Sold", "Required Buy", "Overage", "Expected Purchase",
    "Pack Order", "Supplier #", "Supplier Name",
    "Qty (after mys)",
}

# CCI allocation target (loaded from .env)
CCI_BASE_PERCENT = float(os.environ.get("CCI_BASE_PERCENT", "0.05"))
CCI_GIVING_MULTIPLIER = float(os.environ.get("CCI_GIVING_MULTIPLIER", "1.35"))

# Category IDs from part_categories table
CATEGORY_FRUIT = 2
CATEGORY_VEGETABLES = 3

# Structured preference keywords (from buy_options)
PREFERENCE_FRUIT_ONLY = "Fruit mystery box (no veg)"
PREFERENCE_VEG_ONLY = "Veggie mystery box (no fruit)"
PREFERENCE_MIX = "Mix of everything (best value!)"

# Diversity dimension weights (for multi-dimensional diversity scoring)
DIVERSITY_WEIGHTS = {
    "sub_category": 0.50,
    "usage": 0.20,
    "colour": 0.15,
    "shape": 0.15,
}

# Item classifications: key -> (prefixes, sub_category, usage, colour, shape)
# Prefix matching uses same pattern as FUNGIBLE_GROUPS.
ITEM_CLASSIFICATIONS = {
    # --- Pome fruit ---
    "apple":             (["Apples -"],                                       "pome_fruit",   "snacking", "red",           "round"),
    "pear":              (["Pears -"],                                        "pome_fruit",   "snacking", "green",         "chunky"),
    # --- Citrus ---
    "orange":            (["Orange"],                                         "citrus",       "snacking", "orange_yellow", "round"),
    "mandarin":          (["Mandarins -"],                                    "citrus",       "snacking", "orange_yellow", "round"),
    "lemon":             (["Lemons"],                                         "citrus",       "garnish",  "orange_yellow", "round"),
    "lime":              (["Limes"],                                          "citrus",       "garnish",  "green",         "round"),
    # --- Stone fruit ---
    "apricot":           (["Apricot"],                                        "stone_fruit",  "snacking", "orange_yellow", "round"),
    "nectarine":         (["Nectarines"],                                     "stone_fruit",  "snacking", "orange_yellow", "round"),
    "peach":             (["Peaches"],                                        "stone_fruit",  "snacking", "orange_yellow", "round"),
    "plum":              (["Plum"],                                           "stone_fruit",  "snacking", "purple",        "round"),
    "cherry":            (["Cherries"],                                       "stone_fruit",  "snacking", "red",           "small"),
    # --- Berry ---
    "strawberry_cls":    (["Strawberries"],                                   "berry",        "snacking", "red",           "small"),
    "blueberry":         (["Blueberry", "Blueberries"],                       "berry",        "snacking", "purple",        "small"),
    "raspberry":         (["Raspberries"],                                    "berry",        "snacking", "red",           "small"),
    "blackberry":        (["Blackberries"],                                   "berry",        "snacking", "purple",        "small"),
    # --- Tropical ---
    "banana_cls":        (["Bananas -"],                                      "tropical",     "snacking", "orange_yellow", "long"),
    "mango":             (["Mangoes"],                                        "tropical",     "snacking", "orange_yellow", "chunky"),
    "pineapple":         (["Pineapple"],                                      "tropical",     "snacking", "orange_yellow", "chunky"),
    "kiwi":              (["Kiwi"],                                           "tropical",     "snacking", "green",         "round"),
    # --- Grape ---
    "grape_cls":         (["Grapes -"],                                       "grape",        "snacking", "red",           "small"),
    # --- Avocado ---
    "avocado":           (["Avocado"],                                        "avocado",      "salad",    "green",         "round"),
    # --- Melon ---
    "watermelon":        (["Watermelon"],                                     "melon",        "snacking", "red",           "chunky"),
    "honeydew":          (["Honey Dew"],                                      "melon",        "snacking", "green",         "chunky"),
    "rockmelon":         (["Rock Melon"],                                     "melon",        "snacking", "orange_yellow", "chunky"),
    # --- Leafy green ---
    "baby_spinach":      (["Baby Spinach"],                                   "leafy_green",  "salad",    "green",         "leafy"),
    "spinach_bunch":     (["Spinach -"],                                      "leafy_green",  "cooking",  "green",         "leafy"),
    "roquette":          (["Roquette"],                                       "leafy_green",  "salad",    "green",         "leafy"),
    "salad":             (["Salad"],                                          "leafy_green",  "salad",    "green",         "leafy"),
    "iceberg":           (["Iceberg"],                                        "leafy_green",  "salad",    "green",         "leafy"),
    "lettuce":           (["Baby Cos", "Cos Lettuce", "Lettuce"],             "leafy_green",  "salad",    "green",         "leafy"),
    "kale":              (["Kale"],                                           "leafy_green",  "cooking",  "green",         "leafy"),
    "silverbeet":        (["Silverbeet"],                                     "leafy_green",  "cooking",  "green",         "leafy"),
    "rainbow_chard":     (["Rainbow Chard"],                                  "leafy_green",  "cooking",  "green",         "leafy"),
    # --- Root veg ---
    "potato_cls":        (["Potatoes -"],                                     "root_veg",     "cooking",  "white_brown",   "round"),
    "carrot":            (["Carrots"],                                        "root_veg",     "cooking",  "orange_yellow", "long"),
    "beetroot":          (["Beetroot"],                                       "root_veg",     "cooking",  "purple",        "round"),
    "radish":            (["Radish"],                                         "root_veg",     "salad",    "red",           "small"),
    "ginger":            (["Ginger"],                                         "root_veg",     "garnish",  "orange_yellow", "small"),
    "turmeric":          (["Turmeric"],                                       "root_veg",     "garnish",  "orange_yellow", "small"),
    # --- Fruiting veg ---
    "field_tomato_cls":  (["Truss Tomato", "Tomatoes - Roma",
                           "Tomatoes - Field", "Field"],                      "fruiting_veg", "cooking",  "red",           "round"),
    "cherry_tomato_cls": (["Cherry Tomato", "Grape Tomato",
                           "Tomatoes - Mini", "Tomatoes - Cherry"],           "fruiting_veg", "salad",    "red",           "small"),
    "capsicum":          (["Capsicum"],                                       "fruiting_veg", "cooking",  "red",           "chunky"),
    "eggplant":          (["Eggplant"],                                       "fruiting_veg", "cooking",  "purple",        "long"),
    "chilli_cls":        (["Chillies -"],                                     "fruiting_veg", "garnish",  "green",         "long"),
    "zucchini":          (["Zucchini"],                                       "fruiting_veg", "cooking",  "green",         "long"),
    "cucumber_cls":      (["Cucumber -", "Continental Cucumber"],             "fruiting_veg", "salad",    "green",         "long"),
    "corn":              (["Corn"],                                           "fruiting_veg", "cooking",  "orange_yellow", "long"),
    # --- Cucurbit ---
    "pumpkin":           (["Pumpkin"],                                        "cucurbit",     "cooking",  "orange_yellow", "chunky"),
    # --- Asian green ---
    "pak_choy":          (["Pak Choy"],                                       "asian_green",  "cooking",  "green",         "leafy"),
    "chinese_broccoli":  (["Chinese Broccoli"],                               "asian_green",  "cooking",  "green",         "leafy"),
    # --- Brassica ---
    "broccoli":          (["Broccoli"],                                       "brassica",     "cooking",  "green",         "chunky"),
    "broccolini":        (["Broccolini"],                                     "brassica",     "cooking",  "green",         "long"),
    "cauliflower":       (["Cauliflower"],                                    "brassica",     "cooking",  "white_brown",   "chunky"),
    "brussels":          (["Brussels"],                                       "brassica",     "cooking",  "green",         "small"),
    "cabbage":           (["Cabbage"],                                        "brassica",     "cooking",  "green",         "chunky"),
    # --- Allium ---
    "onion_cls":         (["Onion -"],                                        "allium",       "cooking",  "white_brown",   "round"),
    "garlic":            (["Garlic"],                                         "allium",       "garnish",  "white_brown",   "small"),
    "leek":              (["Leek"],                                           "allium",       "cooking",  "green",         "long"),
    "spring_onion":      (["Spring Onion"],                                   "allium",       "garnish",  "green",         "long"),
    # --- Legume ---
    "beans":             (["Beans"],                                          "legume",       "cooking",  "green",         "long"),
    "snowpeas":          (["Snowpeas"],                                       "legume",       "salad",    "green",         "small"),
    "sugar_snaps":       (["Sugar snaps", "Sugar Snaps"],                     "legume",       "snacking", "green",         "small"),
    # --- Stalk ---
    "celery":            (["Celery"],                                         "stalk",        "salad",    "green",         "long"),
    "asparagus":         (["Asparagus"],                                      "stalk",        "cooking",  "green",         "long"),
    "rhubarb":           (["Rhubarb"],                                        "stalk",        "cooking",  "red",           "long"),
    "fennel":            (["Fennel"],                                         "stalk",        "cooking",  "white_brown",   "chunky"),
    # --- Mushroom ---
    "mushroom_cls":      (["Mushrooms", "Mushroom"],                          "mushroom",     "cooking",  "white_brown",   "small"),
    # --- Herb ---
    "parsley":           (["Parsley"],                                        "herb",         "garnish",  "green",         "small"),
    "chives":            (["Chives"],                                         "herb",         "garnish",  "green",         "small"),
    "coriander":         (["Coriander"],                                      "herb",         "garnish",  "green",         "small"),
    "dill":              (["Dill"],                                           "herb",         "garnish",  "green",         "small"),
    "rosemary":          (["Rosemary"],                                       "herb",         "garnish",  "green",         "small"),
    "sage":              (["Sage"],                                           "herb",         "garnish",  "green",         "small"),
    "thyme":             (["Thyme"],                                          "herb",         "garnish",  "green",         "small"),
    "mint":              (["Mint"],                                           "herb",         "garnish",  "green",         "small"),
    "oregano":           (["Oregano"],                                        "herb",         "garnish",  "green",         "small"),
    "basil":             (["Basil"],                                          "herb",         "garnish",  "green",         "small"),
    "lemongrass":        (["Lemongrass"],                                     "herb",         "garnish",  "green",         "long"),
}

# ---------------------------------------------------------------------------
# Composite scoring (compare.py)
# ---------------------------------------------------------------------------

# Value penalty: sweet spot and zone thresholds (as % of target)
VALUE_SWEET_SPOT_LOW = 114
VALUE_SWEET_SPOT_HIGH = 117
VALUE_HEAVY_PENALTY_THRESHOLD = 110
VALUE_OVER_SOFT_THRESHOLD = 120
VALUE_OVER_HARD_THRESHOLD = 130

# Value penalty: rates per percentage point
VALUE_NEAR_PENALTY_RATE = 1.5     # within 4% of sweet spot
VALUE_FAR_PENALTY_RATE = 5.0      # below 110% or above 130%
VALUE_OVER_MODERATE_RATE = 3.0    # 120-130% range

# Aggregate penalty multipliers
DUPE_PENALTY_MULTIPLIER = 8.0       # weighted_dupe_penalty * this
DIVERSITY_PENALTY_MULTIPLIER = 10.0  # (1 - score) * this, max penalty
FAIRNESS_PENALTY_MULTIPLIER = 0.5   # stddev(value_pct) * this
PREF_VIOLATION_PENALTY = 20.0       # per violation, hard penalty
MAX_COMPOSITE_SCORE = 100.0

# Pack price detection tolerance (cents)
PACK_PRICE_TOLERANCE_CENTS = 5

# Diversity fallback when no reference tags available
DIVERSITY_FALLBACK_SCORE = 0.5

# ---------------------------------------------------------------------------
# Top-up scorer (scorer.py)
# ---------------------------------------------------------------------------

FUNGIBLE_NEUTRAL_SCORE = 0.5   # non-fungible items: neither bonus nor penalty
FUNGIBLE_NEW_GROUP_BONUS = 1.0  # first item from a new fungible group

# ---------------------------------------------------------------------------
# Strategy constants
# ---------------------------------------------------------------------------

# deal-topup
TOPUP_MAX_PASSES = 50

# local-search
LOCAL_SEARCH_MAX_ITERATIONS = 500

# Safety limits
GREEDY_MAX_ROUNDS = 500
ROUND_ROBIN_MAX_PASSES = 200
MINMAX_MAX_PASSES = 100

# ILP: piecewise-linear share^2 approximation breakpoints
ILP_HHI_BREAKPOINTS = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]

# ILP: relative weights for coverage vs concentration in diversity penalty
# alpha * DPM * (1 - coverage) + beta * DPM * hhi_approx
ILP_COVERAGE_WEIGHT = 0.4
ILP_BALANCE_WEIGHT = 0.6

# Fallback classification for unrecognized items (by category_id)
CLASSIFICATION_FALLBACK = {
    CATEGORY_FRUIT:      ("other_fruit", "cooking", "green", "round"),
    CATEGORY_VEGETABLES: ("other_veg",   "cooking", "green", "round"),
}

