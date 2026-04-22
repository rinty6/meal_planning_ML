from __future__ import annotations


# Recommendation engine constants are centralized here so behavior can be tuned
# quickly without changing business logic spread across files.
MEAL_SLOTS = ("breakfast", "lunch", "dinner")


DEFAULT_MEAL_ALLOCATION = {
    # NOTE: Default ratios to avoid overshooting daily calories.
    "breakfast": 0.20,
    "lunch": 0.30,
    "dinner": 0.50,
}


MEAL_HINTS = {
    "breakfast": ["healthy breakfast", "egg", "oatmeal", "latte", "banh mi"],
    "lunch": ["balanced lunch", "rice bowl", "lean protein", "coke", "chicken rice"],
    "dinner": ["dinner plate", "high protein", "whole foods", "banana", "water"],
}

MISMATCH_GROUPS = {
    "breakfast": [
        # NOTE: Cuisine and plating terms that usually don't belong to breakfast.
        "platter",
        "szechuan",
        "papadum",
        "papadums",
        "pizza",
        "burger",
        "fries",
        "fried chicken",
        "wings",
        "steak",
        "bbq",
        "barbecue",
        "pasta",
        "lasagna",
        "curry",
        "taco",
        "burrito",
        "ramen",
        "pho",
        "sushi",
        "pad thai",
        "biryani",
        "tandoori",
        "chili",
        "roast",
        "casserole",
    ],
    "lunch": [
        "pancake",
        "waffle",
        "cereal",
        "granola",
        "oatmeal",
        "porridge",
        "toast",
        "bagel",
        "muffin",
        "smoothie",
    ],
    "dinner": [
        "cereal",
        "granola",
        "oatmeal",
        "porridge",
        "pancake",
        "waffle",
        "toast",
        "bagel",
        "muffin",
        "smoothie",
    ],
}

CATEGORICAL_MISMATCH_PENALTY = 0.2

# NOTE: Breakfast-only penalty when no breakfast keyword is detected.
BREAKFAST_MISSING_HINT_PENALTY = 0.35
BREAKFAST_MAIN_SKIP_PENALTY_FLOOR = 0.82
BREAKFAST_MAIN_BIAS_MULTIPLIER_FLOOR = 0.7

MEAL_KEYWORDS = {
    "breakfast": [
        "breakfast",
        "breakfasts",
        "brunch",
        "oat",
        "oatmeal",
        "porridge",
        "cereal",
        "granola",
        "toast",
        "bagel",
        "muffin",
        "pancake",
        "waffle",
        "omelet",
        "omelette",
        "egg",
        "yogurt",
        "smoothie",
        "coffee",
        "latte",
        "tea",
        "bacon",
        "sausage",
    ],
    "lunch": [
        "lunch",
        "sandwich",
        "wrap",
        "salad",
        "soup",
        "bowl",
        "rice bowl",
        "noodle",
        "pasta",
        "taco",
        "burrito",
        "quesadilla",
        "burger",
        "poke",
        "sushi",
        "pho",
    ],
    "dinner": [
        "dinner",
        "steak",
        "roast",
        "salmon",
        "fish",
        "curry",
        "stir fry",
        "bbq",
        "barbecue",
        "casserole",
        "grilled",
        "chicken",
        "beef",
        "pork",
        "rice",
        "pasta",
        "noodle",
        "lasagna",
    ],
}

# NOTE: Combo category keywords for main/side/drink classification.
COMBO_BEVERAGE_KEYWORDS = (
    "coffee",
    "latte",
    "tea",
    "water",
    "juice",
    "smoothie",
    "shake",
    "soda",
    "soft drink",
    "coke",
    "espresso",
    "cappuccino",
    "matcha",
    "beverage",
    "seltzer",
    "sparkling water",
    "kombucha",
    "energy drink",
    "sports drink",
    "hot chocolate",
    "cocoa",
)

COMBO_DRINK_KEYWORDS = COMBO_BEVERAGE_KEYWORDS

COMBO_DAIRY_SNACK_KEYWORDS = (
    "yogurt",
    "yogurts",
    "yoghurt",
    "yoghurts",
    "jogurt",
    "joghurts",
    "joghurt",
    "joghurts",
    "yaourt",
    "yaourts",
    "baby milk",
    "baby milks",
    "follow on milk",
    "follow-on milk",
    "follow on formula",
    "follow-on formula",
    "fromage blanc",
    "fromages blancs",
    "petits filous",
    "petit filous",
    "dairy snack",
    "drinkable yogurt",
    "drinkable yoghurt",
    "dairy-free yogurt",
    "dairy free yogurt",
    "greek yogurt",
    "greek yoghurt",
    "skyr",
    "quark",
    "formula",
    "pudding",
    "custard",
    "dessert",
    "mousse",
    "ice cream",
    "parfait",
    "vanukas",
    "granola",
    "muesli",
    "cereal",
    "porridge",
    "oatmeal",
    "chia",
    "creams",
    "nutritional drink",
    "nutritional drinks",
    "nutritional-drinks",
    "nutrition shake",
    "nutrition shakes",
    "meal replacement",
    "meal replacements",
    "protein shake",
    "protein shakes",
    "protein milk",
    "protein milks",
    "supplement drink",
    "supplement drinks",
    "dietary supplement",
    "dietary supplements",
    "dietary-supplements",
    "bodybuilding supplement",
    "bodybuilding supplements",
    "bodybuilding-supplements",
    "diabetes care",
)

COMBO_SIDE_KEYWORDS = (
    "banana",
    "apple",
    "orange",
    "berries",
    "grapes",
    "watermelon",
    "salad",
    "vegetable",
    "broccoli",
    "carrot",
    "fries",
    "chips",
    "bread",
    "toast",
    "rice",
    "soup",
)

# NOTE: Default calorie split for combo assembly (main gets leftovers).
COMBO_CATEGORY_TARGETS = {"main": 0.65, "side": 0.20, "drink": 0.15}
COMBO_MMR_LAMBDA = 0.5
COMBO_MMR_LAMBDA_MIN = 0.35
COMBO_MMR_LAMBDA_MAX = 0.7
COMBO_REUSE_PENALTY_BASE = 0.6
COMBO_REUSE_PENALTY_MIN = 0.45
COMBO_REUSE_PENALTY_MAX = 0.8
# NOTE: Keep combo calories within +/-10% of the slot target.
COMBO_TARGET_TOLERANCE = 0.10

# NOTE: Feedback-driven adjustments.
SKIP_PENALTY = 0.6
LOVE_BOOST = 1.15
FEEDBACK_DECAY_LAMBDA = 0.08
BIAS_TABLE_MIN_WEIGHT = -0.75
BIAS_TABLE_MAX_WEIGHT = 0.75
BIAS_TABLE_MIN_ABS_WEIGHT = 0.03

# NOTE: Health/processing guardrails for ranking.
MIN_HEALTH_SCORE = 3.0
HEALTH_SCORE_CLAMP_MIN = 0.5
HEALTH_SCORE_CLAMP_MAX = 1.6
HEALTH_SCORE_BOOST_MIN = 3.5
HEALTH_SCORE_BOOST_MAX_MULTIPLIER = 1.6
SUGAR_LIMIT_PER_MEAL = 10.0
SUGAR_PENALTY_FACTOR = 0.5
DISCRETIONARY_PENALTY = 0.5
AU_LOCAL_BOOST = 1.5
ULTRA_PROCESSED_PENALTY = 0.7
ULTRA_PROCESSED_ADDITIVE_COUNT = 5
ULTRA_PROCESSED_HABIT_WEIGHT_FACTOR = 0.5

DISCRETIONARY_KEYWORDS = [
    "bacon",
    "biscuit",
    "biscuits",
    "cookies",
    "burger",
    "pizza",
    "fries",
    "fried",
    "donut",
    "cake",
    "pastry",
    "soda",
    "soft drink",
    "candy",
    "chocolate",
]

INDUSTRIAL_ADDITIVE_MARKERS = [
    "syrup",
    "hydrogenated",
    "maltodextrin",
    "artificial flavor",
    "artificial flavour",
    "emulsifier",
    "preservative",
    "color",
    "colour",
    "stabilizer",
    "stabiliser",
    "sweetener",
    "thickener",
    "modified starch",
]


GOAL_LABELS = {
    "lose_weight": "weight loss",
    "gain_muscle": "muscle gain",
    "maintain": "maintenance",
}


VALID_GOALS = set(GOAL_LABELS.keys())


DEFAULT_DAILY_CALORIES = 2000.0
MIN_DAILY_CALORIES = 1200.0


HISTORY_LOOKBACK_DAYS = 90
MAX_HISTORY_ROWS = 800
# NOTE: Require at least this many rows before adapting meal ratios.
MIN_HISTORY_ROWS_FOR_DYNAMIC_RATIO = 50

DEFAULT_TOP_CONSUMED_LIMIT = 8
DEFAULT_SLOT_CONSUMED_LIMIT = 5

RECOMMENDED_ITEMS_PER_MEAL = 10
KNN_NEIGHBORS = 40
VARIETY_PENALTY_FACTOR = 1.5
RECENT_VARIETY_WINDOW_HOURS = 48
MAX_SERVING_SCALE_FACTOR = 2.0
SERVING_FIT_TOLERANCE = 0.15
CALORIE_MISMATCH_THRESHOLD = 0.20
# Relaxed mapping threshold allows high-title-similarity matches to be cached
# when real-world nutrition data varies more than strict 20%.
RELAXED_MAPPING_CALORIE_THRESHOLD = 0.38
MAPPING_TITLE_SIMILARITY_FLOOR = 0.42
MIN_FRONTEND_OPTIONS = 20
# Live route diagnostics currently land around 70-80 usable candidates per slot,
# and the fresh AUSNUT rerun still selected candidate_pool_size=80. Keep runtime
# search aligned to that evidence instead of overfetching to the earlier 100/240 budget.
LOCAL_PREFETCH_POOL = 160
LOCAL_CANDIDATE_POOL_PER_MEAL = 80
LOCAL_CANDIDATE_POOL_EXPERIMENT_MIN = 80
LOCAL_CANDIDATE_POOL_EXPERIMENT_MAX = 180
MAX_NEW_MAPPING_LOOKUPS = 20
MAX_MAPPING_LOOKUP_ATTEMPTS = 60
ASYNC_MAPPING_ENABLED = True
# Stale-while-revalidate: return local candidates immediately, enrich macros/images in the background.
STALE_WHILE_REVALIDATE_ENABLED = True
# Keep first response fast: do zero synchronous hidden-search lookups by default.
# Mapping enrichment still runs in background workers and persists to JSON.
SYNC_MAPPING_LOOKUPS = 0
SYNC_MAPPING_LOOKUPS_PER_SLOT = SYNC_MAPPING_LOOKUPS
# Queue a bounded number of promising local candidates per slot for async enrichment.
ASYNC_MAPPING_PREFETCH_PER_SLOT = 8
# One worker is safer for API rate limits and avoids overloading FatSecret.
ASYNC_MAPPING_WORKER_COUNT = 1
ASYNC_MAPPING_QUEUE_SIZE = 240
# Stop retrying impossible mappings forever to avoid 30-minute background loops.
ASYNC_MAPPING_MISS_MAX_RETRIES = 2
ASYNC_MAPPING_MISS_BACKOFF_SECONDS = 1800
# Keep recommendation response fast by minimizing synchronous image enrichment work.
OVERALL_CONSUMED_IMAGE_LOOKUPS = 0
SLOT_CONSUMED_IMAGE_LOOKUPS = 0
CONSUMED_IMAGE_USE_DETAIL_LOOKUP = False
MAX_DUPLICATES_PER_TITLE_PER_SLOT = 1
STOCHASTIC_RANKING_STRENGTH = 0.035

EVAL_K_VALUES = (3, 5, 7, 9)

TIME_DECAY_LAMBDA = 0.12
MIN_HISTORY_ROWS_FOR_CLUSTERING = 24
MAX_ARCHETYPE_CLUSTERS = 3
MIN_CLUSTER_WEIGHT = 0.15
QUERY_EXPANSION_MIN_UNIQUE_RATIO = 0.72
QUERY_EXPANSION_MIN_CANDIDATES = 40
QUERY_EXPANSION_MAX_TOP_DISTANCE = 0.9
QUERY_EXPANSION_QUERY_LIMIT = 4

DEFAULT_RANKING_WEIGHTS = {
    "nutritional_match": 0.32,
    "preference_similarity": 0.20,
    "habit_affinity": 0.25,
    "calorie_fit": 0.13,
    "meal_hint_affinity": 0.10,
}

EXPERIMENT_VARIANTS = {
    "control": {
        "candidate_pool_multiplier": 1.0,
        "ranking_weights": DEFAULT_RANKING_WEIGHTS,
        "mmr_lambda": COMBO_MMR_LAMBDA,
        "combo_reuse_penalty_base": COMBO_REUSE_PENALTY_BASE,
    },
    "diversity_focus": {
        "candidate_pool_multiplier": 1.2,
        "ranking_weights": {
            "nutritional_match": 0.29,
            "preference_similarity": 0.18,
            "habit_affinity": 0.20,
            "calorie_fit": 0.11,
            "meal_hint_affinity": 0.10,
        },
        "mmr_lambda": 0.42,
        "combo_reuse_penalty_base": 0.52,
    },
    "precision_focus": {
        "candidate_pool_multiplier": 0.9,
        "ranking_weights": {
            "nutritional_match": 0.38,
            "preference_similarity": 0.22,
            "habit_affinity": 0.23,
            "calorie_fit": 0.12,
            "meal_hint_affinity": 0.05,
        },
        "mmr_lambda": 0.58,
        "combo_reuse_penalty_base": 0.68,
    },
}
