from __future__ import annotations


import os
import re
import time
from pathlib import Path
from threading import Lock, local
from typing import Any


import numpy as np
import pandas as pd


from .constants import MEAL_KEYWORDS, MEAL_SLOTS
from .ranking import build_text_role_candidate, infer_combo_category, is_candidate_role_compatible
from .utils import canonicalize_title, is_english_title, normalize_text, to_float


try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    faiss = None

try:
    from sklearn.preprocessing import StandardScaler  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    StandardScaler = None

try:
    import duckdb  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    duckdb = None


# Local retrieval is performance-critical for recommendation latency.
# NOTE: DuckDB-backed reads avoid loading the entire dataset into memory.
_DUCKDB_CONN = None
_DUCKDB_CONN_PATH = ""
_DUCKDB_CONN_READ_ONLY = False
_DUCKDB_LOCK = Lock()
_DUCKDB_THREAD_LOCAL = local()
_FEATURE_MEAN = None
_FEATURE_STD = None

MEAL_BITS = {
    "breakfast": 1,
    "lunch": 2,
    "dinner": 4,
}
GENERAL_MEAL_BIT = 8
ALL_MEALS_MASK = MEAL_BITS["breakfast"] | MEAL_BITS["lunch"] | MEAL_BITS["dinner"] | GENERAL_MEAL_BIT


BREAKFAST_HINTS = tuple(MEAL_KEYWORDS.get("breakfast", ()))
LUNCH_HINTS = tuple(MEAL_KEYWORDS.get("lunch", ()))
DINNER_HINTS = tuple(MEAL_KEYWORDS.get("dinner", ()))


def _to_text(value: Any) -> str:
    return str(value or "").strip()


def _env_flag_enabled(name: str) -> bool:
    return normalize_text(os.getenv(name, "")) in {"1", "true", "yes", "on"}

def _parse_first_image(raw_images: Any) -> str | None:
    text = _to_text(raw_images)
    if not text:
        return None
    match = re.search(r"https?://[^\"'\s,)]+", text)
    return match.group(0) if match else None

def _build_hint_pattern(hints: tuple[str, ...]) -> str:
    escaped = [re.escape(token) for token in hints]
    return "|".join(escaped)


def _build_drink_exclusion_pattern(meal_type: str, role_hint: str) -> str:
    meal_key = normalize_text(meal_type)
    role_key = normalize_text(role_hint)
    if role_key != "drink":
        return ""

    dessert_patterns = (
        r"\bcocoa\b",
        r"hot[^a-z0-9]*chocolate",
        r"chocolate[^a-z0-9]*drink",
    )

    if meal_key == "lunch":
        coffee_patterns = (
            r"\bcoffee\b",
            r"\blatte\b",
            r"\bcappuccino\b",
            r"\bespresso\b",
            r"\bmacchiato\b",
            r"cold[^a-z0-9]*brew",
        )
        return "|".join((*coffee_patterns, *dessert_patterns))

    if meal_key == "dinner":
        return "|".join(dessert_patterns)

    return ""


def _contains_terms(series: pd.Series, terms: tuple[str, ...]) -> pd.Series:
    pattern = _build_hint_pattern(tuple(term for term in terms if str(term).strip()))
    if not pattern:
        return pd.Series(False, index=series.index, dtype=bool)
    return series.str.contains(pattern, regex=True, na=False)


def _apply_role_eligibility_filter(working: pd.DataFrame, role_hint: str) -> pd.DataFrame:
    normalized_role = normalize_text(role_hint)
    if working.empty or normalized_role not in {"main", "drink", "side"}:
        return working

    eligibility_mask = working.apply(
        lambda row: (
            lambda candidate: infer_combo_category(candidate) == "main"
            and is_candidate_role_compatible(candidate, "main")
            if normalized_role == "main"
            else infer_combo_category(candidate) == "drink"
            and is_candidate_role_compatible(candidate, "drink")
            if normalized_role == "drink"
            else infer_combo_category(candidate) == "side"
            and is_candidate_role_compatible(candidate, "side")
        )(
            build_text_role_candidate(
                title=row.get("title") or row.get("food_name") or "",
                recipe_category=row.get("recipe_category") or row.get("RecipeCategory") or "",
                keywords=row.get("keywords") or row.get("Keywords") or "",
                ingredient_text=row.get("ingredient_text") or "",
                serving_description=row.get("serving_text") or row.get("RecipeServings") or "",
                food_type=row.get("food_type") or "",
                calories=row.get("serving_calories") or row.get("Calories_100g") or 0.0,
            )
        ),
        axis=1,
    )
    filtered = working.loc[eligibility_mask].reset_index(drop=True)
    return filtered if not filtered.empty else working.iloc[0:0].copy()


def _is_phase7_generic_lunch_fallback_candidate_allowed(candidate: dict[str, Any]) -> bool:
    text = normalize_text(
        " ".join(
            str(candidate.get(key) or "")
            for key in ("title", "recipe_category", "keywords", "ingredient_text")
        )
    )
    if not text:
        return False

    condiment_terms = (
        "aioli",
        "chutney",
        "dip",
        "dressing",
        "gravy",
        "hummus",
        "hommus",
        "mayo",
        "mayonnaise",
        "mustard",
        "pesto",
        "relish",
        "salsa",
        "sauce",
        "spread",
        "vinaigrette",
    )
    sweet_snack_terms = (
        "brownie",
        "cake",
        "chocolate",
        "confection",
        "cookie",
        "donut",
        "doughnut",
        "fudge",
        "mousse",
        "nut bar",
        "protein bar",
        "slice",
        "snack",
        "sweet snacks",
    )
    oil_or_spray_terms = (
        "avocado oil",
        "coconut oil",
        "oil spray",
        "olive oil",
        "spray",
    )

    if any(term in text for term in condiment_terms):
        return False
    if any(term in text for term in sweet_snack_terms):
        return False
    if any(term in text for term in oil_or_spray_terms):
        return False
    return True


def _apply_phase7_generic_lunch_fallback_filter(
    working: pd.DataFrame,
    meal_type: str,
    role_hint: str,
    *,
    enabled: bool,
) -> pd.DataFrame:
    if working.empty or not enabled:
        return working
    if normalize_text(meal_type) != "lunch" or normalize_text(role_hint):
        return working

    eligibility_mask = working.apply(
        lambda row: _is_phase7_generic_lunch_fallback_candidate_allowed(
            {
                "title": row.get("title") or "",
                "recipe_category": row.get("recipe_category") or "",
                "keywords": row.get("keywords") or "",
                "ingredient_text": row.get("ingredient_text") or "",
            }
        ),
        axis=1,
    )
    filtered = working.loc[eligibility_mask].reset_index(drop=True)
    return filtered if not filtered.empty else working


def _is_safe_main_candidate(candidate: dict[str, Any], meal_type: str) -> bool:
    meal_key = normalize_text(meal_type)
    if meal_key not in {"breakfast", "lunch", "dinner"}:
        return True

    title_text = normalize_text(candidate.get("title") or "")
    category_text = normalize_text(candidate.get("recipe_category") or "")
    keyword_text = normalize_text(candidate.get("keywords") or "")
    ingredient_text = normalize_text(candidate.get("ingredient_text") or "")
    combined_text = " ".join(part for part in (title_text, category_text, keyword_text, ingredient_text) if part)
    breakfast_context_text = " ".join(part for part in (title_text, category_text, keyword_text) if part)
    if not combined_text:
        return False

    breakfast_markers = (
        "breakfast",
        "cereal",
        "egg",
        "eggs",
        "french toast",
        "granola",
        "muesli",
        "oatmeal",
        "omelet",
        "omelette",
        "pancake",
        "pancakes",
        "parfait",
        "porridge",
        "toast",
        "waffle",
        "waffles",
        "yogurt",
        "yoghurt",
    )
    breakfast_true_main_markers = (
        "acai",
        "avocado toast",
        "breakfast burrito",
        "egg",
        "eggs",
        "hash brown",
        "oatmeal",
        "omelet",
        "omelette",
        "parfait",
        "porridge",
        "quiche",
        "smoothie bowl",
        "toast",
    )
    breakfast_product_terms = (
        "biscuit",
        "biscuits",
        "breakfast cereal",
        "cereal",
        "cereal bar",
        "cereal bars",
        "cookie",
        "cookies",
        "croissant",
        "croissants",
        "flapjack",
        "flapjacks",
        "granola",
        "macaroon",
        "macaroons",
        "meal bar",
        "muesli",
        "pastry",
        "pastries",
        "protein bar",
        "protein bars",
        "protein powder",
        "protein-powders",
        "viennoiseries",
    )
    breakfast_blocked_meal_terms = (
        "burger",
        "hoagie",
        "mac and cheese",
        "macaroni",
        "panini",
        "pasta",
        "pizza",
        "quesadilla",
        "ravioli",
        "sandwich",
        "sub",
        "taco",
        "tacos",
        "tortellini",
        "wrap",
    )
    breakfast_pack_terms = (
        "keep refrigerated",
        "per pack",
        "serves 1",
        "servings per pack",
        "energy fat",
        "protein bar",
        "fruits and nuts bar",
    )
    supplement_terms = (
        "bodybuilding",
        "bodybuilding-supplements",
        "compléments alimentaires",
        "complements alimentaires",
        "dietary-supplements",
        "nutrition shake",
        "protein powder",
        "protein powders",
        "protein-powders",
        "protéines en poudre",
        "supplement",
        "supplements",
        "whey",
    )
    breakfast_fast_food_brand_terms = (
        "burger king",
        "dunkin",
        "hardee",
        "jack in the box",
        "mcdonald",
        "perkins",
        "quiznos",
        "starbucks",
        "tim hortons",
        "wawa",
    )
    breakfast_handheld_terms = (
        "bagel",
        "biscuit",
        "croissant",
        "mcgriddles",
        "sandwich",
        "sizzli",
        "sub",
        "wrap",
    )
    generic_block_terms = (
        "appetizers & sides",
        "appetizers-sides",
        "build your sampler",
        "choose 2",
        "choose 3",
        "combo",
        "for build your sampler",
        "for mto",
        "ingredients",
        "kids",
        "meal replacement",
        "meal-replacements",
        "nutritionally complete food",
        "sampler",
        "toppings & ingredients",
        "toppings-ingredients",
    )
    snack_family_terms = (
        "appetizers",
        "chips",
        "chips and fries",
        "corn chips",
        "crisps",
        "salty snacks",
        "salty-snacks",
        "snack",
        "snacks",
        "sweet snack",
        "sweet snacks",
        "sweet-snacks",
        "trail mix",
    )
    pasta_product_terms = (
        "cheddar elbows",
        "elbows",
        "mac and cheese",
        "mac n cheese",
        "shells",
        "white cheddar shells",
    )
    lunch_blocked_terms = (
        "alfredo",
        "carbonara",
        "fettuccine",
        "mac and cheese",
        "pasta",
        "quesadilla",
        "ravioli",
        "soft taco",
        "spaghetti",
        "taco",
        "tacos",
        "tortellini",
        "tortelloni",
    )
    private_label_prefix_terms = (
        "tesco ",
        "tesco finest",
        "sainsbury",
        "waitrose",
        "marks and spencer",
        "m&s ",
    )
    branded_meal_context_terms = (
        "meals",
        "meals-with-meat",
        "poultry-meals",
        "prepared-meals",
        "prepared-meats",
        "pasta-dishes",
    )
    dinner_blocked_terms = (
        "burger",
        "hoagie",
        "sandwich",
        "sub",
        "taco",
        "tacos",
        "wrap",
    )
    dinner_packaged_family_terms = (
        "fiskegrateng",
        "fish gratin",
        "gratin",
        "lasagna",
        "lasagne",
        "torti",
        "tortellini",
        "tortelloni",
    )

    has_breakfast_marker = any(term in breakfast_context_text for term in breakfast_markers)
    has_breakfast_true_main_marker = any(term in breakfast_context_text for term in breakfast_true_main_markers)

    if any(term in combined_text for term in supplement_terms):
        return False

    if any(term in combined_text for term in snack_family_terms):
        return False

    if meal_key == "breakfast":
        if any(term in combined_text for term in breakfast_product_terms) and not has_breakfast_true_main_marker:
            return False
        if any(term in combined_text for term in breakfast_pack_terms) and not has_breakfast_marker:
            return False
        if any(term in combined_text for term in breakfast_blocked_meal_terms) and not has_breakfast_marker:
            return False
        if (
            has_breakfast_marker
            and any(term in combined_text for term in breakfast_handheld_terms)
            and any(term in combined_text for term in breakfast_fast_food_brand_terms)
        ):
            return False
        return True

    if any(term in combined_text for term in generic_block_terms):
        return False

    if any(term in combined_text for term in pasta_product_terms):
        return False

    if meal_key in {"lunch", "dinner"} and combined_text.startswith(private_label_prefix_terms) and any(
        term in combined_text for term in branded_meal_context_terms
    ):
        return False

    if meal_key == "lunch" and any(term in combined_text for term in lunch_blocked_terms):
        return False

    if meal_key == "dinner" and any(term in combined_text for term in dinner_blocked_terms):
        return False

    if meal_key == "dinner" and any(term in combined_text for term in dinner_packaged_family_terms):
        return False

    return True


def _is_safe_side_candidate(candidate: dict[str, Any], meal_type: str) -> bool:
    meal_key = normalize_text(meal_type)
    if meal_key not in {"breakfast", "lunch", "dinner"}:
        return True

    text = normalize_text(
        " ".join(
            str(candidate.get(key) or "")
            for key in ("title", "recipe_category", "keywords", "ingredient_text")
        )
    )
    if not text:
        return False

    breakfast_safe_terms = (
        "apple",
        "banana",
        "berries",
        "fruit",
        "fruit cup",
        "granola",
        "greek yogurt",
        "muesli",
        "nuts",
        "toast",
        "yogurt",
        "yoghurt",
    )
    breakfast_blocked_terms = (
        "cheddar cheese",
        "fruit with cheddar cheese",
        "french toast",
        "hotcakes",
        "muffin",
        "pancake",
        "pancakes",
        "waffle",
        "waffles",
    )
    lunch_safe_terms = (
        "apple",
        "banana",
        "bean",
        "beans",
        "broccoli",
        "fruit",
        "garden salad",
        "greens",
        "lentil",
        "nuts",
        "salad",
        "side salad",
        "slaw",
        "soup",
        "vegetable",
        "yogurt",
        "yoghurt",
    )
    lunch_blocked_terms = (
        "burger",
        "carbonara",
        "mac and cheese",
        "pasta",
        "pizza",
        "quesadilla",
        "ravioli",
        "sandwich",
        "soft taco",
        "spaghetti",
        "taco",
        "tacos",
        "tortellini",
        "tortelloni",
        "wrap",
    )
    lunch_protein_or_meal_salad_terms = (
        "asiatisch",
        "beef",
        "chicken",
        "composed",
        "compos",
        "cobb",
        "meal",
        "pollo",
        "pork",
        "poulet",
        "protein",
        "repas",
        "rice",
        "riz",
        "salmon",
        "shrimp",
        "thon",
        "thubfisch",
        "thunfisch",
        "tuna",
        "turkey",
    )
    lunch_beverage_side_terms = (
        "beverage",
        "coffee",
        "drink",
        "iced tea",
        "juice",
        "kombucha",
        "latte",
        "lemonade",
        "milk",
        "smoothie",
        "soda",
        "soft drink",
        "soy beverage",
        "water",
    )
    lunch_pastry_or_snack_terms = (
        "crudda bar",
        "granola bar",
        "protein bar",
        "samosa",
        "samosas",
        "samoussa",
        "samoussas",
        "spring roll",
        "spring rolls",
    )
    lunch_main_like_side_terms = (
        "branzino",
        "bolognese",
        "gratin",
        "lasagna",
        "lasagne",
        "pasta",
        "risotto",
        "schnitzel",
        "spaghetti",
        "torti",
        "tortellini",
        "tortelloni",
    )
    lunch_bread_side_terms = (
        "whole grain bread",
        "whole wheat bread",
        "wholegrain bread",
        "wholemeal bread",
    )
    lunch_packaged_soup_terms = (
        "cup soup",
        "diet soup",
        "instant",
        "miso-cup",
        "ramen",
        "soup cup",
    )

    if meal_key == "breakfast":
        if any(term in text for term in breakfast_blocked_terms):
            return False
        return any(term in text for term in breakfast_safe_terms)

    if meal_key == "lunch":
        if any(term in text for term in lunch_blocked_terms):
            return False
        if any(term in text for term in lunch_beverage_side_terms):
            return False
        if any(term in text for term in lunch_pastry_or_snack_terms):
            return False
        if any(term in text for term in lunch_main_like_side_terms):
            return False
        if any(term in text for term in lunch_bread_side_terms):
            return False
        if (
            any(term in text for term in ("salad", "salade", "slaw", "soup", "soupe"))
            and any(term in text for term in lunch_protein_or_meal_salad_terms)
        ):
            return False
        if any(term in text for term in lunch_packaged_soup_terms) and any(term in text for term in ("soup", "soupe", "broth")):
            return False
        return any(term in text for term in lunch_safe_terms)

    return _is_safe_dinner_side_candidate(candidate)


def _apply_main_meal_safety_filter(working: pd.DataFrame, meal_type: str, role_hint: str) -> pd.DataFrame:
    if working.empty or normalize_text(role_hint) != "main":
        return working

    meal_key = normalize_text(meal_type)
    if meal_key not in {"breakfast", "lunch", "dinner"}:
        return working

    safety_column = f"{meal_key}_main_safe"
    if safety_column in working.columns:
        tagged = working[working[safety_column].fillna(False).astype(bool)].reset_index(drop=True)
        if not tagged.empty:
            working = tagged

    eligibility_mask = working.apply(
        lambda row: _is_safe_main_candidate(
            {
                "title": row.get("title") or "",
                "recipe_category": row.get("recipe_category") or "",
                "keywords": row.get("keywords") or "",
                "ingredient_text": row.get("ingredient_text") or "",
            },
            meal_key,
        ),
        axis=1,
    )
    filtered = working.loc[eligibility_mask].reset_index(drop=True)
    return filtered if not filtered.empty else working.iloc[0:0].copy()


def _is_safe_dinner_side_candidate(candidate: dict[str, Any]) -> bool:
    title_text = normalize_text(
        " ".join(
            str(candidate.get(key) or "")
            for key in ("title", "recipe_category", "keywords", "ingredient_text")
        )
    )
    if not title_text:
        return False

    blocked_main_terms = (
        "avocado toast",
        "bowl",
        "burger",
        "fillet",
        "grilled salmon",
        "pasta",
        "plate",
        "platter",
        "power bowl",
        "protein bowl",
        "proteinbowl",
        "ramen",
        "sandwich",
        "steak",
        "toast",
        "wrap",
    )
    safe_family_terms = (
        "bean salad",
        "broccoli",
        "broccoli slaw",
        "cauliflower",
        "chickpea salad",
        "clear soup",
        "coleslaw",
        "cucumber salad",
        "garden salad",
        "green beans",
        "lentil salad",
        "miso soup",
        "roasted vegetables",
        "salad",
        "side salad",
        "slaw",
        "soup",
        "steamed vegetables",
        "tomato soup",
        "vegetable",
        "vegetable medley",
        "vegetable soup",
        "veggies",
    )
    mainish_meal_terms = (
        "chicken",
        "beef",
        "pork",
        "turkey",
        "salmon",
        "shrimp",
        "sausage",
        "steak",
        "chicken strips",
        "mac and cheese",
        "grain",
        "grains",
        "fried rice",
        "jasmine rice",
        "pilaf",
        "risotto",
    )
    safe_context_terms = (
        "bean",
        "beans",
        "broth",
        "chickpea",
        "lentil",
        "miso",
        "salad",
        "slaw",
        "soup",
        "tomato soup",
        "vegetable",
        "vegetable soup",
    )

    if any(term in title_text for term in blocked_main_terms) and not any(term in title_text for term in safe_family_terms):
        return False
    if any(term in title_text for term in mainish_meal_terms) and not any(term in title_text for term in safe_context_terms):
        return False
    if "soup" in title_text:
        protein_soup_terms = ("chicken", "beef", "pork", "turkey", "sausage", "steak")
        safe_soup_terms = ("miso", "broth", "vegetable", "tomato")
        if any(term in title_text for term in protein_soup_terms) and not any(term in title_text for term in safe_soup_terms):
            return False

    return any(term in title_text for term in safe_family_terms)


def _apply_side_meal_safety_filter(working: pd.DataFrame, meal_type: str, role_hint: str) -> pd.DataFrame:
    if working.empty:
        return working
    if normalize_text(role_hint) != "side":
        return working

    meal_key = normalize_text(meal_type)
    if meal_key not in {"breakfast", "lunch", "dinner"}:
        return working

    safety_column = f"{meal_key}_side_safe"
    if safety_column in working.columns:
        tagged = working[working[safety_column].fillna(False).astype(bool)].reset_index(drop=True)
        if not tagged.empty:
            return tagged

    if meal_key == "dinner" and "dinner_side_safe" in working.columns:
        tagged = working[working["dinner_side_safe"].fillna(False).astype(bool)].reset_index(drop=True)
        if not tagged.empty:
            return tagged

    eligibility_mask = working.apply(
        lambda row: _is_safe_side_candidate(
            {
                "title": row.get("title") or "",
                "recipe_category": row.get("recipe_category") or "",
                "keywords": row.get("keywords") or "",
                "ingredient_text": row.get("ingredient_text") or "",
            },
            meal_key,
        ),
        axis=1,
    )
    filtered = working.loc[eligibility_mask].reset_index(drop=True)
    return filtered if not filtered.empty else working.iloc[0:0].copy()


def _series_to_bool(series: pd.Series | Any, default: bool = False) -> pd.Series:
    if not isinstance(series, pd.Series):
        return pd.Series(dtype=bool)
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().any():
        return numeric.fillna(1 if default else 0).astype(int).astype(bool)
    normalized = series.astype(str).str.strip().str.lower()
    truthy = {"1", "true", "t", "yes", "y"}
    falsy = {"0", "false", "f", "no", "n", "", "none", "nan", "null"}
    fallback = normalized.map(lambda value: default if value in falsy else value in truthy)
    return fallback.astype(bool)

def _safe_std(values: np.ndarray) -> np.ndarray:
    # Use float64 stats for stability on million-row datasets with outliers.
    std = values.std(axis=0, dtype=np.float64).astype(np.float64)
    std[~np.isfinite(std)] = 1.0
    std[std < 1e-9] = 1.0
    return std

def _extract_float(text: str) -> float | None:
    normalized = str(text or "").replace(",", "").strip()
    if not normalized:
        return None
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)", normalized)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def _parse_serving_grams(raw_serving: Any) -> float:
    """
    Parse serving text into grams to convert per-100g nutrients into per-serving nutrients.
    Common patterns: "100g", "1 Tbsp (14 g)", "8 OZA (240 ml)", "240 ml".
    """
    text = _to_text(raw_serving).lower()
    if not text:
        return 100.0

    in_parentheses_g = re.search(r"\(([\d.,]+)\s*g\)", text)
    if in_parentheses_g:
        grams = _extract_float(in_parentheses_g.group(1))
        if grams and grams > 0:
            return float(np.clip(grams, 5.0, 800.0))

    plain_g = re.search(r"([\d.,]+)\s*g\b", text)
    if plain_g:
        grams = _extract_float(plain_g.group(1))
        if grams and grams > 0:
            return float(np.clip(grams, 5.0, 800.0))

    in_parentheses_ml = re.search(r"\(([\d.,]+)\s*ml\)", text)
    if in_parentheses_ml:
        ml = _extract_float(in_parentheses_ml.group(1))
        if ml and ml > 0:
            return float(np.clip(ml, 5.0, 800.0))

    plain_ml = re.search(r"([\d.,]+)\s*ml\b", text)
    if plain_ml:
        ml = _extract_float(plain_ml.group(1))
        if ml and ml > 0:
            return float(np.clip(ml, 5.0, 800.0))

    fallback = _extract_float(text)
    if fallback and fallback > 0:
        return float(np.clip(fallback, 5.0, 800.0))
    return 100.0


def _default_dataset_path() -> str:
    base_dir = Path(__file__).resolve().parent.parent
    return str(base_dir / "dataset_process" / "cleaned_food_dataset.parquet")


def _default_db_path() -> str:
    base_dir = Path(__file__).resolve().parent.parent
    return str(base_dir / "dataset_process" / "off.db")

def _open_duckdb_connection(db_path: str):
    if duckdb is None:
        return None, False

    read_only_error = None
    for read_only in (True, False):
        try:
            conn = duckdb.connect(db_path, read_only=read_only)
            if not read_only and read_only_error is not None:
                print(
                    "**** Local Dataset: read-only DuckDB open failed; "
                    f"falling back to writable access ({read_only_error})."
                )
            return conn, read_only
        except Exception as exc:  # pragma: no cover - defensive
            if read_only:
                read_only_error = exc
            continue

    final_error = read_only_error or "unknown error"
    print(f"**** Local Dataset: failed to open DuckDB ({final_error}).")
    return None, False

def _get_duckdb_connection(db_path: str):
    if duckdb is None:
        return None
    global _DUCKDB_CONN, _DUCKDB_CONN_PATH, _DUCKDB_CONN_READ_ONLY
    with _DUCKDB_LOCK:
        if _DUCKDB_CONN is not None and _DUCKDB_CONN_PATH == db_path:
            return _DUCKDB_CONN

        # NOTE: Keep a single shared connection to avoid cold-start overhead.
        _DUCKDB_CONN, _DUCKDB_CONN_READ_ONLY = _open_duckdb_connection(db_path)
        if _DUCKDB_CONN is None:
            _DUCKDB_CONN_PATH = ""
            _DUCKDB_CONN_READ_ONLY = False
            return None

        _DUCKDB_CONN_PATH = db_path
        return _DUCKDB_CONN

def _get_thread_duckdb_connection(db_path: str):
    if duckdb is None:
        return None

    cached_conn = getattr(_DUCKDB_THREAD_LOCAL, "conn", None)
    cached_path = getattr(_DUCKDB_THREAD_LOCAL, "path", "")
    if cached_conn is not None and cached_path == db_path:
        return cached_conn

    conn, read_only = _open_duckdb_connection(db_path)
    if conn is None:
        return None

    _DUCKDB_THREAD_LOCAL.conn = conn
    _DUCKDB_THREAD_LOCAL.path = db_path
    _DUCKDB_THREAD_LOCAL.read_only = read_only
    return conn

def build_query_vector(slot_target: float, user_vec: np.ndarray | None) -> np.ndarray:
    target_kcal = max(1.0, float(slot_target))


    if user_vec is not None and len(user_vec) >= 4 and to_float(user_vec[0], 0.0) > 0.0:
        history_calories = max(1.0, to_float(user_vec[0], 1.0))
        ratio = target_kcal / history_calories
        protein = max(0.0, to_float(user_vec[1], 0.0) * ratio)
        carbs = max(0.0, to_float(user_vec[2], 0.0) * ratio)
        fats = max(0.0, to_float(user_vec[3], 0.0) * ratio)
    else:
        protein = (target_kcal * 0.25) / 4.0
        carbs = (target_kcal * 0.45) / 4.0
        fats = (target_kcal * 0.30) / 9.0


    return np.asarray([target_kcal, protein, carbs, fats], dtype=np.float32)




class LocalFoodDataset:
    def __init__(self, dataset_path: str | None = None):
        self.dataset_path = dataset_path or os.getenv("LOCAL_FOOD_DATASET_PATH", "").strip() or _default_dataset_path()
        self.is_ready = False
        self.search_backend = "duckdb"
        # NOTE: Prefer DuckDB off.db for faster local candidate retrieval.
        self.db_path = (
            os.getenv("LOCAL_FOOD_DB_PATH", "").strip()
            or (self.dataset_path if str(self.dataset_path).lower().endswith(".db") else "")
            or _default_db_path()
        )
        self.db_table = os.getenv("LOCAL_FOOD_DB_TABLE", "cleaned_food_data").strip() or "cleaned_food_data"
        self.conn = None
        self._available_columns: set[str] = set()
        self._sugar_column = ""
        self._ingredient_column = ""
        self._vector_table = self.db_table
        self._vss_enabled = False


        self.frame = pd.DataFrame()
        self.features = np.zeros((0, 4), dtype=np.float64)
        self.normalized_features = np.zeros((0, 4), dtype=np.float32)
        self.feature_mean = np.zeros((4,), dtype=np.float64)
        self.feature_std = np.ones((4,), dtype=np.float64)
        self.meal_masks = np.zeros((0,), dtype=np.uint8)
        self.recipe_ids = np.array([], dtype=str)
        self.index = None


        self._load()

    def _get_search_connection(self, dedicated_connection: bool = False):
        if not dedicated_connection:
            return self.conn

        thread_conn = _get_thread_duckdb_connection(self.db_path)
        return thread_conn if thread_conn is not None else self.conn


    def _load(self) -> None:
        if duckdb is None:
            print("**** Local Dataset: duckdb is not installed, unable to load off.db.")
            return

        db_file = Path(self.db_path)
        if not db_file.exists():
            print(f"**** Local Dataset: db file not found at {db_file}.")
            return

        self.conn = _get_duckdb_connection(str(db_file))
        if self.conn is None:
            return

        try:
            self.conn.execute(f"SELECT 1 FROM {self.db_table} LIMIT 1").fetchone()
        except Exception as exc:
            print(f"**** Local Dataset: missing table '{self.db_table}' ({exc}).")
            return

        # NOTE: Cache the available columns so we can adapt to schema changes.
        try:
            self._available_columns = {
                str(row[1]) for row in self.conn.execute(f"PRAGMA table_info('{self.db_table}')").fetchall()
            }
        except Exception:
            self._available_columns = set()
        self._ingredient_column = "ingredient_text" if "ingredient_text" in self._available_columns else ""
        self._breakfast_main_safe_column = "breakfast_main_safe" if "breakfast_main_safe" in self._available_columns else ""
        self._lunch_main_safe_column = "lunch_main_safe" if "lunch_main_safe" in self._available_columns else ""
        self._dinner_main_safe_column = "dinner_main_safe" if "dinner_main_safe" in self._available_columns else ""
        self._breakfast_side_safe_column = "breakfast_side_safe" if "breakfast_side_safe" in self._available_columns else ""
        self._lunch_side_safe_column = "lunch_side_safe" if "lunch_side_safe" in self._available_columns else ""
        self._dinner_side_safe_column = "dinner_side_safe" if "dinner_side_safe" in self._available_columns else ""
        self._dinner_side_family_column = "dinner_side_family" if "dinner_side_family" in self._available_columns else ""
        self._dinner_side_priority_column = "dinner_side_priority" if "dinner_side_priority" in self._available_columns else ""
        self._dinner_side_reason_column = "dinner_side_reason" if "dinner_side_reason" in self._available_columns else ""
        self._serving_grams_column = "serving_grams" if "serving_grams" in self._available_columns else ""
        self._serving_calories_column = "serving_calories" if "serving_calories" in self._available_columns else ""
        self._serving_protein_column = "serving_protein" if "serving_protein" in self._available_columns else ""
        self._serving_carbs_column = "serving_carbs" if "serving_carbs" in self._available_columns else ""
        self._serving_fats_column = "serving_fats" if "serving_fats" in self._available_columns else ""
        if "sugar_100g" in self._available_columns:
            self._sugar_column = "sugar_100g"
        elif "sugars_100g" in self._available_columns:
            self._sugar_column = "sugars_100g"
        else:
            self._sugar_column = ""

        # NOTE: Pre-build SQL fragments for fast candidate queries.
        self._title_text_expr = "lower(coalesce(food_name, ''))"
        self._combined_text_expr = (
            "lower(coalesce(food_name, '') || ' ' || coalesce(RecipeCategory, '') || ' ' || coalesce(Keywords, ''))"
        )
        # NOTE: SQL-based serving grams parsing mirrors _parse_serving_grams.
        serving_text = "lower(coalesce(RecipeServings, ''))"
        grams_in_paren = f"NULLIF(regexp_extract({serving_text}, '\\\\(([\\\\d.,]+)\\\\s*g\\\\)', 1), '')"
        grams_plain = f"NULLIF(regexp_extract({serving_text}, '([\\\\d.,]+)\\\\s*g\\\\b', 1), '')"
        ml_in_paren = f"NULLIF(regexp_extract({serving_text}, '\\\\(([\\\\d.,]+)\\\\s*ml\\\\)', 1), '')"
        ml_plain = f"NULLIF(regexp_extract({serving_text}, '([\\\\d.,]+)\\\\s*ml\\\\b', 1), '')"
        fallback_num = f"NULLIF(regexp_extract({serving_text}, '([\\\\d.,]+)', 1), '')"
        parsed_serving_grams_expr = (
            "LEAST(GREATEST(COALESCE("
            f"try_cast(replace({grams_in_paren}, ',', '') AS DOUBLE), "
            f"try_cast(replace({grams_plain}, ',', '') AS DOUBLE), "
            f"try_cast(replace({ml_in_paren}, ',', '') AS DOUBLE), "
            f"try_cast(replace({ml_plain}, ',', '') AS DOUBLE), "
            f"try_cast(replace({fallback_num}, ',', '') AS DOUBLE), "
            "100.0"
            "), 5.0), 800.0)"
        )
        self._serving_grams_expr = self._serving_grams_column or parsed_serving_grams_expr
        self._serving_calories_expr = self._serving_calories_column or f"(Calories_100g * ({self._serving_grams_expr} / 100.0))"
        self._serving_protein_expr = self._serving_protein_column or f"(protein_100g * ({self._serving_grams_expr} / 100.0))"
        self._serving_carbs_expr = self._serving_carbs_column or f"(carbs_100g * ({self._serving_grams_expr} / 100.0))"
        self._serving_fats_expr = self._serving_fats_column or f"(fat_100g * ({self._serving_grams_expr} / 100.0))"
        ingredient_expr = self._ingredient_column or "''"
        breakfast_main_safe_expr = self._breakfast_main_safe_column or "NULL"
        lunch_main_safe_expr = self._lunch_main_safe_column or "NULL"
        dinner_main_safe_expr = self._dinner_main_safe_column or "NULL"
        breakfast_side_safe_expr = self._breakfast_side_safe_column or "NULL"
        lunch_side_safe_expr = self._lunch_side_safe_column or "NULL"
        is_au_expr = "is_australian" if "is_australian" in self._available_columns else "FALSE"
        health_expr = "health_score" if "health_score" in self._available_columns else "AggregatedRating" if "AggregatedRating" in self._available_columns else "NULL"
        sugar_expr = self._sugar_column or "NULL"
        dinner_side_safe_expr = self._dinner_side_safe_column or "NULL"
        dinner_side_family_expr = self._dinner_side_family_column or "NULL"
        dinner_side_priority_expr = self._dinner_side_priority_column or "NULL"
        dinner_side_reason_expr = self._dinner_side_reason_column or "NULL"
        # NOTE: Centralize base-table select expressions and vector-table column names separately.
        self._base_select_columns = (
            "RecipeId, food_name, RecipeCategory, Keywords, "
            f"{ingredient_expr} AS ingredient_text, "
            f"{breakfast_main_safe_expr} AS breakfast_main_safe, "
            f"{lunch_main_safe_expr} AS lunch_main_safe, "
            f"{dinner_main_safe_expr} AS dinner_main_safe, "
            f"{breakfast_side_safe_expr} AS breakfast_side_safe, "
            f"{lunch_side_safe_expr} AS lunch_side_safe, "
            f"{is_au_expr} AS is_australian, "
            f"{health_expr} AS health_score, "
            f"{dinner_side_safe_expr} AS dinner_side_safe, "
            f"{dinner_side_family_expr} AS dinner_side_family, "
            f"{dinner_side_priority_expr} AS dinner_side_priority, "
            f"{dinner_side_reason_expr} AS dinner_side_reason, "
            "Calories_100g, protein_100g, carbs_100g, fat_100g, RecipeServings, "
            f"{sugar_expr} AS sugar_100g, "
            f"{self._serving_grams_expr} AS serving_grams, "
            f"{self._serving_calories_expr} AS serving_calories, "
            f"{self._serving_protein_expr} AS serving_protein, "
            f"{self._serving_carbs_expr} AS serving_carbs, "
            f"{self._serving_fats_expr} AS serving_fats"
        )
        self._vector_select_columns = (
            "RecipeId, food_name, RecipeCategory, Keywords, "
            "ingredient_text, breakfast_main_safe, lunch_main_safe, dinner_main_safe, breakfast_side_safe, lunch_side_safe, is_australian, health_score, dinner_side_safe, dinner_side_family, dinner_side_priority, dinner_side_reason, "
            "Calories_100g, protein_100g, carbs_100g, fat_100g, RecipeServings, "
            "sugar_100g, serving_grams, serving_calories, serving_protein, serving_carbs, serving_fats"
        )
        self._vector_column_names = [
            "RecipeId",
            "food_name",
            "RecipeCategory",
            "Keywords",
            "ingredient_text",
            "breakfast_main_safe",
            "lunch_main_safe",
            "dinner_main_safe",
            "breakfast_side_safe",
            "lunch_side_safe",
            "is_australian",
            "health_score",
            "dinner_side_safe",
            "dinner_side_family",
            "dinner_side_priority",
            "dinner_side_reason",
            "Calories_100g",
            "protein_100g",
            "carbs_100g",
            "fat_100g",
            "RecipeServings",
            "sugar_100g",
            "serving_grams",
            "serving_calories",
            "serving_protein",
            "serving_carbs",
            "serving_fats",
            "embedding",
        ]
        self._base_where_clause = (
            "Calories_100g BETWEEN 1.0 AND 2000.0 "
            "AND protein_100g BETWEEN 0.0 AND 130.0 "
            "AND carbs_100g BETWEEN 0.0 AND 180.0 "
            "AND fat_100g BETWEEN 0.0 AND 130.0 "
            "AND food_name IS NOT NULL"
        )
        self._meal_regex = {
            "breakfast": _build_hint_pattern(BREAKFAST_HINTS),
            "lunch": _build_hint_pattern(LUNCH_HINTS),
            "dinner": _build_hint_pattern(DINNER_HINTS),
        }

        # NOTE: Pre-compute StandardScaler stats once to avoid per-request fitting.
        self._compute_feature_scaler()
        # NOTE: Attempt to enable DuckDB VSS/HNSW acceleration (falls back if unavailable).
        self._try_enable_vss()

        self.is_ready = True
        self.search_backend = "duckdb_vss" if self._vss_enabled else "duckdb"
        print(
            "**** Local Dataset Ready:",
            f"backend={self.search_backend}",
            f"db={db_file}",
            f"table={self.db_table}",
            f"vss={int(self._vss_enabled)}",
        )

    def warmup(self, dedicated_connection: bool = False) -> None:
        if not self.conn:
            return
        conn = self._get_search_connection(dedicated_connection)
        if conn is None:
            return
        try:
            # NOTE: Warm-up query to hydrate DuckDB metadata/page cache.
            conn.execute(f"SELECT COUNT(*) FROM {self.db_table}").fetchone()
            if self._vss_enabled and self._vector_table:
                conn.execute(f"SELECT 1 FROM {self._vector_table} LIMIT 1").fetchone()
        except Exception:
            return

    def _compute_feature_scaler(self) -> None:
        if self.conn is None:
            return
        global _FEATURE_MEAN, _FEATURE_STD
        if _FEATURE_MEAN is not None and _FEATURE_STD is not None:
            self.feature_mean = np.asarray(_FEATURE_MEAN, dtype=np.float64)
            self.feature_std = np.asarray(_FEATURE_STD, dtype=np.float64)
            return

        stats_sql = (
            "SELECT "
            f"avg({self._serving_calories_expr}) AS cal_mean, stddev_pop({self._serving_calories_expr}) AS cal_std, "
            f"avg({self._serving_protein_expr}) AS pro_mean, stddev_pop({self._serving_protein_expr}) AS pro_std, "
            f"avg({self._serving_carbs_expr}) AS carb_mean, stddev_pop({self._serving_carbs_expr}) AS carb_std, "
            f"avg({self._serving_fats_expr}) AS fat_mean, stddev_pop({self._serving_fats_expr}) AS fat_std "
            f"FROM {self.db_table} WHERE {self._base_where_clause}"
        )
        try:
            row = self.conn.execute(stats_sql).fetchone()
        except Exception:
            row = None
        if not row:
            return

        means = np.asarray([row[0], row[2], row[4], row[6]], dtype=np.float64)
        stds = np.asarray([row[1], row[3], row[5], row[7]], dtype=np.float64)
        if not np.all(np.isfinite(means)):
            means = np.zeros((4,), dtype=np.float64)
        if not np.all(np.isfinite(stds)):
            stds = _safe_std(np.zeros((1, 4), dtype=np.float64))
        stds[stds < 1e-9] = 1.0

        self.feature_mean = means
        self.feature_std = stds
        _FEATURE_MEAN = means.copy()
        _FEATURE_STD = stds.copy()

    def _try_enable_vss(self) -> None:
        self._vss_enabled = False
        self._vector_table = self.db_table
        if self.conn is None:
            return
        try:
            self.conn.execute("LOAD vss")
        except Exception:
            try:
                # NOTE: INSTALL can fail offline; ignore and fall back to SQL distance.
                self.conn.execute("INSTALL vss")
                self.conn.execute("LOAD vss")
            except Exception as exc:
                print(f"**** DuckDB VSS disabled: {exc}")
                return

        vector_table = f"{self.db_table}_vss"
        try:
            tables = {row[0] for row in self.conn.execute("SHOW TABLES").fetchall()}
            recreate_table = vector_table not in tables
            if recreate_table and bool(_DUCKDB_CONN_READ_ONLY):
                print("**** DuckDB VSS disabled: prebuilt vector table missing on read-only connection.")
                return
            if not recreate_table:
                try:
                    info_rows = self.conn.execute(f"PRAGMA table_info('{vector_table}')").fetchall()
                    existing_cols = {str(row[1]) for row in info_rows}
                    type_map = {str(row[1]): str(row[2]) for row in info_rows}
                    missing = [name for name in self._vector_column_names if name not in existing_cols]
                    embedding_type = normalize_text(type_map.get("embedding"))
                    if missing or embedding_type != "float[4]":
                        # NOTE: Schema drift between base and vector tables; rebuild vector table.
                        if bool(_DUCKDB_CONN_READ_ONLY):
                            print(
                                "**** DuckDB VSS disabled: stale read-only vector table is missing required columns; "
                                "falling back to SQL distance."
                            )
                            return
                        self.conn.execute(f"DROP TABLE IF EXISTS {vector_table}")
                        recreate_table = True
                except Exception:
                    if bool(_DUCKDB_CONN_READ_ONLY):
                        print("**** DuckDB VSS disabled: unable to validate read-only vector table schema.")
                        return
                    recreate_table = True

            if recreate_table:
                embedding_expr = (
                    "CAST(list_value("
                    f"{self._serving_calories_expr}, {self._serving_protein_expr}, "
                    f"{self._serving_carbs_expr}, {self._serving_fats_expr}"
                    ") AS FLOAT[4])"
                )
                create_sql = (
                    f"CREATE TABLE {vector_table} AS SELECT {self._base_select_columns}, "
                    f"{embedding_expr} AS embedding FROM {self.db_table}"
                )
                self.conn.execute(create_sql)
            if not bool(_DUCKDB_CONN_READ_ONLY):
                try:
                    self.conn.execute("SET hnsw_enable_experimental_persistence = true")
                except Exception:
                    pass
                try:
                    self.conn.execute(
                        f"CREATE INDEX {vector_table}_hnsw ON {vector_table} USING HNSW (embedding)"
                    )
                except Exception:
                    pass

            self._vector_table = vector_table
            self._vss_enabled = True
        except Exception as exc:
            print(f"**** DuckDB VSS fallback to SQL distance ({exc})")
            self._vector_table = self.db_table
            self._vss_enabled = False

    def _build_safety_pushdown_clause(self, meal_type: str, role_hint: str | None) -> str:
        meal_key = normalize_text(meal_type)
        role_key = normalize_text(role_hint or "")
        if meal_key not in {"breakfast", "lunch", "dinner"}:
            return ""

        safety_column = ""
        if role_key == "main":
            safety_column = {
                "breakfast": self._breakfast_main_safe_column,
                "lunch": self._lunch_main_safe_column,
                "dinner": self._dinner_main_safe_column,
            }.get(meal_key, "")
        elif role_key == "side":
            if meal_key == "dinner":
                safety_column = self._dinner_side_safe_column
            else:
                safety_column = {
                    "breakfast": self._breakfast_side_safe_column,
                    "lunch": self._lunch_side_safe_column,
                }.get(meal_key, "")

        if not safety_column:
            return ""
        return f" AND coalesce({safety_column}, FALSE)"


    def _search_indices(self, normalized_query: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        if self.normalized_features.shape[0] == 0:
            return np.array([], dtype=np.int64), np.array([], dtype=np.float32)


        search_k = max(1, min(int(top_k), self.normalized_features.shape[0]))
        if self.index is not None:
            if hasattr(self.index, "hnsw"):
                self.index.hnsw.efSearch = max(int(self.hnsw_ef_search), int(search_k))
            distances, indices = self.index.search(np.asarray([normalized_query], dtype=np.float32), search_k)
            return indices[0], distances[0]


        # Numpy fallback optimization:
        # - compute distances in float64 for numerical stability,
        # - use argpartition to avoid full O(n log n) sort on millions of rows.
        features64 = self.normalized_features.astype(np.float64, copy=False)
        query64 = np.asarray(normalized_query, dtype=np.float64)
        deltas = features64 - query64
        distances = np.einsum("ij,ij->i", deltas, deltas, dtype=np.float64)

        if search_k >= distances.shape[0]:
            indices = np.argsort(distances)
        else:
            partial = np.argpartition(distances, search_k - 1)[:search_k]
            order = np.argsort(distances[partial])
            indices = partial[order]
        return indices.astype(np.int64), distances[indices].astype(np.float32)


    def search(
        self,
        meal_type: str,
        query_vector: np.ndarray,
        top_k: int = 10,
        prefetch: int = 100,
        exclude_recipe_ids: set[str] | None = None,
        is_australian_user: bool = False,
        text_query: str | None = None,
        role_hint: str | None = None,
        dedicated_connection: bool = False,
        log_search: bool = True,
    ) -> list[dict[str, Any]]:
        if not self.is_ready or self.conn is None:
            return []
        conn = self._get_search_connection(dedicated_connection)
        if conn is None:
            return []
        started_at = time.perf_counter()
        slot = normalize_text(meal_type)
        meal_bit = MEAL_BITS.get(slot, ALL_MEALS_MASK)
        prefer_specific = (slot in MEAL_BITS) or bool(text_query)
        phase7_generic_lunch_fallback_filter_enabled = _env_flag_enabled("PHASE7_GENERIC_LUNCH_FALLBACK_FILTER")
        excluded = {str(value).strip() for value in (exclude_recipe_ids or set()) if str(value).strip()}

        # Trust the caller's explicit prefetch budget so narrow expansion and
        # supplement searches do not pay for a hard 100-row SQL fetch.
        candidate_limit = max(int(prefetch), int(top_k))
        table_name = self._vector_table if self._vss_enabled else self.db_table
        columns = self._vector_select_columns if self._vss_enabled else self._base_select_columns
        drink_exclusion_regex = _build_drink_exclusion_pattern(slot, role_hint or "")
        drink_exclusion_clause = ""
        drink_exclusion_params: list[Any] = []
        if drink_exclusion_regex:
            drink_exclusion_clause = f" AND NOT regexp_matches({self._title_text_expr}, ?)"
            drink_exclusion_params = [drink_exclusion_regex]
        australian_clause = ""
        if is_australian_user and "is_australian" in self._available_columns:
            australian_clause = " AND is_australian = TRUE"
        safety_clause = self._build_safety_pushdown_clause(slot, role_hint)
        if self._vss_enabled:
            # NOTE: Use FLOAT[4] to match the persisted DuckDB VSS index key type.
            distance_expr = (
                "array_distance("
                "CAST(embedding AS FLOAT[4]), "
                "CAST(list_value(CAST(? AS FLOAT), CAST(? AS FLOAT), CAST(? AS FLOAT), CAST(? AS FLOAT)) AS FLOAT[4])"
                ")"
            )
        else:
            distance_expr = (
                f"(({self._serving_calories_expr} - ?) * ({self._serving_calories_expr} - ?) "
                f"+ ({self._serving_protein_expr} - ?) * ({self._serving_protein_expr} - ?) "
                f"+ ({self._serving_carbs_expr} - ?) * ({self._serving_carbs_expr} - ?) "
                f"+ ({self._serving_fats_expr} - ?) * ({self._serving_fats_expr} - ?))"
            )

        query = np.asarray(query_vector, dtype=np.float64).reshape(-1)
        if query.shape[0] != 4:
            raise ValueError("query_vector has invalid dimension for local dataset search.")
        q_cal, q_protein, q_carbs, q_fats = [float(value) for value in query.tolist()]
        if self._vss_enabled:
            distance_params = [q_cal, q_protein, q_carbs, q_fats]
        else:
            distance_params = [q_cal, q_cal, q_protein, q_protein, q_carbs, q_carbs, q_fats, q_fats]

        df = pd.DataFrame()
        if prefer_specific:
            regex = text_query if text_query else self._meal_regex.get(slot, "")
            if regex:
                use_single_pass_regex_priority = not (
                    slot == "lunch" and phase7_generic_lunch_fallback_filter_enabled
                )
                if use_single_pass_regex_priority:
                    # NOTE: Keep regex-matched rows ahead of fallback rows in one pass so
                    # breakfast/dinner cold retrieval avoids a second ordered DuckDB scan.
                    query_sql = (
                        f"SELECT {columns}, "
                        f"CASE WHEN regexp_matches({self._combined_text_expr}, ?) THEN 1 ELSE 0 END AS regex_priority "
                        f"FROM {table_name} "
                        f"WHERE {self._base_where_clause}{australian_clause}{safety_clause} "
                        f"{drink_exclusion_clause} "
                        f"ORDER BY regex_priority DESC, {distance_expr} ASC "
                        f"LIMIT {candidate_limit}"
                    )
                    df = conn.execute(query_sql, [regex, *drink_exclusion_params, *distance_params]).fetchdf()
                else:
                    query_sql = (
                        f"SELECT {columns} FROM {table_name} "
                        f"WHERE {self._base_where_clause}{australian_clause}{safety_clause} "
                        f"AND regexp_matches({self._combined_text_expr}, ?) "
                        f"{drink_exclusion_clause} "
                        f"ORDER BY {distance_expr} ASC "
                        f"LIMIT {candidate_limit}"
                    )
                    df = conn.execute(query_sql, [regex, *drink_exclusion_params, *distance_params]).fetchdf()

                    if len(df) < candidate_limit:
                        remaining = candidate_limit - len(df)
                        fill_sql = (
                            f"SELECT {columns} FROM {table_name} "
                            f"WHERE {self._base_where_clause}{australian_clause}{safety_clause} "
                            f"AND NOT regexp_matches({self._combined_text_expr}, ?) "
                            f"{drink_exclusion_clause} "
                            f"ORDER BY {distance_expr} ASC "
                            f"LIMIT {remaining}"
                        )
                        fallback = conn.execute(fill_sql, [regex, *drink_exclusion_params, *distance_params]).fetchdf()
                        if not fallback.empty:
                            # Phase 7 dev-only: keep generic lunch fallback rows from reintroducing
                            # sauce, spread, oil-spray, and sweet-snack noise when text-matched rows already exist.
                            fallback = _apply_phase7_generic_lunch_fallback_filter(
                                fallback,
                                slot,
                                role_hint or "",
                                enabled=phase7_generic_lunch_fallback_filter_enabled and len(df) > 0,
                            )
                            df = pd.concat([df, fallback], ignore_index=True)

        if df.empty:
            base_sql = (
                f"SELECT {columns} FROM {table_name} "
                f"WHERE {self._base_where_clause}{australian_clause}{safety_clause} "
                f"{drink_exclusion_clause} "
                f"ORDER BY {distance_expr} ASC "
                f"LIMIT {candidate_limit}"
            )
            df = conn.execute(base_sql, [*drink_exclusion_params, *distance_params]).fetchdf()

        if df.empty:
            return []

        working = pd.DataFrame()
        working["recipe_id"] = df.get("RecipeId", "").astype(str).str.strip()
        working["title"] = df.get("food_name", "").astype(str).str.strip()
        working["recipe_category"] = df.get("RecipeCategory", "").astype(str).str.strip()
        working["keywords"] = df.get("Keywords", "").astype(str).str.strip()
        # NOTE: Images column removed from DuckDB; keep placeholders for API shape.
        working["image"] = None
        working["ingredient_text"] = df.get("ingredient_text", "").astype(str).str.strip()
        if "breakfast_main_safe" in df.columns:
            working["breakfast_main_safe"] = _series_to_bool(df.get("breakfast_main_safe"), default=False)
        if "lunch_main_safe" in df.columns:
            working["lunch_main_safe"] = _series_to_bool(df.get("lunch_main_safe"), default=False)
        if "dinner_main_safe" in df.columns:
            working["dinner_main_safe"] = _series_to_bool(df.get("dinner_main_safe"), default=False)
        if "breakfast_side_safe" in df.columns:
            working["breakfast_side_safe"] = _series_to_bool(df.get("breakfast_side_safe"), default=False)
        if "lunch_side_safe" in df.columns:
            working["lunch_side_safe"] = _series_to_bool(df.get("lunch_side_safe"), default=False)
        working["is_australian"] = df.get("is_australian", False)
        working["health_score"] = pd.to_numeric(df.get("health_score"), errors="coerce").fillna(0.0)
        if "dinner_side_safe" in df.columns:
            working["dinner_side_safe"] = _series_to_bool(df.get("dinner_side_safe"), default=False)
        if "dinner_side_family" in df.columns:
            working["dinner_side_family"] = df.get("dinner_side_family", "").astype(str).str.strip()
        if "dinner_side_priority" in df.columns:
            working["dinner_side_priority"] = pd.to_numeric(df.get("dinner_side_priority"), errors="coerce").fillna(9).astype(int)
        if "dinner_side_reason" in df.columns:
            working["dinner_side_reason"] = df.get("dinner_side_reason", "").astype(str).str.strip()

        serving_grams = pd.to_numeric(df.get("serving_grams"), errors="coerce").astype(np.float64)
        serving_grams = serving_grams.where(np.isfinite(serving_grams), 100.0).clip(lower=5.0, upper=800.0)

        raw_calories = pd.to_numeric(df.get("Calories_100g"), errors="coerce").astype(np.float64)
        raw_protein = pd.to_numeric(df.get("protein_100g"), errors="coerce").astype(np.float64)
        raw_carbs = pd.to_numeric(df.get("carbs_100g"), errors="coerce").astype(np.float64)
        raw_fats = pd.to_numeric(df.get("fat_100g"), errors="coerce").astype(np.float64)
        raw_sugar = pd.to_numeric(df.get("sugar_100g"), errors="coerce").astype(np.float64)

        for series in (raw_calories, raw_protein, raw_carbs, raw_fats, raw_sugar):
            series.replace([np.inf, -np.inf], np.nan, inplace=True)
            series.fillna(0.0, inplace=True)

        serving_factor = (serving_grams / 100.0).astype(np.float64)
        working["serving_grams"] = serving_grams
        serving_calories = pd.to_numeric(df.get("serving_calories"), errors="coerce").astype(np.float64)
        serving_protein = pd.to_numeric(df.get("serving_protein"), errors="coerce").astype(np.float64)
        serving_carbs = pd.to_numeric(df.get("serving_carbs"), errors="coerce").astype(np.float64)
        serving_fats = pd.to_numeric(df.get("serving_fats"), errors="coerce").astype(np.float64)
        if serving_calories.isna().any():
            serving_calories = (raw_calories * serving_factor).fillna(0.0)
        if serving_protein.isna().any():
            serving_protein = (raw_protein * serving_factor).fillna(0.0)
        if serving_carbs.isna().any():
            serving_carbs = (raw_carbs * serving_factor).fillna(0.0)
        if serving_fats.isna().any():
            serving_fats = (raw_fats * serving_factor).fillna(0.0)
        serving_sugar = raw_sugar * serving_factor
        working["serving_calories"] = serving_calories
        working["serving_protein"] = serving_protein
        working["serving_carbs"] = serving_carbs
        working["serving_fats"] = serving_fats
        working["serving_sugar"] = serving_sugar
        working["calories_100g"] = raw_calories
        working["protein_100g"] = raw_protein
        working["carbs_100g"] = raw_carbs
        working["fats_100g"] = raw_fats
        working["sugar_100g"] = raw_sugar

        working = working[
            (working["recipe_id"] != "")
            & (working["title"] != "")
            & (working["serving_calories"] > 0.0)
        ].reset_index(drop=True)

        english_mask = working["title"].map(is_english_title)
        working = working.loc[english_mask].reset_index(drop=True)
        if working.empty:
            return []

        working = _apply_role_eligibility_filter(working, role_hint or "")
        working = _apply_main_meal_safety_filter(working, slot, role_hint or "")
        working = _apply_side_meal_safety_filter(working, slot, role_hint or "")
        if working.empty:
            return []

        if slot == "dinner" and normalize_text(role_hint or "") == "side" and "dinner_side_priority" in working.columns:
            working = working.sort_values(
                by=["dinner_side_priority", "serving_calories", "title"],
                ascending=[True, True, True],
                kind="stable",
            ).reset_index(drop=True)

        combined_text = (
            working["title"].fillna("")
            + " "
            + working["recipe_category"].fillna("")
            + " "
            + working["keywords"].fillna("")
        ).str.lower()

        breakfast_mask = combined_text.str.contains(_build_hint_pattern(BREAKFAST_HINTS), regex=True, na=False)
        lunch_mask = combined_text.str.contains(_build_hint_pattern(LUNCH_HINTS), regex=True, na=False)
        dinner_mask = combined_text.str.contains(_build_hint_pattern(DINNER_HINTS), regex=True, na=False)

        meal_mask = (
            breakfast_mask.astype(np.uint8) * MEAL_BITS["breakfast"]
            + lunch_mask.astype(np.uint8) * MEAL_BITS["lunch"]
            + dinner_mask.astype(np.uint8) * MEAL_BITS["dinner"]
        ).astype(np.uint8)
        meal_mask = np.where(meal_mask == 0, GENERAL_MEAL_BIT, meal_mask).astype(np.uint8)

        features = np.asarray(
            working[["serving_calories", "serving_protein", "serving_carbs", "serving_fats"]].to_numpy(dtype=np.float64),
            dtype=np.float64,
        )
        mean = np.asarray(self.feature_mean, dtype=np.float64)
        std = np.asarray(self.feature_std, dtype=np.float64)
        if mean.shape != (4,) or std.shape != (4,) or not np.all(np.isfinite(mean)):
            mean = features.mean(axis=0, dtype=np.float64).astype(np.float64)
        if std.shape != (4,) or not np.all(np.isfinite(std)):
            std = _safe_std(features)
        std[std < 1e-9] = 1.0

        if StandardScaler is not None:
            # NOTE: Use precomputed StandardScaler stats instead of fitting every request.
            scaler = StandardScaler()
            scaler.mean_ = mean
            scaler.scale_ = std
            scaler.var_ = std**2
            scaler.n_features_in_ = 4
            normalized = scaler.transform(features).astype(np.float32)
        else:
            normalized = np.asarray((features - mean) / std, dtype=np.float32)

        normalized_query = np.asarray((query - mean) / std, dtype=np.float32)

        deltas = normalized.astype(np.float64, copy=False) - normalized_query.astype(np.float64)
        distances = np.einsum("ij,ij->i", deltas, deltas, dtype=np.float64)

        if top_k >= distances.shape[0]:
            indices = np.argsort(distances)
        else:
            partial = np.argpartition(distances, top_k - 1)[:top_k]
            order = np.argsort(distances[partial])
            indices = partial[order]

        output: list[dict[str, Any]] = []
        general_pool: list[dict[str, Any]] = []
        seen_recipe_ids: set[str] = set()

        for idx in indices.tolist():
            row = working.iloc[idx]
            recipe_id = str(row.get("recipe_id") or "").strip()
            if not recipe_id or recipe_id in excluded or recipe_id in seen_recipe_ids:
                continue

            mask = int(meal_mask[idx])
            if prefer_specific:
                if (mask & meal_bit) == 0 and (mask & GENERAL_MEAL_BIT) == 0:
                    continue
            else:
                if (mask & ALL_MEALS_MASK) == 0:
                    continue

            serving_calories = to_float(row.get("serving_calories"), 0.0)
            if serving_calories <= 0.0:
                continue

            title = str(row.get("title") or "").strip()
            if not title:
                continue

            seen_recipe_ids.add(recipe_id)
            candidate_payload = {
                "id": f"recipe-{recipe_id}",
                "item_id": f"recipe-{recipe_id}",
                "recipe_id": recipe_id,
                "title": title,
                "original_title": title,
                "canonical_title": canonicalize_title(title),
                "dataset_title": title,
                "image": str(row.get("image") or "").strip() or None,
                "meal_type": slot if slot in MEAL_SLOTS else "lunch",
                "source_keyword": str(row.get("recipe_category") or "").strip(),
                "recipe_category": str(row.get("recipe_category") or "").strip(),
                "keywords": str(row.get("keywords") or "").strip(),
                # NOTE: Health and locale metadata are used by ranking penalties/boosts.
                "health_score": round(to_float(row.get("health_score"), 0.0), 3),
                "aggregated_rating": round(to_float(row.get("health_score"), 0.0), 3),
                "is_australian": bool(row.get("is_australian")),
                "normalized_ingredients": normalize_text(row.get("ingredient_text")),
                "review_count": 0,
                "serving_description": f"1 serving ({int(round(to_float(row.get('serving_grams'), 100.0)))} g)",
                "metric_serving_amount": round(to_float(row.get("serving_grams"), 100.0), 3),
                "metric_serving_unit": "g",
                "number_of_units": 1.0,
                "measurement_description": "serving",
                "serving_calories": round(serving_calories, 3),
                "serving_protein": round(to_float(row.get("serving_protein"), 0.0), 3),
                "serving_carbs": round(to_float(row.get("serving_carbs"), 0.0), 3),
                "serving_fats": round(to_float(row.get("serving_fats"), 0.0), 3),
                "serving_sugar": round(to_float(row.get("serving_sugar"), 0.0), 3),
                "dataset_serving_calories": round(serving_calories, 3),
                "dataset_serving_protein": round(to_float(row.get("serving_protein"), 0.0), 3),
                "dataset_serving_carbs": round(to_float(row.get("serving_carbs"), 0.0), 3),
                "dataset_serving_fats": round(to_float(row.get("serving_fats"), 0.0), 3),
                "dataset_serving_sugar": round(to_float(row.get("serving_sugar"), 0.0), 3),
                "per100": {
                    "calories": round(to_float(row.get("calories_100g"), 0.0), 3),
                    "protein": round(to_float(row.get("protein_100g"), 0.0), 3),
                    "carbs": round(to_float(row.get("carbs_100g"), 0.0), 3),
                    "fats": round(to_float(row.get("fats_100g"), 0.0), 3),
                    "sugar": round(to_float(row.get("sugar_100g"), 0.0), 3),
                },
                "knn_distance": round(float(distances[idx]), 6),
            }

            if prefer_specific and (mask & meal_bit) == 0:
                general_pool.append(candidate_payload)
            else:
                output.append(candidate_payload)

            if len(output) >= top_k:
                break

        if len(output) < top_k and general_pool:
            needed = max(0, int(top_k) - len(output))
            output.extend(general_pool[:needed])

        if log_search:
            print(
                f"**** Local Search ({slot}): requested={top_k}, "
                f"returned={len(output)}, candidates={candidate_limit}, backend={self.search_backend}, "
                f"elapsed_ms={(time.perf_counter() - started_at) * 1000:.1f}"
            )
        return output
