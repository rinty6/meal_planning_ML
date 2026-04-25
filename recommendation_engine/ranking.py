from __future__ import annotations


from functools import lru_cache
from typing import Any

import math
import re
import numpy as np

from .constants import (
    AU_LOCAL_BOOST,
    BREAKFAST_MISSING_HINT_PENALTY,
    BREAKFAST_MAIN_BIAS_MULTIPLIER_FLOOR,
    BREAKFAST_MAIN_SKIP_PENALTY_FLOOR,
    COMBO_BEVERAGE_KEYWORDS,
    CATEGORICAL_MISMATCH_PENALTY,
    COMBO_DAIRY_SNACK_KEYWORDS,
    COMBO_CATEGORY_TARGETS,
    COMBO_DRINK_KEYWORDS,
    COMBO_MMR_LAMBDA,
    COMBO_SIDE_KEYWORDS,
    DEFAULT_RANKING_WEIGHTS,
    DISCRETIONARY_KEYWORDS,
    DISCRETIONARY_PENALTY,
    HEALTH_SCORE_BOOST_MAX_MULTIPLIER,
    HEALTH_SCORE_BOOST_MIN,
    HEALTH_SCORE_CLAMP_MIN,
    INDUSTRIAL_ADDITIVE_MARKERS,
    LOVE_BOOST,
    MEAL_HINTS,
    MEAL_KEYWORDS,
    MIN_HEALTH_SCORE,
    MISMATCH_GROUPS,
    SKIP_PENALTY,
    SUGAR_LIMIT_PER_MEAL,
    SUGAR_PENALTY_FACTOR,
    ULTRA_PROCESSED_ADDITIVE_COUNT,
    ULTRA_PROCESSED_HABIT_WEIGHT_FACTOR,
    ULTRA_PROCESSED_PENALTY,
)
from .utils import (
    build_display_title,
    canonical_title_key,
    canonicalize_title,
    clamp,
    cosine_similarity_safe,
    dedupe_strings,
    is_ingredient_like_title,
    normalize_text,
    tokenize,
    tokenize_canonical_title,
    to_float,
)


class Retriever:
    GOAL_FAVORITES = {
        "lose_weight": ["low calorie", "high fiber", "lean protein"],
        "gain_muscle": ["high protein", "complex carbs", "recovery meal"],
        "maintain": ["balanced meal", "mediterranean", "whole foods"],
    }

    ACTIVITY_FAVORITES = {
        "sedentary": ["low calorie"],
        "lightly_active": ["balanced meal"],
        "moderately_active": ["whole foods"],
        "very_active": ["high protein"],
        "super_active": ["high protein"],
        "extra_active": ["high protein"],
    }

    GLOBAL_FALLBACK = ["balanced meal", "high protein", "whole foods", "mediterranean", "lean protein"]

    def demographic_favorites(self, demographics: dict[str, Any]) -> list[str]:
        goal = normalize_text(demographics.get("goal") or "maintain")
        activity = normalize_text(demographics.get("activityLevel") or "moderately_active")

        favorites = []
        favorites.extend(self.GOAL_FAVORITES.get(goal, self.GOAL_FAVORITES["maintain"]))
        favorites.extend(self.ACTIVITY_FAVORITES.get(activity, []))

        profile_text = " ".join(
            normalize_text(demographics.get(key))
            for key in ("activityLevel", "lifestyle", "preferredWorkout", "favoriteActivity")
        )
        if "gym" in profile_text:
            favorites.append("high protein")

        favorite_cuisine = demographics.get("favoriteCuisine")
        if favorite_cuisine:
            favorites.append(str(favorite_cuisine))

        return dedupe_strings(favorites, limit=8)

    def build_keywords(self, top_foods: list[str], demographics: dict[str, Any], meal_type: str) -> list[str]:
        candidates = []
        candidates.extend(top_foods)
        candidates.extend(self.demographic_favorites(demographics))
        candidates.extend(MEAL_HINTS.get(meal_type, []))
        candidates.extend(self.GLOBAL_FALLBACK)
        return dedupe_strings(candidates, limit=10)


def _candidate_display_title(candidate: dict[str, Any]) -> str:
    return build_display_title(
        candidate.get("title") or candidate.get("original_title") or candidate.get("canonical_title") or "",
        candidate.get("mapped_title"),
    )


@lru_cache(maxsize=32768)
def _candidate_title_key_cached(canonical_title: str, title: str, original_title: str) -> str:
    return canonical_title_key(canonical_title or title or original_title)


def _candidate_title_key(candidate: dict[str, Any]) -> str:
    return _candidate_title_key_cached(
        str(candidate.get("canonical_title") or ""),
        str(candidate.get("title") or ""),
        str(candidate.get("original_title") or ""),
    )


def merge_candidates(*candidate_lists: list[dict[str, Any]], max_items: int = 80) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen = set()

    for candidate_list in candidate_lists:
        for candidate in candidate_list or []:
            candidate_id = str(candidate.get("id") or candidate.get("food_id") or candidate.get("recipe_id") or "").strip()
            title_key = _candidate_title_key(candidate)
            dedupe_key = candidate_id or title_key
            if not dedupe_key or dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            merged.append(candidate)
            if len(merged) >= max_items:
                return merged

    return merged


def _title_affinity_score(title: str, top_foods: list[str]) -> float:
    title_tokens = tokenize_canonical_title(title)
    if not title_tokens:
        return 0.0
    top_tokens: set[str] = set()
    for item in top_foods:
        top_tokens.update(tokenize_canonical_title(item))
    if not top_tokens:
        return 0.0

    overlap = len(title_tokens.intersection(top_tokens))
    union = len(title_tokens.union(top_tokens)) or 1
    return overlap / union


def _history_affinity_score(title: str, top_foods: list[str], top_food_counts: dict[str, float] | None) -> float:
    title_tokens = tokenize_canonical_title(title)
    if not title_tokens:
        return 0.0

    if not top_food_counts:
        return _title_affinity_score(title, top_foods)

    total = float(sum(max(0.0, float(count)) for count in top_food_counts.values()))
    if total <= 0.0:
        return _title_affinity_score(title, top_foods)

    best = 0.0
    for food_name, count in top_food_counts.items():
        food_tokens = tokenize_canonical_title(food_name)
        if not food_tokens:
            continue
        overlap = len(title_tokens.intersection(food_tokens))
        union = len(title_tokens.union(food_tokens)) or 1
        similarity = overlap / union
        weight = math.sqrt(max(0.0, float(count))) / math.sqrt(total)
        best = max(best, similarity * weight)

    return best


def _meal_hint_affinity(meal_type: str, candidate: dict[str, Any]) -> float:
    if not meal_type:
        return 0.0
    keywords = MEAL_KEYWORDS.get(meal_type, [])
    if not keywords:
        return 0.0

    candidate_text = " ".join(
        str(candidate.get(key) or "")
        for key in ("canonical_title", "title", "source_keyword", "recipe_category", "keywords")
    ).lower()
    if not candidate_text:
        return 0.0

    for keyword in keywords:
        normalized = normalize_text(keyword)
        if normalized and normalized in candidate_text:
            return 1.0

    return 0.0


def _is_feedback_protected_breakfast_main(
    meal_slot: str,
    category: str,
    meal_hint_affinity: float,
    ingredient_like: bool,
    is_ultra_processed: bool,
) -> bool:
    return (
        meal_slot == "breakfast"
        and category == "main"
        and meal_hint_affinity > 0.0
        and not ingredient_like
        and not is_ultra_processed
    )


def _positive_feedback_quality_cap(
    candidate: dict[str, Any],
    meal_slot: str,
    category: str,
    meal_hint_affinity: float,
    ingredient_like: bool,
    is_ultra_processed: bool,
    health_score: float,
) -> float:
    title_text = _candidate_title_text(candidate)
    raw_title = str(candidate.get("title") or candidate.get("original_title") or "")
    food_type_text = normalize_text(candidate.get("food_type"))
    brand_name = normalize_text(candidate.get("brand_name"))

    cap = 1.0
    if ingredient_like:
        cap *= 0.5
    if is_ultra_processed:
        cap *= 0.68
    if health_score < float(HEALTH_SCORE_BOOST_MIN):
        cap *= 0.8

    if category == "main":
        role_quality = 1.0 if meal_slot == "breakfast" else main_role_quality_multiplier(candidate, meal_slot)
        cap *= clamp(role_quality, 0.35, 1.0)
        if meal_slot == "breakfast" and meal_hint_affinity <= 0.0:
            cap *= 0.72
        if meal_slot in {"lunch", "dinner"} and 0.0 < to_float(candidate.get("serving_calories"), to_float(candidate.get("calories"), 0.0)) <= 220.0:
            cap *= 0.7
    elif category == "side":
        cap *= clamp(side_role_quality_multiplier(candidate, meal_slot), 0.35, 1.0)
    elif category == "drink":
        cap *= clamp(drink_role_quality_multiplier(candidate, meal_slot), 0.3, 1.0)

    if food_type_text == "brand" or brand_name or "," in raw_title or " - " in raw_title:
        cap *= 0.82
    if any(term in title_text for term in _PACKAGED_MAIN_MARKERS):
        cap *= 0.8
    if any(term in title_text for term in _BRANDED_MAIN_PRODUCT_MARKERS):
        cap *= 0.78

    return clamp(cap, 0.2, 1.0)


def _is_feedback_resilient_candidate(
    candidate: dict[str, Any],
    meal_slot: str,
    category: str,
    meal_hint_affinity: float,
    ingredient_like: bool,
    is_ultra_processed: bool,
    health_score: float,
) -> bool:
    if ingredient_like or is_ultra_processed or health_score < float(HEALTH_SCORE_BOOST_MIN):
        return False

    if category == "main":
        if meal_slot == "breakfast":
            return meal_hint_affinity > 0.0
        return main_role_quality_multiplier(candidate, meal_slot) >= 0.72
    if category == "side":
        side_floor = 0.8 if meal_slot == "dinner" else 0.68
        return side_role_quality_multiplier(candidate, meal_slot) >= side_floor
    if category == "drink":
        drink_floor = 0.92 if meal_slot == "breakfast" else 0.74 if meal_slot == "lunch" else 0.82
        return drink_role_quality_multiplier(candidate, meal_slot) >= drink_floor
    return False


def _categorical_fit(meal_type: str, candidate: dict[str, Any]) -> float:
    if not meal_type:
        return 1.0
    mismatch_terms = MISMATCH_GROUPS.get(meal_type, [])
    if not mismatch_terms:
        return 1.0

    candidate_text = " ".join(
        str(candidate.get(key) or "")
        for key in ("canonical_title", "title", "source_keyword", "recipe_category", "keywords")
    ).lower()
    if not candidate_text:
        return 1.0

    for term in mismatch_terms:
        normalized = normalize_text(term)
        if normalized and normalized in candidate_text:
            return float(CATEGORICAL_MISMATCH_PENALTY)

    return 1.0


@lru_cache(maxsize=32768)
def _candidate_text_cached(
    canonical_title: str,
    title: str,
    original_title: str,
    source_keyword: str,
    recipe_category: str,
    food_type: str,
    ingredient_text: str,
    serving_description: str,
    measurement_description: str,
    keywords: str,
) -> str:
    resolved_title = canonical_title or canonicalize_title(title or original_title) or title or original_title or ""
    return " ".join(
        str(part)
        for part in (
            resolved_title,
            source_keyword,
            recipe_category,
            food_type,
            ingredient_text,
            serving_description,
            measurement_description,
            keywords,
        )
        if str(part).strip()
    ).lower()


def _candidate_text(candidate: dict[str, Any]) -> str:
    return _candidate_text_cached(
        str(candidate.get("canonical_title") or ""),
        str(candidate.get("title") or ""),
        str(candidate.get("original_title") or ""),
        str(candidate.get("source_keyword") or ""),
        str(candidate.get("recipe_category") or ""),
        str(candidate.get("food_type") or ""),
        str(candidate.get("ingredient_text") or ""),
        str(candidate.get("serving_description") or ""),
        str(candidate.get("measurement_description") or ""),
        str(candidate.get("keywords") or ""),
    )


@lru_cache(maxsize=32768)
def _candidate_title_text_cached(
    canonical_title: str,
    title: str,
    original_title: str,
    brand_name: str,
    food_type: str,
) -> str:
    return normalize_text(
        " ".join(
            part
            for part in (
                canonical_title,
                title,
                original_title,
                brand_name,
                food_type,
            )
            if str(part).strip()
        )
    )


def _candidate_title_text(candidate: dict[str, Any]) -> str:
    return _candidate_title_text_cached(
        str(candidate.get("canonical_title") or ""),
        str(candidate.get("title") or ""),
        str(candidate.get("original_title") or ""),
        str(candidate.get("brand_name") or ""),
        str(candidate.get("food_type") or ""),
    )


@lru_cache(maxsize=32768)
def _candidate_taxonomy_text_cached(source_keyword: str, recipe_category: str, keywords: str) -> str:
    return normalize_text(
        " ".join(
            part
            for part in (
                source_keyword,
                recipe_category,
                keywords,
            )
            if str(part).strip()
        )
    )


def _candidate_taxonomy_text(candidate: dict[str, Any]) -> str:
    return _candidate_taxonomy_text_cached(
        str(candidate.get("source_keyword") or ""),
        str(candidate.get("recipe_category") or ""),
        str(candidate.get("keywords") or ""),
    )


_NON_DRINK_ROLE_MARKERS = {
    "asparagus",
    "bar",
    "baguette",
    "bamboo",
    "bread",
    "breads",
    "bean",
    "beef",
    "beetroot",
    "biscuit",
    "bok",
    "bowl",
    "broccoli",
    "broccolini",
    "brussels",
    "burger",
    "burrito",
    "cake",
    "capsicum",
    "carrot",
    "cauliflower",
    "cheese",
    "chicken",
    "ciabatta",
    "cracker",
    "crispbread",
    "crispbreads",
    "curry",
    "entree",
    "fish",
    "flatbread",
    "fruit",
    "granola",
    "hoagie",
    "macaroni",
    "muesli",
    "muffin",
    "muffins",
    "noodle",
    "noodles",
    "pancake",
    "pancakes",
    "papadum",
    "papadums",
    "pasta",
    "panini",
    "pita",
    "pizza",
    "rice",
    "roll",
    "salad",
    "sandwich",
    "soup",
    "sprout",
    "steak",
    "sub",
    "taco",
    "tosta",
    "tostas",
    "vegetable",
    "waffle",
    "waffles",
    "wrap",
}

_STRONG_BEVERAGE_KEYWORDS = tuple(
    term for term in COMBO_BEVERAGE_KEYWORDS if normalize_text(term) not in {"tea", "water", "juice", "shake"}
)
_DRINK_ROLE_DEMOTION_PHRASES = (
    "breakfast shake",
    "flavored shake",
    "flavoured shake",
    "meal replacement",
    "nutrition shake",
    "nutritional drink",
    "promour shake",
    "protein shake",
    "slim shake",
    "supplement drink",
)
_MILK_DRINK_DEMOTION_PHRASES = (
    "fat free milk",
    "flavored milk",
    "flavoured milk",
    "low fat milk",
    "reduced fat milk",
    "skim milk",
    "whole milk",
)
_NON_DRINK_LATTE_PHRASES = (
    "fior di latte",
)
_WEAK_BEVERAGE_PHRASES = (
    "iced tea",
    "herbal tea",
    "green tea",
    "black tea",
    "chai tea",
    "sparkling water",
    "mineral water",
    "flavored water",
    "flavoured water",
    "coconut water",
    "orange juice",
    "apple juice",
    "grape juice",
    "cranberry juice",
    "pineapple juice",
    "tomato juice",
    "vegetable juice",
    "fruit juice",
    "juice only",
)
_WEAK_BEVERAGE_BLOCKERS = {
    "biscuit",
    "cake",
    "cookie",
    "cracker",
    "fruit",
    "loaf",
    "salad",
}
_NON_DRINK_BEVERAGE_TITLE_PATTERNS = (
    r"\bcanned\s+in\s+.+\s+juice\b",
    r"\bcocoa butter\b",
    r"\bhigh cocoa solids\b",
    r"\bspinach,\s*water\b",
    r"\btiramisu\b",
)
_BREAKFAST_SOLID_SUBSTRING_MARKERS = (
    "fromage blanc",
    "granola",
    "joghurt",
    "jogurt",
    "kefir",
    "muesli",
    "oatmeal",
    "porridge",
    "parfait",
    "quark",
    "skyr",
    "yaourt",
    "yogurt",
    "yoghurt",
)
_MAIN_TITLE_MARKERS = {
    "bagel",
    "baguette",
    "bowl",
    "brisket",
    "burger",
    "burrito",
    "ciabatta",
    "club",
    "deli",
    "flatbread",
    "hoagie",
    "panini",
    "pita",
    "pizza",
    "plate",
    "platter",
    "roll",
    "sandwich",
    "steak",
    "sub",
    "taco",
    "pasta",
    "pho",
    "ramen",
    "powerbowl",
    "proteinbowl",
    "wrap",
}
_MAIN_PROTEIN_MARKERS = (
    "beef",
    "chicken",
    "fish",
    "lamb",
    "pork",
    "salmon",
    "shrimp",
    "steak",
    "tofu",
    "turkey",
)
_MAIN_PREPARED_MEAL_MARKERS = (
    "bowl",
    "burger",
    "curry",
    "fillet",
    "grilled",
    "noodle",
    "noodles",
    "pasta",
    "plate",
    "platter",
    "rice",
    "salad",
    "sandwich",
    "soft taco",
    "soup",
    "stir fry",
    "taco",
    "wrap",
)
_DINNER_HANDHELD_MAIN_MARKERS = (
    "burger",
    "sandwich",
    "sub",
    "taco",
    "wrap",
)
_PACKAGED_MAIN_MARKERS = (
    "crunchy",
    "frozen",
    "instant meal",
    "instant meals",
    "meal kit",
    "microwavable",
    "nibbles",
    "packet",
    "pocket",
    "pouches",
    "ready meal",
    "snack",
)
_BRANDED_MAIN_PRODUCT_MARKERS = (
    "huel",
    "mygrandma",
    "treasures",
)
_PACKAGED_COFFEE_DRINK_MARKERS = (
    "extra cremoso",
    "intenso",
    "latte macchiato",
    "macchiato",
    "ready to drink",
    "shakissimo",
)
_FLAVORED_COFFEE_DRINK_MARKERS = (
    "caramel",
    "choco",
    "flavor",
    "flavour",
    "mocha",
    "original",
    "vanilla",
)
_PLANT_MILK_DRINK_MARKERS = (
    "plant based beverage",
    "plant based milk",
    "barista series",
    "non-dairy beverage",
    "non dairy beverage",
    "plant-based beverage",
    "plant-based milk",
    "soy plant based beverage",
    "soy plant-based beverage",
    "soy beverage",
    "soy milk",
    "soya beverage",
    "soya milk",
    "soymilk beverage",
    "vanilla soy beverage",
)
_SIDE_SNACK_MARKERS = (
    "chickpea snack",
    "chips",
    "crisps",
    "cracker",
    "crackers",
    "fries",
    "jerky",
    "peanut",
    "peanuts",
    "popcorn",
    "pretzel",
    "pretzels",
    "pork rinds",
    "trail mix",
)
_BREAKFAST_SIDE_MARKERS = (
    "berries",
    "cereal",
    "cracker",
    "crackers",
    "crispbread",
    "crispbreads",
    "fromage blanc",
    "fruit",
    "granola",
    "joghurt",
    "jogurt",
    "kefir",
    "muesli",
    "oat",
    "quark",
    "skyr",
    "tosta",
    "tostas",
    "toast",
    "yaourt",
    "yogurt",
    "yoghurt",
)
_BREAKFAST_PRODUCT_MAIN_BLOCKERS = (
    "breakfast cereal",
    "cereal",
    "cracker",
    "crackers",
    "crispbread",
    "crispbreads",
    "granola",
    "granola bar",
    "muesli",
    "protein bar",
    "protein bars",
    "tosta",
    "tostas",
)
_BREAKFAST_TRUE_MAIN_MARKERS = (
    "avocado toast",
    "bagel",
    "breakfast sandwich",
    "egg",
    "eggs",
    "french toast",
    "muffin",
    "omelet",
    "omelette",
    "oatmeal",
    "pancake",
    "parfait",
    "peanut butter",
    "porridge",
    "scrambled",
    "smoothie bowl",
    "toast",
    "waffle",
)
_DINNER_SIDE_MARKERS = (
    "beans",
    "broccoli",
    "coleslaw",
    "greens",
    "potato",
    "rice",
    "salad",
    "slaw",
    "soup",
    "vegetable",
    "veggies",
)
_VEGETABLE_SIDE_MARKERS = (
    "broccoli",
    "coleslaw",
    "greens",
    "salad",
    "slaw",
    "vegetable",
    "veggies",
)
_PACKAGED_SIDE_MARKERS = (
    "family size",
    "flavor",
    "flavour",
    "instant",
    "ounce",
    "oz",
    "paper",
    "packet",
    "rice a roni",
)
_SIDE_CONDIMENT_MARKERS = (
    "balsamic",
    "condiment",
    "dip",
    "dressing",
    "gravy",
    "marinade",
    "mayo",
    "mayonnaise",
    "mustard",
    "ranch",
    "relish",
    "salsa",
    "sauce",
    "vinaigrette",
    "vinegar",
)
_SIDE_DAIRY_MARKERS = (
    "all natural yogurt",
    "cheese stick",
    "cheese sticks",
    "cream cheese",
    "fruit with cheddar cheese",
    "fromage blanc",
    "joghurt",
    "jogurt",
    "kefir",
    "quark",
    "skyr",
    "string cheese",
    "cheddar cheese",
    "yaourt",
    "yogurt",
    "yoghurt",
)
_SIDE_DESSERT_MARKERS = (
    "cake",
    "cheesecake",
    "cookie",
    "dessert",
    "macaroon",
    "macaroons",
    "sweet",
)
_SIDE_PREPARED_MAIN_MARKERS = (
    "bake",
    "nuggets",
    "pasta salad",
    "quiche",
    "rice & sauce",
    "rice and pasta",
)
_SIDE_NON_SIDE_SOUP_MARKERS = (
    "barley soup",
    "beef broth",
    "beef stock soup",
    "bisque",
    "bone broth",
    "chicken broth",
    "chicken flavored noodle soup",
    "chicken noodle",
    "chowder",
    "condensed",
    "noodle soup",
    "ramen",
)
_SIDE_PICKLED_SALAD_MARKERS = (
    "kraut",
    "matbucha",
    "miracle whip",
    "pickled",
    "salad peppers",
)
_SIDE_PACKAGED_GRAIN_VEGETABLE_MARKERS = (
    "almondine",
    "au gratin",
    "broccoli flavored rice",
    "broccoli rice side",
    "brown rice",
    "cottage cheese",
    "crispins",
    "croutons",
    "cup-a-soup",
    "curry rice",
    "flavored rice",
    "instant brown rice",
    "instant soup",
    "long grain",
    "medley",
    "microwavable",
    "pate",
    "rice medley",
    "rice side",
    "salad topper",
    "salad topping",
    "split pea soup",
    "wild rice",
)
_SIDE_BRANDED_SALAD_MARKERS = (
    "greek salad",
    "salad accents",
    "salad blend",
    "salad elegance",
    "salad fixin",
    "salad kickers",
    "salad kit",
    "salad topping",
    "salad toppings",
    "salad toppins",
    "southwest kit",
    "taylor farms",
)
_SIDE_OILY_VEGETABLE_MARKERS = (
    "almondine",
    "breaded",
    "butter",
    "buttery",
    "chips",
    "creamy",
    "dried",
    "flavored",
    "flavoured",
    "fried",
    "oil",
    "oysters",
    "roasted & salted",
    "salted",
    "skillet dinner",
    "snack",
    "tuna",
)
_SIDE_LANGUAGE_AWARE_SALAD_MARKERS = (
    "salade",
    "salata",
)
_SIDE_LANGUAGE_AWARE_MAIN_MARKERS = (
    "baby food",
    "caesar",
    "citron vert",
    "jambon",
    "kip",
    "pecel",
    "piri-piri",
    "poulet",
    "riz complet",
    "tikka",
)
_SIDE_LANGUAGE_AWARE_SOUP_MARKERS = (
    "barley & vegetable soup",
    "fazool soup",
    "lentil soup",
)
_SIDE_MAINISH_TITLE_PHRASES = (
    "avocado toast",
    "burger",
    "fillet",
    "grilled salmon",
    "pasta",
    "pho",
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
_SUPPLEMENT_TITLE_MARKERS = (
    "isolate",
    "mass gainer",
    "meal replacement",
    "nutrition shake",
    "powder",
    "pre workout",
    "protein shake",
    "supplement",
    "whey",
)

_BLOCKED_DRINK_ROLE_MARKERS = (
    "dayup power protein",
    "energy drink",
    "ensure",
    "fresubin",
    "gut shot",
    "jello shot",
    "nutrition shake",
    "power protein",
    "pre workout",
    "probiotic beverage",
    "protein energy",
    "smoothie cubes",
    "wellness probiotic beverage",
)


def build_text_role_candidate(
    *,
    title: Any = "",
    recipe_category: Any = "",
    keywords: Any = "",
    ingredient_text: Any = "",
    serving_description: Any = "",
    metric_serving_unit: Any = "",
    category: Any = "",
    food_type: Any = "",
    calories: Any = 0.0,
) -> dict[str, Any]:
    serving_text = str(serving_description or "").strip()
    inferred_unit = normalize_text(metric_serving_unit)
    if not inferred_unit and "ml" in normalize_text(serving_text):
        inferred_unit = "ml"
    return {
        "title": str(title or "").strip(),
        "original_title": str(title or "").strip(),
        "canonical_title": canonicalize_title(title),
        "recipe_category": str(recipe_category or "").strip(),
        "source_keyword": str(recipe_category or "").strip(),
        "keywords": str(keywords or "").strip(),
        "ingredient_text": str(ingredient_text or "").strip(),
        "serving_description": serving_text,
        "metric_serving_unit": inferred_unit,
        "category": str(category or "").strip(),
        "food_type": str(food_type or "").strip(),
        "serving_calories": to_float(calories, 0.0),
    }


def _has_explicit_weak_beverage_title(title_text: str, title_tokens: set[str]) -> bool:
    if not title_text:
        return False
    if re.search(r"\bin\s+(water|juice)\b", title_text):
        return False
    if _matches_any_keyword(title_text, _WEAK_BEVERAGE_PHRASES):
        return True
    if title_tokens.intersection(_WEAK_BEVERAGE_BLOCKERS):
        return False
    if "tea" in title_tokens and len(title_tokens) <= 3:
        return True
    if "water" in title_tokens and len(title_tokens) <= 3:
        return True
    if "juice" in title_tokens and len(title_tokens) <= 4:
        return True
    return False


@lru_cache(maxsize=32768)
def _matches_any_keyword(text: str, keywords: tuple[str, ...]) -> bool:
    normalized_text = normalize_text(text)
    tokens = set(tokenize(normalized_text))
    for term in keywords:
        normalized_term = normalize_text(term)
        if not normalized_term:
            continue
        if " " in normalized_term:
            pattern = rf"(?<![a-z0-9]){re.escape(normalized_term)}(?![a-z0-9])"
            if re.search(pattern, normalized_text):
                return True
            continue
        if normalized_term in tokens:
            return True
    return False


@lru_cache(maxsize=32768)
def _infer_breakfast_side_family(title_text: str, text: str) -> str:
    combined_text = " ".join(part for part in (normalize_text(title_text), normalize_text(text)) if part)
    if not combined_text:
        return "other"
    if any(
        term in combined_text
        for term in (
            "actifidus",
            "fromage",
            "fromage blanc",
            "iogurt",
            "joghurt",
            "jogurt",
            "kefir",
            "quark",
            "rahka",
            "skyr",
            "yaourt",
            "yoghourt",
            "yogourt",
            "yogur",
            "yogurt",
            "yoghurt",
        )
    ):
        return "cultured_dairy"
    if any(
        term in combined_text
        for term in (
            "apple",
            "banana",
            "berries",
            "berry",
            "blaubeere",
            "blueberry",
            "erdbeer",
            "framboise",
            "fruit",
            "kirsche",
            "mango",
            "melon",
            "orange",
            "pineapple",
        )
    ):
        return "fruit"
    if any(term in combined_text for term in ("almond", "cashew", "hazelnut", "nut", "pecan", "seed", "trail mix", "walnut")):
        return "nuts"
    if any(term in combined_text for term in ("bagel", "bread", "cracker", "crispbread", "croissant", "english muffin", "toast", "tosta")):
        return "bread"
    if any(term in combined_text for term in ("cereal", "granola", "muesli", "oat", "porridge")):
        return "grain"
    return "other"


def _semantic_candidate_args(candidate: dict[str, Any]) -> tuple[str, str, str, str, str, str, str, str, str, str, str, str, float, float]:
    return (
        str(candidate.get("canonical_title") or ""),
        str(candidate.get("title") or ""),
        str(candidate.get("original_title") or ""),
        str(candidate.get("brand_name") or ""),
        str(candidate.get("source_keyword") or ""),
        str(candidate.get("recipe_category") or ""),
        str(candidate.get("food_type") or ""),
        str(candidate.get("ingredient_text") or ""),
        str(candidate.get("serving_description") or ""),
        str(candidate.get("measurement_description") or ""),
        str(candidate.get("keywords") or ""),
        str(candidate.get("metric_serving_unit") or ""),
        to_float(candidate.get("serving_calories"), 0.0),
        to_float(candidate.get("calories"), 0.0),
    )


@lru_cache(maxsize=32768)
def _drink_semantics_cached(
    canonical_title: str,
    title: str,
    original_title: str,
    brand_name: str,
    source_keyword: str,
    recipe_category: str,
    food_type: str,
    ingredient_text: str,
    serving_description: str,
    measurement_description: str,
    keywords: str,
    metric_serving_unit: str,
    serving_calories: float,
    calories_value: float,
) -> dict[str, Any]:
    resolved_title = canonical_title or title or original_title
    text = _candidate_text_cached(
        canonical_title,
        title,
        original_title,
        source_keyword,
        recipe_category,
        food_type,
        ingredient_text,
        serving_description,
        measurement_description,
        keywords,
    )
    title_text = _candidate_title_text_cached(canonical_title, title, original_title, brand_name, food_type)
    taxonomy_text = _candidate_taxonomy_text_cached(source_keyword, recipe_category, keywords)
    title_tokens = tokenize_canonical_title(resolved_title)
    ingredient_like = is_ingredient_like_title(resolved_title)
    unit = normalize_text(metric_serving_unit)
    calories = to_float(serving_calories, to_float(calories_value, 0.0))

    has_strong_beverage_keyword = _matches_any_keyword(title_text, _STRONG_BEVERAGE_KEYWORDS)
    has_weak_beverage_keyword = _has_explicit_weak_beverage_title(title_text, title_tokens)
    # Treat explicit generic drink labels as beverage markers for role classification.
    has_explicit_generic_beverage_title = any(re.search(rf"\b{term}\b", title_text) for term in ("drink", "beverage"))
    has_beverage_keyword = has_strong_beverage_keyword or has_weak_beverage_keyword or has_explicit_generic_beverage_title
    has_taxonomy_beverage_keyword = _matches_any_keyword(taxonomy_text, COMBO_BEVERAGE_KEYWORDS)
    has_side_keyword = _matches_any_keyword(text, COMBO_SIDE_KEYWORDS)
    has_shake_title = "shake" in title_tokens and "smoothie" not in title_tokens
    # Keep explicit milk drinks in the drink role instead of demoting them into breakfast sides.
    has_plain_milk_title = (
        "milk" in title_tokens
        and not has_beverage_keyword
        and not bool(title_tokens.intersection(_NON_DRINK_ROLE_MARKERS))
    )
    has_non_drink_beverage_title = any(
        re.search(pattern, title_text) for pattern in _NON_DRINK_BEVERAGE_TITLE_PATTERNS
    ) and "juice only" not in title_text
    has_non_drink_latte_phrase = any(phrase in title_text for phrase in _NON_DRINK_LATTE_PHRASES)
    has_role_demotion_marker = (
        has_shake_title
        or has_plain_milk_title
        or _matches_any_keyword(title_text, _DRINK_ROLE_DEMOTION_PHRASES)
        or _matches_any_keyword(title_text, _MILK_DRINK_DEMOTION_PHRASES)
    )
    has_breakfast_solid_substring = any(marker in title_text for marker in _BREAKFAST_SOLID_SUBSTRING_MARKERS)
    has_dairy_snack_marker = (
        _matches_any_keyword(text, COMBO_DAIRY_SNACK_KEYWORDS)
        or has_breakfast_solid_substring
        or has_non_drink_latte_phrase
        or has_role_demotion_marker
    )
    has_non_drink_marker = (
        "entree" in text
        or has_non_drink_beverage_title
        or has_non_drink_latte_phrase
        or bool(title_tokens.intersection(_NON_DRINK_ROLE_MARKERS))
    )
    beverage_like = (
        not ingredient_like
        and not has_non_drink_marker
        and not has_dairy_snack_marker
        and (
            unit == "ml"
            or has_beverage_keyword
            or (has_taxonomy_beverage_keyword and unit == "ml")
        )
    )

    return {
        "calories": calories,
        "ingredient_like": ingredient_like,
        "unit": unit,
        "has_beverage_keyword": has_beverage_keyword,
        "has_strong_beverage_keyword": has_strong_beverage_keyword,
        "has_weak_beverage_keyword": has_weak_beverage_keyword,
        "has_taxonomy_beverage_keyword": has_taxonomy_beverage_keyword,
        "has_side_keyword": has_side_keyword,
        "has_role_demotion_marker": has_role_demotion_marker,
        "has_dairy_snack_marker": has_dairy_snack_marker,
        "has_non_drink_marker": has_non_drink_marker,
        "beverage_like": beverage_like,
    }


def _drink_semantics(candidate: dict[str, Any]) -> dict[str, Any]:
    args = _semantic_candidate_args(candidate)
    return _drink_semantics_cached(*args)


@lru_cache(maxsize=32768)
def _side_semantics_cached(
    canonical_title: str,
    title: str,
    original_title: str,
    brand_name: str,
    source_keyword: str,
    recipe_category: str,
    food_type: str,
    ingredient_text: str,
    serving_description: str,
    measurement_description: str,
    keywords: str,
    metric_serving_unit: str,
    serving_calories: float,
    calories_value: float,
) -> dict[str, Any]:
    semantics = _drink_semantics_cached(
        canonical_title,
        title,
        original_title,
        brand_name,
        source_keyword,
        recipe_category,
        food_type,
        ingredient_text,
        serving_description,
        measurement_description,
        keywords,
        metric_serving_unit,
        serving_calories,
        calories_value,
    )
    resolved_title = canonical_title or title or original_title
    title_text = _candidate_title_text_cached(canonical_title, title, original_title, brand_name, food_type)
    text = _candidate_text_cached(
        canonical_title,
        title,
        original_title,
        source_keyword,
        recipe_category,
        food_type,
        ingredient_text,
        serving_description,
        measurement_description,
        keywords,
    )
    title_tokens = tokenize_canonical_title(resolved_title)
    calories = float(semantics["calories"])
    breakfast_family = _infer_breakfast_side_family(title_text, text)

    has_main_marker = bool(title_tokens.intersection(_MAIN_TITLE_MARKERS))
    has_side_keyword = _matches_any_keyword(text, COMBO_SIDE_KEYWORDS)
    has_snack_marker = _matches_any_keyword(title_text, _SIDE_SNACK_MARKERS)
    has_breakfast_marker = _matches_any_keyword(title_text, _BREAKFAST_SIDE_MARKERS) or breakfast_family != "other"
    has_dinner_marker = _matches_any_keyword(title_text, _DINNER_SIDE_MARKERS)
    has_vegetable_marker = _matches_any_keyword(title_text, _VEGETABLE_SIDE_MARKERS)
    has_supplement_marker = _matches_any_keyword(title_text, _SUPPLEMENT_TITLE_MARKERS)
    has_packaged_marker = _matches_any_keyword(title_text, _PACKAGED_SIDE_MARKERS)
    has_condiment_marker = _matches_any_keyword(text, _SIDE_CONDIMENT_MARKERS)
    has_dairy_marker = _matches_any_keyword(text, _SIDE_DAIRY_MARKERS)
    has_dessert_marker = _matches_any_keyword(text, _SIDE_DESSERT_MARKERS)
    has_prepared_main_marker = _matches_any_keyword(text, _SIDE_PREPARED_MAIN_MARKERS)
    has_non_side_soup_marker = _matches_any_keyword(text, _SIDE_NON_SIDE_SOUP_MARKERS)
    has_rice_marker = "rice" in title_tokens
    has_pickled_salad_marker = _matches_any_keyword(text, _SIDE_PICKLED_SALAD_MARKERS)
    has_packaged_grain_vegetable_marker = _matches_any_keyword(text, _SIDE_PACKAGED_GRAIN_VEGETABLE_MARKERS) and (
        has_packaged_marker or has_rice_marker or has_vegetable_marker or has_side_keyword
    )
    has_branded_salad_marker = _matches_any_keyword(text, _SIDE_BRANDED_SALAD_MARKERS)
    has_oily_vegetable_marker = _matches_any_keyword(text, _SIDE_OILY_VEGETABLE_MARKERS) and (
        has_vegetable_marker
        or "green beans" in text
        or "broccoli" in text
        or "cauliflower" in text
        or "carrot" in text
        or "veggie" in text
        or "vegetable" in text
    )
    has_language_aware_salad_marker = _matches_any_keyword(text, _SIDE_LANGUAGE_AWARE_SALAD_MARKERS) and _matches_any_keyword(text, _SIDE_LANGUAGE_AWARE_MAIN_MARKERS)
    has_language_aware_soup_marker = _matches_any_keyword(text, _SIDE_LANGUAGE_AWARE_SOUP_MARKERS)
    has_count_marker = any(token.isdigit() for token in title_tokens)

    side_like = (
        not semantics["ingredient_like"]
        and not semantics["beverage_like"]
        and not has_supplement_marker
        and not has_condiment_marker
        and not (has_dairy_marker and breakfast_family != "cultured_dairy")
        and not has_dessert_marker
        and not has_prepared_main_marker
        and not has_non_side_soup_marker
        and not has_pickled_salad_marker
        and not has_packaged_grain_vegetable_marker
        and not has_branded_salad_marker
        and not has_oily_vegetable_marker
        and not has_language_aware_salad_marker
        and not has_language_aware_soup_marker
        and (
            has_side_keyword
            or has_breakfast_marker
            or has_dinner_marker
            or (
                0.0 < calories <= 220.0
                and not has_main_marker
                and not has_snack_marker
                and not has_count_marker
                and len(title_tokens) <= 4
            )
        )
    )

    return {
        "has_main_marker": has_main_marker,
        "has_side_keyword": has_side_keyword,
        "has_snack_marker": has_snack_marker,
        "has_breakfast_marker": has_breakfast_marker,
        "has_dinner_marker": has_dinner_marker,
        "has_vegetable_marker": has_vegetable_marker,
        "has_supplement_marker": has_supplement_marker,
        "has_packaged_marker": has_packaged_marker,
        "has_condiment_marker": has_condiment_marker,
        "has_dairy_marker": has_dairy_marker,
        "has_dessert_marker": has_dessert_marker,
        "has_prepared_main_marker": has_prepared_main_marker,
        "has_non_side_soup_marker": has_non_side_soup_marker,
        "has_pickled_salad_marker": has_pickled_salad_marker,
        "has_packaged_grain_vegetable_marker": has_packaged_grain_vegetable_marker,
        "has_branded_salad_marker": has_branded_salad_marker,
        "has_oily_vegetable_marker": has_oily_vegetable_marker,
        "has_language_aware_salad_marker": has_language_aware_salad_marker,
        "has_language_aware_soup_marker": has_language_aware_soup_marker,
        "breakfast_family": breakfast_family,
        "has_rice_marker": has_rice_marker,
        "has_count_marker": has_count_marker,
        "side_like": side_like,
    }


def _side_semantics(candidate: dict[str, Any], drink_semantics: dict[str, Any] | None = None) -> dict[str, Any]:
    args = _semantic_candidate_args(candidate)
    return _side_semantics_cached(*args)


def infer_combo_category(candidate: dict[str, Any]) -> str:
    explicit = normalize_text(candidate.get("category"))
    semantics = _drink_semantics(candidate)
    side_semantics = _side_semantics(candidate, semantics)
    breakfast_product_side_candidate = _is_breakfast_product_side_candidate(candidate)
    if explicit in {"main", "side"}:
        if explicit == "main" and breakfast_product_side_candidate:
            return "side"
        if explicit == "side" and not side_semantics["side_like"] and semantics["beverage_like"]:
            return "drink"
        return explicit
    if explicit == "drink":
        if semantics["beverage_like"]:
            return "drink"
        if semantics["has_dairy_snack_marker"] or semantics["has_side_keyword"]:
            return "side"
        return "main"
    text = _candidate_text(candidate)
    for term in COMBO_DRINK_KEYWORDS:
        normalized = normalize_text(term)
        if normalized and normalized in text:
            if semantics["has_dairy_snack_marker"]:
                return "side"
            return "drink"
    for term in COMBO_SIDE_KEYWORDS:
        normalized = normalize_text(term)
        if normalized and normalized in text:
            return "side"

    if semantics["has_dairy_snack_marker"]:
        return "side"

    if breakfast_product_side_candidate:
        return "side"

    if semantics["beverage_like"]:
        return "drink"

    if side_semantics["side_like"]:
        return "side"

    calories = semantics["calories"]
    if 0 < calories < 220.0 and not side_semantics["has_main_marker"] and not side_semantics["has_supplement_marker"]:
        return "side"

    return "main"


def _is_breakfast_product_side_candidate(candidate: dict[str, Any]) -> bool:
    title_text = _candidate_title_text(candidate)
    if not title_text:
        return False
    if not _matches_any_keyword(title_text, _BREAKFAST_PRODUCT_MAIN_BLOCKERS):
        return False
    return not _matches_any_keyword(title_text, _BREAKFAST_TRUE_MAIN_MARKERS)


def is_candidate_role_compatible(candidate: dict[str, Any], category: str) -> bool:
    normalized_category = normalize_text(category)
    if normalized_category not in {"main", "side", "drink"}:
        return True

    semantics = _drink_semantics(candidate)
    ingredient_like = bool(semantics["ingredient_like"])
    unit = str(semantics["unit"])
    calories = float(semantics["calories"])
    has_drink_keyword = bool(semantics["has_beverage_keyword"])
    has_side_keyword = bool(semantics["has_side_keyword"])
    has_non_drink_marker = bool(semantics["has_non_drink_marker"])
    has_dairy_snack_marker = bool(semantics["has_dairy_snack_marker"])
    beverage_like = bool(semantics["beverage_like"])
    side_semantics = _side_semantics(candidate, semantics)
    title_text = _candidate_title_text(candidate)

    if normalized_category == "main":
        return not ingredient_like and not beverage_like

    if normalized_category == "drink":
        if (
            ingredient_like
            or has_non_drink_marker
            or has_dairy_snack_marker
            or _matches_any_keyword(title_text, _BLOCKED_DRINK_ROLE_MARKERS)
            or side_semantics["has_supplement_marker"]
        ):
            return False
        return beverage_like

    if beverage_like:
        return False
    if ingredient_like or side_semantics["has_supplement_marker"]:
        return False
    allow_cultured_dairy_side = side_semantics.get("breakfast_family") == "cultured_dairy"
    if (
        side_semantics["has_condiment_marker"]
        or (side_semantics["has_dairy_marker"] and not allow_cultured_dairy_side)
        or side_semantics["has_dessert_marker"]
        or side_semantics["has_prepared_main_marker"]
        or side_semantics["has_non_side_soup_marker"]
        or side_semantics["has_pickled_salad_marker"]
        or side_semantics["has_packaged_grain_vegetable_marker"]
        or side_semantics["has_branded_salad_marker"]
        or side_semantics["has_oily_vegetable_marker"]
        or side_semantics["has_language_aware_salad_marker"]
        or side_semantics["has_language_aware_soup_marker"]
    ):
        return False
    if has_drink_keyword and unit != "ml" and calories <= 80.0:
        return False
    if has_non_drink_marker and calories > 450.0:
        return False
    if any(phrase in title_text for phrase in _SIDE_MAINISH_TITLE_PHRASES):
        has_explicit_side_context = (
            side_semantics["has_side_keyword"]
            or side_semantics["has_vegetable_marker"]
            or side_semantics["has_dinner_marker"]
        )
        if not has_explicit_side_context:
            return False
    strict_side_context_terms = (
        "bean salad",
        "broccoli",
        "cauliflower",
        "chickpea salad",
        "clear soup",
        "coleslaw",
        "cucumber salad",
        "garden salad",
        "green beans",
        "lentil salad",
        "miso soup",
        "broth",
        "beans",
        "lentil",
        "chickpea",
        "romaine",
        "side salad",
        "slaw",
        "tomato",
        "vegetable soup",
        "tomato soup",
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
        "grains",
        "grain",
        "ham",
        "tuna",
    )
    breakfast_family = str(side_semantics.get("breakfast_family") or "other")
    if (
        any(term in title_text for term in mainish_meal_terms)
        and not any(term in title_text for term in strict_side_context_terms)
        and breakfast_family not in {"bread", "grain"}
    ):
        return False
    if side_semantics["has_main_marker"] and not side_semantics["has_side_keyword"]:
        return False
    if side_semantics["has_snack_marker"] and calories > 180.0:
        return False
    return bool(side_semantics["side_like"])


def side_role_quality_multiplier(candidate: dict[str, Any], meal_type: str) -> float:
    meal_key = normalize_text(meal_type)
    semantics = _side_semantics(candidate)
    title_text = _candidate_title_text(candidate)
    taxonomy_text = _candidate_taxonomy_text(candidate)
    breakfast_family = str(semantics.get("breakfast_family") or "other")
    has_speculoos_marker = any(term in title_text for term in ("speculoos", "biscoff", "cookie butter"))
    has_spread_marker = any(term in taxonomy_text for term in ("spread", "spreads", "cookie butter"))
    if not semantics["side_like"]:
        return 0.35

    score = 1.0
    if not (
        semantics["has_side_keyword"]
        or semantics["has_breakfast_marker"]
        or semantics["has_dinner_marker"]
    ):
        score *= 0.62
    if semantics["has_main_marker"]:
        score *= 0.55
    if semantics["has_count_marker"]:
        score *= 0.78
    if semantics["has_packaged_marker"]:
        score *= 0.72 if meal_key == "dinner" else 0.82
    if semantics["has_snack_marker"]:
        score *= 0.68 if meal_key == "breakfast" else 0.52

    if meal_key == "breakfast":
        if semantics["has_breakfast_marker"]:
            score *= 1.15
        if "breakfast side safety" in taxonomy_text and breakfast_family in {"fruit", "nuts", "bread", "grain"}:
            score *= 1.08
        if breakfast_family == "cultured_dairy":
            score *= 0.76
        if semantics["has_side_keyword"] and not semantics["has_breakfast_marker"]:
            score *= 0.55
        if semantics["has_dinner_marker"] and not semantics["has_breakfast_marker"]:
            score *= 0.38
        if has_speculoos_marker or has_spread_marker:
            score *= 0.18
        if any(term in title_text for term in ("french toast", "hotcakes", "muffin", "pancake", "pancakes", "waffle", "waffles")):
            score *= 0.32
    elif meal_key in {"lunch", "dinner"}:
        if semantics["has_dinner_marker"] or semantics["has_side_keyword"]:
            score *= 1.12
        if semantics["has_breakfast_marker"] and not semantics["has_dinner_marker"]:
            score *= 0.45 if meal_key == "dinner" else 0.78
        if meal_key == "lunch" and any(term in title_text for term in ("salade", "salad", "slaw", "soup", "soupe")) and any(
            term in title_text
            for term in ("asiatisch", "beef", "chicken", "compos", "meal", "poulet", "protein", "repas", "rice", "riz", "thon", "thubfisch", "thunfisch", "tuna", "turkey")
        ):
            score *= 0.42
        if meal_key == "lunch" and any(term in title_text for term in ("soup", "soupe", "broth")) and any(
            term in title_text for term in ("cup soup", "instant", "miso-cup", "ramen", "soup cup")
        ):
            score *= 0.36
        if meal_key == "dinner" and any(phrase in title_text for phrase in _SIDE_MAINISH_TITLE_PHRASES):
            score *= 0.38
        if meal_key == "dinner" and semantics["has_vegetable_marker"]:
            score *= 1.18
        if meal_key == "dinner" and semantics["has_rice_marker"] and semantics["has_packaged_marker"]:
            score *= 0.7
        if meal_key == "dinner" and any(term in title_text for term in ("chicken", "beef", "pork", "turkey", "salmon", "shrimp", "sausage", "mac and cheese", "grains", "grain")):
            score *= 0.42

    return clamp(score, 0.2, 1.25)


def main_role_quality_multiplier(candidate: dict[str, Any], meal_type: str) -> float:
    meal_key = normalize_text(meal_type)
    if meal_key not in {"breakfast", "lunch", "dinner"}:
        return 1.0

    title_text = _candidate_title_text(candidate)
    category_text = normalize_text(candidate.get("recipe_category"))
    taxonomy_text = _candidate_taxonomy_text(candidate)
    main_context_text = " ".join(part for part in (title_text, category_text) if part)
    breakfast_quality_text = " ".join(part for part in (main_context_text, taxonomy_text) if part)
    raw_title = str(candidate.get("title") or candidate.get("original_title") or "")
    food_type_text = normalize_text(candidate.get("food_type"))
    brand_name = normalize_text(candidate.get("brand_name"))
    calories = to_float(candidate.get("serving_calories"), to_float(candidate.get("calories"), 0.0))

    if not is_candidate_role_compatible(candidate, "main"):
        return 0.2

    score = 1.0
    has_protein_marker = any(term in title_text for term in _MAIN_PROTEIN_MARKERS)
    has_prepared_meal_marker = any(term in title_text for term in _MAIN_PREPARED_MEAL_MARKERS)
    has_packaged_marker = any(term in title_text for term in _PACKAGED_MAIN_MARKERS)
    has_branded_product_marker = any(term in title_text for term in _BRANDED_MAIN_PRODUCT_MARKERS)
    has_dinner_handheld_marker = any(term in title_text for term in _DINNER_HANDHELD_MAIN_MARKERS)
    has_breakfast_sandwich_marker = (
        any(term in title_text for term in ("egg", "egg white", "breakfast"))
        and any(
            term in title_text
            for term in ("bagel", "biscuit", "croissant", "mcgriddles", "sandwich", "sizzli", "sub", "wrap")
        )
    )
    has_breakfast_named_meal_marker = (
        "breakfast" in title_text
        and any(
            term in title_text
            for term in ("bowl", "burrito", "plate", "platter", "sandwich", "skillet", "wrap")
        )
    )
    has_breakfast_dessert_main_marker = any(
        term in title_text for term in ("frozen custard", "gelato", "ice cream", "sundae", "waffle cone")
    ) or (
        "scoop" in title_text and any(term in title_text for term in ("cone", "custard", "gelato", "ice cream"))
    )
    has_snack_marker = any(term in title_text for term in ("chips", "crisps", "nibbles", "popcorn", "snack"))
    has_brand_prefix = "," in raw_title or " - " in raw_title
    is_brand_candidate = food_type_text == "brand" or bool(brand_name)

    if meal_key == "breakfast":
        has_breakfast_wholesome_marker = any(
            term in title_text
            for term in ("toast", "oatmeal", "omelet", "omelette", "egg", "eggs", "parfait", "scramble", "skillet", "yogurt")
        )
        has_breakfast_pastry_marker = any(
            term in title_text for term in ("brioche", "french toast", "hotcakes", "pancake", "pancakes", "waffle", "waffles")
        )
        has_breakfast_packaged_marker = any(
            term in title_text for term in ("bar", "bars", "cereal", "granola", "stromboli", "smoothie cubes", "protein whey")
        )
        has_breakfast_nonmeal_product_marker = any(
            term in breakfast_quality_text
            for term in (
                "baking mix",
                "egg replacement",
                "egg replacer",
                "egg substitute",
                "ei ersatz",
                "ei-ersatz",
                "muffin mix",
            )
        )
        has_breakfast_condiment_marker = any(
            term in breakfast_quality_text
            for term in (
                "apple butter",
                "caviar",
                "caviars",
                "chutney",
                "cod caviar",
                "curd",
                "curds",
                "fish eggs",
                "foie gras",
                "jam",
                "jelly",
                "marmalade",
                "pate",
                "pat\u00e9",
                "p\u00e2t\u00e9",
                "preserve",
                "preserves",
                "roe",
                "spread",
                "spreads",
            )
        )

        if has_breakfast_wholesome_marker:
            score *= 1.08
        if has_breakfast_nonmeal_product_marker:
            score *= 0.18
        if has_breakfast_condiment_marker and not has_breakfast_wholesome_marker:
            score *= 0.22
        if has_breakfast_pastry_marker:
            score *= 0.52
        if has_breakfast_dessert_main_marker:
            score *= 0.18
        if has_breakfast_packaged_marker:
            score *= 0.4
        if has_breakfast_sandwich_marker and (is_brand_candidate or has_brand_prefix):
            score *= 0.52
        if "wrap" in title_text and not has_breakfast_wholesome_marker:
            score *= 0.58
        if has_brand_prefix and not has_breakfast_wholesome_marker:
            score *= 0.72
        if is_brand_candidate and not has_breakfast_wholesome_marker:
            score *= 0.78
        if 0.0 < calories <= 180.0 and not has_breakfast_wholesome_marker:
            score *= 0.62
        return clamp(score, 0.2, 1.2)

    if has_prepared_meal_marker:
        score *= 1.05 if meal_key == "lunch" else 1.08
    if has_protein_marker and has_prepared_meal_marker:
        score *= 1.08

    if has_packaged_marker:
        score *= 0.68 if meal_key == "lunch" else 0.58
    if has_branded_product_marker and has_packaged_marker:
        score *= 0.72
    if has_snack_marker:
        score *= 0.42 if meal_key == "dinner" else 0.48
    if "pocket" in title_text:
        score *= 0.55
    if "instant meal" in title_text or "instant meals" in title_text:
        score *= 0.48
    if has_brand_prefix and has_packaged_marker and not has_prepared_meal_marker:
        score *= 0.76
    if 0.0 < calories <= 220.0 and not has_prepared_meal_marker:
        score *= 0.55
    if meal_key == "lunch" and any(
        term in title_text
        for term in (
            "alfredo",
            "carbonara",
            "fettuccine",
            "mac and cheese",
            "pasta",
            "ravioli",
            "soft taco",
            "spaghetti",
            "taco",
            "tacos",
            "tortellini",
            "tortelloni",
        )
    ):
        score *= 0.34
    if meal_key == "lunch" and has_breakfast_sandwich_marker:
        score *= 0.3
    if meal_key == "lunch" and has_breakfast_named_meal_marker:
        score *= 0.24
    if meal_key == "lunch" and is_brand_candidate and any(term in title_text for term in ("sandwich", "sub", "wrap")):
        score *= 0.52
    if meal_key == "dinner" and has_dinner_handheld_marker and not has_packaged_marker:
        score *= 0.56
        if is_brand_candidate:
            score *= 0.82
    if meal_key == "dinner" and has_breakfast_named_meal_marker:
        score *= 0.18
    if meal_key == "dinner" and any(term in title_text for term in ("instant noodle", "instant noodles", "ramen")):
        score *= 0.24
    if meal_key == "dinner" and any(term in title_text for term in ("mac and cheese", "stroganoff")):
        score *= 0.28
    if meal_key == "dinner" and any(
        term in title_text
        for term in (
            "bandnudeln",
            "carbonara",
            "fettuccine",
            "gnocchi",
            "noodle",
            "noodles",
            "nudeln",
            "pasta",
            "penne",
            "ravioli",
            "schupfnudeln",
            "spaghetti",
            "tagliatelle",
            "tortellini",
            "tortelloni",
        )
    ):
        score *= 0.6 if has_protein_marker else 0.36
    if meal_key == "dinner" and any(term in title_text for term in ("popper", "poppers")):
        score *= 0.26
    if meal_key == "dinner" and any(term in title_text for term in ("pizza", "flatbread")):
        score *= 0.34

    return clamp(score, 0.2, 1.2)


def drink_role_quality_multiplier(candidate: dict[str, Any], meal_type: str) -> float:
    meal_key = normalize_text(meal_type)
    semantics = _drink_semantics(candidate)
    title_text = _candidate_title_text(candidate)
    calories = float(semantics["calories"])

    if not semantics["beverage_like"] or not is_candidate_role_compatible(candidate, "drink"):
        return 0.2

    score = 1.0
    if semantics["has_strong_beverage_keyword"]:
        score *= 1.08
    if semantics["has_weak_beverage_keyword"]:
        score *= 0.72 if meal_key == "breakfast" else 0.58 if meal_key == "lunch" else 0.48

    # Keep plain filler beverages from crowding out better meal drinks at lunch and dinner.
    if any(term in title_text for term in ("water", "sparkling water", "mineral water", "seltzer")):
        score *= 0.62 if meal_key == "breakfast" else 0.42
    if "juice" in title_text and "vegetable juice" not in title_text and "smoothie" not in title_text:
        score *= 0.88 if meal_key == "breakfast" else 0.68 if meal_key == "lunch" else 0.58
    if "tea" in title_text and not any(term in title_text for term in ("matcha", "latte", "milk tea", "chai")):
        score *= 0.84 if meal_key == "breakfast" else 0.7 if meal_key == "lunch" else 0.55
    if any(term in title_text for term in ("cocoa", "hot chocolate", "chocolate drink")):
        score *= 0.92 if meal_key == "breakfast" else 0.45 if meal_key == "lunch" else 0.28

    has_coffee_marker = any(term in title_text for term in ("coffee", "latte", "cappuccino", "espresso"))
    has_packaged_coffee_marker = any(term in title_text for term in _PACKAGED_COFFEE_DRINK_MARKERS)
    has_flavored_coffee_marker = any(term in title_text for term in _FLAVORED_COFFEE_DRINK_MARKERS)
    has_plant_milk_marker = any(term in title_text for term in _PLANT_MILK_DRINK_MARKERS)
    has_latte_marker = "latte" in title_text
    has_cappuccino_marker = "cappuccino" in title_text
    has_espresso_marker = "espresso" in title_text
    has_plain_iced_coffee_marker = "iced coffee" in title_text and not any(
        term in title_text for term in ("latte", "cappuccino", "espresso", "frappe", "mocha")
    )
    if has_coffee_marker:
        score *= 1.14 if meal_key == "breakfast" else 0.55 if meal_key == "lunch" else 0.42
        if meal_key == "breakfast":
            if has_latte_marker or has_cappuccino_marker or has_espresso_marker:
                score *= 1.08
            elif has_plain_iced_coffee_marker and not has_flavored_coffee_marker:
                score *= 0.78
        if meal_key in {"lunch", "dinner"} and (has_packaged_coffee_marker or has_flavored_coffee_marker):
            score *= 0.55 if meal_key == "lunch" else 0.4
    if has_plant_milk_marker:
        score *= 0.92 if meal_key == "breakfast" else 0.36 if meal_key == "lunch" else 0.28
    if any(term in title_text for term in ("smoothie", "kombucha", "kefir")):
        score *= 1.08 if meal_key in {"breakfast", "lunch"} else 0.94
    if any(
        term in title_text
        for term in (
            "gut shot",
            "jello shot",
            "probiotic beverage",
            "smoothie cubes",
            "wellness probiotic beverage",
        )
    ):
        score *= 0.22
    if "shot" in title_text and not any(term in title_text for term in ("espresso", "ginger shot")):
        score *= 0.28
    if calories <= 20.0:
        score *= 0.86 if meal_key == "breakfast" else 0.68 if meal_key == "lunch" else 0.52

    return clamp(score, 0.2, 1.2)


def _ingredient_text(candidate: dict[str, Any]) -> str:
    return normalize_text(
        candidate.get("normalized_ingredients")
        or candidate.get("ingredient_text")
        or candidate.get("ingredients")
        or ""
    )


def _discretionary_penalty(candidate: dict[str, Any]) -> float:
    health_score = to_float(candidate.get("health_score"), to_float(candidate.get("aggregated_rating"), 0.0))
    if health_score >= float(HEALTH_SCORE_BOOST_MIN):
        return 1.0
    text = _candidate_text(candidate)
    if not text:
        return 1.0
    for term in DISCRETIONARY_KEYWORDS:
        normalized = normalize_text(term)
        if normalized and normalized in text:
            return float(DISCRETIONARY_PENALTY)
    return 1.0


def _additive_hits(ingredients_text: str) -> int:
    if not ingredients_text:
        return 0
    hits = 0
    for marker in INDUSTRIAL_ADDITIVE_MARKERS:
        normalized = normalize_text(marker)
        if normalized and normalized in ingredients_text:
            hits += 1
    return hits


def _normalize_weight_map(ranking_weights: dict[str, float] | None) -> dict[str, float]:
    base = dict(DEFAULT_RANKING_WEIGHTS)
    if ranking_weights:
        for key, value in ranking_weights.items():
            if key in base:
                base[key] = max(0.0, float(value))

    total = sum(base.values()) or 1.0
    return {key: value / total for key, value in base.items()}


def rank_candidates(
    candidates: list[dict[str, Any]],
    user_vec: np.ndarray | None,
    archetype_vectors: list[np.ndarray] | None,
    preference_tokens: set[str],
    preference_token_weights: dict[str, float] | None,
    recent_titles: set[str],
    top_foods: list[str],
    meal_target: float,
    top_food_counts: dict[str, float] | None = None,
    meal_type: str | None = None,
    top_n: int = 40,
    serving_fit_tolerance: float = 0.15,
    stochastic_strength: float = 0.0,
    rng: np.random.Generator | None = None,
    skipped_titles: set[str] | None = None,
    loved_titles: set[str] | None = None,
    favorite_titles: set[str] | None = None,
    ranking_weights: dict[str, float] | None = None,
    mmr_lambda: float | None = None,
    title_bias: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    ranked: list[dict[str, Any]] = []
    reference_calories = max(1.0, float(meal_target))
    stochastic_strength = max(0.0, float(stochastic_strength))
    meal_slot = normalize_text(meal_type) if meal_type else ""
    if rng is None:
        rng = np.random.default_rng()

    weights = _normalize_weight_map(ranking_weights)
    bias_table = {canonical_title_key(key): float(value) for key, value in (title_bias or {}).items() if canonical_title_key(key)}
    skipped = {canonical_title_key(title) for title in (skipped_titles or set()) if canonical_title_key(title)}
    loved = {canonical_title_key(title) for title in (loved_titles or set()) if canonical_title_key(title)}
    favorites = {canonical_title_key(title) for title in (favorite_titles or set()) if canonical_title_key(title)}
    recent = {canonical_title_key(title) for title in (recent_titles or set()) if canonical_title_key(title)}

    for candidate in candidates:
        per100 = candidate.get("per100", {})
        calories = to_float(candidate.get("serving_calories"), to_float(per100.get("calories"), 0.0))
        protein = to_float(candidate.get("serving_protein"), to_float(per100.get("protein"), 0.0))
        carbs = to_float(candidate.get("serving_carbs"), to_float(per100.get("carbs"), 0.0))
        fats = to_float(candidate.get("serving_fats"), to_float(per100.get("fats"), 0.0))
        category = infer_combo_category(candidate)
        candidate["category"] = category
        title_text = str(candidate.get("canonical_title") or candidate.get("title") or "")
        ingredient_like = is_ingredient_like_title(title_text)
        role_compatible = is_candidate_role_compatible(candidate, category)
        if calories <= 0:
            if category == "drink" and role_compatible:
                calories = 1.0
            else:
                continue
        if category == "main" and ingredient_like:
            continue
        adjusted_tolerance = float(serving_fit_tolerance)

        if category == "main":
            comp_target = reference_calories * float(COMBO_CATEGORY_TARGETS.get("main", 0.65))
            local_only_candidate = normalize_text(candidate.get("ml_tag")) == "local_only"
            adjusted_tolerance = max(adjusted_tolerance, 0.6 if local_only_candidate else 0.35)
        elif category == "side":
            comp_target = reference_calories * float(COMBO_CATEGORY_TARGETS.get("side", 0.20))
            adjusted_tolerance = max(adjusted_tolerance, 0.50)
        else:
            comp_target = reference_calories * float(COMBO_CATEGORY_TARGETS.get("drink", 0.15))
            adjusted_tolerance = max(adjusted_tolerance, 0.50)

        comp_fit_ratio = abs(calories - comp_target) / max(1.0, comp_target)
        full_fit_ratio = abs(calories - reference_calories) / reference_calories
        serving_fit_ratio = min(comp_fit_ratio, full_fit_ratio)
        is_low_calorie_filler = category in {"side", "drink"} and calories <= 200.0
        if not is_low_calorie_filler and serving_fit_ratio > max(0.0, adjusted_tolerance):
            continue

        item_vec = np.array([calories, protein, carbs, fats], dtype=float)
        archetypes = archetype_vectors or []
        if archetypes:
            best_match = 0.0
            for archetype in archetypes:
                raw_cosine = cosine_similarity_safe(item_vec, np.array(archetype, dtype=float))
                best_match = max(best_match, (raw_cosine + 1.0) / 2.0)
            nutritional_match = best_match
        elif user_vec is None:
            nutritional_match = 0.55
        else:
            raw_cosine = cosine_similarity_safe(item_vec, np.array(user_vec, dtype=float))
            nutritional_match = (raw_cosine + 1.0) / 2.0

        title_tokens = tokenize_canonical_title(title_text)
        source_tokens = tokenize(candidate.get("source_keyword", ""))
        all_tokens = title_tokens.union(source_tokens)

        if not preference_tokens or not all_tokens:
            preference_similarity = 0.5
        else:
            overlap_tokens = preference_tokens.intersection(all_tokens)
            if preference_token_weights:
                overlap = sum(preference_token_weights.get(token, 1.0) for token in overlap_tokens)
                base = sum(preference_token_weights.get(token, 1.0) for token in preference_tokens) or 1.0
                preference_similarity = overlap / base
            else:
                overlap = len(overlap_tokens)
                base = len(preference_tokens.union(all_tokens)) or 1
                preference_similarity = overlap / base

        habit_affinity = _history_affinity_score(title_text, top_foods, top_food_counts)
        discretionary_penalty = _discretionary_penalty(candidate)
        ingredients_text = _ingredient_text(candidate)
        additive_hits = _additive_hits(ingredients_text)
        is_ultra_processed = additive_hits >= int(ULTRA_PROCESSED_ADDITIVE_COUNT) or discretionary_penalty < 1.0
        health_score = to_float(candidate.get("health_score"), to_float(candidate.get("aggregated_rating"), 0.0))
        if health_score <= 0.0:
            health_score = float(MIN_HEALTH_SCORE)

        health_habit_factor = clamp(float(health_score) / float(MIN_HEALTH_SCORE), 0.4, 1.0)
        habit_weight = weights["habit_affinity"] * (
            float(ULTRA_PROCESSED_HABIT_WEIGHT_FACTOR) if is_ultra_processed else 1.0
        ) * health_habit_factor
        meal_hint_affinity = _meal_hint_affinity(meal_slot, candidate)
        prefetched_main_role_quality: float | None = None
        if meal_slot == "breakfast" and category == "main":
            prefetched_main_role_quality = main_role_quality_multiplier(candidate, meal_slot)
            if prefetched_main_role_quality <= 0.24 and meal_hint_affinity <= 0.0:
                continue

        tolerance = max(0.01, float(adjusted_tolerance))
        calorie_fit = 1.0 - min(1.0, serving_fit_ratio / tolerance)
        calorie_fit = max(0.0, calorie_fit)

        title_key = _candidate_title_key(candidate)
        recency_factor = 1.05 if title_key in recent else 1.0

        base_score = recency_factor * (
            (weights["nutritional_match"] * nutritional_match)
            + (weights["preference_similarity"] * preference_similarity)
            + (habit_weight * habit_affinity)
            + (weights["calorie_fit"] * calorie_fit)
            + (weights["meal_hint_affinity"] * meal_hint_affinity)
        )
        exploration_noise = float(rng.uniform(-stochastic_strength, stochastic_strength))
        base_score = max(0.0, base_score + exploration_noise)

        raw_hsr_multiplier = float(health_score) / float(MIN_HEALTH_SCORE)
        if float(health_score) >= float(HEALTH_SCORE_BOOST_MIN):
            hsr_multiplier = clamp(raw_hsr_multiplier, 1.0, float(HEALTH_SCORE_BOOST_MAX_MULTIPLIER))
        else:
            hsr_multiplier = clamp(raw_hsr_multiplier, float(HEALTH_SCORE_CLAMP_MIN), 1.0)
        nova_penalty = float(ULTRA_PROCESSED_PENALTY) if is_ultra_processed else 1.0
        local_boost = float(AU_LOCAL_BOOST) if bool(candidate.get("is_australian")) else 1.0
        serving_sugar = to_float(candidate.get("serving_sugar"), 0.0)
        sugar_penalty = float(SUGAR_PENALTY_FACTOR) if serving_sugar > float(SUGAR_LIMIT_PER_MEAL) else 1.0

        health_adjustment = hsr_multiplier * nova_penalty * local_boost * discretionary_penalty * sugar_penalty
        adjusted_distance = max(0.0, 1.0 - min(1.0, float(health_adjustment)))
        final_score = max(0.0, base_score * health_adjustment)
        categorical_fit = _categorical_fit(meal_slot, candidate)
        final_score = max(0.0, final_score * categorical_fit)

        breakfast_hint_penalty = 1.0
        if meal_slot == "breakfast" and meal_hint_affinity <= 0.0:
            breakfast_hint_penalty = float(BREAKFAST_MISSING_HINT_PENALTY)
            final_score = max(0.0, final_score * breakfast_hint_penalty)

        breakfast_role_penalty = 1.0
        if meal_slot == "breakfast":
            breakfast_side_semantics = _side_semantics(candidate)
            if category == "main" and meal_hint_affinity > 0.0:
                breakfast_role_penalty *= 1.12
            if category == "side" and _is_breakfast_product_side_candidate(candidate):
                breakfast_role_penalty *= 0.58
            if category == "side" and not breakfast_side_semantics["has_breakfast_marker"]:
                breakfast_role_penalty *= 0.45
            if category == "side" and breakfast_side_semantics["has_dinner_marker"] and not breakfast_side_semantics["has_breakfast_marker"]:
                breakfast_role_penalty *= 0.5
            if category == "drink" and calories <= 20.0 and meal_hint_affinity <= 0.0:
                breakfast_role_penalty *= 0.28
            final_score = max(0.0, final_score * breakfast_role_penalty)

        breakfast_relaxed_mapping_penalty = 1.0
        if meal_slot == "breakfast" and category in {"side", "drink"}:
            mapping_acceptance_mode = normalize_text(candidate.get("mapping_acceptance_mode"))
            mapping_gap_ratio = to_float(candidate.get("calorie_diff_ratio"), 0.0)
            mapping_title_similarity = to_float(candidate.get("title_similarity"), 0.0)
            if mapping_acceptance_mode == "relaxed_title_fallback":
                if category == "side":
                    if mapping_gap_ratio >= 0.18:
                        breakfast_relaxed_mapping_penalty = 0.34 if mapping_title_similarity < 0.58 else 0.46
                else:
                    if mapping_gap_ratio >= 0.3:
                        breakfast_relaxed_mapping_penalty = 0.46 if mapping_title_similarity < 0.68 else 0.58
                    elif mapping_gap_ratio >= 0.22:
                        breakfast_relaxed_mapping_penalty = 0.66 if mapping_title_similarity < 0.72 else 0.78
            final_score = max(0.0, final_score * breakfast_relaxed_mapping_penalty)

        main_role_penalty = 1.0
        if category == "main":
            main_role_penalty = (
                prefetched_main_role_quality
                if prefetched_main_role_quality is not None
                else main_role_quality_multiplier(candidate, meal_slot)
            )
            final_score = max(0.0, final_score * main_role_penalty)

        dinner_relaxed_mapping_penalty = 1.0
        if meal_slot == "dinner" and category == "main":
            mapping_acceptance_mode = normalize_text(candidate.get("mapping_acceptance_mode"))
            mapping_gap_ratio = to_float(candidate.get("calorie_diff_ratio"), 0.0)
            mapping_title_similarity = to_float(candidate.get("title_similarity"), 0.0)
            if mapping_acceptance_mode == "relaxed_title_fallback":
                if mapping_gap_ratio >= 0.6:
                    dinner_relaxed_mapping_penalty = 0.32 if mapping_title_similarity < 0.78 else 0.42
                elif mapping_gap_ratio >= 0.5:
                    dinner_relaxed_mapping_penalty = 0.42 if mapping_title_similarity < 0.78 else 0.55
                elif mapping_gap_ratio >= 0.4:
                    dinner_relaxed_mapping_penalty = 0.62 if mapping_title_similarity < 0.8 else 0.72
                elif mapping_gap_ratio >= 0.3:
                    dinner_relaxed_mapping_penalty = 0.78 if mapping_title_similarity < 0.82 else 0.88
            final_score = max(0.0, final_score * dinner_relaxed_mapping_penalty)

        ingredient_penalty = 0.45 if ingredient_like else 1.0
        role_penalty = 1.0 if role_compatible else (0.15 if category == "drink" else 0.45)
        final_score = max(0.0, final_score * ingredient_penalty * role_penalty)

        feedback_protected_breakfast_main = _is_feedback_protected_breakfast_main(
            meal_slot,
            category,
            meal_hint_affinity,
            ingredient_like,
            is_ultra_processed,
        )
        positive_feedback_cap = _positive_feedback_quality_cap(
            candidate,
            meal_slot,
            category,
            meal_hint_affinity,
            ingredient_like,
            is_ultra_processed,
            float(health_score),
        )
        feedback_resilient_candidate = _is_feedback_resilient_candidate(
            candidate,
            meal_slot,
            category,
            meal_hint_affinity,
            ingredient_like,
            is_ultra_processed,
            float(health_score),
        )

        skip_penalty = float(SKIP_PENALTY) if title_key in skipped else 1.0
        love_boost = float(LOVE_BOOST) if title_key in loved else 1.0
        if positive_feedback_cap < 0.6:
            love_boost = 1.0
        elif love_boost > 1.0:
            love_boost = 1.0 + ((love_boost - 1.0) * positive_feedback_cap)
        favorite_boost = 1.08 if title_key in favorites else 1.0
        if positive_feedback_cap < 0.6:
            favorite_boost = 1.0
        elif favorite_boost > 1.0:
            favorite_boost = 1.0 + ((favorite_boost - 1.0) * positive_feedback_cap)
        bias_weight = float(bias_table.get(title_key, 0.0))
        bias_multiplier = clamp(1.0 + bias_weight, 0.25, 1.75)
        if bias_weight > 0.0 and positive_feedback_cap < 0.6:
            bias_multiplier = 1.0
        elif bias_weight > 0.0:
            bias_multiplier = 1.0 + ((bias_multiplier - 1.0) * positive_feedback_cap)
        if feedback_resilient_candidate:
            skip_penalty = max(skip_penalty, 0.9)
            if bias_weight < 0.0:
                bias_multiplier = max(bias_multiplier, 0.9)
        if feedback_protected_breakfast_main:
            skip_penalty = max(skip_penalty, float(BREAKFAST_MAIN_SKIP_PENALTY_FLOOR))
            bias_multiplier = max(bias_multiplier, float(BREAKFAST_MAIN_BIAS_MULTIPLIER_FLOOR))
        final_score = max(0.0, final_score * skip_penalty * favorite_boost * love_boost * bias_multiplier)

        ranked.append(
            {
                **candidate,
                "title": _candidate_display_title(candidate),
                "score": round(float(final_score), 4),
                "scores": {
                    "nutritional_match": round(float(nutritional_match), 4),
                    "preference_similarity": round(float(preference_similarity), 4),
                    "habit_affinity": round(float(habit_affinity), 4),
                    "habit_weight": round(float(habit_weight), 4),
                    "health_habit_factor": round(float(health_habit_factor), 4),
                    "calorie_fit": round(float(calorie_fit), 4),
                    "serving_fit_ratio": round(float(serving_fit_ratio), 4),
                    "recency_factor": round(float(recency_factor), 4),
                    "exploration_noise": round(float(exploration_noise), 4),
                    "meal_hint_affinity": round(float(meal_hint_affinity), 4),
                    "categorical_fit": round(float(categorical_fit), 4),
                    "health_score": round(float(health_score), 4),
                    "hsr_multiplier": round(float(hsr_multiplier), 4),
                    "nova_penalty": round(float(nova_penalty), 4),
                    "local_boost": round(float(local_boost), 4),
                    "discretionary_penalty": round(float(discretionary_penalty), 4),
                    "sugar_penalty": round(float(sugar_penalty), 4),
                    "health_adjustment": round(float(health_adjustment), 4),
                    "additive_hits": int(additive_hits),
                    "ultra_processed": int(is_ultra_processed),
                    "breakfast_hint_penalty": round(float(breakfast_hint_penalty), 4),
                    "breakfast_role_penalty": round(float(breakfast_role_penalty), 4),
                    "main_role_penalty": round(float(main_role_penalty), 4),
                    "dinner_relaxed_mapping_penalty": round(float(dinner_relaxed_mapping_penalty), 4),
                    "ingredient_penalty": round(float(ingredient_penalty), 4),
                    "ingredient_like": int(ingredient_like),
                    "skip_penalty": round(float(skip_penalty), 4),
                    "favorite_boost": round(float(favorite_boost), 4),
                    "love_boost": round(float(love_boost), 4),
                    "bias_weight": round(float(bias_weight), 4),
                    "bias_multiplier": round(float(bias_multiplier), 4),
                    "positive_feedback_cap": round(float(positive_feedback_cap), 4),
                    "feedback_resilient_candidate": int(feedback_resilient_candidate),
                    "role_penalty": round(float(role_penalty), 4),
                    "role_compatible": int(role_compatible),
                },
                "serving_fit_ratio": round(float(serving_fit_ratio), 4),
                "adjusted_distance": round(float(adjusted_distance), 4),
            }
        )

    return _mmr_rerank(ranked, top_n=top_n, lambda_param=float(mmr_lambda or COMBO_MMR_LAMBDA))


def _mmr_rerank(
    candidates: list[dict[str, Any]],
    top_n: int,
    lambda_param: float = 0.5,
) -> list[dict[str, Any]]:
    if not candidates:
        return []
    n = len(candidates)
    lambda_param = float(np.clip(lambda_param, 0.0, 1.0))

    # Pre-compute all nutritional vectors as a matrix (n × 4) for batch cosine ops.
    vecs = np.array(
        [
            [
                to_float(c.get("serving_calories"), 0.0),
                to_float(c.get("serving_protein"), 0.0),
                to_float(c.get("serving_carbs"), 0.0),
                to_float(c.get("serving_fats"), 0.0),
            ]
            for c in candidates
        ],
        dtype=float,
    )
    norms = np.linalg.norm(vecs, axis=1)  # shape (n,)

    # Pre-compute title keys once — O(n) instead of O(n²) inside the loop.
    all_title_keys = [_candidate_title_key(c) for c in candidates]

    scores = np.array([to_float(c.get("score"), 0.0) for c in candidates])

    remaining_pos = list(range(n))   # indices into candidates / vecs / norms / all_title_keys
    selected_pos: list[int] = []
    selected_title_keys: set[str] = set()

    limit = max(1, int(top_n))

    while remaining_pos and len(selected_pos) < limit:
        if not selected_pos:
            # First pick: highest relevance score.
            rem = np.array(remaining_pos)
            best_local = int(np.argmax(scores[rem]))
        else:
            rem = np.array(remaining_pos)      # shape (m,)
            sel = np.array(selected_pos)       # shape (k,)

            rem_vecs = vecs[rem]               # (m, 4)
            sel_vecs = vecs[sel]               # (k, 4)
            rem_norms = norms[rem]             # (m,)
            sel_norms = norms[sel]             # (k,)

            # Batch cosine similarities: shape (m, k)
            dots = rem_vecs @ sel_vecs.T
            norm_prod = np.outer(rem_norms, sel_norms)
            valid = norm_prod > 0.0
            cosines = np.where(valid, dots / np.where(valid, norm_prod, 1.0), 0.0)
            max_sims = cosines.max(axis=1)     # shape (m,)

            relevances = scores[rem]
            title_penalties = np.array(
                [0.18 if all_title_keys[i] in selected_title_keys else 0.0 for i in remaining_pos]
            )
            mmr_scores = (lambda_param * relevances) - ((1.0 - lambda_param) * max_sims) - title_penalties
            best_local = int(np.argmax(mmr_scores))

        chosen_pos = remaining_pos.pop(best_local)
        selected_pos.append(chosen_pos)
        chosen_key = all_title_keys[chosen_pos]
        if chosen_key:
            selected_title_keys.add(chosen_key)

    return [candidates[i] for i in selected_pos]
