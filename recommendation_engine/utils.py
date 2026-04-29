from __future__ import annotations


import html
import re
from typing import Any, Iterable

import numpy as np

_ENGLISH_TITLE_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9\s'\",.\-()/+%&:]*$")
_ENGLISH_ALPHA_PATTERN = re.compile(r"[A-Za-z]")
_TITLE_PACK_SIZE_PATTERN = re.compile(r"\b\d+(?:[.,]\d+)?\s*(?:g|kg|mg|ml|l|oz|lb|lbs|pack|pk|pcs|ct)\b", re.IGNORECASE)
_TITLE_TRAILING_PACKAGING_PATTERN = re.compile(r"\b(?:pos|pkt|pack|package|pouch|sachet|bottle|jar|tray|box|bag)\b", re.IGNORECASE)
_TITLE_BRAND_PREFIXES = {
    "basket",
    "mcdonald",
    "mcdonalds",
    "perkins",
    "kfc",
    "burger king",
    "sonic",
    "subway",
    "starbucks",
    "taco bell",
    "tesco",
    "tesco finest",
    "wendy",
    "ihop",
    "domino",
    "pizza hut",
    "papa johns",
    "papa john's",
    "dunkin",
    "jason's deli",
    "jasons deli",
    "outback steakhouse",
    "panera bread",
    "steak 'n shake",
    "steak n shake",
    "wawa",
}
_BRAND_LIKE_PREFIX_MARKERS = {
    "bakery",
    "bar",
    "bistro",
    "cafe",
    "coffee",
    "company",
    "deli",
    "foods",
    "general",
    "grill",
    "house",
    "inc",
    "kitchen",
    "llc",
    "market",
    "restaurant",
    "roasters",
    "sons",
    "store",
    "subs",
}
_TITLE_PLACEHOLDER_KEYS = {
    "food item",
    "food product",
    "item",
    "meal item",
    "product",
    "unknown",
    "unknown item",
    "unknown meal",
    "unknown product",
}
_TITLE_ALIAS_MAP = {
    "almond milk imp": "Almond Milk",
    "ah 75% groente smoothie spinazie komkommer": "Spinach Cucumber Smoothie",
    "almond milky": "Almond Milk",
    "activia au bifidus kiwi": "Bifidus Yogurt",
    "activia bifidus cereales saveur noix": "Nut Cereal Activia Yogurt",
    "activia prune": "Bifidus Yogurt",
    "bami goreng au poulet": "Chicken Bami Goreng",
    "beetroot and feta salad kit": "Beetroot and Feta Salad",
    "bifidus": "Bifidus Yogurt",
    "bifidus coco": "Coconut Bifidus Yogurt",
    "bifidus coco 4x125g": "Coconut Bifidus Yogurt",
    "bioland frische bio eier": "Fresh Organic Eggs",
    "bio-yogurt mild - latte macchiato": "Latte Macchiato Yogurt",
    "brebis sur lit de fruit myrtille": "Blueberry Sheep Milk Yogurt",
    "butter chicken mit huhn und reis in tomaten-kokos-sauce": "Butter Chicken with Rice",
    "butter chicken mit huhn und reis in tomaten kokos sauce": "Butter Chicken with Rice",
    "butter chicken with cauliflower rice": "Butter Chicken with Cauliflower Rice",
    "cocos yogurt": "Coconut Yogurt",
    "couscous fe": "Couscous",
    "crisp": "Crunchy Yogurt",
    "cold brew coffee": "Cold Brew Coffee",
    "crackers aux 3 graines bio": "Three Seed Crackers",
    "duo yogurt and crisp": "Duo Yogurt Crisp",
    "edamame and butterbean salad": "Bean Salad",
    "emmi strawberry moments": "Strawberry Yogurt",
    "ensalada quinoa": "Quinoa Salad",
    "fleischschnackas 4 tranches": "Meat Crepes",
    "flaki po zamojski": "Polish Tripe Soup",
    "flaki po zamojsku": "Polish Tripe Soup",
    "fromage blanc framboise": "Raspberry Yogurt",
    "fromage blanc rabache": "Fruit Yogurt",
    "garden greens and shaved parm side salad": "Garden Salad",
    "granola bio nature and graines": "Natural Seed Granola",
    "granola chocolat noir": "Dark Chocolate Granola",
    "griego con papaya": "Papaya Greek Yogurt",
    "hachis parmentier": "French Cottage Pie",
    "hafer cappuccino": "Oat Cappuccino",
    "indian style chicken tikka": "Chicken Tikka Masala",
    "iogurte morango e nozes cremoso": "Creamy Strawberry Walnut Yogurt",
    "knickis banane": "Banana Yogurt",
    "knicks knusper": "Crunchy Yogurt",
    "kycklingschnitzel med potatis": "Chicken Schnitzel with Potatoes",
    "le poulet granscendant sauteed chicken breast": "Sauteed Chicken Breast",
    "little salad spirelli feta and caviar de carottes": "Spirelli Feta Carrot Salad",
    "merendina al miele": "Honey Snack Cake",
    "moments": "Yogurt Cup",
    "mrs mcgregor's margarine spread": "Margarine Spread",
    "no sugar added white chocolate": "White Chocolate",
    "black and white true iced espresso": "Black and White Iced Espresso",
    "papas garden salad": "Garden Salad",
    "panier de yoplait": "Yoplait Yogurt",
    "protein pasta golden": "Protein Pasta",
    "poulet aigre doux et riz": "Sweet and Sour Chicken with Rice",
    "salade de riz": "Rice Salad",
    "salade de riz au thon": "Tuna Rice Salad",
    "salade de riz noir bio edamame and feta": "Black Rice Edamame and Feta Salad",
    "salade de riz noir bio, edamame and feta": "Black Rice Edamame and Feta Salad",
    "salade riz et thon": "Rice and Tuna Salad",
    "salade de poulpe": "Octopus Salad",
    "salade de poulpes": "Octopus Salad",
    "salade de fruits de mer": "Seafood Salad",
    "salade saumon": "Salmon Salad",
    "semi di finocchio": "Fennel Seeds",
    "shrimp california roll qbr": "Shrimp California Roll",
    "simple and fit 2 egg breakfast": "2 Egg Breakfast",
    "soup instant white miso": "Instant White Miso Soup",
    "soupe de moules": "Mussel Soup",
    "stjerneanis hel": "Star Anise",
    "stjerneanis hel hindu": "Star Anise",
    "stonebaked sourdough bbq chicken and bacon": "BBQ Chicken and Bacon Pizza",
    "spring onion and garlic potato salad": "Spring Onion and Garlic Potato Salad",
    "sweet and sour chicken, lunch": "Sweet and Sour Chicken",
    "scrambled egg breakfast bowl": "Breakfast Bowl",
    "tesco finest edamame and sprouting pra salad": "Edamame Salad",
    "wok panang curry": "Panang Curry",
    "55 pot roast": "Pot Roast",
    "55+ pot roast": "Pot Roast",
    "yaourt a la chataigne": "Chestnut Yogurt",
    "yaourt au lait de brebis bio": "Organic Sheep Milk Yogurt",
    "yaourt brebis brasse": "Sheep Milk Yogurt",
    "yogurt cremoso stracciatella": "Creamy Stracciatella Yogurt",
    "yogurt crisp": "Yogurt Crisp",
}
_TITLE_PHRASE_ALIAS_MAP = {
    "almond beverage": "Almond Milk",
    "cashew beverage": "Cashew Milk",
    "coconut beverage": "Coconut Milk",
    "oat beverage": "Oat Milk",
    "rice beverage": "Rice Milk",
    "soy beverage": "Soy Milk",
}
_CANDIDATE_ONLY_TITLE_ALIAS_MAP = {
    # Keep the Phase 11 dev-only exact-title bridges off the benchmark-side canonical title surface.
    "hermesetas original 300 tablets": "Intense sweetener, containing saccharin and sucralose, tablet",
    "milk": "Milk, cow, fluid, rich or creamy",
}
_TITLE_TOKEN_REPLACEMENTS = {
    "bidifus": "bifidus",
    "fetta": "feta",
    "joghurt": "yogurt",
    "jogurt": "yogurt",
    "yoghurt": "yogurt",
}
_NON_ENGLISH_TITLE_MARKERS = (
    "aigre doux",
    "bami goreng",
    "banane",
    "bifidus coco",
    "brebis",
    "caviar de carottes",
    "chataigne",
    "cereales saveur",
    "eier",
    "fleischschnackas",
    "frische",
    "framboise",
    "fruits de mer",
    "fromage blanc",
    "griego",
    "groente",
    "hachis parmentier",
    "hafer",
    "joghurt",
    "jogurt",
    "knickis",
    "knicks",
    "knusper",
    "komkommer",
    "morango",
    "myrtille",
    "nozes",
    "panier de",
    "potatis",
    "poulet",
    "salade",
    "spinazie",
    "stracciatella",
    "yaourt",
    "zamojsk",
)
_TITLE_TRAILING_MENU_SIZE_ABBREVIATIONS = {
    "lg",
    "med",
    "min",
    "reg",
    "sm",
}
_MEAL_LIKE_TITLE_MARKERS = {
    "bagel",
    "beef",
    "bowl",
    "burger",
    "burrito",
    "cappuccino",
    "cereal",
    "chicken",
    "coffee",
    "curry",
    "egg",
    "eggs",
    "hoagie",
    "juice",
    "latte",
    "muesli",
    "noodle",
    "noodles",
    "oatmeal",
    "omelet",
    "omelette",
    "pancake",
    "pancakes",
    "pasta",
    "parfait",
    "pizza",
    "platter",
    "potato",
    "roast",
    "rice",
    "roll",
    "salad",
    "sandwich",
    "skillet",
    "smoothie",
    "soup",
    "spud",
    "toast",
    "waffle",
    "waffles",
    "wrap",
    "yogurt",
}
_INGREDIENT_LIKE_TITLE_MARKERS = {
    "essence",
    "extract",
    "herb",
    "herbs",
    "mix",
    "morsel",
    "paste",
    "powder",
    "sauce",
    "seasoning",
    "seasonings",
    "seed",
    "seeds",
    "spice",
    "spices",
}
_INGREDIENT_LIKE_TITLE_PHRASES = {
    "curry powder",
    "filling mix",
    "herb mix",
    "seasoning mix",
    "spice mix",
}
_INGREDIENT_LIKE_SPICE_NOUNS = {
    "cinnamon",
    "cumin",
    "nutmeg",
    "oregano",
    "paprika",
    "pepper",
    "turmeric",
}
_INGREDIENT_LIKE_SPICE_DESCRIPTORS = {
    "black",
    "cracked",
    "ground",
    "mixed",
    "white",
}


def ensure_array(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if value is None:
        return []
    return [value]


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def to_positive_int(value: Any, default: int = 0) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def normalize_text(value: Any) -> str:
    return str(value or "").strip().lower()


def clean_title_text(value: Any) -> str:
    return re.sub(r"\s+", " ", html.unescape(str(value or ""))).strip()


def strip_title_wrappers(value: Any) -> str:
    text = clean_title_text(value)
    if not text:
        return ""
    text = re.sub(r"\([^)]*\)", " ", text)
    text = re.sub(r"\[[^\]]*\]", " ", text)
    return re.sub(r"\s+", " ", text).strip(" ,:-")


def strip_brand_prefix(value: Any) -> str:
    text = clean_title_text(value)
    if not text:
        return ""

    if "," in text:
        prefix, remainder = text.split(",", 1)
        if normalize_text(prefix) in _TITLE_BRAND_PREFIXES:
            text = remainder.strip()
    elif " - " in text:
        prefix, remainder = text.split(" - ", 1)
        if normalize_text(prefix) in _TITLE_BRAND_PREFIXES:
            text = remainder.strip()
    elif ":" in text:
        prefix, remainder = text.split(":", 1)
        if normalize_text(prefix) in _TITLE_BRAND_PREFIXES:
            text = remainder.strip()

    normalized = normalize_text(text)
    for brand in _TITLE_BRAND_PREFIXES:
        if normalized.startswith(f"{brand} "):
            text = text[len(brand) :].strip(" ,:-")
            break

    if "," in text:
        prefix, remainder = text.split(",", 1)
        normalized_prefix = normalize_text(prefix)
        normalized_remainder = normalize_text(remainder)
        prefix_tokens = re.findall(r"[a-z0-9']+", normalized_prefix)
        remainder_tokens = tokenize_canonical_title(remainder)
        looks_like_brand_prefix = 0 < len(prefix_tokens) <= 6 and (
            (
                bool(remainder_tokens.intersection(_MEAL_LIKE_TITLE_MARKERS))
                and (
                    "'" in prefix
                    or "&" in prefix
                    or any(marker in normalized_prefix for marker in _BRAND_LIKE_PREFIX_MARKERS)
                )
            )
            or (
                len(remainder_tokens) >= 2
                and (
                    "'" in prefix
                    or any(marker in normalized_prefix for marker in _BRAND_LIKE_PREFIX_MARKERS)
                )
            )
        )
        if looks_like_brand_prefix and normalized_remainder:
            text = remainder.strip(" ,:-")

    return clean_title_text(text)


def _apply_title_token_replacements(value: Any) -> str:
    text = clean_title_text(value)
    if not text:
        return ""

    text = re.sub(r"\s*&\s*", " and ", text)
    text = re.sub(r"\bw\s*/\s*", " with ", text, flags=re.IGNORECASE)
    text = re.sub(r"\bw\b", " with ", text, flags=re.IGNORECASE)
    for source, target in _TITLE_TOKEN_REPLACEMENTS.items():
        text = re.sub(rf"\b{re.escape(source)}\b", target, text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip(" ,:-")


def _phrase_alias_for_title(normalized_title: str) -> str | None:
    for phrase, alias in _TITLE_PHRASE_ALIAS_MAP.items():
        if normalized_title == phrase or normalized_title.startswith(phrase):
            return alias
    return None


def _contains_title_phrase(normalized_title: str, phrases: set[str]) -> bool:
    return any(phrase in normalized_title for phrase in phrases)


def strip_packaging_noise(value: Any) -> str:
    text = clean_title_text(value)
    if not text:
        return ""

    text = _TITLE_PACK_SIZE_PATTERN.sub(" ", text)
    text = _TITLE_TRAILING_PACKAGING_PATTERN.sub(" ", text)
    text = re.sub(r"\b(?:x|×)\s*\d+\b", " ", text)
    return re.sub(r"\s+", " ", text).strip(" ,:-")


def strip_menu_size_suffix(value: Any) -> str:
    text = clean_title_text(value)
    if not text or "," not in text:
        return text

    parts = [part.strip() for part in text.split(",") if part.strip()]
    if len(parts) < 2:
        return text

    trailing = normalize_text(parts[-1])
    if trailing not in _TITLE_TRAILING_MENU_SIZE_ABBREVIATIONS:
        return text

    base = ", ".join(parts[:-1]).strip(" ,:-")
    if not base:
        return text

    if not tokenize(base).intersection(_MEAL_LIKE_TITLE_MARKERS):
        return text

    return clean_title_text(base)


def is_placeholder_title(value: Any) -> bool:
    normalized = normalize_text(value)
    if not normalized:
        return True
    return normalized in _TITLE_PLACEHOLDER_KEYS or normalized.startswith("unknown ")


def is_ingredient_like_title(value: Any) -> bool:
    normalized_title = normalize_text(canonicalize_title(value))
    if _contains_title_phrase(normalized_title, _INGREDIENT_LIKE_TITLE_PHRASES):
        return True

    title_tokens = tokenize_canonical_title(value)
    if not title_tokens:
        return False

    # Catch short standalone spice titles before they leak into meal-role pools.
    if not title_tokens.intersection(_MEAL_LIKE_TITLE_MARKERS):
        spice_hits = title_tokens.intersection(_INGREDIENT_LIKE_SPICE_NOUNS)
        spice_descriptors = title_tokens.intersection(_INGREDIENT_LIKE_SPICE_DESCRIPTORS)
        if spice_hits and (len(title_tokens) <= 3 or spice_descriptors):
            return True

    ingredient_hits = title_tokens.intersection(_INGREDIENT_LIKE_TITLE_MARKERS)
    if not ingredient_hits:
        return False

    strong_ingredient_hits = ingredient_hits.intersection({"extract", "mix", "paste", "powder", "seasoning", "spice", "spices"})
    if title_tokens.intersection(_MEAL_LIKE_TITLE_MARKERS) and not strong_ingredient_hits:
        return False
    return len(title_tokens) <= 5 or bool(strong_ingredient_hits)


def canonicalize_title(value: Any, include_candidate_only_aliases: bool = False) -> str:
    original = clean_title_text(value)
    if not original:
        return ""

    stripped = strip_menu_size_suffix(strip_packaging_noise(strip_brand_prefix(strip_title_wrappers(original))))
    stripped = re.sub(r"[_/]+", " ", stripped)
    stripped = _apply_title_token_replacements(stripped)
    stripped = re.sub(r"\s+", " ", stripped).strip(" ,:-")
    normalized = normalize_text(stripped)
    alias = _TITLE_ALIAS_MAP.get(normalized)
    if not alias and include_candidate_only_aliases:
        alias = _CANDIDATE_ONLY_TITLE_ALIAS_MAP.get(normalized)
    if alias:
        return alias
    phrase_alias = _phrase_alias_for_title(normalized)
    if phrase_alias:
        return phrase_alias
    return stripped or original


def build_display_title(primary_title: Any, mapped_title: Any = None) -> str:
    primary_clean = canonicalize_title(primary_title)
    mapped_clean = canonicalize_title(mapped_title)

    if is_placeholder_title(primary_clean) and mapped_clean and not is_placeholder_title(mapped_clean):
        return mapped_clean
    if not primary_clean and mapped_clean:
        return mapped_clean
    if (
        primary_clean
        and mapped_clean
        and not is_placeholder_title(mapped_clean)
        and not is_english_title(primary_clean)
        and is_english_title(mapped_clean)
    ):
        return mapped_clean
    return primary_clean or mapped_clean or clean_title_text(primary_title) or clean_title_text(mapped_title)


def canonical_title_key(value: Any, include_candidate_only_aliases: bool = False) -> str:
    return normalize_text(canonicalize_title(value, include_candidate_only_aliases=include_candidate_only_aliases))


def tokenize(value: Any) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9]+", normalize_text(value)) if token}


def tokenize_canonical_title(value: Any) -> set[str]:
    return tokenize(canonicalize_title(value))


def is_english_title(value: Any) -> bool:
    text = str(value or "").strip()
    if not text:
        return False
    if not _ENGLISH_ALPHA_PATTERN.search(text):
        return False
    normalized = normalize_text(text)
    if any(marker in normalized for marker in _NON_ENGLISH_TITLE_MARKERS):
        return False
    return bool(_ENGLISH_TITLE_PATTERN.fullmatch(text))


def cosine_similarity_safe(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    norm_a = float(np.linalg.norm(vec_a))
    norm_b = float(np.linalg.norm(vec_b))
    if norm_a <= 0.0 or norm_b <= 0.0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


def parse_force_exploration(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return normalize_text(value) in {"1", "true", "yes", "y", "on"}
    return False


def dedupe_strings(values: Iterable[Any], limit: int | None = None) -> list[str]:
    deduped: list[str] = []
    seen = set()

    for value in values:
        as_str = str(value or "").strip()
        normalized = normalize_text(as_str)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(as_str)
        if limit is not None and len(deduped) >= limit:
            break

    return deduped


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def normalize_to_per100(
    calories: Any,
    protein: Any,
    carbs: Any,
    fats: Any,
    metric_amount: Any,
) -> dict[str, float]:
    amount = to_float(metric_amount, 0.0)
    if amount <= 0:
        amount = 100.0
    factor = 100.0 / amount
    return {
        "calories": round(to_float(calories, 0.0) * factor, 4),
        "protein": round(to_float(protein, 0.0) * factor, 4),
        "carbs": round(to_float(carbs, 0.0) * factor, 4),
        "fats": round(to_float(fats, 0.0) * factor, 4),
    }


def normalize_macros(
    protein: Any,
    carbs: Any,
    fats: Any,
    calories: Any,
) -> np.ndarray:
    kcal = max(0.0, to_float(calories, 0.0))
    if kcal <= 0.0:
        return np.array([0.0, 0.0, 0.0], dtype=float)

    return np.array(
        [
            max(0.0, to_float(protein, 0.0)) / kcal,
            max(0.0, to_float(carbs, 0.0)) / kcal,
            max(0.0, to_float(fats, 0.0)) / kcal,
        ],
        dtype=float,
    )
