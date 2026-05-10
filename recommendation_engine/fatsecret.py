# The Data Fetcher. 
# It communicates with the FatSecret API to get real-world nutritional data.


from __future__ import annotations


import base64
import hashlib
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from .utils import canonicalize_title, dedupe_strings, ensure_array, normalize_text, to_float, to_positive_int

_BRAND_PREFIXES = {
    "mcdonald",
    "mcdonalds",
    "kfc",
    "burger king",
    "sonic",
    "subway",
    "starbucks",
    "taco bell",
    "wendy",
    "ihop",
    "domino",
    "pizza hut",
    "papa johns",
    "dunkin",
}


def _strip_brand_prefix(query: str) -> str:
    text = str(query or "").strip()
    if not text:
        return ""
    # NOTE: Drop brand prefixes (e.g., "Sonic, Chicken Sandwich" -> "Chicken Sandwich").
    if "," in text:
        text = text.split(",", 1)[1].strip()
    if " - " in text:
        text = text.split(" - ", 1)[1].strip()
    if ":" in text:
        text = text.split(":", 1)[1].strip()

    normalized = normalize_text(text)
    for brand in _BRAND_PREFIXES:
        if normalized.startswith(f"{brand} "):
            text = text[len(brand) :].strip()
            break
    return " ".join(text.split())


class FatSecretClient:
    OAUTH_URL = "https://goodhealthmate-fs.fly.dev/connect/token"
    API_URL = "https://goodhealthmate-fs.fly.dev/rest/server.api"

    def __init__(
        self,
        client_id: str | None,
        client_secret: str | None,
        cache_ttl_seconds: int = 1800,
    ):
        self.client_id = (client_id or "").strip()
        self.client_secret = (client_secret or "").strip()
        self.cache_ttl_seconds = cache_ttl_seconds
        self.token: str | None = None
        self.token_expires_at = 0.0
        self.response_cache: dict[str, tuple[float, Any]] = {}

    def _cache_get(self, key: str) -> Any:
        payload = self.response_cache.get(key)
        if not payload:
            return None

        expires_at, value = payload
        if expires_at <= time.time():
            self.response_cache.pop(key, None)
            return None
        return value

    def _cache_set(self, key: str, value: Any) -> Any:
        self.response_cache[key] = (time.time() + self.cache_ttl_seconds, value)
        return value

    @staticmethod
    def _build_cache_key(method_name: str, params: dict[str, Any]) -> str:
        normalized = "&".join(f"{key}={params[key]}" for key in sorted(params.keys()))
        return f"{method_name}:{normalized}"

    def _get_access_token(self, force_refresh: bool = False) -> str | None:
        if not self.client_id or not self.client_secret:
            return None

        now = time.time()
        if not force_refresh and self.token and now < (self.token_expires_at - 30):
            return self.token

        credentials = f"{self.client_id}:{self.client_secret}".encode("utf-8")
        encoded = base64.b64encode(credentials).decode("utf-8")
        body = urllib.parse.urlencode({"grant_type": "client_credentials", "scope": "premier"}).encode("utf-8")

        req = urllib.request.Request(
            self.OAUTH_URL,
            data=body,
            headers={
                "Authorization": f"Basic {encoded}",
                "Content-Type": "application/x-www-form-urlencoded",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=15) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except Exception:
            self.token = None
            self.token_expires_at = 0.0
            return None

        token = payload.get("access_token")
        if not token:
            self.token = None
            self.token_expires_at = 0.0
            return None

        self.token = str(token)
        expires_in = to_positive_int(payload.get("expires_in"), 3600)
        self.token_expires_at = now + expires_in
        return self.token

    def prime_token(self) -> bool:
        # NOTE: Force-refresh OAuth token to avoid first-request latency.
        return bool(self._get_access_token(force_refresh=True))

    def _request(self, method_name: str, params: dict[str, Any], retry_on_401: bool = True) -> Any:
        token = self._get_access_token()
        if not token:
            return None

        final_params = {"method": method_name, "format": "json", **params}
        cache_key = self._build_cache_key(method_name, final_params)

        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        query = urllib.parse.urlencode(final_params)
        req = urllib.request.Request(
            f"{self.API_URL}?{query}",
            headers={"Authorization": f"Bearer {token}"},
            method="GET",
        )

        try:
            with urllib.request.urlopen(req, timeout=20) as response:
                payload = json.loads(response.read().decode("utf-8"))
                return self._cache_set(cache_key, payload)
        except urllib.error.HTTPError as error:
            if error.code == 401 and retry_on_401:
                self._get_access_token(force_refresh=True)
                return self._request(method_name, params, retry_on_401=False)
            return None
        except Exception:
            return None

    def search_foods(
        self,
        query: str,
        max_results: int = 50,
        category: str | None = None,
    ) -> list[dict[str, Any]]:
        normalized_query = _strip_brand_prefix(str(query or "").strip())
        if not normalized_query:
            return []

        category_hint = normalize_text(category)
        search_expression = normalized_query
        params: dict[str, Any] = {
            "search_expression": search_expression,
            "max_results": str(max_results),
            "include_food_images": "true",
        }
        if category_hint in {"recipe", "main"}:
            params["food_type"] = "recipe"
        elif category_hint in {"generic", "side"}:
            params["food_type"] = "generic"
        elif category_hint in {"beverage", "drink"}:
            params["food_type"] = "generic"
            if "drink" not in search_expression and "beverage" not in search_expression:
                params["search_expression"] = f"{search_expression} drink"

        payload = self._request(
            "foods.search",
            params,
        )
        if not payload:
            return []

        foods = payload.get("foods", {}).get("food")
        return [item for item in ensure_array(foods) if isinstance(item, dict)]

    def get_food(self, food_id: Any) -> dict[str, Any] | None:
        if food_id is None:
            return None

        payload = self._request(
            "food.get.v5",
            {
                "food_id": str(food_id),
                "include_food_images": "true",
            },
        )
        if not payload or not isinstance(payload, dict):
            return None
        return payload


def extract_image(food_obj: dict[str, Any]) -> str | None:
    imgs = food_obj.get("food_images", {}).get("food_image")
    for img in ensure_array(imgs):
        if isinstance(img, dict) and img.get("image_url"):
            return str(img["image_url"])

    image = food_obj.get("food_image")
    if isinstance(image, str):
        return image
    if isinstance(image, dict):
        maybe_url = image.get("image_url")
        return str(maybe_url) if maybe_url else None
    return None


def parse_description_macros(description: Any) -> dict[str, float]:
    text = str(description or "")
    patterns = {
        "calories": r"calories:\s*([\d.]+)",
        "protein": r"protein:\s*([\d.]+)",
        "carbs": r"carbs?:\s*([\d.]+)",
        "fats": r"fat:\s*([\d.]+)",
    }

    output = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, text, flags=re.IGNORECASE)
        output[key] = to_float(match.group(1), 0.0) if match else 0.0
    return output


def extract_per100_macros(food_obj: dict[str, Any]) -> dict[str, float] | None:
    servings = ensure_array(food_obj.get("servings", {}).get("serving"))

    for serving in servings:
        if not isinstance(serving, dict):
            continue

        metric_amount = to_float(serving.get("metric_serving_amount"), 0.0)
        if metric_amount <= 0:
            continue

        calories = to_float(serving.get("calories"), 0.0)
        protein = to_float(serving.get("protein"), 0.0)
        carbs = to_float(serving.get("carbohydrate"), 0.0)
        fats = to_float(serving.get("fat"), 0.0)
        factor = 100.0 / metric_amount

        return {
            "calories": calories * factor,
            "protein": protein * factor,
            "carbs": carbs * factor,
            "fats": fats * factor,
        }

    fallback = parse_description_macros(food_obj.get("food_description"))
    if any(value > 0 for value in fallback.values()):
        return fallback
    return None


def map_food_hit_to_candidate(
    food_hit: dict[str, Any],
    detailed_food_payload: dict[str, Any] | None,
    source_keyword: str,
    meal_type: str,
) -> dict[str, Any] | None:
    detailed_food = (detailed_food_payload or {}).get("food", {})
    food_obj = detailed_food if isinstance(detailed_food, dict) and detailed_food else food_hit

    macros = extract_per100_macros(food_obj)
    if not macros:
        macros = parse_description_macros(food_hit.get("food_description"))
    if not any(value > 0 for value in macros.values()):
        return None

    title = str(food_obj.get("food_name") or food_hit.get("food_name") or source_keyword).strip()
    canonical_title = canonicalize_title(title)
    food_id = food_obj.get("food_id") or food_hit.get("food_id")
    if food_id is None:
        food_id = hashlib.md5(f"{title}|{source_keyword}".encode("utf-8")).hexdigest()[:12]

    servings = ensure_array(food_obj.get("servings", {}).get("serving"))
    primary_serving = None
    for serving in servings:
        if isinstance(serving, dict) and to_float(serving.get("metric_serving_amount"), 0.0) > 0:
            primary_serving = serving
            break
    if primary_serving is None and servings:
        primary_serving = servings[0] if isinstance(servings[0], dict) else {}
    primary_serving = primary_serving if isinstance(primary_serving, dict) else {}
    serving_calories = to_float(primary_serving.get("calories"), 0.0)
    serving_protein = to_float(primary_serving.get("protein"), 0.0)
    serving_carbs = to_float(primary_serving.get("carbohydrate"), 0.0)
    serving_fats = to_float(primary_serving.get("fat"), 0.0)
    if serving_calories <= 0.0:
        metric_amount = to_float(primary_serving.get("metric_serving_amount"), 0.0)
        if metric_amount > 0.0:
            if to_float(macros.get("calories"), 0.0) > 0.0:
                serving_calories = (to_float(macros.get("calories"), 0.0) * metric_amount) / 100.0
            if to_float(macros.get("protein"), 0.0) > 0.0:
                serving_protein = (to_float(macros.get("protein"), 0.0) * metric_amount) / 100.0
            if to_float(macros.get("carbs"), 0.0) > 0.0:
                serving_carbs = (to_float(macros.get("carbs"), 0.0) * metric_amount) / 100.0
            if to_float(macros.get("fats"), 0.0) > 0.0:
                serving_fats = (to_float(macros.get("fats"), 0.0) * metric_amount) / 100.0

    allergens = [
        {
            "id": allergen.get("id"),
            "name": allergen.get("name"),
            "value": allergen.get("value"),
        }
        for allergen in ensure_array(food_obj.get("allergens", {}).get("allergen"))
        if isinstance(allergen, dict)
    ]
    preferences = [
        {
            "id": pref.get("id"),
            "name": pref.get("name"),
            "value": pref.get("value"),
        }
        for pref in ensure_array(food_obj.get("preferences", {}).get("preference"))
        if isinstance(pref, dict)
    ]
    sub_categories = []
    for entry in ensure_array(food_obj.get("food_sub_categories", {}).get("food_sub_category")):
        if isinstance(entry, dict):
            value = entry.get("food_sub_category")
        else:
            value = entry
        if value is not None and str(value).strip():
            sub_categories.append(str(value).strip())

    return {
        "id": str(food_id),
        "food_id": str(food_id),
        "title": title,
        "original_title": title,
        "canonical_title": canonical_title,
        "mapped_title": title,
        "mapped_canonical_title": canonical_title,
        "image": extract_image(food_obj) or extract_image(food_hit) or None,
        "meal_type": meal_type,
        "source_keyword": source_keyword,
        "per100": {
            "calories": round(to_float(macros.get("calories"), 0.0), 3),
            "protein": round(to_float(macros.get("protein"), 0.0), 3),
            "carbs": round(to_float(macros.get("carbs"), 0.0), 3),
            "fats": round(to_float(macros.get("fats"), 0.0), 3),
        },
        "food_type": food_obj.get("food_type") or food_hit.get("food_type"),
        "food_url": food_obj.get("food_url") or food_hit.get("food_url"),
        "brand_name": food_obj.get("brand_name") or food_hit.get("brand_name"),
        "serving_id": primary_serving.get("serving_id"),
        "serving_description": primary_serving.get("serving_description"),
        "metric_serving_amount": primary_serving.get("metric_serving_amount"),
        "metric_serving_unit": primary_serving.get("metric_serving_unit"),
        "number_of_units": primary_serving.get("number_of_units"),
        "measurement_description": primary_serving.get("measurement_description"),
        "serving_calories": round(serving_calories, 3),
        "serving_protein": round(serving_protein, 3),
        "serving_carbs": round(serving_carbs, 3),
        "serving_fats": round(serving_fats, 3),
        "allergens": allergens,
        "preferences": preferences,
        "food_sub_categories": sub_categories,
    }


def build_safety_candidates(meal_type: str, keywords: list[str]) -> list[dict[str, Any]]:
    normalized_meal = normalize_text(meal_type)
    if normalized_meal == "breakfast":
        base = [
            {"title": "Greek Yogurt Bowl", "category": "main", "per100": {"calories": 95, "protein": 10, "carbs": 8, "fats": 3}},
            {"title": "Oatmeal with Milk", "category": "main", "per100": {"calories": 138, "protein": 6, "carbs": 22, "fats": 3}},
            {"title": "Egg and Avocado Toast", "category": "main", "per100": {"calories": 182, "protein": 9, "carbs": 14, "fats": 10}},
            {"title": "Scrambled Eggs on Toast", "category": "main", "per100": {"calories": 176, "protein": 11, "carbs": 13, "fats": 9}},
            {"title": "Peanut Butter Banana Toast", "category": "main", "per100": {"calories": 205, "protein": 8, "carbs": 24, "fats": 9}},
            {"title": "Berry Yogurt Parfait", "category": "main", "per100": {"calories": 128, "protein": 8, "carbs": 18, "fats": 3}},
            {"title": "Fruit and Nuts Mix", "category": "side", "per100": {"calories": 210, "protein": 5, "carbs": 19, "fats": 12}},
            {"title": "Mixed Berries", "category": "side", "per100": {"calories": 57, "protein": 1, "carbs": 14, "fats": 0}},
        ]
    elif normalized_meal == "lunch":
        base = [
            {"title": "Chicken and Brown Rice", "category": "main", "per100": {"calories": 165, "protein": 17, "carbs": 15, "fats": 4}},
            {"title": "Vegetable Stir Fry", "category": "side", "per100": {"calories": 110, "protein": 5, "carbs": 12, "fats": 4}},
            {"title": "Mediterranean Salad", "category": "side", "per100": {"calories": 120, "protein": 6, "carbs": 9, "fats": 6}},
            {"title": "Turkey Wrap", "category": "main", "per100": {"calories": 160, "protein": 12, "carbs": 16, "fats": 5}},
            {"title": "Tuna and Quinoa Bowl", "category": "main", "per100": {"calories": 170, "protein": 14, "carbs": 15, "fats": 5}},
            {"title": "Fruit and Nuts Mix", "category": "side", "per100": {"calories": 210, "protein": 5, "carbs": 19, "fats": 12}},
        ]
    else:
        base = [
            {"title": "Chicken and Brown Rice", "category": "main", "per100": {"calories": 165, "protein": 17, "carbs": 15, "fats": 4}},
            {"title": "Grilled Salmon Plate", "category": "main", "per100": {"calories": 195, "protein": 20, "carbs": 3, "fats": 11}},
            {"title": "Vegetable Stir Fry", "category": "side", "per100": {"calories": 110, "protein": 5, "carbs": 12, "fats": 4}},
            {"title": "Mediterranean Salad", "category": "side", "per100": {"calories": 120, "protein": 6, "carbs": 9, "fats": 6}},
            {"title": "Turkey Wrap", "category": "main", "per100": {"calories": 160, "protein": 12, "carbs": 16, "fats": 5}},
            {"title": "Tuna and Quinoa Bowl", "category": "main", "per100": {"calories": 170, "protein": 14, "carbs": 15, "fats": 5}},
            {"title": "Tofu Rice Bowl", "category": "main", "per100": {"calories": 150, "protein": 9, "carbs": 18, "fats": 5}},
            {"title": "Chicken Pho Style Soup", "category": "side", "per100": {"calories": 92, "protein": 8, "carbs": 11, "fats": 2}},
        ]

    output: list[dict[str, Any]] = []
    for index, item in enumerate(base):
        source = keywords[index % len(keywords)] if keywords else "balanced meal"
        output.append(
            {
                "id": f"safety-{meal_type}-{index}",
                "title": item["title"],
                "image": None,
                "meal_type": meal_type,
                "category": item.get("category"),
                "source_keyword": source,
                "per100": item["per100"],
            }
        )
    return output


def retrieve_food_candidates(
    client: FatSecretClient,
    queries: list[str],
    meal_type: str,
    max_results_per_query: int = 50,
    hard_cap: int = 50,
    top_foods: list[str] | None = None,
) -> list[dict[str, Any]]:
    if not queries:
        return build_safety_candidates(meal_type, [])

    top_foods = top_foods or []
    expanded_queries = dedupe_strings([*queries, *top_foods], limit=8)
    if not expanded_queries:
        return build_safety_candidates(meal_type, [])

    candidates: list[dict[str, Any]] = []
    seen_titles = set()
    detail_cache: dict[str, dict[str, Any] | None] = {}

    for query in expanded_queries:
        hits = client.search_foods(query, max_results=max_results_per_query)
        for hit in hits[:max_results_per_query]:
            if len(candidates) >= hard_cap:
                break
            if not isinstance(hit, dict):
                continue

            food_id = str(hit.get("food_id") or "").strip()
            if food_id:
                if food_id not in detail_cache:
                    detail_cache[food_id] = client.get_food(food_id)
                detail = detail_cache[food_id]
            else:
                detail = None

            mapped = map_food_hit_to_candidate(hit, detail, query, meal_type)
            if not mapped:
                continue

            title_key = normalize_text(mapped.get("title"))
            if not title_key or title_key in seen_titles:
                continue
            seen_titles.add(title_key)
            candidates.append(mapped)

        if len(candidates) >= hard_cap:
            break

    return candidates or build_safety_candidates(meal_type, expanded_queries)
