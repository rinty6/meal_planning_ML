# This file calculates BMR 
# builds a "User Vector" based on what the user previously ate. 


from __future__ import annotations


from collections import Counter
from datetime import date
from typing import Any

import json
import os
from pathlib import Path
from threading import Lock

import numpy as np
import pandas as pd

try:
    from sklearn.cluster import KMeans  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    KMeans = None

from .constants import (
    DEFAULT_DAILY_CALORIES,
    DEFAULT_MEAL_ALLOCATION,
    HISTORY_LOOKBACK_DAYS,
    MAX_ARCHETYPE_CLUSTERS,
    MEAL_SLOTS,
    MIN_CLUSTER_WEIGHT,
    MIN_HISTORY_ROWS_FOR_CLUSTERING,
    MIN_HISTORY_ROWS_FOR_DYNAMIC_RATIO,
    MIN_DAILY_CALORIES,
    TIME_DECAY_LAMBDA,
)
from .utils import canonical_title_key, canonicalize_title, clean_title_text, normalize_text, to_float, tokenize_canonical_title

_EATING_HISTORY_CACHE: list[dict[str, Any]] | None = None
_EATING_HISTORY_LOCK = Lock()
_PROFILE_RUNTIME_WARMED = False
_PROFILE_RUNTIME_WARMUP_LOCK = Lock()


def _load_eating_history_snapshot() -> list[dict[str, Any]]:
    global _EATING_HISTORY_CACHE
    if _EATING_HISTORY_CACHE is not None:
        return _EATING_HISTORY_CACHE
    with _EATING_HISTORY_LOCK:
        if _EATING_HISTORY_CACHE is not None:
            return _EATING_HISTORY_CACHE

        candidate_paths = [
            os.getenv("EATING_HISTORY_FILE", "").strip(),
            str(Path(__file__).resolve().parent.parent / "dataset_process" / "eating_history.json"),
            os.path.join(os.path.expanduser("~"), "Downloads", "eating_history.json"),
        ]

        for path in candidate_paths:
            if not path:
                continue
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    payload = json.load(handle)
                if isinstance(payload, list):
                    _EATING_HISTORY_CACHE = [row for row in payload if isinstance(row, dict)]
                    return _EATING_HISTORY_CACHE
            except Exception:
                continue

        _EATING_HISTORY_CACHE = []
        return _EATING_HISTORY_CACHE


def _build_time_decay_weights(history_df: pd.DataFrame) -> np.ndarray:
    if history_df is None or history_df.empty:
        return np.array([], dtype=float)
    if "created_at" not in history_df.columns or not history_df["created_at"].notna().any():
        return np.ones((len(history_df),), dtype=float)

    created_series = pd.to_datetime(history_df["created_at"], errors="coerce", utc=True)
    reference_time = pd.Timestamp.now(tz="UTC")
    age_days = ((reference_time - created_series).dt.total_seconds().fillna(0.0) / 86400.0).clip(lower=0.0)
    lambda_value = max(0.0, float(TIME_DECAY_LAMBDA))
    if lambda_value <= 0.0:
        return np.ones((len(history_df),), dtype=float)

    weights = np.exp(-lambda_value * age_days.to_numpy(dtype=float))
    weights[~np.isfinite(weights)] = 1.0
    weights = np.clip(weights, 1e-4, 1.0)
    return weights


def _weighted_macro_mean(frame: pd.DataFrame, weights: np.ndarray) -> np.ndarray:
    values = frame[["calories", "protein", "carbs", "fats"]].to_numpy(dtype=float)
    if values.size == 0:
        return np.zeros((4,), dtype=float)

    safe_weights = np.asarray(weights, dtype=float)
    if safe_weights.shape[0] != values.shape[0] or np.sum(safe_weights) <= 0.0:
        return values.mean(axis=0).astype(float)

    return np.average(values, axis=0, weights=safe_weights).astype(float)


def _build_rule_based_archetypes(frame: pd.DataFrame, weights: np.ndarray) -> list[np.ndarray]:
    if frame.empty:
        return []

    plant_keywords = {"tofu", "tempeh", "lentil", "bean", "vegetable", "salad", "fruit", "veggie"}
    protein_keywords = {"chicken", "beef", "pork", "fish", "salmon", "egg", "turkey", "protein", "steak"}

    name_series = frame["canonical_title"].astype(str).str.lower()
    plant_mask = name_series.str.contains("|".join(sorted(plant_keywords)), regex=True, na=False)
    protein_mask = name_series.str.contains("|".join(sorted(protein_keywords)), regex=True, na=False)
    protein_ratio = frame["protein"].clip(lower=0) / frame["calories"].replace(0, np.nan)
    protein_mask = protein_mask | (protein_ratio.fillna(0) >= 0.25)

    archetype_vectors: list[np.ndarray] = []
    for mask in (plant_mask, protein_mask):
        if not mask.any():
            continue
        mask_values = mask.to_numpy(dtype=bool)
        vec = _weighted_macro_mean(frame.loc[mask_values], weights[mask_values])
        if np.isfinite(vec).all():
            archetype_vectors.append(np.array(vec, dtype=float))

    return archetype_vectors


def _build_cluster_archetypes(frame: pd.DataFrame, weights: np.ndarray) -> tuple[list[np.ndarray], dict[str, Any]]:
    metadata = {
        "enabled": False,
        "raw_cluster_count": 0,
        "cluster_count": 0,
    }
    if KMeans is None or frame.empty or len(frame) < int(MIN_HISTORY_ROWS_FOR_CLUSTERING):
        return [], metadata

    features = frame[["calories", "protein", "carbs", "fats"]].to_numpy(dtype=float)
    safe_weights = np.asarray(weights, dtype=float)
    valid_mask = np.isfinite(features).all(axis=1) & np.isfinite(safe_weights)
    if valid_mask.sum() < int(MIN_HISTORY_ROWS_FOR_CLUSTERING):
        return [], metadata

    working = frame.loc[valid_mask].reset_index(drop=True)
    features = features[valid_mask]
    safe_weights = np.clip(safe_weights[valid_mask], 1e-4, None)
    unique_rows = np.unique(np.round(features, 3), axis=0)
    cluster_count = min(int(MAX_ARCHETYPE_CLUSTERS), len(working), len(unique_rows))
    metadata["enabled"] = True
    metadata["raw_cluster_count"] = int(cluster_count)
    if cluster_count < 2:
        return [], metadata

    model = KMeans(n_clusters=cluster_count, n_init=8, random_state=42)
    try:
        labels = model.fit_predict(features, sample_weight=safe_weights)
    except TypeError:
        labels = model.fit_predict(features)

    cluster_vectors: list[np.ndarray] = []
    cluster_weights: list[float] = []
    total_weight = float(np.sum(safe_weights)) or 1.0
    for cluster_idx in range(cluster_count):
        cluster_mask = labels == cluster_idx
        if int(np.sum(cluster_mask)) < 2:
            continue
        cluster_weight = float(np.sum(safe_weights[cluster_mask])) / total_weight
        if cluster_weight < float(MIN_CLUSTER_WEIGHT):
            continue
        vec = _weighted_macro_mean(working.loc[cluster_mask], safe_weights[cluster_mask])
        if not np.isfinite(vec).all():
            continue
        cluster_vectors.append(np.array(vec, dtype=float))
        cluster_weights.append(cluster_weight)

    if cluster_vectors:
        ordered = np.argsort(np.asarray(cluster_weights, dtype=float))[::-1]
        cluster_vectors = [cluster_vectors[index] for index in ordered.tolist()]
    metadata["cluster_count"] = len(cluster_vectors)
    return cluster_vectors, metadata


def normalize_history_df(history_df: pd.DataFrame | None) -> pd.DataFrame:
    if history_df is None or history_df.empty:
        return pd.DataFrame(
            columns=[
                "meal_type",
                "food_name",
                "original_title",
                "canonical_title",
                "calories",
                "protein",
                "carbs",
                "fats",
                "image",
                "date",
                "created_at",
            ]
        )

    working = history_df.copy()
    for col in ("calories", "protein", "carbs", "fats"):
        if col in working.columns:
            working[col] = pd.to_numeric(working[col], errors="coerce").fillna(0.0)
        else:
            working[col] = 0.0

    working["meal_type"] = working.get("meal_type", "").astype(str).str.lower()
    working = working[working["meal_type"].isin(MEAL_SLOTS)]

    if "food_name" not in working.columns:
        working["food_name"] = ""
    working["food_name"] = working["food_name"].astype(str).str.strip()
    working["original_title"] = working["food_name"].map(clean_title_text)
    working["canonical_title"] = working["original_title"].map(canonicalize_title)

    if "image" not in working.columns:
        working["image"] = ""
    working["image"] = working["image"].fillna("").astype(str)

    if "created_at" in working.columns:
        working["created_at"] = pd.to_datetime(working["created_at"], errors="coerce")
    else:
        working["created_at"] = pd.NaT

    if "date" in working.columns:
        working["date"] = pd.to_datetime(working["date"], errors="coerce")
    else:
        working["date"] = pd.NaT

    return working


def get_age_from_dob(dob_str: Any) -> int:
    if not dob_str:
        return 30
    try:
        dob = date.fromisoformat(str(dob_str)[:10])
        today = date.today()
        return today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
    except Exception:
        return 30


def calculate_bmr(weight_kg: Any, height_cm: Any, age: Any, gender: Any, activity_level: Any, goal: str = "maintain") -> float:
    weight = to_float(weight_kg, 0.0)
    height = to_float(height_cm, 0.0)
    age_value = max(1.0, to_float(age, 30.0))
    gender_text = normalize_text(gender)

    if weight <= 0 or height <= 0:
        daily = DEFAULT_DAILY_CALORIES
        if goal == "lose_weight":
            daily -= 500
        elif goal == "gain_muscle":
            daily += 300
        return max(MIN_DAILY_CALORIES, float(daily))
    if gender_text == "male":
        bmr = 88.36 + (13.4 * weight) + (4.8 * height) - (5.7 * age_value)
        daily = bmr
    else:
        bmr = 447.6 + (9.2 * weight) + (3.1 * height) - (4.3 * age_value)
        daily = bmr

    multipliers = {
        "sedentary": 1.2,
        "lightly_active": 1.375,
        "moderately_active": 1.55,
        "very_active": 1.725,
        "super_active": 1.9,
        "extra_active": 1.9,
    }
    daily *= multipliers.get(normalize_text(activity_level), 1.55)

    if goal == "lose_weight":
        daily -= 500
    elif goal == "gain_muscle":
        daily += 300

    return max(MIN_DAILY_CALORIES, float(daily))


def resolve_daily_calories(calorie_override: Any, active_goal_calories: Any, demographics: dict[str, Any], goal: str) -> float:
    override = to_float(calorie_override, 0.0)
    if override > 0:
        return max(MIN_DAILY_CALORIES, override)

    active_goal = to_float(active_goal_calories, 0.0)
    if active_goal > 0:
        return max(MIN_DAILY_CALORIES, active_goal)

    age = get_age_from_dob(demographics.get("dateOfBirth"))
    return calculate_bmr(
        weight_kg=demographics.get("weight"),
        height_cm=demographics.get("height"),
        age=age,
        gender=demographics.get("gender", "male"),
        activity_level=demographics.get("activityLevel", "moderately_active"),
        goal=goal,
    )


def compute_dynamic_meal_allocation(history_df: pd.DataFrame) -> dict[str, float]:
    if history_df.empty:
        return DEFAULT_MEAL_ALLOCATION.copy()

    working = history_df.copy()
    if working["created_at"].notna().any():
        cutoff = pd.Timestamp.now() - pd.Timedelta(days=HISTORY_LOOKBACK_DAYS)
        working = working[working["created_at"].notna() & (working["created_at"] >= cutoff)]
        if working.empty:
            return DEFAULT_MEAL_ALLOCATION.copy()

    # NOTE: Only adapt ratios once we have enough usage history.
    if len(working) <= MIN_HISTORY_ROWS_FOR_DYNAMIC_RATIO:
        return DEFAULT_MEAL_ALLOCATION.copy()

    calories_by_meal = working.groupby("meal_type", as_index=True)["calories"].sum()
    total = float(calories_by_meal.sum())
    if total <= 0.0:
        return DEFAULT_MEAL_ALLOCATION.copy()

    observed = {meal: float(calories_by_meal.get(meal, 0.0)) / total for meal in MEAL_SLOTS}
    norm = sum(observed.values()) or 1.0
    return {meal: observed[meal] / norm for meal in MEAL_SLOTS}


def warm_profile_runtime() -> None:
    global _PROFILE_RUNTIME_WARMED
    if _PROFILE_RUNTIME_WARMED:
        return

    with _PROFILE_RUNTIME_WARMUP_LOCK:
        if _PROFILE_RUNTIME_WARMED:
            return

        warm_row_count = max(int(MIN_HISTORY_ROWS_FOR_CLUSTERING), 30)
        warm_rows: list[dict[str, Any]] = []
        base_created_at = pd.Timestamp("2026-01-01", tz="UTC")

        for index in range(warm_row_count):
            warm_rows.append(
                {
                    "meal_type": "breakfast",
                    "food_name": f"Profile Warmup Food {index}",
                    "original_title": f"Profile Warmup Food {index}",
                    "canonical_title": f"profile warmup food {index}",
                    "calories": 260 + (index * 7),
                    "protein": 16 + (index % 6),
                    "carbs": 24 + (index % 9),
                    "fats": 8 + (index % 5),
                    "image": "",
                    "date": base_created_at,
                    "created_at": base_created_at,
                }
            )

        warm_df = pd.DataFrame(warm_rows)
        build_user_profile(warm_df, meal_type="breakfast")
        _PROFILE_RUNTIME_WARMED = True


def build_user_profile(history_df: pd.DataFrame, meal_type: str | None = None) -> dict[str, Any]:
    subset = history_df
    if meal_type in MEAL_SLOTS:
        subset = history_df[history_df["meal_type"] == meal_type]
        if subset.empty:
            subset = history_df

    if subset.empty:
        return {
            "user_vec": None,
            "top_foods": [],
            "preference_tokens": set(),
            "recent_titles": set(),
        }

    time_decay_weights = _build_time_decay_weights(subset)
    user_vec = _weighted_macro_mean(subset, time_decay_weights)

    display_title_by_key: dict[str, str] = {}
    food_counts: dict[str, float] = {}
    original_titles = subset["original_title"].tolist() if "original_title" in subset.columns else subset["food_name"].tolist()
    canonical_titles = subset["canonical_title"].tolist() if "canonical_title" in subset.columns else subset["food_name"].tolist()
    for idx, (original_title, canonical_title) in enumerate(zip(original_titles, canonical_titles)):
        original_name = clean_title_text(original_title)
        name_key = canonical_title_key(canonical_title or original_title)
        if not name_key:
            continue
        weight = float(time_decay_weights[idx]) if idx < len(time_decay_weights) else 1.0
        if weight <= 0.0:
            continue
        display_title_by_key.setdefault(name_key, original_name or clean_title_text(name_key) or name_key)
        food_counts[name_key] = food_counts.get(name_key, 0.0) + weight

    # NOTE: Blend in static eating history to reduce cold-start and improve habits.
    snapshot_counts: dict[str, float] = {}
    for row in _load_eating_history_snapshot():
        raw_name = clean_title_text(row.get("food_name") or "")
        name_key = canonical_title_key(raw_name)
        if not name_key:
            continue
        row_meal = normalize_text(row.get("meal_type"))
        if meal_type in MEAL_SLOTS and row_meal and row_meal != meal_type:
            continue
        count = int(to_float(row.get("number_appearance"), 0.0))
        if count <= 0:
            continue
        display_title_by_key.setdefault(name_key, raw_name or name_key)
        snapshot_counts[name_key] = snapshot_counts.get(name_key, 0.0) + float(count)

    for name_key, count in snapshot_counts.items():
        food_counts[name_key] = food_counts.get(name_key, 0.0) + float(count)

    sorted_food_counts = sorted(food_counts.items(), key=lambda item: item[1], reverse=True)
    top_food_counts = {name: round(float(count), 6) for name, count in sorted_food_counts[:12]}
    top_food_keys = [name for name, _count in sorted_food_counts[:8]]
    top_foods = [display_title_by_key.get(name, name) for name in top_food_keys]

    preference_tokens: set[str] = set()
    token_counts = Counter()
    for food_name, count in food_counts.items():
        for token in tokenize_canonical_title(food_name):
            token_counts[token] += float(count)
    for food_name in top_food_keys:
        preference_tokens.update(tokenize_canonical_title(food_name))
    for token in token_counts.keys():
        preference_tokens.add(token)

    preference_token_weights = {
        token: 1.5 for token, count in token_counts.items() if float(count) >= 3.0
    }

    recent_subset = subset.sort_values("created_at", ascending=False, na_position="last").head(12)
    recent_titles = {
        canonical_title_key(food_name) for food_name in recent_subset["canonical_title"].tolist() if str(food_name).strip()
    }

    rule_based_archetypes = _build_rule_based_archetypes(subset, time_decay_weights)
    clustered_archetypes, cluster_metadata = _build_cluster_archetypes(subset, time_decay_weights)
    archetype_vectors = clustered_archetypes or rule_based_archetypes or [np.array(user_vec, dtype=float)]
    archetype_strategy = "clustered" if clustered_archetypes else "rule_based"

    return {
        "user_vec": np.array(user_vec, dtype=float),
        "archetype_vectors": archetype_vectors,
        "top_foods": top_foods,
        "top_food_keys": top_food_keys,
        "top_food_counts": top_food_counts,
        "preference_tokens": preference_tokens,
        "preference_token_weights": preference_token_weights,
        "recent_titles": recent_titles,
        "archetype_strategy": archetype_strategy,
        "archetype_cluster_count": int(cluster_metadata.get("cluster_count", 0)),
        "history_rows": int(len(subset)),
    }


def get_top_consumed_items(history_df: pd.DataFrame, meal_type: str | None = None, top_n: int = 8) -> list[dict[str, Any]]:
    if history_df is None or history_df.empty:
        return []

    working = history_df.copy()
    if meal_type in MEAL_SLOTS:
        working = working[working["meal_type"] == meal_type]
    if working.empty:
        return []

    working["norm_name"] = working["food_name"].map(normalize_text)
    working = working[working["norm_name"] != ""]
    if working.empty:
        return []

    latest = (
        working.sort_values("created_at", ascending=False)
        .drop_duplicates(subset=["norm_name"])
        .set_index("norm_name")
    )
    counts = working.groupby("norm_name").size().sort_values(ascending=False)

    output: list[dict[str, Any]] = []
    for norm_name, count in counts.head(top_n).items():
        row = latest.loc[norm_name] if norm_name in latest.index else None
        title = str(row["food_name"]).strip() if row is not None else norm_name
        image = str(row["image"]).strip() if row is not None and row.get("image") is not None else ""
        source_meal = str(row["meal_type"]).strip() if row is not None else ""
        output.append(
            {
                "title": title or norm_name,
                "original_title": title or norm_name,
                "canonical_title": canonicalize_title(title or norm_name),
                "food_name": title or norm_name,
                "count": int(count),
                "number_appearance": int(count),
                "meal_type": source_meal or (meal_type or ""),
                "image": image,
            }
        )

    return output


def build_behavioral_insight(
    meal_type: str,
    slot_weights: dict[str, float],
    top_foods: list[str],
    health_hint: str | None = None,
) -> str:
    observed = to_float(slot_weights.get(meal_type), DEFAULT_MEAL_ALLOCATION[meal_type])
    baseline = DEFAULT_MEAL_ALLOCATION[meal_type]
    observed_pct = int(round(observed * 100))

    if observed >= baseline + 0.04:
        habit = "you usually eat a larger share in this timeline"
    elif observed <= baseline - 0.04:
        habit = "you usually eat a lighter share in this timeline"
    else:
        habit = "your intake is stable for this timeline"

    if top_foods:
        top_text = ", ".join(top_foods[:2])
        insight = (
            f"{habit}. We prioritized foods similar to what you consume most often, "
            f"like {top_text}, while targeting ~{observed_pct}% of daily calories."
        )
    else:
        insight = f"{habit}. This recommendation targets about {observed_pct}% of your daily calories."

    if health_hint:
        # NOTE: Append health alignment context when available.
        return f"{insight} {health_hint}"
    return insight
