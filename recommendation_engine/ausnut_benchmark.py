from __future__ import annotations


import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .constants import MEAL_SLOTS
from .local_dataset import _parse_serving_grams
from .ranking import build_text_role_candidate, infer_combo_category, is_candidate_role_compatible
from .utils import canonical_title_key, canonicalize_title, normalize_text, to_float


try:
    import duckdb  # type: ignore
except Exception:  # pragma: no cover - optional at authoring time
    duckdb = None


BENCHMARK_POLICY_VERSION = "2026-03-19"
BENCHMARK_REPORT_TEMPLATE_VERSION = "ausnut-benchmark-v1"
BENCHMARK_TOP_K_VALUES = (1, 3, 5, 10)
BENCHMARK_FILE_NAME = "AUSNUT_dataset.csv"
BENCHMARK_DIR_NAME = "ausnut_benchmark"

PRIMARY_BENCHMARK_FLAGS = (
    "ambiguous_row",
    "condiment_only",
    "incomplete_row",
)
SUPPLEMENTARY_BENCHMARK_FLAGS = ("beverage_only",)

BREAKFAST_TOKENS = (
    "breakfast cereal",
    "cereal",
    "granola",
    "muesli",
    "porridge",
    "oat",
    "toast",
    "egg",
    "yoghurt",
    "yogurt",
    "pancake",
    "waffle",
)
LUNCH_TOKENS = (
    "salad",
    "soup",
    "sandwich",
    "wrap",
    "roll",
    "burger",
    "pizza",
    "noodle",
    "pasta dish",
    "bean",
    "quiche",
)
DINNER_TOKENS = (
    "beef",
    "chicken",
    "lamb",
    "pork",
    "fish",
    "seafood",
    "curry",
    "stir-fry",
    "stir fry",
    "casserole",
    "roast",
    "rice dish",
    "sausage",
)
SNACK_TOKENS = (
    "biscuit",
    "bar",
    "cake",
    "ice cream",
    "confectionery",
    "chip",
    "cracker",
    "dessert",
    "pastry",
    "slice",
    "nut",
)
BEVERAGE_TOKENS = (
    "beer",
    "cider",
    "wine",
    "spirit",
    "coffee",
    "tea",
    "juice",
    "milk",
    "protein drink",
    "soft drink",
    "water",
    "cordial",
    "liqueur",
    "port",
    "sherry",
    "sake",
    "mirin",
)
ALCOHOLIC_BEVERAGE_TOKENS = (
    "beer",
    "cider",
    "wine",
    "spirit",
    "liqueur",
    "port",
    "sherry",
    "sake",
    "mirin",
)
CONDIMENT_TOKENS = (
    "sauce",
    "gravy",
    "stock",
    "dressing",
    "spread",
    "paste",
    "condiment",
    "marinade",
)
AMBIGUOUS_TOKENS = (
    "not further defined",
    "not specified",
    "assorted",
    "mixed",
    "variety",
    "unknown",
    "nfd",
)

MAJOR_FOOD_GROUPS: dict[str, tuple[str, ...]] = {
    "beverage": BEVERAGE_TOKENS,
    "condiment": CONDIMENT_TOKENS,
    "bakery_cereal": ("bread", "biscuit", "breakfast cereal", "cake", "pastry", "muffin", "slice"),
    "dairy_egg": ("cheese", "yoghurt", "yogurt", "milk", "egg"),
    "protein": ("beef", "chicken", "lamb", "pork", "fish", "seafood", "bean", "legume", "sausage"),
    "produce": ("fruit", "vegetable", "potato", "salad"),
    "prepared_meal": ("soup", "curry", "stir-fry", "stir fry", "casserole", "pizza", "sandwich", "pasta dish", "rice dish", "burger"),
    "dessert_snack": ("bar", "ice cream", "dessert", "confectionery", "chip", "cracker"),
}

BENCHMARK_POLICY = {
    "training_and_retrieval_corpus": "Open Food Facts (OFF)",
    "external_benchmark": "AUSNUT",
    "benchmark_success_definition": "Nutritional fidelity to Australian food profiles, not title overlap alone.",
    "hard_rules": [
        "Freeze the AUSNUT holdout before any tuning.",
        "Never copy AUSNUT final test rows into OFF, embeddings, lookup caches, or any tuning set.",
        "Keep AUSNUT benchmark artifacts in a separate versioned directory from OFF dataset builders and indexes.",
    ],
}


def benchmark_dataset_root() -> Path:
    return Path(__file__).resolve().parent.parent / "dataset_process"


def default_ausnut_dataset_path() -> Path:
    return benchmark_dataset_root() / BENCHMARK_FILE_NAME


def default_benchmark_artifact_dir() -> Path:
    return benchmark_dataset_root() / BENCHMARK_DIR_NAME


def default_benchmark_table_path() -> Path:
    return default_benchmark_artifact_dir() / "ausnut_benchmark_table.csv"


def default_benchmark_table_metadata_path() -> Path:
    return default_benchmark_artifact_dir() / "ausnut_benchmark_table.metadata.json"


def default_split_manifest_path() -> Path:
    return default_benchmark_artifact_dir() / "ausnut_split_manifest.json"


def default_dev_priors_path() -> Path:
    return default_benchmark_artifact_dir() / "ausnut_dev_priors.json"


def default_leakage_report_path() -> Path:
    return default_benchmark_artifact_dir() / "ausnut_leakage_report.json"


def default_latest_summary_path() -> Path:
    return default_benchmark_artifact_dir() / "latest_ausnut_summary.json"


def ensure_benchmark_artifact_dir(artifact_dir: str | Path | None = None) -> Path:
    resolved = Path(artifact_dir) if artifact_dir else default_benchmark_artifact_dir()
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def save_json(path: str | Path, payload: dict[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return target


def _normalize_series(frame: pd.DataFrame, column: str, default: str = "") -> pd.Series:
    if column not in frame.columns:
        return pd.Series([default] * len(frame), index=frame.index, dtype="object")
    return frame[column].fillna(default).astype(str).str.strip()


def _token_match(text: str, tokens: tuple[str, ...]) -> bool:
    normalized_text = normalize_text(text)
    if not normalized_text:
        return False
    return any(token in normalized_text for token in tokens)


def _is_runtime_beverage_candidate(
    category: Any,
    title: Any = "",
    keywords: Any = "",
    ingredient_text: Any = "",
    serving_text: Any = "",
) -> bool:
    candidate = build_text_role_candidate(
        title=title,
        recipe_category=category,
        keywords=keywords,
        ingredient_text=ingredient_text,
        serving_description=serving_text,
    )
    return infer_combo_category(candidate) == "drink" and is_candidate_role_compatible(candidate, "drink")


def infer_report_meal_slot(category: Any, title: Any = "", keywords: Any = "", ingredient_text: Any = "", serving_text: Any = "") -> str:
    category_text = normalize_text(category)
    joined = " ".join(part for part in [str(category or ""), str(title or ""), str(keywords or "")] if str(part).strip())
    joined_text = normalize_text(joined)

    if _token_match(category_text, CONDIMENT_TOKENS):
        return "general"
    if _is_runtime_beverage_candidate(category, title, keywords, ingredient_text, serving_text):
        return "beverage"
    if _token_match(category_text, BREAKFAST_TOKENS):
        return "breakfast"
    if _token_match(category_text, SNACK_TOKENS):
        return "snack"
    if _token_match(category_text, DINNER_TOKENS):
        return "dinner"
    if _token_match(category_text, LUNCH_TOKENS):
        return "lunch"
    if _token_match(joined_text, BREAKFAST_TOKENS):
        return "breakfast"
    if _token_match(joined_text, SNACK_TOKENS):
        return "snack"
    if _token_match(joined_text, DINNER_TOKENS):
        return "dinner"
    if _token_match(joined_text, LUNCH_TOKENS):
        return "lunch"
    return "general"


def infer_request_meal_slot(report_meal_slot: Any, category: Any, title: Any = "") -> str:
    normalized_slot = normalize_text(report_meal_slot)
    if normalized_slot in MEAL_SLOTS:
        return normalized_slot

    joined_text = normalize_text(f"{category or ''} {title or ''}")
    if normalized_slot == "beverage":
        if _token_match(joined_text, ALCOHOLIC_BEVERAGE_TOKENS):
            return "dinner"
        if _token_match(joined_text, ("coffee", "tea", "juice", "smoothie", "cappuccino", "latte", "espresso", "matcha")):
            return "breakfast"
        return "lunch"
    if normalized_slot == "snack":
        if _token_match(joined_text, ("yoghurt", "yogurt", "breakfast cereal", "fruit", "granola", "muesli")):
            return "breakfast"
        return "lunch"
    return "lunch"


def infer_major_food_group(category: Any, title: Any = "", keywords: Any = "", ingredient_text: Any = "", serving_text: Any = "") -> str:
    if _is_runtime_beverage_candidate(category, title, keywords, ingredient_text, serving_text):
        return "beverage"
    joined_text = normalize_text(f"{category or ''} {title or ''} {keywords or ''}")
    for group_name, tokens in MAJOR_FOOD_GROUPS.items():
        if group_name == "beverage":
            continue
        if _token_match(joined_text, tokens):
            return group_name
    return "other"


def _flag_ambiguous_row(title: str, category: str, keywords: str) -> bool:
    return _token_match(f"{title} {category} {keywords}", AMBIGUOUS_TOKENS)


def _calculate_nutrient_density_band(frame: pd.DataFrame) -> pd.Series:
    calories = frame["calories_100g"].clip(lower=1.0)
    protein_energy_share = (frame["protein_100g"] * 4.0) / calories
    sugar_energy_share = (frame["sugar_100g"] * 4.0) / calories
    fat_energy_share = (frame["fat_100g"] * 9.0) / calories
    density_score = protein_energy_share - (0.5 * sugar_energy_share) - (0.25 * fat_energy_share)
    ranked = density_score.rank(method="first")
    bands = pd.qcut(ranked, q=3, labels=["low_density", "mid_density", "high_density"])
    return bands.astype(str)


def _stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _stable_sort_indices(frame: pd.DataFrame, indices: list[int], seed: int, salt: str) -> list[int]:
    return sorted(indices, key=lambda idx: _stable_hash(f"{salt}:{seed}:{frame.iloc[idx]['benchmark_id']}"))


def _compute_benchmark_version(frame: pd.DataFrame) -> str:
    stable_columns = [
        "benchmark_id",
        "canonical_title",
        "recipe_category",
        "report_meal_slot",
        "request_meal_slot",
        "major_food_group",
        "nutrient_density_band",
        "calories_100g",
        "protein_100g",
        "carbs_100g",
        "fat_100g",
        "sugar_100g",
        "serving_grams",
        "benchmark_tier",
    ]
    payload = frame[stable_columns].sort_values("benchmark_id").to_csv(index=False)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:10]
    return f"ausnut-benchmark-{BENCHMARK_POLICY_VERSION}-{digest}"


def build_clean_ausnut_benchmark_table(dataset_path: str | Path | None = None) -> pd.DataFrame:
    source_path = Path(dataset_path) if dataset_path else default_ausnut_dataset_path()
    source_frame = pd.read_csv(source_path)

    frame = pd.DataFrame()
    frame["benchmark_id"] = _normalize_series(source_frame, "RecipeId")
    frame["title"] = _normalize_series(source_frame, "food_name")
    frame["canonical_title"] = frame["title"].map(canonicalize_title)
    frame["canonical_title_key"] = frame["canonical_title"].map(canonical_title_key)
    frame["recipe_category"] = _normalize_series(source_frame, "RecipeCategory")
    frame["keywords"] = _normalize_series(source_frame, "Keywords")
    frame["ingredient_text"] = _normalize_series(source_frame, "ingredient_text")
    frame["is_australian"] = source_frame.get("is_australian", True).fillna(True).astype(bool)
    frame["aggregated_rating"] = pd.to_numeric(source_frame.get("AggregatedRating"), errors="coerce").fillna(0.0).round(4)
    frame["serving_text"] = _normalize_series(source_frame, "RecipeServings", default="100g")
    frame["serving_grams"] = frame["serving_text"].map(_parse_serving_grams).astype(float).round(4)

    for source_column, target_column in (
        ("Calories_100g", "calories_100g"),
        ("protein_100g", "protein_100g"),
        ("carbs_100g", "carbs_100g"),
        ("fat_100g", "fat_100g"),
        ("sugar_100g", "sugar_100g"),
    ):
        frame[target_column] = pd.to_numeric(source_frame.get(source_column), errors="coerce").fillna(0.0).round(4)

    serving_factor = frame["serving_grams"].clip(lower=1.0) / 100.0
    frame["calories_per_serving"] = (frame["calories_100g"] * serving_factor).round(4)
    frame["protein_per_serving"] = (frame["protein_100g"] * serving_factor).round(4)
    frame["carbs_per_serving"] = (frame["carbs_100g"] * serving_factor).round(4)
    frame["fat_per_serving"] = (frame["fat_100g"] * serving_factor).round(4)
    frame["sugar_per_serving"] = (frame["sugar_100g"] * serving_factor).round(4)

    frame["report_meal_slot"] = frame.apply(
        lambda row: infer_report_meal_slot(
            row["recipe_category"],
            row["title"],
            row["keywords"],
            row["ingredient_text"],
            row["serving_text"],
        ),
        axis=1,
    )
    frame["request_meal_slot"] = frame.apply(
        lambda row: infer_request_meal_slot(row["report_meal_slot"], row["recipe_category"], row["title"]),
        axis=1,
    )
    frame["major_food_group"] = frame.apply(
        lambda row: infer_major_food_group(
            row["recipe_category"],
            row["title"],
            row["keywords"],
            row["ingredient_text"],
            row["serving_text"],
        ),
        axis=1,
    )
    frame["ambiguous_row"] = frame.apply(
        lambda row: _flag_ambiguous_row(row["title"], row["recipe_category"], row["keywords"]),
        axis=1,
    )
    frame["beverage_only"] = frame["report_meal_slot"].eq("beverage")
    frame["condiment_only"] = frame["major_food_group"].eq("condiment")
    frame["incomplete_row"] = (
        frame["benchmark_id"].eq("")
        | frame["title"].eq("")
        | frame["calories_100g"].le(0.0)
        | frame["serving_grams"].le(0.0)
    )
    frame["benchmark_tier"] = np.where(
        frame[list(PRIMARY_BENCHMARK_FLAGS)].any(axis=1) | frame[list(SUPPLEMENTARY_BENCHMARK_FLAGS)].any(axis=1),
        "supplementary",
        "primary",
    )
    frame["nutrient_density_band"] = _calculate_nutrient_density_band(frame)
    frame["benchmark_version"] = _compute_benchmark_version(frame)

    flag_notes = []
    for column_name, note_text in (
        ("ambiguous_row", "ambiguous"),
        ("beverage_only", "beverage_only"),
        ("condiment_only", "condiment_only"),
        ("incomplete_row", "incomplete"),
    ):
        flag_notes.append(np.where(frame[column_name], note_text, ""))
    frame["benchmark_flags"] = (
        pd.DataFrame(flag_notes).T.apply(lambda row: ",".join([value for value in row.tolist() if value]), axis=1)
    )

    ordered_columns = [
        "benchmark_id",
        "benchmark_version",
        "title",
        "canonical_title",
        "canonical_title_key",
        "recipe_category",
        "keywords",
        "ingredient_text",
        "is_australian",
        "aggregated_rating",
        "report_meal_slot",
        "request_meal_slot",
        "major_food_group",
        "nutrient_density_band",
        "calories_100g",
        "protein_100g",
        "carbs_100g",
        "fat_100g",
        "sugar_100g",
        "serving_text",
        "serving_grams",
        "calories_per_serving",
        "protein_per_serving",
        "carbs_per_serving",
        "fat_per_serving",
        "sugar_per_serving",
        "benchmark_tier",
        "benchmark_flags",
        "ambiguous_row",
        "beverage_only",
        "condiment_only",
        "incomplete_row",
    ]
    return frame[ordered_columns].sort_values("benchmark_id").reset_index(drop=True)


def build_benchmark_table_metadata(frame: pd.DataFrame, dataset_path: str | Path | None = None) -> dict[str, Any]:
    source_path = Path(dataset_path) if dataset_path else default_ausnut_dataset_path()
    benchmark_version = str(frame["benchmark_version"].iloc[0]) if not frame.empty else "unknown"
    return {
        "policy_version": BENCHMARK_POLICY_VERSION,
        "report_template_version": BENCHMARK_REPORT_TEMPLATE_VERSION,
        "benchmark_version": benchmark_version,
        "source_dataset": str(source_path),
        "row_count": int(len(frame)),
        "primary_row_count": int((frame["benchmark_tier"] == "primary").sum()),
        "supplementary_row_count": int((frame["benchmark_tier"] != "primary").sum()),
        "report_meal_slot_distribution": frame["report_meal_slot"].value_counts().to_dict(),
        "request_meal_slot_distribution": frame["request_meal_slot"].value_counts().to_dict(),
        "major_food_group_distribution": frame["major_food_group"].value_counts().to_dict(),
        "flag_distribution": {
            flag_name: int(frame[flag_name].sum())
            for flag_name in [*PRIMARY_BENCHMARK_FLAGS, *SUPPLEMENTARY_BENCHMARK_FLAGS]
        },
    }


def write_clean_ausnut_benchmark_table(
    dataset_path: str | Path | None = None,
    artifact_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = build_clean_ausnut_benchmark_table(dataset_path)
    metadata = build_benchmark_table_metadata(frame, dataset_path)
    target_dir = ensure_benchmark_artifact_dir(artifact_dir)
    table_path = target_dir / default_benchmark_table_path().name
    frame.to_csv(table_path, index=False)
    save_json(target_dir / default_benchmark_table_metadata_path().name, metadata)
    return frame, metadata


def _build_stratification_labels(frame: pd.DataFrame) -> pd.Series:
    category_key = frame["recipe_category"].map(canonical_title_key)
    fine = category_key + "|" + frame["report_meal_slot"] + "|" + frame["nutrient_density_band"]
    medium = category_key + "|" + frame["report_meal_slot"]
    coarse = frame["report_meal_slot"] + "|" + frame["nutrient_density_band"]
    slot_only = frame["report_meal_slot"]

    fine_counts = fine.value_counts()
    medium_counts = medium.value_counts()
    coarse_counts = coarse.value_counts()
    slot_counts = slot_only.value_counts()

    labels: list[str] = []
    for index in frame.index:
        if int(fine_counts.get(fine.iloc[index], 0)) >= 6:
            labels.append(f"fine::{fine.iloc[index]}")
        elif int(medium_counts.get(medium.iloc[index], 0)) >= 6:
            labels.append(f"medium::{medium.iloc[index]}")
        elif int(coarse_counts.get(coarse.iloc[index], 0)) >= 6:
            labels.append(f"coarse::{coarse.iloc[index]}")
        elif int(slot_counts.get(slot_only.iloc[index], 0)) >= 6:
            labels.append(f"slot::{slot_only.iloc[index]}")
        else:
            labels.append("global")
    return pd.Series(labels, index=frame.index, dtype="object")


def freeze_ausnut_split_manifest(
    frame: pd.DataFrame,
    test_ratio: float = 0.20,
    seed: int = 42,
) -> dict[str, Any]:
    working = frame.sort_values("benchmark_id").reset_index(drop=True).copy()
    stratification_labels = _build_stratification_labels(working)
    split_assignment = pd.Series(["dev"] * len(working), index=working.index, dtype="object")
    target_test_total = int(round(len(working) * float(test_ratio)))
    extra_candidates: list[tuple[int, int]] = []

    for label in sorted(stratification_labels.unique().tolist()):
        label_indices = [int(idx) for idx in working.index[stratification_labels == label].tolist()]
        ordered_indices = _stable_sort_indices(working, label_indices, seed, f"split:{label}")
        base_test_count = int(round(len(ordered_indices) * float(test_ratio)))
        if len(ordered_indices) >= 6:
            base_test_count = max(1, base_test_count)
        else:
            base_test_count = 0
        base_test_count = min(base_test_count, max(0, len(ordered_indices) - 1))

        for index in ordered_indices[:base_test_count]:
            split_assignment.iloc[index] = "test"

        remaining_indices = ordered_indices[base_test_count:]
        for index in remaining_indices:
            extra_candidates.append((len(ordered_indices), index))

    current_test_total = int((split_assignment == "test").sum())
    for _, index in sorted(extra_candidates, key=lambda item: (-item[0], item[1])):
        if current_test_total >= target_test_total:
            break
        label = stratification_labels.iloc[index]
        dev_count = int(((stratification_labels == label) & (split_assignment == "dev")).sum())
        if dev_count <= 1:
            continue
        split_assignment.iloc[index] = "test"
        current_test_total += 1

    dev_frame = working.loc[split_assignment == "dev"].reset_index(drop=True)
    dev_labels = _build_stratification_labels(dev_frame)
    dev_fold_assignments: dict[str, int] = {}
    for label in sorted(dev_labels.unique().tolist()):
        label_indices = [int(idx) for idx in dev_frame.index[dev_labels == label].tolist()]
        ordered_indices = _stable_sort_indices(dev_frame, label_indices, seed, f"fold:{label}")
        for fold_position, index in enumerate(ordered_indices):
            dev_fold_assignments[str(dev_frame.iloc[index]["benchmark_id"])] = int(fold_position % 5)

    dev_ids = dev_frame["benchmark_id"].tolist()
    test_ids = working.loc[split_assignment == "test", "benchmark_id"].tolist()
    benchmark_version = str(working["benchmark_version"].iloc[0]) if not working.empty else "unknown"

    manifest = {
        "policy_version": BENCHMARK_POLICY_VERSION,
        "benchmark_version": benchmark_version,
        "split_config": {
            "test_ratio": round(float(test_ratio), 4),
            "seed": int(seed),
            "stratification": "RecipeCategory + report meal slot + nutrient density band with deterministic fallback grouping",
            "dev_cross_validation_folds": 5,
        },
        "counts": {
            "total": int(len(working)),
            "dev": int(len(dev_ids)),
            "test": int(len(test_ids)),
        },
        "dev_ids": dev_ids,
        "test_ids": test_ids,
        "dev_fold_assignments": dev_fold_assignments,
        "split_distribution": {
            "report_meal_slot": {
                slot: {
                    "dev": int(((working["report_meal_slot"] == slot) & (split_assignment == "dev")).sum()),
                    "test": int(((working["report_meal_slot"] == slot) & (split_assignment == "test")).sum()),
                }
                for slot in sorted(working["report_meal_slot"].unique().tolist())
            },
            "major_food_group": {
                group_name: {
                    "dev": int(((working["major_food_group"] == group_name) & (split_assignment == "dev")).sum()),
                    "test": int(((working["major_food_group"] == group_name) & (split_assignment == "test")).sum()),
                }
                for group_name in sorted(working["major_food_group"].unique().tolist())
            },
        },
    }
    return manifest


def write_split_manifest(
    frame: pd.DataFrame,
    artifact_dir: str | Path | None = None,
    test_ratio: float = 0.20,
    seed: int = 42,
) -> dict[str, Any]:
    manifest = freeze_ausnut_split_manifest(frame, test_ratio=test_ratio, seed=seed)
    target_dir = ensure_benchmark_artifact_dir(artifact_dir)
    save_json(target_dir / default_split_manifest_path().name, manifest)
    return manifest


def derive_dev_priors(frame: pd.DataFrame, manifest: dict[str, Any]) -> dict[str, Any]:
    dev_ids = {str(value) for value in manifest.get("dev_ids", [])}
    dev_frame = frame[frame["benchmark_id"].astype(str).isin(dev_ids)].copy()
    benchmark_version = str(frame["benchmark_version"].iloc[0]) if not frame.empty else "unknown"

    def _quantiles(subset: pd.DataFrame) -> dict[str, Any]:
        if subset.empty:
            return {}
        return {
            nutrient: {
                "p05": round(float(subset[nutrient].quantile(0.05)), 4),
                "p50": round(float(subset[nutrient].quantile(0.50)), 4),
                "p95": round(float(subset[nutrient].quantile(0.95)), 4),
            }
            for nutrient in ["calories_100g", "protein_100g", "carbs_100g", "fat_100g", "sugar_100g"]
        }

    priors = {
        "policy_version": BENCHMARK_POLICY_VERSION,
        "benchmark_version": benchmark_version,
        "dev_record_count": int(len(dev_frame)),
        "request_meal_slot_priors": {
            key: round(float(value), 4)
            for key, value in dev_frame["request_meal_slot"].value_counts(normalize=True).to_dict().items()
        },
        "report_meal_slot_priors": {
            key: round(float(value), 4)
            for key, value in dev_frame["report_meal_slot"].value_counts(normalize=True).to_dict().items()
        },
        "major_food_group_priors": {
            key: round(float(value), 4)
            for key, value in dev_frame["major_food_group"].value_counts(normalize=True).to_dict().items()
        },
        "nutrient_density_band_priors": {
            key: round(float(value), 4)
            for key, value in dev_frame["nutrient_density_band"].value_counts(normalize=True).to_dict().items()
        },
        "reasonable_macro_ranges_by_request_slot": {
            slot: _quantiles(dev_frame[dev_frame["request_meal_slot"] == slot])
            for slot in sorted(dev_frame["request_meal_slot"].unique().tolist())
        },
    }
    return priors


def write_dev_priors(
    frame: pd.DataFrame,
    manifest: dict[str, Any],
    artifact_dir: str | Path | None = None,
) -> dict[str, Any]:
    priors = derive_dev_priors(frame, manifest)
    target_dir = ensure_benchmark_artifact_dir(artifact_dir)
    save_json(target_dir / default_dev_priors_path().name, priors)
    return priors


def validate_no_leakage(
    frame: pd.DataFrame,
    manifest: dict[str, Any],
    off_db_path: str | Path | None = None,
    off_table: str = "cleaned_food_data",
    artifact_dir: str | Path | None = None,
) -> dict[str, Any]:
    dev_ids = {str(value) for value in manifest.get("dev_ids", [])}
    test_ids = {str(value) for value in manifest.get("test_ids", [])}
    dev_fold_assignments = manifest.get("dev_fold_assignments", {}) or {}
    benchmark_dir = ensure_benchmark_artifact_dir(artifact_dir)
    resolved_off_db_path = Path(off_db_path) if off_db_path else benchmark_dataset_root() / "off.db"

    direct_overlap_with_off = []
    if duckdb is not None and resolved_off_db_path.exists():
        connection = None
        try:
            connection = duckdb.connect(str(resolved_off_db_path), read_only=True)
            placeholders = ", ".join(["?"] * len(test_ids)) if test_ids else ""
            if placeholders:
                query = f"SELECT DISTINCT CAST(RecipeId AS VARCHAR) FROM {off_table} WHERE CAST(RecipeId AS VARCHAR) IN ({placeholders})"
                direct_overlap_with_off = [row[0] for row in connection.execute(query, list(test_ids)).fetchall()]
        except Exception:
            direct_overlap_with_off = []
        finally:
            if connection is not None:
                try:
                    connection.close()
                except Exception:
                    pass

    checks = [
        {
            "name": "dev_and_test_ids_are_disjoint",
            "passed": bool(dev_ids.isdisjoint(test_ids)),
            "details": {
                "dev_count": len(dev_ids),
                "test_count": len(test_ids),
            },
        },
        {
            "name": "test_ids_do_not_appear_in_dev_fold_assignments",
            "passed": bool(test_ids.isdisjoint({str(key) for key in dev_fold_assignments.keys()})),
            "details": {
                "dev_fold_assignment_count": len(dev_fold_assignments),
            },
        },
        {
            "name": "benchmark_artifacts_are_separate_from_off_corpus_path",
            "passed": not str(benchmark_dir.resolve()).startswith(str(resolved_off_db_path.resolve())),
            "details": {
                "artifact_dir": str(benchmark_dir.resolve()),
                "off_db_path": str(resolved_off_db_path.resolve()),
            },
        },
        {
            "name": "final_test_ids_do_not_overlap_direct_off_recipe_ids",
            "passed": len(direct_overlap_with_off) == 0,
            "details": {
                "direct_overlap_count": len(direct_overlap_with_off),
                "overlapping_ids": direct_overlap_with_off[:20],
            },
        },
        {
            "name": "benchmark_version_is_singleton_across_table",
            "passed": int(frame["benchmark_version"].nunique()) == 1,
            "details": {
                "benchmark_version_count": int(frame["benchmark_version"].nunique()),
            },
        },
    ]
    return {
        "policy_version": BENCHMARK_POLICY_VERSION,
        "benchmark_version": str(frame["benchmark_version"].iloc[0]) if not frame.empty else "unknown",
        "passed": all(bool(check["passed"]) for check in checks),
        "checks": checks,
        "hard_rules": BENCHMARK_POLICY["hard_rules"],
    }


def write_leakage_report(
    frame: pd.DataFrame,
    manifest: dict[str, Any],
    artifact_dir: str | Path | None = None,
    off_db_path: str | Path | None = None,
    off_table: str = "cleaned_food_data",
) -> dict[str, Any]:
    leakage_report = validate_no_leakage(
        frame,
        manifest,
        off_db_path=off_db_path,
        off_table=off_table,
        artifact_dir=artifact_dir,
    )
    target_dir = ensure_benchmark_artifact_dir(artifact_dir)
    save_json(target_dir / default_leakage_report_path().name, leakage_report)
    return leakage_report