from __future__ import annotations


import argparse
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .ausnut_benchmark import (
    BENCHMARK_POLICY,
    BENCHMARK_POLICY_VERSION,
    BENCHMARK_REPORT_TEMPLATE_VERSION,
    BENCHMARK_TOP_K_VALUES,
    default_benchmark_artifact_dir,
    default_benchmark_table_path,
    default_latest_summary_path,
    default_split_manifest_path,
    ensure_benchmark_artifact_dir,
    infer_major_food_group,
    infer_report_meal_slot,
    save_json,
    write_clean_ausnut_benchmark_table,
    write_dev_priors,
    write_leakage_report,
    write_split_manifest,
)
from .local_dataset import LocalFoodDataset
from .metrics import (
    build_ausnut_benchmark_report,
    compute_category_agreement_metrics,
    compute_coverage_metrics,
    compute_macro_vector_distance,
    compute_nutrient_profile_error,
    compute_topk_retrieval_metrics,
)
from .ranking import merge_candidates
from .utils import canonical_title_key, canonicalize_title, dedupe_strings, normalize_text, tokenize_canonical_title, to_float


BASELINE_NAMES = ("current_off_only", "nutrient_nn", "category_constrained")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _env_flag_enabled(name: str) -> bool:
    return normalize_text(os.getenv(name, "")) in {"1", "true", "yes", "on"}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the AUSNUT benchmark against the OFF recommendation corpus.")
    parser.add_argument("--dataset-path", default=str(Path(default_benchmark_artifact_dir()).parent / "AUSNUT_dataset.csv"))
    parser.add_argument("--artifact-dir", default=str(default_benchmark_artifact_dir()))
    parser.add_argument("--off-db-path", default=str(Path(default_benchmark_artifact_dir()).parent / "off.db"))
    parser.add_argument("--off-db-table", default="cleaned_food_data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-ratio", type=float, default=0.20)
    parser.add_argument("--candidate-pool-options", default="60,80,100")
    parser.add_argument("--tuning-sample-size", type=int, default=600)
    return parser.parse_args()


def _parse_candidate_pool_options(raw_value: str) -> list[int]:
    values = []
    for token in str(raw_value or "").split(","):
        token = token.strip()
        if not token:
            continue
        try:
            values.append(max(20, int(token)))
        except ValueError:
            continue
    deduped = sorted({value for value in values if value > 0})
    return deduped or [80]


def _benchmark_per100_profile(record: dict[str, Any]) -> dict[str, float]:
    return {
        "calories": to_float(record.get("calories_100g"), 0.0),
        "protein": to_float(record.get("protein_100g"), 0.0),
        "carbs": to_float(record.get("carbs_100g"), 0.0),
        "fats": to_float(record.get("fat_100g"), 0.0),
        "sugar": to_float(record.get("sugar_100g"), 0.0),
    }


def _benchmark_serving_profile(record: dict[str, Any]) -> dict[str, float]:
    return {
        "calories": to_float(record.get("calories_per_serving"), 0.0),
        "protein": to_float(record.get("protein_per_serving"), 0.0),
        "carbs": to_float(record.get("carbs_per_serving"), 0.0),
        "fats": to_float(record.get("fat_per_serving"), 0.0),
        "sugar": to_float(record.get("sugar_per_serving"), 0.0),
    }


def _candidate_per100_profile(candidate: dict[str, Any]) -> dict[str, float]:
    per100 = candidate.get("per100") or {}
    return {
        "calories": to_float(per100.get("calories"), 0.0),
        "protein": to_float(per100.get("protein"), 0.0),
        "carbs": to_float(per100.get("carbs"), 0.0),
        "fats": to_float(per100.get("fats"), 0.0),
        "sugar": to_float(per100.get("sugar"), 0.0),
    }


def _candidate_serving_profile(candidate: dict[str, Any]) -> dict[str, float]:
    return {
        "calories": to_float(candidate.get("serving_calories"), 0.0),
        "protein": to_float(candidate.get("serving_protein"), 0.0),
        "carbs": to_float(candidate.get("serving_carbs"), 0.0),
        "fats": to_float(candidate.get("serving_fats"), 0.0),
        "sugar": to_float(candidate.get("serving_sugar"), 0.0),
    }


def _title_token_overlap(candidate_title: str, benchmark_title: str) -> float:
    candidate_tokens = tokenize_canonical_title(candidate_title)
    benchmark_tokens = tokenize_canonical_title(benchmark_title)
    if not candidate_tokens or not benchmark_tokens:
        return 0.0
    union = candidate_tokens.union(benchmark_tokens)
    if not union:
        return 0.0
    return round(float(len(candidate_tokens.intersection(benchmark_tokens)) / len(union)), 4)


def default_acceptance_thresholds() -> dict[str, Any]:
    return {
        "max_mean_relative_error": 0.35,
        "max_macro_vector_distance": 0.60,
        "min_title_overlap": 0.18,
        "minimum_relevance_grade": 2.0,
        "require_category_or_major_group_match": True,
    }


def default_alias_promotion_thresholds() -> dict[str, Any]:
    return {
        "minimum_support_count": 2,
        "minimum_relevance_grade": 3.0,
        "max_mean_relative_error": 0.12,
        "min_mean_title_token_overlap": 0.25,
        "min_source_target_token_overlap": 0.35,
        "min_recipe_category_match_rate": 1.0,
        "min_meal_slot_match_rate": 1.0,
    }


def evaluate_candidate_match(
    candidate: dict[str, Any],
    record: dict[str, Any],
    acceptance_thresholds: dict[str, Any],
) -> dict[str, Any]:
    benchmark_profile = _benchmark_per100_profile(record)
    benchmark_serving_profile = _benchmark_serving_profile(record)
    candidate_profile = _candidate_per100_profile(candidate)
    candidate_serving_profile = _candidate_serving_profile(candidate)

    nutrient_profile_error = compute_nutrient_profile_error(candidate_profile, benchmark_profile, comparison_basis="per100")
    serving_profile_error = compute_nutrient_profile_error(
        candidate_serving_profile,
        benchmark_serving_profile,
        comparison_basis="per_serving",
    )
    macro_vector_distance = compute_macro_vector_distance(candidate_profile, benchmark_profile)
    predicted_meal_slot = infer_report_meal_slot(
        candidate.get("recipe_category") or candidate.get("source_keyword"),
        candidate.get("title"),
        candidate.get("keywords"),
    )
    predicted_major_food_group = infer_major_food_group(
        candidate.get("recipe_category") or candidate.get("source_keyword"),
        candidate.get("title"),
        candidate.get("keywords"),
    )
    category_agreement = compute_category_agreement_metrics(
        predicted_category=str(candidate.get("recipe_category") or candidate.get("source_keyword") or ""),
        benchmark_category=str(record.get("recipe_category") or ""),
        predicted_meal_slot=predicted_meal_slot,
        benchmark_meal_slot=str(record.get("report_meal_slot") or ""),
        predicted_major_food_group=predicted_major_food_group,
        benchmark_major_food_group=str(record.get("major_food_group") or ""),
    )
    title_token_overlap = _title_token_overlap(
        str(candidate.get("title") or candidate.get("canonical_title") or ""),
        str(record.get("canonical_title") or record.get("title") or ""),
    )
    category_or_group_match = bool(
        to_float(category_agreement.get("recipe_category_match"), 0.0) >= 1.0
        or to_float(category_agreement.get("major_food_group_match"), 0.0) >= 1.0
    )

    max_mean_relative_error = float(acceptance_thresholds.get("max_mean_relative_error", 0.35))
    max_macro_vector_distance = float(acceptance_thresholds.get("max_macro_vector_distance", 0.60))
    min_title_overlap = float(acceptance_thresholds.get("min_title_overlap", 0.18))
    require_category_gate = bool(acceptance_thresholds.get("require_category_or_major_group_match", True))

    acceptable_match = (
        to_float(nutrient_profile_error.get("mean_relative_error"), 0.0) <= max_mean_relative_error
        and to_float(macro_vector_distance.get("normalized_l2_distance"), 0.0) <= max_macro_vector_distance
        and (not require_category_gate or category_or_group_match)
        and (title_token_overlap >= min_title_overlap or category_or_group_match)
    )

    relevance_grade = 0.0
    if to_float(nutrient_profile_error.get("mean_relative_error"), 0.0) <= max_mean_relative_error:
        relevance_grade += 1.0
    elif to_float(nutrient_profile_error.get("mean_relative_error"), 0.0) <= max(max_mean_relative_error * 1.35, 0.45):
        relevance_grade += 0.5

    if to_float(macro_vector_distance.get("normalized_l2_distance"), 0.0) <= max_macro_vector_distance:
        relevance_grade += 1.0
    elif to_float(macro_vector_distance.get("normalized_l2_distance"), 0.0) <= max(max_macro_vector_distance * 1.25, 0.75):
        relevance_grade += 0.5

    if category_or_group_match:
        relevance_grade += 0.5
    if to_float(category_agreement.get("meal_slot_match"), 0.0) >= 1.0:
        relevance_grade += 0.5

    if title_token_overlap >= max(min_title_overlap, 0.45):
        relevance_grade += 1.0
    elif title_token_overlap >= min_title_overlap:
        relevance_grade += 0.5

    if bool(candidate.get("is_australian")):
        relevance_grade += 0.25

    relevance_grade = min(4.0, round(relevance_grade * 2.0) / 2.0)
    if acceptable_match:
        relevance_grade = max(relevance_grade, float(acceptance_thresholds.get("minimum_relevance_grade", 2.0)))

    return {
        "nutrient_profile_error": nutrient_profile_error,
        "serving_profile_error": serving_profile_error,
        "macro_vector_distance": macro_vector_distance,
        "category_agreement": category_agreement,
        "title_token_overlap": round(float(title_token_overlap), 4),
        "acceptable_match": bool(acceptable_match),
        "relevance_grade": round(float(relevance_grade), 4),
        "predicted_meal_slot": predicted_meal_slot,
        "predicted_major_food_group": predicted_major_food_group,
    }


def _current_off_only_score(candidate: dict[str, Any], match: dict[str, Any]) -> float:
    nutrient_similarity = 1.0 - min(1.0, to_float((match.get("nutrient_profile_error") or {}).get("mean_relative_error"), 0.0) / 2.0)
    macro_similarity = 1.0 - min(1.0, to_float((match.get("macro_vector_distance") or {}).get("normalized_l2_distance"), 0.0))
    category_agreement = match.get("category_agreement") or {}
    health_score = min(5.0, max(0.0, to_float(candidate.get("health_score") or candidate.get("aggregated_rating"), 0.0))) / 5.0
    score = (
        0.45 * nutrient_similarity
        + 0.25 * macro_similarity
        + 0.15 * to_float(match.get("title_token_overlap"), 0.0)
        + 0.10 * to_float(category_agreement.get("recipe_category_match"), 0.0)
        + 0.05 * to_float(category_agreement.get("meal_slot_match"), 0.0)
        + (0.05 if bool(candidate.get("is_australian")) else 0.0)
        + (0.03 * health_score)
    )
    return round(float(score), 6)


def _distance_only_score(match: dict[str, Any], category_weight: float = 0.0) -> float:
    category_agreement = match.get("category_agreement") or {}
    penalty = (
        to_float((match.get("nutrient_profile_error") or {}).get("mean_relative_error"), 0.0)
        + to_float((match.get("macro_vector_distance") or {}).get("normalized_l2_distance"), 0.0)
    )
    return round(
        -float(penalty)
        + (category_weight * to_float(category_agreement.get("major_food_group_match"), 0.0))
        + (0.05 * to_float(category_agreement.get("recipe_category_match"), 0.0)),
        6,
    )


def _make_query_vector(record: dict[str, Any]) -> np.ndarray:
    return np.asarray(
        [
            to_float(record.get("calories_per_serving"), 0.0),
            to_float(record.get("protein_per_serving"), 0.0),
            to_float(record.get("carbs_per_serving"), 0.0),
            to_float(record.get("fat_per_serving"), 0.0),
        ],
        dtype=np.float32,
    )


def _build_bounded_query_pattern(term: Any) -> str:
    normalized_term = normalize_text(term)
    if not normalized_term:
        return ""
    return rf"(?:^|[^a-z0-9]){re.escape(normalized_term)}(?:[^a-z0-9]|$)"


def _build_title_query_regex(record: dict[str, Any], token_cap: int = 6) -> str:
    title_tokens = [token for token in tokenize_canonical_title(record.get("canonical_title") or record.get("title") or "") if len(token) >= 3]
    category_tokens = [token for token in tokenize_canonical_title(record.get("recipe_category") or "") if len(token) >= 3]
    keywords_tokens = [token for token in tokenize_canonical_title(record.get("keywords") or "") if len(token) >= 3]
    terms = dedupe_strings([*title_tokens, *category_tokens, *keywords_tokens], limit=token_cap)
    patterns: list[str] = []
    for term in terms:
        pattern = _build_bounded_query_pattern(term)
        if pattern:
            patterns.append(pattern)
    return "|".join(patterns)


def _build_category_query_regex(record: dict[str, Any]) -> str:
    major_group = normalize_text(record.get("major_food_group") or "")
    recipe_category = normalize_text(record.get("recipe_category") or "")
    report_slot = normalize_text(record.get("report_meal_slot") or "")
    terms = dedupe_strings([recipe_category, major_group, report_slot], limit=4)
    return "|".join(re.escape(normalize_text(term)) for term in terms if normalize_text(term))


def _record_needs_phase7_lunch_family_filter(record: dict[str, Any]) -> bool:
    if not _env_flag_enabled("PHASE7_EVALUATOR_LUNCH_FAMILY_FILTER"):
        return False

    request_slot = normalize_text(record.get("request_meal_slot") or "")
    if request_slot != "lunch":
        return False

    benchmark_major_group = normalize_text(record.get("major_food_group") or "")
    if benchmark_major_group in {"condiment", "dessert_snack", "dairy_egg", "beverage"}:
        return False

    benchmark_text = normalize_text(
        " ".join(
            str(record.get(key) or "")
            for key in ("title", "canonical_title", "recipe_category", "keywords")
        )
    )
    if not benchmark_text:
        return False

    weak_family_terms = (
        "bar",
        "beverage",
        "biscuit",
        "cake",
        "cheese",
        "chocolate",
        "cookie",
        "cream",
        "dip",
        "dressing",
        "drink",
        "doughnut",
        "donut",
        "hummus",
        "hommus",
        "margarine",
        "mayo",
        "mayonnaise",
        "milk",
        "mousse",
        "oil",
        "pesto",
        "salsa",
        "sauce",
        "spread",
        "vinaigrette",
        "yoghurt",
        "yogurt",
    )
    return not any(term in benchmark_text for term in weak_family_terms)


def _candidate_blocked_by_phase7_lunch_mixed_product_filter(candidate: dict[str, Any]) -> bool:
    # Phase 7 generic mixed product blocking criteria
    import re
    title_lower = str(candidate.get("title") or "").lower()
    pattern = r'\b(and|mix|harvest|medley|blend|assorted|salad|platter|mixed)\b|&'
    return bool(re.search(pattern, title_lower))


def _candidate_blocked_by_phase7_supplement_filter(candidate: dict[str, Any]) -> bool:
    # Phase 7 supplement filter: block sports-nutrition / supplement powder items
    # that should never rank above whole foods (energy shots, protein powders, etc.)
    # Deliberately narrow — does NOT block protein bars, energy bars, or whey products
    # which may themselves be valid benchmark foods.
    import re
    title_lower = str(candidate.get("title") or "").lower()
    # Exact multi-word supplement patterns only (avoids false positives)
    pattern = (
        r'\b(energy shot|protein powder|spirulina|pre.?workout|creatine|amino acid'
        r'|electrolyte powder|collagen powder|superfood powder|greens powder)\b'
    )
    return bool(re.search(pattern, title_lower))


def _apply_phase7_supplement_filter(record: dict[str, Any], candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # Apply supplement filter only when env flag is set and benchmark is NOT a supplement
    if not candidates or not _env_flag_enabled("PHASE7_EVALUATOR_SUPPLEMENT_FILTER"):
        return candidates

    request_slot = normalize_text(record.get("request_meal_slot") or "")
    if request_slot not in ("lunch", "dinner"):
        return candidates

    # Exempt: if the benchmark itself is a supplement-type food, do not filter
    import re
    benchmark_title_lower = str(record.get("title") or "").lower()
    supplement_pattern = (
        r'\b(energy shot|protein powder|spirulina|pre.?workout|creatine|amino acid'
        r'|electrolyte powder|collagen powder|superfood powder|greens powder)\b'
    )
    if bool(re.search(supplement_pattern, benchmark_title_lower)):
        return candidates

    filtered = [c for c in candidates if not _candidate_blocked_by_phase7_supplement_filter(c)]
    # Safety: never return an empty pool (fall back to original if all blocked)
    return filtered or candidates

def _candidate_blocked_by_phase7_lunch_family_filter(candidate: dict[str, Any]) -> bool:
    candidate_text = normalize_text(
        " ".join(
            str(candidate.get(key) or "")
            for key in ("title", "recipe_category", "source_keyword", "keywords")
        )
    )
    if not candidate_text:
        return False

    blocked_terms = (
        "aioli",
        "bar",
        "biscuit",
        "brownie",
        "cake",
        "cheese",
        "chocolate",
        "chutney",
        "condiment",
        "confection",
        "cookie",
        "cream",
        "dip",
        "doughnut",
        "donut",
        "dressing",
        "fetta",
        "feta",
        "gravy",
        "hummus",
        "hommus",
        "margarine",
        "mayo",
        "mayonnaise",
        "mousse",
        "mustard",
        "oil",
        "pesto",
        "relish",
        "ricotta",
        "salsa",
        "sauce",
        "snack",
        "sour cream",
        "spread",
        "spray",
        "sweet snacks",
        "vinaigrette",
    )
    return any(term in candidate_text for term in blocked_terms)


def _apply_phase7_lunch_family_filter(record: dict[str, Any], candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not candidates or not _record_needs_phase7_lunch_family_filter(record):
        return candidates

    filtered = [candidate for candidate in candidates if not _candidate_blocked_by_phase7_lunch_family_filter(candidate)]
    return filtered or candidates


def _apply_phase7_lunch_mixed_product_filter(record: dict[str, Any], candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not candidates or not _env_flag_enabled("PHASE7_EVALUATOR_LUNCH_MIXED_PRODUCT_FILTER"):
        return candidates

    request_slot = normalize_text(record.get("request_meal_slot") or "")
    if request_slot != "lunch":
        return candidates

    # Only apply filter if benchmark title is also not a mixed product
    benchmark_title = str(record.get("title") or "")
    import re
    mixed_pattern = r'\b(and|mix|harvest|medley|blend|assorted|salad|platter|mixed)\b|&'
    if bool(re.search(mixed_pattern, benchmark_title.lower())):
        return candidates

    filtered = [candidate for candidate in candidates if not _candidate_blocked_by_phase7_lunch_mixed_product_filter(candidate)]
    return filtered or candidates


def _search_candidates(
    local_dataset: LocalFoodDataset,
    record: dict[str, Any],
    candidate_pool_size: int,
    text_query: str | None,
) -> list[dict[str, Any]]:
    if text_query is not None and not str(text_query).strip():
        return []
    return local_dataset.search(
        meal_type=str(record.get("request_meal_slot") or "lunch"),
        query_vector=_make_query_vector(record),
        top_k=int(candidate_pool_size),
        prefetch=max(int(candidate_pool_size) * 2, 120),
        is_australian_user=True,
        text_query=text_query,
    )


def build_ranked_candidates(
    local_dataset: LocalFoodDataset,
    record: dict[str, Any],
    baseline_name: str,
    candidate_pool_size: int,
    acceptance_thresholds: dict[str, Any],
) -> list[dict[str, Any]]:
    title_query = _build_title_query_regex(record)
    category_query = _build_category_query_regex(record)

    if baseline_name == "current_off_only":
        merged_candidates = merge_candidates(
            _search_candidates(local_dataset, record, candidate_pool_size, title_query),
            _search_candidates(local_dataset, record, max(30, candidate_pool_size // 2), category_query),
            _search_candidates(local_dataset, record, max(20, candidate_pool_size // 3), None),
            max_items=int(candidate_pool_size),
        )
        # Phase 7 dev-only: family filter reverted (net-negative MRE per isolation probe 2026-04-21).
        # Supplement filter also reverted (net-negative MRE 2026-04-21 — blocks nutritionally-close items).
        # Only mixed-product filter remains active.
        merged_candidates = _apply_phase7_lunch_mixed_product_filter(record, merged_candidates)
    elif baseline_name == "nutrient_nn":
        merged_candidates = _search_candidates(local_dataset, record, candidate_pool_size, None)
    elif baseline_name == "category_constrained":
        broad_candidates = merge_candidates(
            _search_candidates(local_dataset, record, candidate_pool_size, category_query),
            _search_candidates(local_dataset, record, max(20, candidate_pool_size // 2), title_query),
            max_items=int(candidate_pool_size),
        )
        filtered_candidates = [
            candidate
            for candidate in broad_candidates
            if infer_major_food_group(
                candidate.get("recipe_category") or candidate.get("source_keyword"),
                candidate.get("title"),
                candidate.get("keywords"),
            )
            == str(record.get("major_food_group") or "other")
            or canonical_title_key(candidate.get("recipe_category") or candidate.get("source_keyword"))
            == canonical_title_key(record.get("recipe_category") or "")
        ]
        merged_candidates = filtered_candidates or broad_candidates or _search_candidates(local_dataset, record, candidate_pool_size, None)
    else:
        raise ValueError(f"Unsupported baseline: {baseline_name}")

    ranked_candidates = []
    for candidate in merged_candidates:
        match = evaluate_candidate_match(candidate, record, acceptance_thresholds)
        if baseline_name == "current_off_only":
            baseline_score = _current_off_only_score(candidate, match)
        elif baseline_name == "category_constrained":
            baseline_score = _distance_only_score(match, category_weight=0.15)
        else:
            baseline_score = _distance_only_score(match)
        ranked_candidates.append({
            **candidate,
            "_benchmark_match": match,
            "_baseline_score": baseline_score,
        })

    ranked_candidates.sort(
        key=lambda candidate: (
            to_float(candidate.get("_baseline_score"), 0.0),
            to_float((candidate.get("_benchmark_match") or {}).get("relevance_grade"), 0.0),
            -to_float(((candidate.get("_benchmark_match") or {}).get("nutrient_profile_error") or {}).get("mean_relative_error"), 0.0),
        ),
        reverse=True,
    )
    return ranked_candidates


def _serialize_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    match = candidate.get("_benchmark_match") or {}
    return {
        "candidate_id": str(candidate.get("id") or candidate.get("recipe_id") or ""),
        "title": str(candidate.get("title") or ""),
        "recipe_category": str(candidate.get("recipe_category") or candidate.get("source_keyword") or ""),
        "major_food_group": str(match.get("predicted_major_food_group") or "other"),
        "predicted_meal_slot": str(match.get("predicted_meal_slot") or "general"),
        "is_australian": bool(candidate.get("is_australian")),
        "baseline_score": to_float(candidate.get("_baseline_score"), 0.0),
        "relevance_grade": to_float(match.get("relevance_grade"), 0.0),
        "acceptable_match": bool(match.get("acceptable_match")),
        "nutrient_profile_error": match.get("nutrient_profile_error") or {},
        "macro_vector_distance": match.get("macro_vector_distance") or {},
        "category_agreement": match.get("category_agreement") or {},
        "title_token_overlap": to_float(match.get("title_token_overlap"), 0.0),
    }


def classify_failure_mode(
    record: dict[str, Any],
    ranked_candidates: list[dict[str, Any]],
    acceptance_thresholds: dict[str, Any],
) -> str:
    if not ranked_candidates:
        return "retrieval_empty"

    top_candidates = ranked_candidates[:10]
    top_match = (top_candidates[0].get("_benchmark_match") or {})
    later_acceptable = any(bool((candidate.get("_benchmark_match") or {}).get("acceptable_match")) for candidate in top_candidates[1:5])
    if later_acceptable and not bool(top_match.get("acceptable_match")):
        return "ranking_order"

    nutrient_error = to_float(((top_match.get("nutrient_profile_error") or {}).get("mean_relative_error")), 0.0)
    serving_error = to_float(((top_match.get("serving_profile_error") or {}).get("mean_relative_error")), 0.0)
    macro_distance = to_float(((top_match.get("macro_vector_distance") or {}).get("normalized_l2_distance")), 0.0)
    title_overlap = to_float(top_match.get("title_token_overlap"), 0.0)
    category_agreement = top_match.get("category_agreement") or {}
    category_or_group_match = bool(
        to_float(category_agreement.get("recipe_category_match"), 0.0) >= 1.0
        or to_float(category_agreement.get("major_food_group_match"), 0.0) >= 1.0
    )

    if nutrient_error <= float(acceptance_thresholds.get("max_mean_relative_error", 0.35)) and serving_error > max(0.55, nutrient_error * 1.5):
        return "serving_conversion"
    if title_overlap < float(acceptance_thresholds.get("min_title_overlap", 0.18)) and nutrient_error <= float(acceptance_thresholds.get("max_mean_relative_error", 0.35)):
        return "title_normalization"
    if not category_or_group_match and any(
        bool(
            to_float(((candidate.get("_benchmark_match") or {}).get("category_agreement") or {}).get("major_food_group_match"), 0.0)
            >= 1.0
        )
        for candidate in top_candidates
    ):
        return "category_assignment"
    if not any(bool((candidate.get("_benchmark_match") or {}).get("acceptable_match")) for candidate in top_candidates):
        if macro_distance <= float(acceptance_thresholds.get("max_macro_vector_distance", 0.60)):
            return "title_or_category_gate"
        return "retrieval_gap"
    return "candidate_quality"


def evaluate_record(
    local_dataset: LocalFoodDataset,
    record: dict[str, Any],
    baseline_name: str,
    candidate_pool_size: int,
    acceptance_thresholds: dict[str, Any],
) -> dict[str, Any]:
    ranked_candidates = build_ranked_candidates(
        local_dataset=local_dataset,
        record=record,
        baseline_name=baseline_name,
        candidate_pool_size=candidate_pool_size,
        acceptance_thresholds=acceptance_thresholds,
    )
    relevance_grades = [to_float((candidate.get("_benchmark_match") or {}).get("relevance_grade"), 0.0) for candidate in ranked_candidates]
    acceptable_matches = [bool((candidate.get("_benchmark_match") or {}).get("acceptable_match")) for candidate in ranked_candidates]
    retrieval_metrics = compute_topk_retrieval_metrics(
        relevance_grades,
        top_k_values=BENCHMARK_TOP_K_VALUES,
        relevance_threshold=float(acceptance_thresholds.get("minimum_relevance_grade", 2.0)),
    )
    coverage_metrics = compute_coverage_metrics(
        acceptable_matches,
        top_k_values=BENCHMARK_TOP_K_VALUES,
        candidate_count=len(ranked_candidates),
    )

    top_candidates = ranked_candidates[:10]
    top_candidate = top_candidates[0] if top_candidates else None
    top_candidate_metrics = (top_candidate.get("_benchmark_match") or {}) if top_candidate else {}
    failure_mode = classify_failure_mode(record, ranked_candidates, acceptance_thresholds)

    return {
        "benchmark_id": str(record.get("benchmark_id") or ""),
        "benchmark_tier": str(record.get("benchmark_tier") or "primary"),
        "title": str(record.get("title") or ""),
        "canonical_title": str(record.get("canonical_title") or ""),
        "recipe_category": str(record.get("recipe_category") or ""),
        "major_food_group": str(record.get("major_food_group") or "other"),
        "report_meal_slot": str(record.get("report_meal_slot") or "general"),
        "request_meal_slot": str(record.get("request_meal_slot") or "lunch"),
        "baseline_name": baseline_name,
        "candidate_pool_size": int(candidate_pool_size),
        "candidate_return_count": int(len(ranked_candidates)),
        "top_candidate_metrics": top_candidate_metrics,
        "retrieval_metrics": retrieval_metrics,
        "coverage_metrics": coverage_metrics,
        "top_candidates": [_serialize_candidate(candidate) for candidate in top_candidates],
        "failure_mode": failure_mode,
    }


def evaluate_records(
    local_dataset: LocalFoodDataset,
    frame: pd.DataFrame,
    baseline_name: str,
    candidate_pool_size: int,
    acceptance_thresholds: dict[str, Any],
    progress_label: str,
) -> list[dict[str, Any]]:
    records = frame.to_dict(orient="records")
    outputs: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        outputs.append(
            evaluate_record(
                local_dataset=local_dataset,
                record=record,
                baseline_name=baseline_name,
                candidate_pool_size=candidate_pool_size,
                acceptance_thresholds=acceptance_thresholds,
            )
        )
        if index % 250 == 0 or index == len(records):
            print(f"[AUSNUT] {progress_label}: {index}/{len(records)}")
    return outputs


def sample_dev_records(frame: pd.DataFrame, sample_size: int, seed: int) -> pd.DataFrame:
    if len(frame) <= sample_size:
        return frame.copy()

    grouped = frame.groupby(["report_meal_slot", "benchmark_tier"], dropna=False, sort=True)
    parts = []
    for _, group in grouped:
        share = len(group) / max(1, len(frame))
        target = max(1, int(round(sample_size * share)))
        target = min(target, len(group))
        parts.append(group.sample(n=target, random_state=seed))

    sample = pd.concat(parts, ignore_index=False).drop_duplicates(subset=["benchmark_id"])
    if len(sample) > sample_size:
        sample = sample.sample(n=sample_size, random_state=seed)
    elif len(sample) < sample_size:
        remaining = frame[~frame["benchmark_id"].isin(sample["benchmark_id"])]
        if not remaining.empty:
            fill_count = min(sample_size - len(sample), len(remaining))
            sample = pd.concat([sample, remaining.sample(n=fill_count, random_state=seed)], ignore_index=False)
    return sample.sort_values("benchmark_id").reset_index(drop=True)


def summarize_tuning_score(report: dict[str, Any]) -> float:
    primary = report.get("overall_primary") or {}
    retrieval = primary.get("retrieval_metrics") or {}
    coverage = primary.get("coverage") or {}
    nutrient = primary.get("nutrient_profile_error") or {}
    return round(
        (0.45 * to_float((retrieval.get("ndcg_at_k") or {}).get("10"), 0.0))
        + (0.35 * to_float((coverage.get("acceptable_at_k") or {}).get("5"), 0.0))
        + (0.10 * to_float((retrieval.get("hit_rate_at_k") or {}).get("5"), 0.0))
        + (0.10 * max(0.0, 1.0 - to_float(nutrient.get("mean_relative_error"), 0.0))),
        6,
    )


def tune_current_off_only_baseline(
    local_dataset: LocalFoodDataset,
    dev_frame: pd.DataFrame,
    benchmark_version: str,
    candidate_pool_options: list[int],
    tuning_sample_size: int,
    seed: int,
) -> dict[str, Any]:
    default_thresholds = default_acceptance_thresholds()
    tuning_sample = sample_dev_records(dev_frame, sample_size=tuning_sample_size, seed=seed)
    trials: list[dict[str, Any]] = []

    for candidate_pool_size in candidate_pool_options:
        results = evaluate_records(
            local_dataset=local_dataset,
            frame=tuning_sample,
            baseline_name="current_off_only",
            candidate_pool_size=candidate_pool_size,
            acceptance_thresholds=default_thresholds,
            progress_label=f"tuning current_off_only pool={candidate_pool_size}",
        )
        report = build_ausnut_benchmark_report(
            results,
            split_name="dev_tuning_sample",
            benchmark_version=benchmark_version,
            acceptance_thresholds=default_thresholds,
        )
        trials.append(
            {
                "candidate_pool_size": int(candidate_pool_size),
                "score": summarize_tuning_score(report),
                "report": report,
            }
        )

    trials.sort(key=lambda item: (item["score"], -item["candidate_pool_size"]), reverse=True)
    selected = trials[0]
    return {
        "selected_candidate_pool_size": int(selected["candidate_pool_size"]),
        "tuning_sample_size": int(len(tuning_sample)),
        "candidate_pool_trials": [
            {
                "candidate_pool_size": int(item["candidate_pool_size"]),
                "score": float(item["score"]),
                "overall_primary": item["report"].get("overall_primary") or {},
            }
            for item in trials
        ],
    }


def derive_acceptance_thresholds(dev_results: list[dict[str, Any]]) -> dict[str, Any]:
    default_thresholds = default_acceptance_thresholds()
    nutrient_errors = [
        to_float(((result.get("top_candidate_metrics") or {}).get("nutrient_profile_error") or {}).get("mean_relative_error"), 0.0)
        for result in dev_results
        if result.get("candidate_return_count", 0) > 0
    ]
    macro_distances = [
        to_float(((result.get("top_candidate_metrics") or {}).get("macro_vector_distance") or {}).get("normalized_l2_distance"), 0.0)
        for result in dev_results
        if result.get("candidate_return_count", 0) > 0
    ]
    title_overlaps = [
        to_float((result.get("top_candidate_metrics") or {}).get("title_token_overlap"), 0.0)
        for result in dev_results
        if result.get("candidate_return_count", 0) > 0 and to_float((result.get("top_candidate_metrics") or {}).get("title_token_overlap"), 0.0) > 0.0
    ]

    thresholds = dict(default_thresholds)
    if nutrient_errors:
        thresholds["max_mean_relative_error"] = round(float(np.clip(np.quantile(nutrient_errors, 0.75), 0.18, 0.45)), 4)
    if macro_distances:
        thresholds["max_macro_vector_distance"] = round(float(np.clip(np.quantile(macro_distances, 0.75), 0.20, 0.75)), 4)
    if title_overlaps:
        thresholds["min_title_overlap"] = round(float(np.clip(np.quantile(title_overlaps, 0.35), 0.12, 0.40)), 4)
    thresholds["minimum_relevance_grade"] = 2.0
    thresholds["require_category_or_major_group_match"] = True
    return thresholds


def flatten_record_results(record_results: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for result in record_results:
        top_candidate = (result.get("top_candidates") or [{}])[0]
        top_candidate_metrics = result.get("top_candidate_metrics") or {}
        rows.append(
            {
                "benchmark_id": result.get("benchmark_id"),
                "benchmark_tier": result.get("benchmark_tier"),
                "title": result.get("title"),
                "canonical_title": result.get("canonical_title"),
                "recipe_category": result.get("recipe_category"),
                "major_food_group": result.get("major_food_group"),
                "report_meal_slot": result.get("report_meal_slot"),
                "request_meal_slot": result.get("request_meal_slot"),
                "baseline_name": result.get("baseline_name"),
                "candidate_pool_size": result.get("candidate_pool_size"),
                "candidate_return_count": result.get("candidate_return_count"),
                "predicted_title": top_candidate.get("title"),
                "predicted_category": top_candidate.get("recipe_category"),
                "predicted_major_food_group": top_candidate.get("major_food_group"),
                "predicted_meal_slot": top_candidate.get("predicted_meal_slot"),
                "acceptable_match": bool(top_candidate_metrics.get("acceptable_match")),
                "relevance_grade": to_float(top_candidate_metrics.get("relevance_grade"), 0.0),
                "mean_relative_error": to_float(((top_candidate_metrics.get("nutrient_profile_error") or {}).get("mean_relative_error")), 0.0),
                "serving_mean_relative_error": to_float(((top_candidate_metrics.get("serving_profile_error") or {}).get("mean_relative_error")), 0.0),
                "macro_vector_distance": to_float(((top_candidate_metrics.get("macro_vector_distance") or {}).get("normalized_l2_distance")), 0.0),
                "title_token_overlap": to_float(top_candidate_metrics.get("title_token_overlap"), 0.0),
                "recipe_category_match": to_float(((top_candidate_metrics.get("category_agreement") or {}).get("recipe_category_match")), 0.0),
                "meal_slot_match": to_float(((top_candidate_metrics.get("category_agreement") or {}).get("meal_slot_match")), 0.0),
                "major_food_group_match": to_float(((top_candidate_metrics.get("category_agreement") or {}).get("major_food_group_match")), 0.0),
                "hit_rate_at_5": to_float(((result.get("retrieval_metrics") or {}).get("hit_rate_at_k") or {}).get("5"), 0.0),
                "ndcg_at_10": to_float(((result.get("retrieval_metrics") or {}).get("ndcg_at_k") or {}).get("10"), 0.0),
                "coverage_at_5": to_float(((result.get("coverage_metrics") or {}).get("acceptable_at_k") or {}).get("5"), 0.0),
                "failure_mode": result.get("failure_mode"),
                "top_candidates": " | ".join(str(candidate.get("title") or "") for candidate in (result.get("top_candidates") or [])[:5]),
            }
        )
    return pd.DataFrame(rows)


def generate_alias_candidate_report(
    dev_results: list[dict[str, Any]],
    acceptance_thresholds: dict[str, Any],
) -> pd.DataFrame:
    buckets: dict[tuple[str, str], dict[str, Any]] = {}

    minimum_relevance = float(acceptance_thresholds.get("minimum_relevance_grade", 2.0))
    maximum_error = float(acceptance_thresholds.get("max_mean_relative_error", 0.45))

    for result in dev_results:
        if str(result.get("failure_mode") or "") != "title_normalization":
            continue

        top_candidates = result.get("top_candidates") or []
        top_candidate = top_candidates[0] if top_candidates else {}
        source_title = str(top_candidate.get("title") or "").strip()
        source_key = canonical_title_key(source_title)
        target_title = canonicalize_title(result.get("canonical_title") or result.get("title") or "")
        target_key = canonical_title_key(target_title)
        if not source_key or not target_key or source_key == target_key:
            continue

        top_metrics = result.get("top_candidate_metrics") or {}
        relevance_grade = to_float(top_metrics.get("relevance_grade"), 0.0)
        nutrient_error = to_float(
            ((top_metrics.get("nutrient_profile_error") or {}).get("mean_relative_error")),
            0.0,
        )
        title_token_overlap = to_float(top_metrics.get("title_token_overlap"), 0.0)
        category_agreement = top_metrics.get("category_agreement") or {}
        recipe_category_match = to_float(category_agreement.get("recipe_category_match"), 0.0)
        meal_slot_match = to_float(category_agreement.get("meal_slot_match"), 0.0)
        source_target_token_overlap = _title_token_overlap(source_title, target_title)
        if relevance_grade < minimum_relevance or nutrient_error > maximum_error:
            continue

        pair_key = (source_key, target_key)
        bucket = buckets.setdefault(
            pair_key,
            {
                "source_key": source_key,
                "target_key": target_key,
                "source_titles": [],
                "target_titles": [],
                "benchmark_titles": [],
                "top_candidate_examples": [],
                "support_count": 0,
                "relevance_total": 0.0,
                "nutrient_error_total": 0.0,
                "title_token_overlap_total": 0.0,
                "recipe_category_match_total": 0.0,
                "meal_slot_match_total": 0.0,
                "source_target_token_overlap": source_target_token_overlap,
            },
        )
        bucket["source_titles"].append(source_title)
        bucket["target_titles"].append(target_title)
        bucket["benchmark_titles"].append(str(result.get("title") or ""))
        bucket["top_candidate_examples"].append(
            " | ".join(str(candidate.get("title") or "") for candidate in top_candidates[:3] if str(candidate.get("title") or "").strip())
        )
        bucket["support_count"] += 1
        bucket["relevance_total"] += relevance_grade
        bucket["nutrient_error_total"] += nutrient_error
        bucket["title_token_overlap_total"] += title_token_overlap
        bucket["recipe_category_match_total"] += recipe_category_match
        bucket["meal_slot_match_total"] += meal_slot_match
        bucket["source_target_token_overlap"] = max(
            to_float(bucket.get("source_target_token_overlap"), 0.0),
            source_target_token_overlap,
        )

    if not buckets:
        return pd.DataFrame(
            columns=[
                "source_title",
                "source_title_key",
                "suggested_alias",
                "suggested_alias_key",
                "support_count",
                "mean_relevance_grade",
                "mean_nutrient_error",
                "mean_title_token_overlap",
                "source_target_token_overlap",
                "recipe_category_match_rate",
                "meal_slot_match_rate",
                "example_benchmark_titles",
                "example_top_candidates",
            ]
        )

    best_by_source: dict[str, dict[str, Any]] = {}
    for bucket in buckets.values():
        source_key = str(bucket["source_key"])
        current_best = best_by_source.get(source_key)
        score = (
            int(bucket["support_count"]),
            round(float(bucket["relevance_total"]) / max(1, int(bucket["support_count"])), 4),
            -round(float(bucket["nutrient_error_total"]) / max(1, int(bucket["support_count"])), 4),
        )
        if not current_best:
            best_by_source[source_key] = {**bucket, "_score": score}
            continue
        if score > current_best["_score"]:
            best_by_source[source_key] = {**bucket, "_score": score}

    rows = []
    for bucket in best_by_source.values():
        support_count = max(1, int(bucket["support_count"]))
        rows.append(
            {
                "source_title": dedupe_strings(bucket["source_titles"], limit=1)[0],
                "source_title_key": bucket["source_key"],
                "suggested_alias": dedupe_strings(bucket["target_titles"], limit=1)[0],
                "suggested_alias_key": bucket["target_key"],
                "support_count": support_count,
                "mean_relevance_grade": round(float(bucket["relevance_total"]) / support_count, 4),
                "mean_nutrient_error": round(float(bucket["nutrient_error_total"]) / support_count, 4),
                "mean_title_token_overlap": round(float(bucket["title_token_overlap_total"]) / support_count, 4),
                "source_target_token_overlap": round(to_float(bucket.get("source_target_token_overlap"), 0.0), 4),
                "recipe_category_match_rate": round(float(bucket["recipe_category_match_total"]) / support_count, 4),
                "meal_slot_match_rate": round(float(bucket["meal_slot_match_total"]) / support_count, 4),
                "example_benchmark_titles": " | ".join(dedupe_strings(bucket["benchmark_titles"], limit=5)),
                "example_top_candidates": " | ".join(dedupe_strings(bucket["top_candidate_examples"], limit=3)),
            }
        )

    return pd.DataFrame(rows).sort_values(
        by=["support_count", "mean_relevance_grade", "mean_nutrient_error", "source_title"],
        ascending=[False, False, True, True],
        kind="stable",
    ).reset_index(drop=True)


def generate_promoteable_alias_report(
    candidate_report: pd.DataFrame,
    promotion_thresholds: dict[str, Any] | None = None,
) -> pd.DataFrame:
    thresholds = {**default_alias_promotion_thresholds(), **(promotion_thresholds or {})}
    if candidate_report.empty:
        return pd.DataFrame(
            columns=[
                "source_title",
                "source_title_key",
                "suggested_alias",
                "suggested_alias_key",
                "support_count",
                "mean_relevance_grade",
                "mean_nutrient_error",
                "mean_title_token_overlap",
                "source_target_token_overlap",
                "recipe_category_match_rate",
                "meal_slot_match_rate",
                "promotion_thresholds",
                "promotion_confidence",
                "example_benchmark_titles",
                "example_top_candidates",
            ]
        )

    promoteable = candidate_report.loc[
        (candidate_report["support_count"] >= int(thresholds["minimum_support_count"]))
        & (candidate_report["mean_relevance_grade"] >= float(thresholds["minimum_relevance_grade"]))
        & (candidate_report["mean_nutrient_error"] <= float(thresholds["max_mean_relative_error"]))
        & (candidate_report["mean_title_token_overlap"] >= float(thresholds["min_mean_title_token_overlap"]))
        & (candidate_report["source_target_token_overlap"] >= float(thresholds["min_source_target_token_overlap"]))
        & (candidate_report["recipe_category_match_rate"] >= float(thresholds["min_recipe_category_match_rate"]))
        & (candidate_report["meal_slot_match_rate"] >= float(thresholds["min_meal_slot_match_rate"]))
    ].copy()

    promoteable["promotion_thresholds"] = str(thresholds)
    promoteable["promotion_confidence"] = np.where(
        promoteable["support_count"] >= max(3, int(thresholds["minimum_support_count"]) + 1),
        "high",
        "guarded",
    )
    return promoteable.sort_values(
        by=[
            "support_count",
            "mean_relevance_grade",
            "mean_title_token_overlap",
            "source_target_token_overlap",
            "mean_nutrient_error",
            "source_title",
        ],
        ascending=[False, False, False, False, True, True],
        kind="stable",
    ).reset_index(drop=True)


def generate_alias_review_queue_report(
    candidate_report: pd.DataFrame,
    promoteable_report: pd.DataFrame,
    promotion_thresholds: dict[str, Any] | None = None,
) -> pd.DataFrame:
    thresholds = {**default_alias_promotion_thresholds(), **(promotion_thresholds or {})}
    if candidate_report.empty:
        return pd.DataFrame(
            columns=[
                "source_title",
                "source_title_key",
                "suggested_alias",
                "suggested_alias_key",
                "support_count",
                "mean_relevance_grade",
                "mean_nutrient_error",
                "mean_title_token_overlap",
                "source_target_token_overlap",
                "recipe_category_match_rate",
                "meal_slot_match_rate",
                "review_priority",
                "review_reason",
                "promotion_gap",
                "example_benchmark_titles",
                "example_top_candidates",
            ]
        )

    promoteable_pairs = {
        (str(row["source_title_key"]), str(row["suggested_alias_key"]))
        for _, row in promoteable_report.iterrows()
    }
    review_queue = candidate_report.loc[
        ~candidate_report.apply(
            lambda row: (str(row["source_title_key"]), str(row["suggested_alias_key"])) in promoteable_pairs,
            axis=1,
        )
    ].copy()
    if review_queue.empty:
        return review_queue

    review_queue = review_queue.loc[
        (review_queue["mean_relevance_grade"] >= max(3.0, float(thresholds["minimum_relevance_grade"])))
        & (review_queue["mean_nutrient_error"] <= max(0.18, float(thresholds["max_mean_relative_error"]) + 0.06))
        & (
            (review_queue["meal_slot_match_rate"] >= 1.0)
            | (review_queue["recipe_category_match_rate"] >= 1.0)
            | (review_queue["mean_title_token_overlap"] >= 0.18)
            | (review_queue["source_target_token_overlap"] >= 0.18)
        )
    ].copy()
    if review_queue.empty:
        return review_queue

    def _review_reason(row: pd.Series) -> str:
        reasons: list[str] = []
        if int(row["support_count"]) < int(thresholds["minimum_support_count"]):
            reasons.append("insufficient_support")
        if float(row["source_target_token_overlap"]) < float(thresholds["min_source_target_token_overlap"]):
            reasons.append("low_source_target_overlap")
        if float(row["mean_title_token_overlap"]) < float(thresholds["min_mean_title_token_overlap"]):
            reasons.append("low_benchmark_overlap")
        if float(row["recipe_category_match_rate"]) < float(thresholds["min_recipe_category_match_rate"]):
            reasons.append("category_mismatch")
        if float(row["meal_slot_match_rate"]) < float(thresholds["min_meal_slot_match_rate"]):
            reasons.append("meal_slot_mismatch")
        if float(row["mean_nutrient_error"]) > float(thresholds["max_mean_relative_error"]):
            reasons.append("nutrient_error_above_promotion_bar")
        return "|".join(reasons) or "manual_review_recommended"

    def _review_priority(row: pd.Series) -> str:
        if int(row["support_count"]) >= max(1, int(thresholds["minimum_support_count"]) - 1) and float(row["source_target_token_overlap"]) >= 0.25:
            return "high"
        if float(row["meal_slot_match_rate"]) >= 1.0 and float(row["mean_nutrient_error"]) <= 0.12:
            return "high"
        return "medium"

    def _promotion_gap(row: pd.Series) -> str:
        gaps: list[str] = []
        support_gap = max(0, int(thresholds["minimum_support_count"]) - int(row["support_count"]))
        if support_gap > 0:
            gaps.append(f"support+{support_gap}")
        overlap_gap = max(0.0, float(thresholds["min_source_target_token_overlap"]) - float(row["source_target_token_overlap"]))
        if overlap_gap > 0:
            gaps.append(f"source_overlap+{overlap_gap:.2f}")
        title_gap = max(0.0, float(thresholds["min_mean_title_token_overlap"]) - float(row["mean_title_token_overlap"]))
        if title_gap > 0:
            gaps.append(f"benchmark_overlap+{title_gap:.2f}")
        return "|".join(gaps) or "close_to_promotion"

    review_queue["review_priority"] = review_queue.apply(_review_priority, axis=1)
    review_queue["review_reason"] = review_queue.apply(_review_reason, axis=1)
    review_queue["promotion_gap"] = review_queue.apply(_promotion_gap, axis=1)
    return review_queue.sort_values(
        by=[
            "review_priority",
            "mean_relevance_grade",
            "mean_nutrient_error",
            "source_target_token_overlap",
            "mean_title_token_overlap",
            "source_title",
        ],
        ascending=[True, False, True, False, False, True],
        kind="stable",
    ).reset_index(drop=True)


def _comparison_row(report: dict[str, Any]) -> dict[str, Any]:
    primary = report.get("overall_primary") or {}
    return {
        "records": int((report.get("summary") or {}).get("record_count", 0)),
        "primary_records": int((report.get("summary") or {}).get("primary_record_count", 0)),
        "mean_relative_error": to_float((primary.get("nutrient_profile_error") or {}).get("mean_relative_error"), 0.0),
        "macro_vector_distance": to_float((primary.get("macro_vector_distance") or {}).get("mean_normalized_l2_distance"), 0.0),
        "hit_rate_at_5": to_float((primary.get("retrieval_metrics") or {}).get("hit_rate_at_k", {}).get("5"), 0.0),
        "ndcg_at_10": to_float((primary.get("retrieval_metrics") or {}).get("ndcg_at_k", {}).get("10"), 0.0),
        "coverage_at_5": to_float((primary.get("coverage") or {}).get("acceptable_at_k", {}).get("5"), 0.0),
        "recipe_category_accuracy": to_float((primary.get("category_agreement") or {}).get("recipe_category_accuracy"), 0.0),
    }


def build_service_summary(
    benchmark_version: str,
    tuning_summary: dict[str, Any],
    acceptance_thresholds: dict[str, Any],
    comparison_reports: dict[str, dict[str, Any]],
    output_paths: dict[str, Any],
) -> dict[str, Any]:
    comparison = {baseline_name: _comparison_row(report) for baseline_name, report in comparison_reports.items()}
    best_baseline = max(
        comparison.items(),
        key=lambda item: (
            to_float(item[1].get("coverage_at_5"), 0.0),
            to_float(item[1].get("ndcg_at_10"), 0.0),
            -to_float(item[1].get("mean_relative_error"), 0.0),
        ),
    )[0]
    return {
        "status": "available",
        "generated_at": _utc_now_iso(),
        "policy_version": BENCHMARK_POLICY_VERSION,
        "report_template_version": BENCHMARK_REPORT_TEMPLATE_VERSION,
        "benchmark_version": benchmark_version,
        "training_and_retrieval_corpus": BENCHMARK_POLICY["training_and_retrieval_corpus"],
        "external_benchmark": BENCHMARK_POLICY["external_benchmark"],
        "holdout_split": "test",
        "selected_tuning": tuning_summary,
        "acceptance_thresholds": acceptance_thresholds,
        "baseline_comparison": comparison,
        "best_holdout_baseline": best_baseline,
        "paths": output_paths,
    }


def render_benchmark_specification_markdown(
    table_metadata: dict[str, Any],
    split_manifest: dict[str, Any],
    dev_priors: dict[str, Any],
    tuning_summary: dict[str, Any],
    acceptance_thresholds: dict[str, Any],
    leakage_report: dict[str, Any],
    output_paths: dict[str, Any],
) -> str:
    benchmark_version = table_metadata.get("benchmark_version", "unknown")
    hard_rules = "\n".join(f"- {rule}" for rule in BENCHMARK_POLICY["hard_rules"])
    request_slot_lines = "\n".join(
        f"- {slot}: {value:.4f}"
        for slot, value in (dev_priors.get("request_meal_slot_priors") or {}).items()
    )
    output_path_lines = "\n".join(f"- {name}: {path}" for name, path in output_paths.items())
    return f"""# AUSNUT Benchmark Specification

Generated: {_utc_now_iso()}
Benchmark version: {benchmark_version}

## Policy

- OFF remains the training and retrieval corpus.
- AUSNUT remains the external Australian ground-truth benchmark.
- Benchmark success is defined as nutritional fidelity to Australian food profiles, not title overlap alone.

## Hard Rules

{hard_rules}

## Frozen Split

- Total rows: {split_manifest.get('counts', {}).get('total', 0)}
- Development rows: {split_manifest.get('counts', {}).get('dev', 0)}
- Holdout test rows: {split_manifest.get('counts', {}).get('test', 0)}
- Dev folds: {split_manifest.get('split_config', {}).get('dev_cross_validation_folds', 5)}
- Stratification: {split_manifest.get('split_config', {}).get('stratification', '')}

## Benchmark Table

- Primary rows: {table_metadata.get('primary_row_count', 0)}
- Supplementary rows: {table_metadata.get('supplementary_row_count', 0)}
- Report template version: {BENCHMARK_REPORT_TEMPLATE_VERSION}
- Leakage validation passed: {bool(leakage_report.get('passed'))}

## Dev Priors

{request_slot_lines or '- no dev priors available'}

## Dev-Only Tuning

- Selected candidate pool size: {tuning_summary.get('selected_candidate_pool_size', 0)}
- Tuning sample size: {tuning_summary.get('tuning_sample_size', 0)}

## Acceptance Gates

- Max mean relative nutrient error: {acceptance_thresholds.get('max_mean_relative_error', 0.0):.4f}
- Max macro vector distance: {acceptance_thresholds.get('max_macro_vector_distance', 0.0):.4f}
- Minimum title overlap or category gate: {acceptance_thresholds.get('min_title_overlap', 0.0):.4f}
- Minimum relevance grade: {acceptance_thresholds.get('minimum_relevance_grade', 0.0):.1f}
- Require category or major-group match: {bool(acceptance_thresholds.get('require_category_or_major_group_match', True))}

## Deliverables

{output_path_lines}
"""


def render_metric_definition_markdown(acceptance_thresholds: dict[str, Any]) -> str:
    return f"""# AUSNUT Metric Definition Sheet

Generated: {_utc_now_iso()}

## Nutrient Profile Error

- Basis: per-100g comparison for calories, protein, carbs, fat, and sugar.
- Field relative error: `abs(predicted - benchmark) / max(benchmark, epsilon)` capped at 2.0.
- Aggregate nutrient error: mean and median of the five field relative errors.

## Macro Vector Distance

- Vector: `[calories, protein, carbs, fat]` on a per-100g basis.
- `normalized_l2_distance`: Euclidean distance divided by benchmark vector norm.
- `cosine_distance`: `1 - cosine_similarity` for the same macro vector.

## Retrieval Metrics

- Hit rate@k: whether at least one AUSNUT-like candidate appears in the top-k ranked window.
- Recall@k: relevant candidates retrieved in top-k divided by relevant candidates inside the evaluated candidate window.
- NDCG@k: graded gain over the ranked candidate window using dev-calibrated relevance grades.

## Category Agreement

- Recipe category accuracy: exact normalized category match.
- Meal slot accuracy: controlled benchmark slot match.
- Major food-group accuracy: curated group match used for broader Australian food-group fidelity.

## Coverage

- Candidate return rate: fraction of benchmark rows with at least one OFF candidate returned.
- Acceptable@k: whether the top-k window contains a candidate passing the dev-only acceptance gates.

## Acceptance Gates

- Max mean relative nutrient error: {acceptance_thresholds.get('max_mean_relative_error', 0.0):.4f}
- Max macro vector distance: {acceptance_thresholds.get('max_macro_vector_distance', 0.0):.4f}
- Minimum title overlap: {acceptance_thresholds.get('min_title_overlap', 0.0):.4f}
- Minimum relevance grade: {acceptance_thresholds.get('minimum_relevance_grade', 0.0):.1f}
- Require category or major-group match: {bool(acceptance_thresholds.get('require_category_or_major_group_match', True))}
"""


def render_baseline_report_markdown(
    benchmark_version: str,
    tuning_summary: dict[str, Any],
    acceptance_thresholds: dict[str, Any],
    comparison_reports: dict[str, dict[str, Any]],
    leakage_report: dict[str, Any],
) -> str:
    table_lines = [
        "| Baseline | Primary Records | Mean Nutrient Error | Macro Distance | Hit@5 | NDCG@10 | Coverage@5 | Category Acc |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for baseline_name, report in comparison_reports.items():
        summary = _comparison_row(report)
        table_lines.append(
            f"| {baseline_name} | {summary['primary_records']} | {summary['mean_relative_error']:.4f} | {summary['macro_vector_distance']:.4f} | {summary['hit_rate_at_5']:.4f} | {summary['ndcg_at_10']:.4f} | {summary['coverage_at_5']:.4f} | {summary['recipe_category_accuracy']:.4f} |"
        )

    return f"""# AUSNUT Baseline Report

Generated: {_utc_now_iso()}
Benchmark version: {benchmark_version}
Leakage validation passed: {bool(leakage_report.get('passed'))}

## Tuning Summary

- Selected candidate pool size: {tuning_summary.get('selected_candidate_pool_size', 0)}
- Tuning sample size: {tuning_summary.get('tuning_sample_size', 0)}
- Acceptance threshold max mean nutrient error: {acceptance_thresholds.get('max_mean_relative_error', 0.0):.4f}

## Holdout Comparison

{chr(10).join(table_lines)}
"""


def render_gap_analysis_markdown(current_report: dict[str, Any]) -> str:
    failure_modes = (current_report.get("overall") or {}).get("failure_modes") or {}
    ordered_modes = list(failure_modes.items())[:5]
    failure_lines = "\n".join(f"- {name}: {count}" for name, count in ordered_modes)

    next_actions = []
    for name, _ in ordered_modes:
        if name == "retrieval_gap":
            next_actions.append("Broaden OFF candidate retrieval for weak Australian food groups before changing ranking weights.")
        elif name == "ranking_order":
            next_actions.append("Tune reranking weights and candidate-pool depth with the AUSNUT dev split because acceptable matches already exist deeper in the ranked list.")
        elif name == "title_normalization":
            next_actions.append("Improve canonical-title and category normalization so Australian foods align before scoring.")
        elif name == "serving_conversion":
            next_actions.append("Tighten serving conversion and per-100g normalization because the candidate is nutritionally close only after serving adjustments.")
        elif name == "category_assignment":
            next_actions.append("Refine meal-slot and food-group assignment logic using the AUSNUT controlled mapping tables.")
        elif name == "title_or_category_gate":
            next_actions.append("Review category and title acceptance gates for borderline cases that are nutritionally close but structurally mismatched.")
    deduped_actions = dedupe_strings(next_actions, limit=5)
    action_lines = "\n".join(f"1. {action}" if index == 0 else f"{index + 1}. {action}" for index, action in enumerate(deduped_actions))

    return f"""# AUSNUT Gap Analysis Report

Generated: {_utc_now_iso()}

## Top Failure Modes

{failure_lines or '- no failures recorded'}

## Next Model Changes

{action_lines or '1. Maintain the current retrieval and ranking configuration until new benchmark failures appear.'}
"""


def main() -> None:
    args = _parse_args()
    artifact_dir = ensure_benchmark_artifact_dir(args.artifact_dir)

    benchmark_frame, table_metadata = write_clean_ausnut_benchmark_table(args.dataset_path, artifact_dir)
    split_manifest = write_split_manifest(benchmark_frame, artifact_dir, test_ratio=args.test_ratio, seed=args.seed)
    dev_priors = write_dev_priors(benchmark_frame, split_manifest, artifact_dir)
    leakage_report = write_leakage_report(
        benchmark_frame,
        split_manifest,
        artifact_dir=artifact_dir,
        off_db_path=args.off_db_path,
        off_table=args.off_db_table,
    )

    dev_ids = {str(value) for value in split_manifest.get("dev_ids", [])}
    test_ids = {str(value) for value in split_manifest.get("test_ids", [])}
    dev_frame = benchmark_frame[benchmark_frame["benchmark_id"].astype(str).isin(dev_ids)].reset_index(drop=True)
    test_frame = benchmark_frame[benchmark_frame["benchmark_id"].astype(str).isin(test_ids)].reset_index(drop=True)
    primary_dev_frame = dev_frame[dev_frame["benchmark_tier"] == "primary"].reset_index(drop=True)

    candidate_pool_options = _parse_candidate_pool_options(args.candidate_pool_options)
    local_dataset = LocalFoodDataset(dataset_path=args.off_db_path)
    local_dataset.warmup()

    benchmark_version = str(
        table_metadata.get("benchmark_version")
        or (benchmark_frame["benchmark_version"].iloc[0] if not benchmark_frame.empty else "unknown")
    )
    tuning_summary = tune_current_off_only_baseline(
        local_dataset=local_dataset,
        dev_frame=primary_dev_frame,
        benchmark_version=benchmark_version,
        candidate_pool_options=candidate_pool_options,
        tuning_sample_size=int(args.tuning_sample_size),
        seed=int(args.seed),
    )

    tuned_candidate_pool_size = int(tuning_summary.get("selected_candidate_pool_size", candidate_pool_options[0]))
    current_dev_results = evaluate_records(
        local_dataset=local_dataset,
        frame=primary_dev_frame,
        baseline_name="current_off_only",
        candidate_pool_size=tuned_candidate_pool_size,
        acceptance_thresholds=default_acceptance_thresholds(),
        progress_label="current_off_only dev",
    )
    acceptance_thresholds = derive_acceptance_thresholds(current_dev_results)
    dev_priors["acceptance_thresholds"] = acceptance_thresholds
    dev_priors["selected_tuning"] = tuning_summary
    save_json(artifact_dir / Path(default_benchmark_table_path()).with_name("ausnut_dev_priors.json").name, dev_priors)

    comparison_reports: dict[str, dict[str, Any]] = {}
    output_paths: dict[str, Any] = {
        "benchmark_table": str(artifact_dir / Path(default_benchmark_table_path()).name),
        "split_manifest": str(artifact_dir / Path(default_split_manifest_path()).name),
    }

    current_dev_report = build_ausnut_benchmark_report(
        current_dev_results,
        split_name="dev",
        benchmark_version=benchmark_version,
        acceptance_thresholds=acceptance_thresholds,
    )
    current_dev_results_path = artifact_dir / "current_off_only_dev_results.csv"
    flatten_record_results(current_dev_results).to_csv(current_dev_results_path, index=False)
    current_dev_alias_candidates = generate_alias_candidate_report(current_dev_results, acceptance_thresholds)
    current_dev_alias_candidates_path = artifact_dir / "current_off_only_dev_alias_candidates.csv"
    current_dev_alias_candidates.to_csv(current_dev_alias_candidates_path, index=False)
    current_dev_promoteable_aliases = generate_promoteable_alias_report(current_dev_alias_candidates)
    current_dev_promoteable_aliases_path = artifact_dir / "current_off_only_dev_promoteable_aliases.csv"
    current_dev_promoteable_aliases.to_csv(
        current_dev_promoteable_aliases_path,
        index=False,
    )
    current_dev_alias_review_queue_path = artifact_dir / "current_off_only_dev_alias_review_queue.csv"
    generate_alias_review_queue_report(
        current_dev_alias_candidates,
        current_dev_promoteable_aliases,
    ).to_csv(
        current_dev_alias_review_queue_path,
        index=False,
    )
    current_dev_report_path = artifact_dir / "current_off_only_dev_report.json"
    save_json(current_dev_report_path, current_dev_report)
    output_paths["current_off_only_dev_results"] = str(current_dev_results_path)
    output_paths["current_off_only_dev_alias_candidates"] = str(current_dev_alias_candidates_path)
    output_paths["current_off_only_dev_promoteable_aliases"] = str(current_dev_promoteable_aliases_path)
    output_paths["current_off_only_dev_alias_review_queue"] = str(current_dev_alias_review_queue_path)
    output_paths["current_off_only_dev_report"] = str(current_dev_report_path)

    for baseline_name in BASELINE_NAMES:
        baseline_results = evaluate_records(
            local_dataset=local_dataset,
            frame=test_frame,
            baseline_name=baseline_name,
            candidate_pool_size=tuned_candidate_pool_size,
            acceptance_thresholds=acceptance_thresholds,
            progress_label=f"{baseline_name} test",
        )
        baseline_report = build_ausnut_benchmark_report(
            baseline_results,
            split_name="test",
            benchmark_version=benchmark_version,
            acceptance_thresholds=acceptance_thresholds,
        )
        comparison_reports[baseline_name] = baseline_report

        results_path = artifact_dir / f"{baseline_name}_test_results.csv"
        report_path = artifact_dir / f"{baseline_name}_test_report.json"
        flatten_record_results(baseline_results).to_csv(results_path, index=False)
        save_json(report_path, baseline_report)
        output_paths[f"{baseline_name}_test_results"] = str(results_path)
        output_paths[f"{baseline_name}_test_report"] = str(report_path)

    latest_summary = build_service_summary(
        benchmark_version=benchmark_version,
        tuning_summary=tuning_summary,
        acceptance_thresholds=acceptance_thresholds,
        comparison_reports=comparison_reports,
        output_paths=output_paths,
    )
    latest_summary_path = artifact_dir / Path(default_latest_summary_path()).name
    save_json(latest_summary_path, latest_summary)
    output_paths["latest_summary"] = str(latest_summary_path)

    specification_path = artifact_dir / "benchmark_specification.md"
    specification_path.write_text(
        render_benchmark_specification_markdown(
            table_metadata=table_metadata,
            split_manifest=split_manifest,
            dev_priors=dev_priors,
            tuning_summary=tuning_summary,
            acceptance_thresholds=acceptance_thresholds,
            leakage_report=leakage_report,
            output_paths=output_paths,
        ),
        encoding="utf-8",
    )
    output_paths["benchmark_specification"] = str(specification_path)

    metric_sheet_path = artifact_dir / "metric_definition_sheet.md"
    metric_sheet_path.write_text(render_metric_definition_markdown(acceptance_thresholds), encoding="utf-8")
    output_paths["metric_definition_sheet"] = str(metric_sheet_path)

    baseline_report_path = artifact_dir / "baseline_report.md"
    baseline_report_path.write_text(
        render_baseline_report_markdown(
            benchmark_version=benchmark_version,
            tuning_summary=tuning_summary,
            acceptance_thresholds=acceptance_thresholds,
            comparison_reports=comparison_reports,
            leakage_report=leakage_report,
        ),
        encoding="utf-8",
    )
    output_paths["baseline_report"] = str(baseline_report_path)

    gap_analysis_path = artifact_dir / "gap_analysis_report.md"
    gap_analysis_path.write_text(
        render_gap_analysis_markdown(comparison_reports["current_off_only"]),
        encoding="utf-8",
    )
    output_paths["gap_analysis_report"] = str(gap_analysis_path)

    save_json(latest_summary_path, {**latest_summary, "paths": output_paths})
    print(
        "[AUSNUT] benchmark complete",
        {
            "benchmark_version": benchmark_version,
            "candidate_pool_size": tuned_candidate_pool_size,
            "latest_summary": str(latest_summary_path),
        },
    )


if __name__ == "__main__":
    main()