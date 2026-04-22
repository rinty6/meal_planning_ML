from __future__ import annotations


import math
from typing import Any

import numpy as np

from .constants import EVAL_K_VALUES
from .utils import canonical_title_key, cosine_similarity_safe, normalize_text, tokenize_canonical_title, to_float


BENCHMARK_TOP_K_VALUES = (1, 3, 5, 10)
BENCHMARK_NUTRIENTS = ("calories", "protein", "carbs", "fats", "sugar")
BENCHMARK_REPORT_TEMPLATE_VERSION = "ausnut-benchmark-v1"


def build_runtime_slot_metric_metadata() -> dict[str, Any]:
    return {
        "scope": "runtime_slot_diagnostics",
        "benchmark_guidance": (
            "proxy_metrics, offline_diagnostics, diversity_dashboard, retrieval_metrics, tuning, "
            "and experimentation describe the live recommendation path. They are not AUSNUT benchmark truth metrics."
        ),
        "groups": {
            "proxy_metrics": "Runtime preference proxy against recent user titles.",
            "offline_diagnostics": "Runtime item-level diagnostics for selected recommendations.",
            "combo_diagnostics": "Runtime combo-level diagnostics for returned meal combinations.",
            "mapping_diagnostics": "Runtime diagnostics for mapped, local-only, and unresolved recommendation identities.",
            "diversity_dashboard": "Runtime diversity diagnostics for returned recommendations and combos.",
            "retrieval_metrics": "Runtime candidate-pool diagnostics before final combo shaping.",
            "timing": "Runtime slot timing breakdown in milliseconds for retrieval, mapping, ranking, combo assembly, combo pool building, combo candidate generation, combo diversity selection, and total slot processing.",
            "tuning": "Runtime adaptive tuning values applied to this request.",
            "experimentation": "Runtime experiment and variant metadata for this request.",
        },
    }


def build_runtime_aggregate_metric_metadata() -> dict[str, Any]:
    return {
        "scope": "runtime_aggregate_diagnostics",
        "benchmark_guidance": (
            "slots and overall_diversity summarize live runtime diagnostics. ausnut_benchmark is a separate offline "
            "external benchmark summary loaded from generated AUSNUT artifacts."
        ),
        "groups": {
            "slots": "Runtime slot diagnostics keyed by meal slot.",
            "overall_combo_diagnostics": "Aggregate combo diagnostics merged across returned slots.",
            "overall_mapping_diagnostics": "Aggregate mapping diagnostics merged across returned slots.",
            "overall_diversity": "Runtime aggregate diversity across returned slots.",
            "overall_timing": "Aggregate runtime timing merged across returned slots.",
            "experimentation": "Runtime experiment metadata for the aggregated response.",
            "ausnut_benchmark": "Offline external AUSNUT benchmark summary for model-quality reference.",
        },
    }


def _average(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(np.median(np.asarray(values, dtype=float)))


def _bounded_relative_error(predicted: float, actual: float, cap: float = 2.0) -> float:
    actual = abs(float(actual))
    predicted = float(predicted)
    if actual <= 1e-9:
        return 0.0 if abs(predicted) <= 1e-9 else float(cap)
    return float(min(cap, abs(predicted - actual) / actual))


def _safe_scope_report(top_k_values: tuple[int, ...]) -> dict[str, Any]:
    return {
        "record_count": 0,
        "nutrient_profile_error": {
            "comparison_basis": "per100",
            "mean_relative_error": 0.0,
            "median_relative_error": 0.0,
            "field_relative_error": {nutrient: 0.0 for nutrient in BENCHMARK_NUTRIENTS},
        },
        "macro_vector_distance": {
            "mean_normalized_l2_distance": 0.0,
            "median_normalized_l2_distance": 0.0,
            "mean_cosine_distance": 0.0,
        },
        "retrieval_metrics": {
            "hit_rate_at_k": {str(k): 0.0 for k in top_k_values},
            "recall_at_k": {str(k): 0.0 for k in top_k_values},
            "ndcg_at_k": {str(k): 0.0 for k in top_k_values},
        },
        "category_agreement": {
            "recipe_category_accuracy": 0.0,
            "meal_slot_accuracy": 0.0,
            "major_food_group_accuracy": 0.0,
        },
        "coverage": {
            "candidate_return_rate": 0.0,
            "acceptable_at_k": {str(k): 0.0 for k in top_k_values},
        },
        "title_alignment": {
            "mean_token_overlap": 0.0,
            "median_token_overlap": 0.0,
        },
        "failure_modes": {},
    }


def _candidate_title_key(candidate: dict[str, Any]) -> str:
    return canonical_title_key(candidate.get("canonical_title") or candidate.get("title") or candidate.get("original_title"))


def _title_matches_preferences(title: str, preferred_titles: list[str]) -> bool:
    normalized_title = canonical_title_key(title)
    if not normalized_title:
        return False

    preferred_normalized = {canonical_title_key(item) for item in preferred_titles if canonical_title_key(item)}
    if normalized_title in preferred_normalized:
        return True

    title_tokens = tokenize_canonical_title(normalized_title)
    if not title_tokens:
        return False

    for preferred in preferred_normalized:
        preferred_tokens = tokenize_canonical_title(preferred)
        if not preferred_tokens:
            continue
        if title_tokens.intersection(preferred_tokens):
            return True

    return False


def compute_proxy_title_metrics(
    ranked_candidates: list[dict[str, Any]],
    recommended_items: list[dict[str, Any]],
    preferred_titles: list[str],
    candidate_cap: int = 40,
) -> dict[str, Any]:
    candidates = (ranked_candidates or [])[:candidate_cap]
    if not candidates:
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "support": 0,
            "precision_at_k": {str(k): 0.0 for k in EVAL_K_VALUES},
        }

    selected_ids = {
        str(item.get("id") or item.get("food_id") or item.get("item_id") or "").strip()
        for item in (recommended_items or [])
        if str(item.get("id") or item.get("food_id") or item.get("item_id") or "").strip()
    }

    y_true: list[int] = []
    y_pred: list[int] = []
    for candidate in candidates:
        title = str(candidate.get("canonical_title") or candidate.get("title") or "")
        candidate_id = str(candidate.get("id") or "").strip()
        y_true.append(1 if _title_matches_preferences(title, preferred_titles) else 0)
        y_pred.append(1 if candidate_id and candidate_id in selected_ids else 0)

    tp = sum(1 for truth, pred in zip(y_true, y_pred) if truth == 1 and pred == 1)
    fp = sum(1 for truth, pred in zip(y_true, y_pred) if truth == 0 and pred == 1)
    tn = sum(1 for truth, pred in zip(y_true, y_pred) if truth == 0 and pred == 0)
    fn = sum(1 for truth, pred in zip(y_true, y_pred) if truth == 1 and pred == 0)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    accuracy = (tp + tn) / max(1, len(y_true))
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    precision_at_k: dict[str, float] = {}
    for k in EVAL_K_VALUES:
        k_value = int(k)
        if k_value <= 0:
            continue
        top_k = (recommended_items or [])[:k_value]
        hits = sum(
            1
            for item in top_k
            if _title_matches_preferences(
                str(item.get("canonical_title") or item.get("title") or ""),
                preferred_titles,
            )
        )
        precision_at_k[str(k_value)] = round(float(hits / max(1, k_value)), 4)

    return {
        "accuracy": round(float(accuracy), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1_score": round(float(f1_score), 4),
        "support": int(sum(y_true)),
        "precision_at_k": precision_at_k,
    }


def compute_candidate_pool_metrics(candidates: list[dict[str, Any]], top_n: int = 20) -> dict[str, Any]:
    pool = (candidates or [])[: max(1, int(top_n))]
    title_keys = [_candidate_title_key(candidate) for candidate in pool if _candidate_title_key(candidate)]
    unique_count = len(set(title_keys))
    total = len(title_keys)
    mean_top_distance = _average([to_float(candidate.get("knn_distance"), 0.0) for candidate in pool[:5]])
    mean_top_score = _average([to_float(candidate.get("score"), 0.0) for candidate in pool[:5]])

    return {
        "candidate_count": int(len(candidates or [])),
        "pool_sample_size": int(total),
        "unique_title_count": int(unique_count),
        "unique_title_ratio": round(float(unique_count / max(1, total)), 4),
        "repeated_title_rate": round(float((total - unique_count) / max(1, total)), 4),
        "mean_top_distance": round(float(mean_top_distance), 4),
        "mean_top_score": round(float(mean_top_score), 4),
    }


def compute_offline_diagnostic_metrics(
    ranked_candidates: list[dict[str, Any]],
    recommended_items: list[dict[str, Any]],
    slot_target: float,
    user_vec: np.ndarray | None = None,
) -> dict[str, Any]:
    items = recommended_items or []
    if not items:
        return {
            "mean_selected_score": 0.0,
            "mean_knn_distance": 0.0,
            "mean_adjusted_distance": 0.0,
            "mean_health_score": 0.0,
            "mean_calorie_gap_ratio": 0.0,
            "within_15pct_target_rate": 0.0,
            "macro_profile_similarity": 0.0,
            "candidate_pool_size": int(len(ranked_candidates or [])),
        }

    slot_target = max(1.0, float(slot_target or 0.0))
    calorie_gap_ratios = [
        abs(to_float(item.get("calories"), 0.0) - slot_target) / slot_target
        for item in items
    ]
    within_15pct_target_rate = sum(1 for gap in calorie_gap_ratios if gap <= 0.15) / max(1, len(calorie_gap_ratios))
    mean_health_score = _average(
        [to_float(item.get("health_score") or item.get("aggregated_rating"), 0.0) for item in items]
    )
    macro_profile_similarity = 0.0
    if user_vec is not None and len(items) > 0:
        macro_mean = np.asarray(
            [
                _average([to_float(item.get("calories"), 0.0) for item in items]),
                _average([to_float(item.get("protein"), 0.0) for item in items]),
                _average([to_float(item.get("carbs"), 0.0) for item in items]),
                _average([to_float(item.get("fats"), 0.0) for item in items]),
            ],
            dtype=float,
        )
        macro_profile_similarity = max(0.0, (cosine_similarity_safe(macro_mean, np.asarray(user_vec, dtype=float)) + 1.0) / 2.0)

    return {
        "mean_selected_score": round(_average([to_float(item.get("score"), 0.0) for item in items]), 4),
        "mean_knn_distance": round(_average([to_float(item.get("knn_distance"), 0.0) for item in items]), 4),
        "mean_adjusted_distance": round(_average([to_float(item.get("adjusted_distance"), 0.0) for item in items]), 4),
        "mean_health_score": round(mean_health_score, 4),
        "mean_calorie_gap_ratio": round(_average(calorie_gap_ratios), 4),
        "within_15pct_target_rate": round(float(within_15pct_target_rate), 4),
        "macro_profile_similarity": round(float(macro_profile_similarity), 4),
        "candidate_pool_size": int(len(ranked_candidates or [])),
    }


def compute_diversity_dashboard(
    recommended_items: list[dict[str, Any]],
    combos: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    combos = combos or []
    flat_items = list(recommended_items or [])
    if combos:
        flat_items = []
        for combo in combos:
            flat_items.extend(combo.get("items") or [])

    title_keys = [_candidate_title_key(item) for item in flat_items if _candidate_title_key(item)]
    unique_titles = len(set(title_keys))
    total_titles = len(title_keys)

    side_titles = [
        _candidate_title_key(item)
        for combo in combos
        for item in (combo.get("items") or [])
        if str(item.get("category") or item.get("role") or "").strip().lower() == "side" and _candidate_title_key(item)
    ]
    drink_titles = [
        _candidate_title_key(item)
        for combo in combos
        for item in (combo.get("items") or [])
        if str(item.get("category") or item.get("role") or "").strip().lower() == "drink" and _candidate_title_key(item)
    ]

    repeated_side_rate = 0.0
    if side_titles:
        repeated_side_rate = (len(side_titles) - len(set(side_titles))) / max(1, len(side_titles))
    repeated_drink_rate = 0.0
    if drink_titles:
        repeated_drink_rate = (len(drink_titles) - len(set(drink_titles))) / max(1, len(drink_titles))

    image_hit_rate = sum(1 for item in flat_items if str(item.get("image") or "").strip()) / max(1, len(flat_items or [1]))

    return {
        "total_items": int(len(flat_items)),
        "total_combos": int(len(combos)),
        "unique_title_count": int(unique_titles),
        "unique_title_ratio": round(float(unique_titles / max(1, total_titles)), 4),
        "repeated_side_rate": round(float(repeated_side_rate), 4),
        "repeated_drink_rate": round(float(repeated_drink_rate), 4),
        "image_hit_rate": round(float(image_hit_rate), 4),
    }


def compute_combo_diagnostic_metrics(
    combos: list[dict[str, Any]] | None,
    slot_target: float,
) -> dict[str, Any]:
    combo_list = [combo for combo in (combos or []) if isinstance(combo, dict)]
    if not combo_list:
        return {
            "combo_count": 0,
            "mean_combo_calorie_gap_ratio": 0.0,
            "within_10pct_target_combo_rate": 0.0,
            "mean_items_per_combo": 0.0,
            "role_coverage_rate": 0.0,
            "duplicate_title_within_combo_rate": 0.0,
        }

    safe_slot_target = max(1.0, float(slot_target or 0.0))
    calorie_gap_ratios: list[float] = []
    item_counts: list[int] = []
    role_coverage_flags: list[float] = []
    duplicate_title_rates: list[float] = []

    for combo in combo_list:
        total_calories = to_float(combo.get("total_calories"), 0.0)
        calorie_gap_ratios.append(abs(total_calories - safe_slot_target) / safe_slot_target)

        items = [item for item in (combo.get("items") or []) if isinstance(item, dict)]
        item_counts.append(float(len(items)))

        roles = {
            normalize_text(item.get("category") or item.get("role") or "")
            for item in items
            if normalize_text(item.get("category") or item.get("role") or "")
        }
        role_coverage_flags.append(1.0 if {"main", "side", "drink"}.issubset(roles) else 0.0)

        title_keys = [_candidate_title_key(item) for item in items if _candidate_title_key(item)]
        duplicate_title_rates.append(float((len(title_keys) - len(set(title_keys))) / max(1, len(title_keys))))

    within_target_rate = sum(1 for value in calorie_gap_ratios if value <= 0.10) / max(1, len(calorie_gap_ratios))
    return {
        "combo_count": int(len(combo_list)),
        "mean_combo_calorie_gap_ratio": round(_average(calorie_gap_ratios), 4),
        "within_10pct_target_combo_rate": round(float(within_target_rate), 4),
        "mean_items_per_combo": round(_average(item_counts), 4),
        "role_coverage_rate": round(_average(role_coverage_flags), 4),
        "duplicate_title_within_combo_rate": round(_average(duplicate_title_rates), 4),
    }


def compute_mapping_diagnostics(
    recommended_items: list[dict[str, Any]],
    combos: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    flat_items = list(recommended_items or [])
    if combos:
        flat_items = []
        for combo in combos:
            flat_items.extend(item for item in (combo.get("items") or []) if isinstance(item, dict))

    if not flat_items:
        return {
            "total_items": 0,
            "mapped_item_rate": 0.0,
            "local_only_item_rate": 0.0,
            "strict_mapping_rate": 0.0,
            "relaxed_mapping_rate": 0.0,
            "unresolved_identity_rate": 0.0,
            "fatsecret_id_rate": 0.0,
            "mean_mapping_calorie_gap_ratio": 0.0,
        }

    total_items = len(flat_items)
    mapped_count = 0
    local_only_count = 0
    strict_count = 0
    relaxed_count = 0
    unresolved_count = 0
    fatsecret_id_count = 0
    calorie_gap_values: list[float] = []

    for item in flat_items:
        ml_tag = normalize_text(item.get("ml_tag") or "")
        food_id = str(item.get("food_id") or "").strip()
        fatsecret_food_id = str(item.get("fatsecret_food_id") or "").strip()
        acceptance_mode = normalize_text(item.get("mapping_acceptance_mode") or "")
        calorie_gap = to_float(item.get("calorie_diff_ratio"), 0.0)
        calorie_gap_values.append(calorie_gap)

        is_local_only = ml_tag == "local_only"
        has_external_identity = bool(fatsecret_food_id)
        is_unresolved_identity = is_local_only or not food_id or food_id.startswith("local-")

        if not is_local_only:
            mapped_count += 1
        if is_local_only:
            local_only_count += 1
        if acceptance_mode == "strict":
            strict_count += 1
        if acceptance_mode == "relaxed_title_fallback":
            relaxed_count += 1
        if is_unresolved_identity:
            unresolved_count += 1
        if has_external_identity:
            fatsecret_id_count += 1

    return {
        "total_items": int(total_items),
        "mapped_item_rate": round(float(mapped_count / max(1, total_items)), 4),
        "local_only_item_rate": round(float(local_only_count / max(1, total_items)), 4),
        "strict_mapping_rate": round(float(strict_count / max(1, total_items)), 4),
        "relaxed_mapping_rate": round(float(relaxed_count / max(1, total_items)), 4),
        "unresolved_identity_rate": round(float(unresolved_count / max(1, total_items)), 4),
        "fatsecret_id_rate": round(float(fatsecret_id_count / max(1, total_items)), 4),
        "mean_mapping_calorie_gap_ratio": round(_average(calorie_gap_values), 4),
    }


def compute_nutrient_profile_error(
    predicted_profile: dict[str, Any],
    benchmark_profile: dict[str, Any],
    nutrients: tuple[str, ...] = BENCHMARK_NUTRIENTS,
    comparison_basis: str = "per100",
) -> dict[str, Any]:
    abs_errors: dict[str, float] = {}
    relative_errors: dict[str, float] = {}

    for nutrient in nutrients:
        predicted = to_float((predicted_profile or {}).get(nutrient), 0.0)
        actual = to_float((benchmark_profile or {}).get(nutrient), 0.0)
        abs_errors[nutrient] = round(abs(predicted - actual), 4)
        relative_errors[nutrient] = round(_bounded_relative_error(predicted, actual), 4)

    raw_relative_errors = [to_float(relative_errors.get(nutrient), 0.0) for nutrient in nutrients]
    return {
        "comparison_basis": comparison_basis,
        "field_abs_error": abs_errors,
        "field_relative_error": relative_errors,
        "mean_abs_error": round(_average([to_float(value, 0.0) for value in abs_errors.values()]), 4),
        "mean_relative_error": round(_average(raw_relative_errors), 4),
        "median_relative_error": round(_median(raw_relative_errors), 4),
    }


def compute_macro_vector_distance(
    predicted_profile: dict[str, Any],
    benchmark_profile: dict[str, Any],
) -> dict[str, Any]:
    predicted = np.asarray(
        [
            to_float((predicted_profile or {}).get("calories"), 0.0),
            to_float((predicted_profile or {}).get("protein"), 0.0),
            to_float((predicted_profile or {}).get("carbs"), 0.0),
            to_float((predicted_profile or {}).get("fats"), 0.0),
        ],
        dtype=float,
    )
    actual = np.asarray(
        [
            to_float((benchmark_profile or {}).get("calories"), 0.0),
            to_float((benchmark_profile or {}).get("protein"), 0.0),
            to_float((benchmark_profile or {}).get("carbs"), 0.0),
            to_float((benchmark_profile or {}).get("fats"), 0.0),
        ],
        dtype=float,
    )

    l2_distance = float(np.linalg.norm(predicted - actual))
    actual_norm = max(float(np.linalg.norm(actual)), 1.0)
    normalized_l2_distance = l2_distance / actual_norm
    cosine_distance = 1.0 - max(-1.0, min(1.0, cosine_similarity_safe(predicted, actual)))

    return {
        "l2_distance": round(l2_distance, 4),
        "normalized_l2_distance": round(float(normalized_l2_distance), 4),
        "cosine_distance": round(float(cosine_distance), 4),
    }


def compute_topk_retrieval_metrics(
    ranked_relevances: list[float],
    top_k_values: tuple[int, ...] = BENCHMARK_TOP_K_VALUES,
    relevance_threshold: float = 2.0,
) -> dict[str, Any]:
    values = [max(0.0, to_float(value, 0.0)) for value in (ranked_relevances or [])]
    relevant_total = sum(1 for value in values if value >= float(relevance_threshold))

    hit_rate_at_k: dict[str, float] = {}
    recall_at_k: dict[str, float] = {}
    ndcg_at_k: dict[str, float] = {}

    for top_k in sorted({int(value) for value in top_k_values if int(value) > 0}):
        top_slice = values[:top_k]
        hits = sum(1 for value in top_slice if value >= float(relevance_threshold))
        dcg = 0.0
        for index, value in enumerate(top_slice):
            dcg += ((2.0 ** float(value)) - 1.0) / math.log2(index + 2.0)

        ideal_slice = sorted(values, reverse=True)[:top_k]
        ideal_dcg = 0.0
        for index, value in enumerate(ideal_slice):
            ideal_dcg += ((2.0 ** float(value)) - 1.0) / math.log2(index + 2.0)

        hit_rate_at_k[str(top_k)] = round(1.0 if hits > 0 else 0.0, 4)
        recall_at_k[str(top_k)] = round(hits / max(1, relevant_total), 4) if relevant_total else 0.0
        ndcg_at_k[str(top_k)] = round(dcg / ideal_dcg, 4) if ideal_dcg > 0 else 0.0

    return {
        "relevant_candidate_count": int(relevant_total),
        "candidate_window": int(len(values)),
        "hit_rate_at_k": hit_rate_at_k,
        "recall_at_k": recall_at_k,
        "ndcg_at_k": ndcg_at_k,
    }


def compute_category_agreement_metrics(
    predicted_category: str,
    benchmark_category: str,
    predicted_meal_slot: str,
    benchmark_meal_slot: str,
    predicted_major_food_group: str | None = None,
    benchmark_major_food_group: str | None = None,
) -> dict[str, Any]:
    normalized_predicted_category = canonical_title_key(predicted_category)
    normalized_benchmark_category = canonical_title_key(benchmark_category)
    normalized_predicted_meal_slot = canonical_title_key(predicted_meal_slot)
    normalized_benchmark_meal_slot = canonical_title_key(benchmark_meal_slot)
    normalized_predicted_group = canonical_title_key(predicted_major_food_group)
    normalized_benchmark_group = canonical_title_key(benchmark_major_food_group)

    return {
        "recipe_category_match": 1.0
        if normalized_predicted_category and normalized_predicted_category == normalized_benchmark_category
        else 0.0,
        "meal_slot_match": 1.0
        if normalized_predicted_meal_slot and normalized_predicted_meal_slot == normalized_benchmark_meal_slot
        else 0.0,
        "major_food_group_match": 1.0
        if normalized_predicted_group and normalized_predicted_group == normalized_benchmark_group
        else 0.0,
    }


def compute_coverage_metrics(
    acceptable_matches: list[bool],
    top_k_values: tuple[int, ...] = BENCHMARK_TOP_K_VALUES,
    candidate_count: int = 0,
) -> dict[str, Any]:
    flags = [bool(value) for value in (acceptable_matches or [])]
    acceptable_at_k: dict[str, float] = {}

    for top_k in sorted({int(value) for value in top_k_values if int(value) > 0}):
        acceptable_at_k[str(top_k)] = round(1.0 if any(flags[:top_k]) else 0.0, 4)

    return {
        "candidate_returned": 1.0 if int(candidate_count) > 0 else 0.0,
        "acceptable_at_k": acceptable_at_k,
    }


def _aggregate_ausnut_scope(
    record_level_results: list[dict[str, Any]],
    top_k_values: tuple[int, ...],
) -> dict[str, Any]:
    if not record_level_results:
        return _safe_scope_report(top_k_values)

    nutrient_blocks = [
        (result.get("top_candidate_metrics") or {}).get("nutrient_profile_error") or {}
        for result in record_level_results
    ]
    macro_blocks = [
        (result.get("top_candidate_metrics") or {}).get("macro_vector_distance") or {}
        for result in record_level_results
    ]
    category_blocks = [
        (result.get("top_candidate_metrics") or {}).get("category_agreement") or {}
        for result in record_level_results
    ]
    title_overlaps = [
        to_float((result.get("top_candidate_metrics") or {}).get("title_token_overlap"), 0.0)
        for result in record_level_results
    ]
    failure_modes: dict[str, int] = {}
    for result in record_level_results:
        failure_mode = str(result.get("failure_mode") or "").strip()
        if not failure_mode:
            continue
        failure_modes[failure_mode] = failure_modes.get(failure_mode, 0) + 1

    report = _safe_scope_report(top_k_values)
    report["record_count"] = int(len(record_level_results))
    report["nutrient_profile_error"] = {
        "comparison_basis": "per100",
        "mean_relative_error": round(
            _average([to_float(block.get("mean_relative_error"), 0.0) for block in nutrient_blocks]),
            4,
        ),
        "median_relative_error": round(
            _median([to_float(block.get("mean_relative_error"), 0.0) for block in nutrient_blocks]),
            4,
        ),
        "field_relative_error": {
            nutrient: round(
                _average(
                    [
                        to_float((block.get("field_relative_error") or {}).get(nutrient), 0.0)
                        for block in nutrient_blocks
                    ]
                ),
                4,
            )
            for nutrient in BENCHMARK_NUTRIENTS
        },
    }
    report["macro_vector_distance"] = {
        "mean_normalized_l2_distance": round(
            _average([to_float(block.get("normalized_l2_distance"), 0.0) for block in macro_blocks]),
            4,
        ),
        "median_normalized_l2_distance": round(
            _median([to_float(block.get("normalized_l2_distance"), 0.0) for block in macro_blocks]),
            4,
        ),
        "mean_cosine_distance": round(
            _average([to_float(block.get("cosine_distance"), 0.0) for block in macro_blocks]),
            4,
        ),
    }
    report["retrieval_metrics"] = {
        "hit_rate_at_k": {
            str(top_k): round(
                _average(
                    [
                        to_float(((result.get("retrieval_metrics") or {}).get("hit_rate_at_k") or {}).get(str(top_k)), 0.0)
                        for result in record_level_results
                    ]
                ),
                4,
            )
            for top_k in top_k_values
        },
        "recall_at_k": {
            str(top_k): round(
                _average(
                    [
                        to_float(((result.get("retrieval_metrics") or {}).get("recall_at_k") or {}).get(str(top_k)), 0.0)
                        for result in record_level_results
                    ]
                ),
                4,
            )
            for top_k in top_k_values
        },
        "ndcg_at_k": {
            str(top_k): round(
                _average(
                    [
                        to_float(((result.get("retrieval_metrics") or {}).get("ndcg_at_k") or {}).get(str(top_k)), 0.0)
                        for result in record_level_results
                    ]
                ),
                4,
            )
            for top_k in top_k_values
        },
    }
    report["category_agreement"] = {
        "recipe_category_accuracy": round(
            _average([to_float(block.get("recipe_category_match"), 0.0) for block in category_blocks]),
            4,
        ),
        "meal_slot_accuracy": round(
            _average([to_float(block.get("meal_slot_match"), 0.0) for block in category_blocks]),
            4,
        ),
        "major_food_group_accuracy": round(
            _average([to_float(block.get("major_food_group_match"), 0.0) for block in category_blocks]),
            4,
        ),
    }
    report["coverage"] = {
        "candidate_return_rate": round(
            _average([to_float((result.get("coverage_metrics") or {}).get("candidate_returned"), 0.0) for result in record_level_results]),
            4,
        ),
        "acceptable_at_k": {
            str(top_k): round(
                _average(
                    [
                        to_float(((result.get("coverage_metrics") or {}).get("acceptable_at_k") or {}).get(str(top_k)), 0.0)
                        for result in record_level_results
                    ]
                ),
                4,
            )
            for top_k in top_k_values
        },
    }
    report["title_alignment"] = {
        "mean_token_overlap": round(_average(title_overlaps), 4),
        "median_token_overlap": round(_median(title_overlaps), 4),
    }
    report["failure_modes"] = dict(sorted(failure_modes.items(), key=lambda item: (-item[1], item[0])))
    return report


def build_ausnut_benchmark_report(
    record_level_results: list[dict[str, Any]],
    split_name: str,
    benchmark_version: str,
    acceptance_thresholds: dict[str, Any] | None = None,
    top_k_values: tuple[int, ...] = BENCHMARK_TOP_K_VALUES,
) -> dict[str, Any]:
    normalized_top_k_values = tuple(sorted({int(value) for value in top_k_values if int(value) > 0}))
    all_results = list(record_level_results or [])
    primary_results = [result for result in all_results if str(result.get("benchmark_tier") or "primary") == "primary"]
    supplementary_results = [result for result in all_results if str(result.get("benchmark_tier") or "primary") != "primary"]

    by_meal_slot: dict[str, Any] = {}
    for slot in sorted({str(result.get("report_meal_slot") or "general") for result in all_results}):
        scoped = [result for result in all_results if str(result.get("report_meal_slot") or "general") == slot]
        by_meal_slot[slot] = _aggregate_ausnut_scope(scoped, normalized_top_k_values)

    by_major_food_group: dict[str, Any] = {}
    for food_group in sorted({str(result.get("major_food_group") or "other") for result in all_results}):
        scoped = [result for result in all_results if str(result.get("major_food_group") or "other") == food_group]
        by_major_food_group[food_group] = _aggregate_ausnut_scope(scoped, normalized_top_k_values)

    return {
        "report_template_version": BENCHMARK_REPORT_TEMPLATE_VERSION,
        "split": str(split_name),
        "benchmark_version": str(benchmark_version),
        "summary": {
            "record_count": int(len(all_results)),
            "primary_record_count": int(len(primary_results)),
            "supplementary_record_count": int(len(supplementary_results)),
        },
        "acceptance_thresholds": acceptance_thresholds or {},
        "overall": _aggregate_ausnut_scope(all_results, normalized_top_k_values),
        "overall_primary": _aggregate_ausnut_scope(primary_results, normalized_top_k_values),
        "overall_supplementary": _aggregate_ausnut_scope(supplementary_results, normalized_top_k_values),
        "by_meal_slot": by_meal_slot,
        "by_major_food_group": by_major_food_group,
    }


def merge_diversity_dashboards(dashboards: list[dict[str, Any]]) -> dict[str, Any]:
    valid_dashboards = [dashboard for dashboard in dashboards if isinstance(dashboard, dict)]
    if not valid_dashboards:
        return compute_diversity_dashboard([], [])

    total_items = sum(int(dashboard.get("total_items", 0)) for dashboard in valid_dashboards)
    total_combos = sum(int(dashboard.get("total_combos", 0)) for dashboard in valid_dashboards)
    unique_title_count = sum(int(dashboard.get("unique_title_count", 0)) for dashboard in valid_dashboards)

    return {
        "total_items": total_items,
        "total_combos": total_combos,
        "unique_title_count": unique_title_count,
        "unique_title_ratio": round(_average([to_float(dashboard.get("unique_title_ratio"), 0.0) for dashboard in valid_dashboards]), 4),
        "repeated_side_rate": round(_average([to_float(dashboard.get("repeated_side_rate"), 0.0) for dashboard in valid_dashboards]), 4),
        "repeated_drink_rate": round(_average([to_float(dashboard.get("repeated_drink_rate"), 0.0) for dashboard in valid_dashboards]), 4),
        "image_hit_rate": round(_average([to_float(dashboard.get("image_hit_rate"), 0.0) for dashboard in valid_dashboards]), 4),
    }


def merge_combo_diagnostics(diagnostics_list: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [diagnostics for diagnostics in diagnostics_list if isinstance(diagnostics, dict)]
    if not valid:
        return compute_combo_diagnostic_metrics([], 0.0)

    return {
        "combo_count": int(sum(int(item.get("combo_count", 0)) for item in valid)),
        "mean_combo_calorie_gap_ratio": round(_average([to_float(item.get("mean_combo_calorie_gap_ratio"), 0.0) for item in valid]), 4),
        "within_10pct_target_combo_rate": round(_average([to_float(item.get("within_10pct_target_combo_rate"), 0.0) for item in valid]), 4),
        "mean_items_per_combo": round(_average([to_float(item.get("mean_items_per_combo"), 0.0) for item in valid]), 4),
        "role_coverage_rate": round(_average([to_float(item.get("role_coverage_rate"), 0.0) for item in valid]), 4),
        "duplicate_title_within_combo_rate": round(_average([to_float(item.get("duplicate_title_within_combo_rate"), 0.0) for item in valid]), 4),
    }


def merge_mapping_diagnostics(diagnostics_list: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [diagnostics for diagnostics in diagnostics_list if isinstance(diagnostics, dict)]
    if not valid:
        return compute_mapping_diagnostics([], [])

    total_items = sum(int(item.get("total_items", 0)) for item in valid)
    return {
        "total_items": int(total_items),
        "mapped_item_rate": round(_average([to_float(item.get("mapped_item_rate"), 0.0) for item in valid]), 4),
        "local_only_item_rate": round(_average([to_float(item.get("local_only_item_rate"), 0.0) for item in valid]), 4),
        "strict_mapping_rate": round(_average([to_float(item.get("strict_mapping_rate"), 0.0) for item in valid]), 4),
        "relaxed_mapping_rate": round(_average([to_float(item.get("relaxed_mapping_rate"), 0.0) for item in valid]), 4),
        "unresolved_identity_rate": round(_average([to_float(item.get("unresolved_identity_rate"), 0.0) for item in valid]), 4),
        "fatsecret_id_rate": round(_average([to_float(item.get("fatsecret_id_rate"), 0.0) for item in valid]), 4),
        "mean_mapping_calorie_gap_ratio": round(_average([to_float(item.get("mean_mapping_calorie_gap_ratio"), 0.0) for item in valid]), 4),
    }


def merge_timing_metrics(timings_list: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [timing for timing in timings_list if isinstance(timing, dict)]
    if not valid:
        return {
            "slot_count": 0,
            "total_slot_ms": 0.0,
            "mean_slot_ms": 0.0,
            "profile_ms": 0.0,
            "retrieval_ms": 0.0,
            "mapping_ms": 0.0,
            "ranking_ms": 0.0,
            "combo_assembly_ms": 0.0,
            "combo_pool_build_ms": 0.0,
            "combo_candidate_generation_ms": 0.0,
            "combo_diversity_ms": 0.0,
            "metrics_ms": 0.0,
        }

    total_slot_ms = sum(to_float(item.get("slot_total_ms"), 0.0) for item in valid)
    return {
        "slot_count": int(len(valid)),
        "total_slot_ms": round(total_slot_ms, 1),
        "mean_slot_ms": round(total_slot_ms / max(1, len(valid)), 1),
        "profile_ms": round(sum(to_float(item.get("profile_ms"), 0.0) for item in valid), 1),
        "retrieval_ms": round(sum(to_float(item.get("retrieval_ms"), 0.0) for item in valid), 1),
        "mapping_ms": round(sum(to_float(item.get("mapping_ms"), 0.0) for item in valid), 1),
        "ranking_ms": round(sum(to_float(item.get("ranking_ms"), 0.0) for item in valid), 1),
        "combo_assembly_ms": round(sum(to_float(item.get("combo_assembly_ms"), 0.0) for item in valid), 1),
        "combo_pool_build_ms": round(sum(to_float(item.get("combo_pool_build_ms"), 0.0) for item in valid), 1),
        "combo_candidate_generation_ms": round(sum(to_float(item.get("combo_candidate_generation_ms"), 0.0) for item in valid), 1),
        "combo_diversity_ms": round(sum(to_float(item.get("combo_diversity_ms"), 0.0) for item in valid), 1),
        "metrics_ms": round(sum(to_float(item.get("metrics_ms"), 0.0) for item in valid), 1),
    }


def build_model_metrics(
    proxy_metrics: dict[str, Any],
    offline_diagnostics: dict[str, Any],
    combo_diagnostics: dict[str, Any],
    mapping_diagnostics: dict[str, Any],
    diversity_dashboard: dict[str, Any],
    retrieval_metrics: dict[str, Any],
    timing: dict[str, Any],
    tuning: dict[str, Any],
    experimentation: dict[str, Any],
) -> dict[str, Any]:
    return {
        "metric_metadata": build_runtime_slot_metric_metadata(),
        "proxy_metrics": proxy_metrics,
        "offline_diagnostics": offline_diagnostics,
        "combo_diagnostics": combo_diagnostics,
        "mapping_diagnostics": mapping_diagnostics,
        "diversity_dashboard": diversity_dashboard,
        "retrieval_metrics": retrieval_metrics,
        "timing": timing,
        "tuning": tuning,
        "experimentation": experimentation,
    }


def print_metrics(slot: str, metrics: dict[str, Any]) -> None:
    proxy = metrics.get("proxy_metrics") or {}
    diagnostics = metrics.get("offline_diagnostics") or {}
    combo = metrics.get("combo_diagnostics") or {}
    mapping = metrics.get("mapping_diagnostics") or {}
    diversity = metrics.get("diversity_dashboard") or {}
    timing = metrics.get("timing") or {}
    tuning = metrics.get("tuning") or {}
    precision_at_k = proxy.get("precision_at_k") or {}
    k_text = " ".join(f"proxy.p@{k}={precision_at_k.get(str(k), 0.0):.4f}" for k in EVAL_K_VALUES)
    print(
        "[Runtime Slot Metrics]"
        f" slot={slot}"
        f" proxy.title_acc={proxy.get('accuracy', 0.0):.4f}"
        f" proxy.title_prec={proxy.get('precision', 0.0):.4f}"
        f" proxy.title_rec={proxy.get('recall', 0.0):.4f}"
        f" proxy.title_f1={proxy.get('f1_score', 0.0):.4f}"
        f" item.cal_gap={diagnostics.get('mean_calorie_gap_ratio', 0.0):.4f}"
        f" item.health={diagnostics.get('mean_health_score', 0.0):.4f}"
        f" combo.count={int(combo.get('combo_count', 0))}"
        f" combo.cal_gap={combo.get('mean_combo_calorie_gap_ratio', 0.0):.4f}"
        f" combo.coverage={combo.get('role_coverage_rate', 0.0):.4f}"
        f" map.mapped={mapping.get('mapped_item_rate', 0.0):.4f}"
        f" map.local_only={mapping.get('local_only_item_rate', 0.0):.4f}"
        f" map.relaxed={mapping.get('relaxed_mapping_rate', 0.0):.4f}"
        f" map.unresolved={mapping.get('unresolved_identity_rate', 0.0):.4f}"
        f" diversity.unique_titles={diversity.get('unique_title_ratio', 0.0):.4f}"
        f" diversity.side_repeat={diversity.get('repeated_side_rate', 0.0):.4f}"
        f" diversity.image_hit={diversity.get('image_hit_rate', 0.0):.4f}"
        f" time.slot_ms={to_float(timing.get('slot_total_ms'), 0.0):.1f}"
        f" time.profile_ms={to_float(timing.get('profile_ms'), 0.0):.1f}"
        f" time.retrieval_ms={to_float(timing.get('retrieval_ms'), 0.0):.1f}"
        f" time.map_ms={to_float(timing.get('mapping_ms'), 0.0):.1f}"
        f" time.rank_ms={to_float(timing.get('ranking_ms'), 0.0):.1f}"
        f" time.combo_ms={to_float(timing.get('combo_assembly_ms'), 0.0):.1f}"
        f" time.metrics_ms={to_float(timing.get('metrics_ms'), 0.0):.1f}"
        f" adaptive.mmr={to_float(tuning.get('mmr_lambda'), 0.0):.4f}"
        f" adaptive.reuse={to_float(tuning.get('combo_reuse_penalty_base'), 0.0):.4f}"
        f" {k_text}"
    )