from __future__ import annotations

import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from queue import Empty, Full, Queue
from threading import Event, Lock, Thread, current_thread
from typing import Any

import numpy as np

from .constants import (
    ASYNC_MAPPING_MISS_BACKOFF_SECONDS,
    ASYNC_MAPPING_MISS_MAX_RETRIES,
    ASYNC_MAPPING_ENABLED,
    ASYNC_MAPPING_PREFETCH_PER_SLOT,
    ASYNC_MAPPING_QUEUE_SIZE,
    ASYNC_MAPPING_WORKER_COUNT,
    CALORIE_MISMATCH_THRESHOLD,
    COMBO_CATEGORY_TARGETS,
    COMBO_DRINK_KEYWORDS,
    COMBO_MMR_LAMBDA_MAX,
    COMBO_MMR_LAMBDA_MIN,
    COMBO_REUSE_PENALTY_BASE,
    COMBO_REUSE_PENALTY_MAX,
    COMBO_REUSE_PENALTY_MIN,
    COMBO_SIDE_KEYWORDS,
    COMBO_TARGET_TOLERANCE,
    CONSUMED_IMAGE_USE_DETAIL_LOOKUP,
    DEFAULT_MEAL_ALLOCATION,
    DEFAULT_SLOT_CONSUMED_LIMIT,
    DEFAULT_TOP_CONSUMED_LIMIT,
    EXPERIMENT_VARIANTS,
    LOCAL_CANDIDATE_POOL_PER_MEAL,
    LOCAL_CANDIDATE_POOL_EXPERIMENT_MAX,
    LOCAL_CANDIDATE_POOL_EXPERIMENT_MIN,
    LOCAL_PREFETCH_POOL,
    MAPPING_TITLE_SIMILARITY_FLOOR,
    MAX_MAPPING_LOOKUP_ATTEMPTS,
    MAX_DUPLICATES_PER_TITLE_PER_SLOT,
    MAX_NEW_MAPPING_LOOKUPS,
    MEAL_SLOTS,
    MIN_HEALTH_SCORE,
    MIN_FRONTEND_OPTIONS,
    OVERALL_CONSUMED_IMAGE_LOOKUPS,
    QUERY_EXPANSION_MAX_TOP_DISTANCE,
    QUERY_EXPANSION_MIN_CANDIDATES,
    QUERY_EXPANSION_MIN_UNIQUE_RATIO,
    QUERY_EXPANSION_QUERY_LIMIT,
    RECOMMENDED_ITEMS_PER_MEAL,
    RELAXED_MAPPING_CALORIE_THRESHOLD,
    SERVING_FIT_TOLERANCE,
    SUGAR_LIMIT_PER_MEAL,
    SLOT_CONSUMED_IMAGE_LOOKUPS,
    SYNC_MAPPING_LOOKUPS_PER_SLOT,
    STOCHASTIC_RANKING_STRENGTH,
    VALID_GOALS,
)
from .db import fetch_active_daily_goal, fetch_user_history_and_goal, fetch_user_meal_history
from .fatsecret import FatSecretClient, build_safety_candidates, extract_image, map_food_hit_to_candidate
from .food_mapping import FoodMappingStore
from .local_dataset import LocalFoodDataset, build_query_vector
from .metrics import (
    build_model_metrics,
    build_runtime_aggregate_metric_metadata,
    compute_candidate_pool_metrics,
    compute_combo_diagnostic_metrics,
    compute_diversity_dashboard,
    compute_mapping_diagnostics,
    compute_offline_diagnostic_metrics,
    compute_proxy_title_metrics,
    merge_combo_diagnostics,
    merge_diversity_dashboards,
    merge_mapping_diagnostics,
    merge_timing_metrics,
    print_metrics,
)
from .profile import (
    build_behavioral_insight,
    build_user_profile,
    compute_dynamic_meal_allocation,
    get_top_consumed_items,
    normalize_history_df,
    resolve_daily_calories,
    warm_profile_runtime,
)
from .ranking import (
    Retriever,
    drink_role_quality_multiplier,
    infer_combo_category,
    is_candidate_role_compatible,
    main_role_quality_multiplier,
    merge_candidates,
    rank_candidates,
    side_role_quality_multiplier,
)
from .utils import (
    build_display_title,
    canonical_title_key,
    canonicalize_title,
    dedupe_strings,
    is_placeholder_title,
    normalize_text,
    normalize_to_per100,
    parse_force_exploration,
    to_float,
    tokenize,
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_env_bool(value: Any, default: bool) -> bool:
    text = normalize_text(value)
    if not text:
        return bool(default)
    return text not in {"0", "false", "no", "off"}


def _elapsed_ms(started_at: float) -> float:
    return round((time.perf_counter() - started_at) * 1000.0, 1)


_AUSNUT_SUMMARY_CACHE: dict[str, Any] = {"path": "", "mtime": -1.0, "value": None}
_MAPPING_MEAL_TITLE_MARKERS = {
    "bagel",
    "burger",
    "burrito",
    "flatbread",
    "hoagie",
    "panini",
    "pita",
    "roll",
    "sandwich",
    "sub",
    "taco",
    "wrap",
}
_MAPPING_SUPPLEMENT_TITLE_MARKERS = (
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


def _default_ausnut_summary_path() -> str:
    return os.path.normpath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "dataset_process",
            "ausnut_benchmark",
            "latest_ausnut_summary.json",
        )
    )


def _default_runtime_slot_comparison_log_path() -> str:
    return os.path.normpath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "dataset_process",
            "ausnut_benchmark",
            "runtime_slot_comparisons.jsonl",
        )
    )


def _load_latest_ausnut_benchmark_summary() -> dict[str, Any]:
    summary_path = os.getenv("AUSNUT_BENCHMARK_SUMMARY_PATH", "").strip() or _default_ausnut_summary_path()
    normalized_path = os.path.normpath(summary_path)

    if not os.path.exists(normalized_path):
        return {
            "status": "not_run",
            "summary_path": normalized_path,
        }

    try:
        modified_at = float(os.path.getmtime(normalized_path))
    except OSError:
        return {
            "status": "not_run",
            "summary_path": normalized_path,
        }

    global _AUSNUT_SUMMARY_CACHE
    if (
        _AUSNUT_SUMMARY_CACHE.get("path") == normalized_path
        and float(_AUSNUT_SUMMARY_CACHE.get("mtime", -1.0)) == modified_at
        and isinstance(_AUSNUT_SUMMARY_CACHE.get("value"), dict)
    ):
        return _AUSNUT_SUMMARY_CACHE["value"]

    try:
        with open(normalized_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception as exc:
        return {
            "status": "error",
            "summary_path": normalized_path,
            "error": str(exc),
        }

    if not isinstance(payload, dict):
        payload = {
            "status": "error",
            "summary_path": normalized_path,
            "error": "Invalid AUSNUT benchmark summary payload.",
        }
    else:
        payload.setdefault("status", "available")
        payload.setdefault("summary_path", normalized_path)

    _AUSNUT_SUMMARY_CACHE = {
        "path": normalized_path,
        "mtime": modified_at,
        "value": payload,
    }
    return payload


# This service coordinates local retrieval, optional FatSecret mapping,
# ranking, and response shaping for the frontend.
class RecommendationService:
    def __init__(
        self,
        fs_client: FatSecretClient,
        local_dataset: LocalFoodDataset,
        mapping_store: FoodMappingStore,
        response_cache_seconds: int = 120,
        serving_fit_tolerance: float = SERVING_FIT_TOLERANCE,
        calorie_mismatch_threshold: float = CALORIE_MISMATCH_THRESHOLD,
        relaxed_mapping_calorie_threshold: float = RELAXED_MAPPING_CALORIE_THRESHOLD,
        mapping_title_similarity_floor: float = MAPPING_TITLE_SIMILARITY_FLOOR,
        max_new_mapping_lookups: int = MAX_NEW_MAPPING_LOOKUPS,
        max_mapping_lookup_attempts: int = MAX_MAPPING_LOOKUP_ATTEMPTS,
        sync_mapping_lookups_per_slot: int = SYNC_MAPPING_LOOKUPS_PER_SLOT,
        max_duplicates_per_title_per_slot: int = MAX_DUPLICATES_PER_TITLE_PER_SLOT,
        stochastic_ranking_strength: float = STOCHASTIC_RANKING_STRENGTH,
        async_mapping_enabled: bool = ASYNC_MAPPING_ENABLED,
        async_mapping_prefetch_per_slot: int = ASYNC_MAPPING_PREFETCH_PER_SLOT,
        async_mapping_worker_count: int = ASYNC_MAPPING_WORKER_COUNT,
        async_mapping_queue_size: int = ASYNC_MAPPING_QUEUE_SIZE,
        async_mapping_miss_max_retries: int = ASYNC_MAPPING_MISS_MAX_RETRIES,
        async_mapping_miss_backoff_seconds: int = ASYNC_MAPPING_MISS_BACKOFF_SECONDS,
        overall_consumed_image_lookups: int = OVERALL_CONSUMED_IMAGE_LOOKUPS,
        slot_consumed_image_lookups: int = SLOT_CONSUMED_IMAGE_LOOKUPS,
        consumed_image_use_detail_lookup: bool = CONSUMED_IMAGE_USE_DETAIL_LOOKUP,
        parallel_slot_execution_enabled: bool = False,
        parallel_primary_role_retrieval_enabled: bool = True,
    ):
        self.fs_client = fs_client
        self.local_dataset = local_dataset
        self.mapping_store = mapping_store
        self.retriever = Retriever()

        self.response_cache_seconds = max(0, int(response_cache_seconds))
        self.serving_fit_tolerance = max(0.01, float(serving_fit_tolerance))
        self.calorie_mismatch_threshold = max(0.05, float(calorie_mismatch_threshold))
        self.relaxed_mapping_calorie_threshold = max(
            self.calorie_mismatch_threshold,
            float(relaxed_mapping_calorie_threshold),
        )
        self.mapping_title_similarity_floor = float(np.clip(float(mapping_title_similarity_floor), 0.0, 1.0))
        self.max_new_mapping_lookups = max(0, int(max_new_mapping_lookups))
        self.max_mapping_lookup_attempts = max(0, int(max_mapping_lookup_attempts))
        self.sync_mapping_lookups_per_slot = max(0, int(sync_mapping_lookups_per_slot))
        self.max_duplicates_per_title_per_slot = max(1, int(max_duplicates_per_title_per_slot))
        self.stochastic_ranking_strength = float(max(0.0, stochastic_ranking_strength))
        self.async_mapping_enabled = bool(async_mapping_enabled)
        self.async_mapping_prefetch_per_slot = max(
            RECOMMENDED_ITEMS_PER_MEAL,
            int(async_mapping_prefetch_per_slot),
        )
        self.async_mapping_worker_count = max(1, int(async_mapping_worker_count))
        self.async_mapping_queue_size = max(10, int(async_mapping_queue_size))
        self.async_mapping_miss_max_retries = max(1, int(async_mapping_miss_max_retries))
        self.async_mapping_miss_backoff_seconds = max(60, int(async_mapping_miss_backoff_seconds))
        self.overall_consumed_image_lookups = max(0, int(overall_consumed_image_lookups))
        self.slot_consumed_image_lookups = max(0, int(slot_consumed_image_lookups))
        self.consumed_image_use_detail_lookup = bool(consumed_image_use_detail_lookup)
        self.parallel_slot_execution_enabled = bool(parallel_slot_execution_enabled)
        self.parallel_primary_role_retrieval_enabled = bool(parallel_primary_role_retrieval_enabled)
        self.has_fatsecret_credentials = bool(
            str(getattr(self.fs_client, "client_id", "")).strip()
            and str(getattr(self.fs_client, "client_secret", "")).strip()
        )

        self.response_cache: dict[str, tuple[float, dict[str, Any]]] = {}
        self._response_cache_lock = Lock()
        self._mapping_title_lookup_cache_lock = Lock()
        self._mapping_title_lookup_cache: dict[str, dict[str, Any] | bool] = {}
        self._mapping_title_lookup_cache_max_entries = 4096
        self._response_build_lock = Lock()
        self._response_build_events: dict[str, Event] = {}
        self._prime_response_warmup_lock = Lock()
        self._prime_response_warmup_inflight: set[str] = set()
        self.history_cache_seconds = max(0, int(os.getenv("HISTORY_CACHE_SECONDS", "30")))
        self.goal_cache_seconds = max(0, int(os.getenv("GOAL_CACHE_SECONDS", "300")))
        self.profile_cache_seconds = max(0, int(os.getenv("PROFILE_CACHE_SECONDS", "120")))
        self._history_cache: dict[str, tuple[float, Any]] = {}
        self._goal_cache: dict[str, tuple[float, float | None]] = {}
        self._profile_cache: dict[str, tuple[float, dict[str, Any]]] = {}
        self._async_queue: Queue[tuple[dict[str, Any], str]] | None = None
        if self.async_mapping_enabled and self.has_fatsecret_credentials:
            self._async_queue = Queue(maxsize=self.async_mapping_queue_size)
        self._async_worker_lock = Lock()
        self._async_inflight_lock = Lock()
        self._async_inflight_recipe_ids: set[str] = set()
        self._async_miss_lock = Lock()
        self._async_miss_state: dict[str, dict[str, float]] = {}
        self.runtime_comparison_log_enabled = _parse_env_bool(
            os.getenv("RUNTIME_COMPARISON_LOG_ENABLED", "1"),
            True,
        )
        self.runtime_comparison_log_path = os.path.normpath(
            os.getenv("RUNTIME_COMPARISON_LOG_PATH", "").strip() or _default_runtime_slot_comparison_log_path()
        )
        self._runtime_comparison_log_lock = Lock()
        self._async_workers: list[Thread] = []
        self._async_workers_started = False
        self._slot_build_max_workers = max(1, len(MEAL_SLOTS))
        self._primary_retrieval_max_workers = max(3, len(MEAL_SLOTS) * 3)
        self._slot_build_executor = ThreadPoolExecutor(
            max_workers=self._slot_build_max_workers,
            thread_name_prefix="slot-build",
        )
        self._primary_retrieval_executor = ThreadPoolExecutor(
            max_workers=self._primary_retrieval_max_workers,
            thread_name_prefix="primary-search",
        )

        print(
            "**** Mapping Policy:",
            f"strict_gap={self.calorie_mismatch_threshold:.3f}",
            f"relaxed_gap={self.relaxed_mapping_calorie_threshold:.3f}",
            f"title_floor={self.mapping_title_similarity_floor:.3f}",
            f"max_dup_title={self.max_duplicates_per_title_per_slot}",
            f"stochastic_strength={self.stochastic_ranking_strength:.4f}",
            f"sync_lookups_per_slot={self.sync_mapping_lookups_per_slot}",
            f"async_enabled={int(self.async_mapping_enabled and self._async_queue is not None)}",
            f"async_prefetch_per_slot={self.async_mapping_prefetch_per_slot}",
            f"async_miss_retries={self.async_mapping_miss_max_retries}",
            f"overall_image_lookups={self.overall_consumed_image_lookups}",
            f"slot_image_lookups={self.slot_consumed_image_lookups}",
            f"parallel_slots={int(self.parallel_slot_execution_enabled)}",
            f"parallel_primary_roles={int(self.parallel_primary_role_retrieval_enabled)}",
        )
        # NOTE: Start async mapping workers during initialization.
        self._ensure_async_workers_started()

    def warm_parallel_search_runtime(self) -> None:
        if not self.local_dataset.is_ready:
            return

        search_shape_warmup_enabled = _parse_env_bool(
            os.getenv("PRIMARY_SEARCH_SHAPE_WARMUP_ENABLED", "1"),
            True,
        )

        def _warm_executor_workers(executor: ThreadPoolExecutor, worker_count: int) -> int:
            if worker_count <= 0:
                return 0

            release_workers = Event()
            started_workers: Queue[str] = Queue()

            def _warm_worker() -> str:
                self.local_dataset.warmup(dedicated_connection=True)
                thread_name = current_thread().name
                started_workers.put(thread_name)
                release_workers.wait(timeout=2.0)
                return thread_name

            futures = [executor.submit(_warm_worker) for _ in range(worker_count)]
            seen_workers: set[str] = set()
            deadline = time.perf_counter() + 5.0
            while len(seen_workers) < worker_count:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                try:
                    seen_workers.add(started_workers.get(timeout=remaining))
                except Empty:
                    break

            release_workers.set()
            for future in futures:
                try:
                    seen_workers.add(future.result(timeout=5.0))
                except Exception:
                    continue
            return len(seen_workers)

        if self.parallel_slot_execution_enabled:
            slot_warmups = _warm_executor_workers(self._slot_build_executor, self._slot_build_max_workers)
        else:
            slot_warmups = 0

        if self.parallel_primary_role_retrieval_enabled:
            primary_warmups = _warm_executor_workers(
                self._primary_retrieval_executor,
                self._primary_retrieval_max_workers,
            )
        else:
            primary_warmups = 0

        search_shape_warmup_ms = 0.0
        if search_shape_warmup_enabled and self.parallel_primary_role_retrieval_enabled:
            search_shape_warmup_ms = self._warm_primary_search_shapes()

        if slot_warmups or primary_warmups:
            warmup_parts = [
                "**** Parallel Search Warmup:",
                f"slot_workers={slot_warmups}",
                f"primary_workers={primary_warmups}",
            ]
            if search_shape_warmup_ms > 0.0:
                warmup_parts.append(f"search_shape_ms={search_shape_warmup_ms:.1f}")
            print(*warmup_parts)

    def _warm_primary_search_shapes(self) -> float:
        started_at = time.perf_counter()
        reference_daily_calories = 2400.0
        candidate_pool_target = int(
            np.clip(
                round(LOCAL_CANDIDATE_POOL_PER_MEAL),
                LOCAL_CANDIDATE_POOL_EXPERIMENT_MIN,
                LOCAL_CANDIDATE_POOL_EXPERIMENT_MAX,
            )
        )
        prefetch_target = max(candidate_pool_target, min(LOCAL_PREFETCH_POOL, 120))
        release_workers = Event()
        started_workers: Queue[str] = Queue()
        search_plans = []

        for meal_type in MEAL_SLOTS:
            slot_target = int(round(reference_daily_calories * DEFAULT_MEAL_ALLOCATION[meal_type]))
            main_target = float(slot_target) * float(COMBO_CATEGORY_TARGETS.get("main", 0.65))
            side_target = float(slot_target) * float(COMBO_CATEGORY_TARGETS.get("side", 0.20))
            drink_target = float(slot_target) * float(COMBO_CATEGORY_TARGETS.get("drink", 0.15))

            main_query_vec = build_query_vector(slot_target=main_target, user_vec=None)
            side_query_vec = build_query_vector(slot_target=side_target, user_vec=None)
            drink_query_vec = build_query_vector(slot_target=drink_target, user_vec=None)
            side_query_str = "|".join([normalize_text(kw) for kw in self._side_query_keywords(meal_type)])
            drink_query_str = self._primary_drink_query_regex(meal_type)
            # NOTE: Warm the same dedicated primary-search shapes used by the cold all-slot route.
            primary_search_budgets = self._primary_search_budgets(meal_type, candidate_pool_target, prefetch_target)

            search_plans.append(
                {
                    "meal_type": meal_type,
                    "query_vector": main_query_vec,
                    "top_k": primary_search_budgets["main_top_k"],
                    "prefetch": primary_search_budgets["main_prefetch"],
                    "text_query": None,
                    "role_hint": "main",
                }
            )
            search_plans.append(
                {
                    "meal_type": meal_type,
                    "query_vector": side_query_vec,
                    "top_k": primary_search_budgets["side_top_k"],
                    "prefetch": primary_search_budgets["side_prefetch"],
                    "text_query": side_query_str,
                    "role_hint": "side",
                }
            )
            search_plans.append(
                {
                    "meal_type": meal_type,
                    "query_vector": drink_query_vec,
                    "top_k": primary_search_budgets["drink_top_k"],
                    "prefetch": primary_search_budgets["drink_prefetch"],
                    "text_query": drink_query_str,
                    "role_hint": "drink",
                }
            )

        def _warm_search_plan(plan: dict[str, Any]) -> str:
            self.local_dataset.search(
                meal_type=plan["meal_type"],
                query_vector=plan["query_vector"],
                top_k=plan["top_k"],
                prefetch=plan["prefetch"],
                is_australian_user=False,
                text_query=plan["text_query"],
                role_hint=plan["role_hint"],
                dedicated_connection=True,
                log_search=False,
            )
            thread_name = current_thread().name
            started_workers.put(thread_name)
            release_workers.wait(timeout=2.0)
            return thread_name

        warm_tasks = [
            self._primary_retrieval_executor.submit(_warm_search_plan, plan)
            for plan in search_plans
        ]
        seen_workers: set[str] = set()
        deadline = time.perf_counter() + 8.0
        while len(seen_workers) < len(search_plans):
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                break
            try:
                seen_workers.add(started_workers.get(timeout=remaining))
            except Empty:
                break

        release_workers.set()
        for warm_task in warm_tasks:
            try:
                seen_workers.add(warm_task.result(timeout=8.0))
            except Exception:
                continue

        return _elapsed_ms(started_at)

    @classmethod
    def from_env(cls) -> "RecommendationService":
        # Support both backend-style and Expo-style env names for shared deployments.
        client_id = (
            os.getenv("FATSECRET_CLIENT_ID")
            or os.getenv("EXPO_PUBLIC_FATSECRET_CLIENT_ID")
            or ""
        )
        client_secret = (
            os.getenv("FATSECRET_CLIENT_SECRET")
            or os.getenv("EXPO_PUBLIC_FATSECRET_CLIENT_SECRET")
            or ""
        )
        fs_client = FatSecretClient(
            client_id=client_id,
            client_secret=client_secret,
            cache_ttl_seconds=1800,
        )
        if not str(client_id).strip() or not str(client_secret).strip():
            pass

        dataset_path = os.getenv("LOCAL_FOOD_DATASET_PATH", "").strip() or None
        local_dataset = LocalFoodDataset(dataset_path=dataset_path)
        # NOTE: Warm-up DuckDB on app start to reduce first-request latency.
        local_dataset.warmup()
        # NOTE: Pay the one-time clustering startup cost before the first live request.
        warm_profile_runtime()

        mapping_file = (
            os.getenv("FOOD_MAPPING_FILE", "").strip()
            or os.path.join(os.path.dirname(__file__), "food_mappings.json")
        )
        mapping_store = FoodMappingStore(mapping_file_path=mapping_file)

        cache_ttl = int(os.getenv("ML_RESPONSE_CACHE_SECONDS", "120"))
        serving_fit_tolerance = float(os.getenv("SERVING_FIT_TOLERANCE", str(SERVING_FIT_TOLERANCE)))
        calorie_mismatch_threshold = float(
            os.getenv("CALORIE_MISMATCH_THRESHOLD", str(CALORIE_MISMATCH_THRESHOLD))
        )
        relaxed_mapping_calorie_threshold = float(
            os.getenv("RELAXED_MAPPING_CALORIE_THRESHOLD", str(RELAXED_MAPPING_CALORIE_THRESHOLD))
        )
        mapping_title_similarity_floor = float(
            os.getenv("MAPPING_TITLE_SIMILARITY_FLOOR", str(MAPPING_TITLE_SIMILARITY_FLOOR))
        )
        max_new_mapping_lookups = int(
            os.getenv("FATSECRET_MAX_NEW_MAPPING_LOOKUPS", str(MAX_NEW_MAPPING_LOOKUPS))
        )
        max_mapping_lookup_attempts = int(
            os.getenv("FATSECRET_MAX_MAPPING_LOOKUP_ATTEMPTS", str(MAX_MAPPING_LOOKUP_ATTEMPTS))
        )
        sync_mapping_lookups_per_slot = int(
            os.getenv("SYNC_MAPPING_LOOKUPS_PER_SLOT", str(SYNC_MAPPING_LOOKUPS_PER_SLOT))
        )
        max_duplicates_per_title_per_slot = int(
            os.getenv("MAX_DUPLICATES_PER_TITLE_PER_SLOT", str(MAX_DUPLICATES_PER_TITLE_PER_SLOT))
        )
        stochastic_ranking_strength = float(
            os.getenv("STOCHASTIC_RANKING_STRENGTH", str(STOCHASTIC_RANKING_STRENGTH))
        )
        async_mapping_enabled = _parse_env_bool(
            os.getenv("ASYNC_MAPPING_ENABLED", str(int(ASYNC_MAPPING_ENABLED))),
            ASYNC_MAPPING_ENABLED,
        )
        async_mapping_prefetch_per_slot = int(
            os.getenv("ASYNC_MAPPING_PREFETCH_PER_SLOT", str(ASYNC_MAPPING_PREFETCH_PER_SLOT))
        )
        async_mapping_worker_count = int(
            os.getenv("ASYNC_MAPPING_WORKER_COUNT", str(ASYNC_MAPPING_WORKER_COUNT))
        )
        async_mapping_queue_size = int(
            os.getenv("ASYNC_MAPPING_QUEUE_SIZE", str(ASYNC_MAPPING_QUEUE_SIZE))
        )
        async_mapping_miss_max_retries = int(
            os.getenv("ASYNC_MAPPING_MISS_MAX_RETRIES", str(ASYNC_MAPPING_MISS_MAX_RETRIES))
        )
        async_mapping_miss_backoff_seconds = int(
            os.getenv("ASYNC_MAPPING_MISS_BACKOFF_SECONDS", str(ASYNC_MAPPING_MISS_BACKOFF_SECONDS))
        )
        overall_consumed_image_lookups = int(
            os.getenv("OVERALL_CONSUMED_IMAGE_LOOKUPS", str(OVERALL_CONSUMED_IMAGE_LOOKUPS))
        )
        slot_consumed_image_lookups = int(
            os.getenv("SLOT_CONSUMED_IMAGE_LOOKUPS", str(SLOT_CONSUMED_IMAGE_LOOKUPS))
        )
        consumed_image_use_detail_lookup = _parse_env_bool(
            os.getenv("CONSUMED_IMAGE_USE_DETAIL_LOOKUP", str(int(CONSUMED_IMAGE_USE_DETAIL_LOOKUP))),
            CONSUMED_IMAGE_USE_DETAIL_LOOKUP,
        )
        parallel_slot_execution_enabled = _parse_env_bool(
            os.getenv("PARALLEL_SLOT_EXECUTION_ENABLED", "1"),
            True,
        )
        parallel_primary_role_retrieval_enabled = _parse_env_bool(
            os.getenv("PARALLEL_PRIMARY_ROLE_RETRIEVAL_ENABLED", "1"),
            True,
        )

        service = cls(
            fs_client=fs_client,
            local_dataset=local_dataset,
            mapping_store=mapping_store,
            response_cache_seconds=cache_ttl,
            serving_fit_tolerance=serving_fit_tolerance,
            calorie_mismatch_threshold=calorie_mismatch_threshold,
            relaxed_mapping_calorie_threshold=relaxed_mapping_calorie_threshold,
            mapping_title_similarity_floor=mapping_title_similarity_floor,
            max_new_mapping_lookups=max_new_mapping_lookups,
            max_mapping_lookup_attempts=max_mapping_lookup_attempts,
            sync_mapping_lookups_per_slot=sync_mapping_lookups_per_slot,
            max_duplicates_per_title_per_slot=max_duplicates_per_title_per_slot,
            stochastic_ranking_strength=stochastic_ranking_strength,
            async_mapping_enabled=async_mapping_enabled,
            async_mapping_prefetch_per_slot=async_mapping_prefetch_per_slot,
            async_mapping_worker_count=async_mapping_worker_count,
            async_mapping_queue_size=async_mapping_queue_size,
            async_mapping_miss_max_retries=async_mapping_miss_max_retries,
            async_mapping_miss_backoff_seconds=async_mapping_miss_backoff_seconds,
            overall_consumed_image_lookups=overall_consumed_image_lookups,
            slot_consumed_image_lookups=slot_consumed_image_lookups,
            consumed_image_use_detail_lookup=consumed_image_use_detail_lookup,
            parallel_slot_execution_enabled=parallel_slot_execution_enabled,
            parallel_primary_role_retrieval_enabled=parallel_primary_role_retrieval_enabled,
        )
        service.warm_parallel_search_runtime()
        return service

    def _response_cache_get(self, key: str) -> dict[str, Any] | None:
        with self._response_cache_lock:
            payload = self.response_cache.get(key)
            if not payload:
                return None
            expires_at, value = payload
            if expires_at <= time.time():
                self.response_cache.pop(key, None)
                return None
            return value

    def _response_cache_set(self, key: str, value: dict[str, Any]) -> dict[str, Any]:
        if self.response_cache_seconds <= 0:
            return value
        with self._response_cache_lock:
            self.response_cache[key] = (time.time() + self.response_cache_seconds, value)
        return value

    def _response_cache_clear(self) -> None:
        with self._response_cache_lock:
            self.response_cache.clear()

    def _clear_mapping_title_lookup_cache(self) -> None:
        with self._mapping_title_lookup_cache_lock:
            self._mapping_title_lookup_cache.clear()

    def _mapping_title_lookup_cache_get(self, key: str) -> dict[str, Any] | bool | None:
        if not key:
            return None
        with self._mapping_title_lookup_cache_lock:
            return self._mapping_title_lookup_cache.get(key)

    def _mapping_title_lookup_cache_set(self, key: str, value: dict[str, Any] | bool) -> None:
        if not key:
            return
        with self._mapping_title_lookup_cache_lock:
            # NOTE: Keep the shared mapping-title cache bounded so cold-path reuse stays cheap.
            if key not in self._mapping_title_lookup_cache and len(self._mapping_title_lookup_cache) >= self._mapping_title_lookup_cache_max_entries:
                self._mapping_title_lookup_cache.clear()
            self._mapping_title_lookup_cache[key] = value

    def _claim_response_build(self, key: str) -> tuple[bool, Event]:
        with self._response_build_lock:
            build_event = self._response_build_events.get(key)
            if build_event is None:
                build_event = Event()
                self._response_build_events[key] = build_event
                return True, build_event
            return False, build_event

    def _release_response_build(self, key: str, build_event: Event) -> None:
        with self._response_build_lock:
            current_event = self._response_build_events.get(key)
            if current_event is build_event:
                self._response_build_events.pop(key, None)
        build_event.set()

    def _get_response_build_event(self, key: str) -> Event | None:
        with self._response_build_lock:
            return self._response_build_events.get(key)

    def _build_prime_response_warmup_context(self, data: dict[str, Any]) -> dict[str, Any]:
        payload = data or {}
        meal_type = self._normalize_meal_type(payload.get("mealType") or payload.get("slot") or "all")
        feedback = payload.get("feedback") if isinstance(payload.get("feedback"), dict) else {}
        favorite_titles = payload.get("favorite_titles")
        if favorite_titles is None:
            favorite_titles = payload.get("favoriteTitles")
        if isinstance(favorite_titles, (list, tuple, set)):
            normalized_favorites = [value for value in favorite_titles if value]
        elif favorite_titles:
            normalized_favorites = [favorite_titles]
        else:
            normalized_favorites = []

        warm_payload = {
            "userId": payload.get("userId"),
            "mealType": meal_type,
            "calorieTarget": payload.get("calorieTarget"),
            "force_exploration": False,
            "demographics": payload.get("demographics") or {},
            "feedback": feedback,
            "favorite_titles": normalized_favorites,
        }
        experiment_variant = payload.get("experiment_variant") or payload.get("experimentVariant")
        if experiment_variant:
            warm_payload["experiment_variant"] = experiment_variant

        goal = self._normalize_goal((warm_payload.get("demographics") or {}).get("goal") or "maintain")
        cache_key = self._build_cache_key(warm_payload, goal)
        return {
            "cache_key": cache_key,
            "meal_type": meal_type,
            "warm_payload": warm_payload,
        }

    def prime_user_context(self, data: dict[str, Any]) -> dict[str, Any]:
        """
        Prime caches on login:
        - warm DuckDB connection,
        - refresh FatSecret OAuth token,
        - prefetch user history and profile vectors.
        """
        payload = data or {}
        user_id = payload.get("userId")
        raw_wait_for_warmup = payload.get("wait_for_warmup")
        if raw_wait_for_warmup is None:
            raw_wait_for_warmup = payload.get("waitForWarmup")
        if isinstance(raw_wait_for_warmup, bool):
            wait_for_warmup = raw_wait_for_warmup
        else:
            wait_for_warmup = normalize_text(raw_wait_for_warmup) in {"1", "true", "yes", "on", "y"}
        raw_wait_timeout_ms = payload.get("wait_timeout_ms")
        if raw_wait_timeout_ms is None:
            raw_wait_timeout_ms = payload.get("waitTimeoutMs")
        try:
            wait_timeout_ms = max(0, int(float(raw_wait_timeout_ms or 0)))
        except (TypeError, ValueError):
            wait_timeout_ms = 0

        self.local_dataset.warmup()
        if self.has_fatsecret_credentials:
            self.fs_client.prime_token()

        primed_history = False
        profile_counts: dict[str, int] = {}
        recommendation_warmup = {
            "queued": False,
            "reason": "missing_user_id",
        }
        if user_id:
            history, cached_goal = fetch_user_history_and_goal(user_id)
            self._history_cache_set(user_id, history)
            history_df = normalize_history_df(history)
            for meal_type in MEAL_SLOTS:
                profile = build_user_profile(history_df, meal_type=meal_type)
                self._profile_cache_set(user_id, meal_type, profile)
                profile_counts[meal_type] = len(profile.get("top_foods", []) or [])
            primed_history = True

            self._goal_cache_set(user_id, cached_goal)

            recommendation_warmup = self._queue_prime_response_warmup(
                {
                    "userId": user_id,
                    "mealType": payload.get("mealType") or payload.get("slot") or "all",
                    "calorieTarget": payload.get("calorieTarget"),
                    "demographics": payload.get("demographics") or {},
                    "feedback": payload.get("feedback") or {},
                    "favorite_titles": payload.get("favorite_titles") or payload.get("favoriteTitles") or [],
                    "experiment_variant": payload.get("experiment_variant") or payload.get("experimentVariant"),
                }
            )
            if wait_for_warmup:
                recommendation_warmup = {
                    **recommendation_warmup,
                    **self.wait_for_prime_response_warmup(payload, timeout_ms=wait_timeout_ms),
                }
            else:
                recommendation_warmup["waited"] = False
                recommendation_warmup["wait_timeout_ms"] = wait_timeout_ms
                recommendation_warmup["wait_timed_out"] = False
                recommendation_warmup["waited_ms"] = 0

        return {
            "primed_history": primed_history,
            "profile_counts": profile_counts,
            "recommendation_warmup": recommendation_warmup,
        }

    def _queue_prime_response_warmup(self, data: dict[str, Any]) -> dict[str, Any]:
        if self.response_cache_seconds <= 0:
            return {
                "queued": False,
                "reason": "response_cache_disabled",
            }

        context = self._build_prime_response_warmup_context(data)
        warm_payload = context["warm_payload"]
        cache_key = context["cache_key"]
        meal_type = context["meal_type"]
        if self._response_cache_get(cache_key) is not None:
            return {
                "queued": False,
                "reason": "already_cached",
                "meal_type": meal_type,
            }

        with self._prime_response_warmup_lock:
            if cache_key in self._prime_response_warmup_inflight:
                return {
                    "queued": False,
                    "reason": "already_warming",
                    "meal_type": meal_type,
                }
            self._prime_response_warmup_inflight.add(cache_key)

        Thread(
            target=self._run_prime_response_warmup,
            args=(cache_key, warm_payload),
            daemon=True,
            name=f"prime-response-{meal_type or 'all'}",
        ).start()
        return {
            "queued": True,
            "reason": "queued",
            "meal_type": meal_type,
        }

    def get_prime_response_warmup_status(self, data: dict[str, Any]) -> dict[str, Any]:
        if self.response_cache_seconds <= 0:
            return {
                "warmed": False,
                "warming": False,
                "reason": "response_cache_disabled",
                "meal_type": self._normalize_meal_type((data or {}).get("mealType") or (data or {}).get("slot") or "all"),
            }

        context = self._build_prime_response_warmup_context(data)
        cache_key = context["cache_key"]
        warmed = self._response_cache_get(cache_key) is not None
        queued = False
        with self._prime_response_warmup_lock:
            queued = cache_key in self._prime_response_warmup_inflight
        build_event = self._get_response_build_event(cache_key)
        warming = bool((queued or build_event is not None) and not warmed)
        reason = "already_cached" if warmed else "warming" if warming else "not_started"
        return {
            "warmed": warmed,
            "warming": warming,
            "reason": reason,
            "meal_type": context["meal_type"],
        }

    def wait_for_prime_response_warmup(self, data: dict[str, Any], timeout_ms: int = 0) -> dict[str, Any]:
        timeout_ms = max(0, int(timeout_ms or 0))
        if self.response_cache_seconds <= 0:
            return {
                "warmed": False,
                "warming": False,
                "reason": "response_cache_disabled",
                "meal_type": self._normalize_meal_type((data or {}).get("mealType") or (data or {}).get("slot") or "all"),
                "waited": False,
                "wait_timeout_ms": timeout_ms,
                "wait_timed_out": False,
                "waited_ms": 0,
            }

        context = self._build_prime_response_warmup_context(data)
        cache_key = context["cache_key"]
        started_at = time.perf_counter()
        deadline = started_at + (timeout_ms / 1000.0 if timeout_ms > 0 else 0.0)

        while True:
            warmed = self._response_cache_get(cache_key) is not None
            if warmed:
                waited_ms = _elapsed_ms(started_at)
                return {
                    "warmed": True,
                    "warming": False,
                    "reason": "already_cached" if waited_ms <= 0 else "warmed",
                    "meal_type": context["meal_type"],
                    "waited": waited_ms > 0,
                    "wait_timeout_ms": timeout_ms,
                    "wait_timed_out": False,
                    "waited_ms": waited_ms,
                }

            build_event = self._get_response_build_event(cache_key)
            queued = False
            with self._prime_response_warmup_lock:
                queued = cache_key in self._prime_response_warmup_inflight
            warming = bool(build_event is not None or queued)
            if not warming:
                waited_ms = _elapsed_ms(started_at)
                return {
                    "warmed": False,
                    "warming": False,
                    "reason": "not_started",
                    "meal_type": context["meal_type"],
                    "waited": waited_ms > 0,
                    "wait_timeout_ms": timeout_ms,
                    "wait_timed_out": False,
                    "waited_ms": waited_ms,
                }

            if timeout_ms <= 0:
                waited_ms = _elapsed_ms(started_at)
                return {
                    "warmed": False,
                    "warming": True,
                    "reason": "warming",
                    "meal_type": context["meal_type"],
                    "waited": False,
                    "wait_timeout_ms": timeout_ms,
                    "wait_timed_out": False,
                    "waited_ms": waited_ms,
                }

            remaining_seconds = deadline - time.perf_counter()
            if remaining_seconds <= 0:
                waited_ms = _elapsed_ms(started_at)
                return {
                    "warmed": False,
                    "warming": True,
                    "reason": "wait_timeout",
                    "meal_type": context["meal_type"],
                    "waited": waited_ms > 0,
                    "wait_timeout_ms": timeout_ms,
                    "wait_timed_out": True,
                    "waited_ms": waited_ms,
                }

            if build_event is None:
                time.sleep(min(0.05, remaining_seconds))
            else:
                build_event.wait(timeout=min(0.25, remaining_seconds))

    def _run_prime_response_warmup(self, cache_key: str, payload: dict[str, Any]) -> None:
        user_id = payload.get("userId")
        meal_type = payload.get("mealType") or "all"
        try:
            self.recommend(payload)
            warmed = self._response_cache_get(cache_key) is not None
            print(
                f"**** Prime Response Warmup Completed: user_id={user_id} meal_type={meal_type} "
                f"warmed={int(warmed)}"
            )
        except Exception as exc:
            print(
                f"**** Prime Response Warmup Error: user_id={user_id} meal_type={meal_type} error={exc}"
            )
        finally:
            with self._prime_response_warmup_lock:
                self._prime_response_warmup_inflight.discard(cache_key)

    def _history_cache_get(self, user_id: Any) -> Any | None:
        if self.history_cache_seconds <= 0:
            return None
        key = str(user_id or "").strip()
        if not key:
            return None
        payload = self._history_cache.get(key)
        if not payload:
            return None
        expires_at, value = payload
        if expires_at <= time.time():
            self._history_cache.pop(key, None)
            return None
        return value

    def _history_cache_set(self, user_id: Any, value: Any) -> None:
        if self.history_cache_seconds <= 0:
            return
        key = str(user_id or "").strip()
        if not key:
            return
        self._history_cache[key] = (time.time() + self.history_cache_seconds, value)

    def _goal_cache_get(self, user_id: Any) -> float | None:
        if self.goal_cache_seconds <= 0:
            return None
        key = str(user_id or "").strip()
        if not key:
            return None
        payload = self._goal_cache.get(key)
        if not payload:
            return None
        expires_at, value = payload
        if expires_at <= time.time():
            self._goal_cache.pop(key, None)
            return None
        return value

    def _goal_cache_set(self, user_id: Any, value: float | None) -> None:
        if self.goal_cache_seconds <= 0:
            return
        key = str(user_id or "").strip()
        if not key:
            return
        self._goal_cache[key] = (time.time() + self.goal_cache_seconds, value)

    def _profile_cache_key(self, user_id: Any, meal_type: str) -> str:
        user_key = str(user_id or "").strip()
        if not user_key:
            return ""
        return f"{user_key}|{normalize_text(meal_type)}"

    def _profile_cache_get(self, user_id: Any, meal_type: str) -> dict[str, Any] | None:
        if self.profile_cache_seconds <= 0:
            return None
        key = self._profile_cache_key(user_id, meal_type)
        if not key.strip("|"):
            return None
        payload = self._profile_cache.get(key)
        if not payload:
            return None
        expires_at, value = payload
        if expires_at <= time.time():
            self._profile_cache.pop(key, None)
            return None
        return value

    def _profile_cache_set(self, user_id: Any, meal_type: str, value: dict[str, Any]) -> None:
        if self.profile_cache_seconds <= 0:
            return
        key = self._profile_cache_key(user_id, meal_type)
        if not key.strip("|"):
            return
        self._profile_cache[key] = (time.time() + self.profile_cache_seconds, value)

    def _build_cache_key(self, data: dict[str, Any], goal: str) -> str:
        user_id = str(data.get("userId") or "").strip()
        meal_type = normalize_text(data.get("mealType") or data.get("slot") or "all")
        calorie_target = str(data.get("calorieTarget") or "")
        force_exploration = str(
            parse_force_exploration(data.get("force_exploration"))
            or parse_force_exploration(data.get("forceExploration"))
        )
        feedback = data.get("feedback") or {}
        skipped = feedback.get("skipped_titles") or []
        loved = feedback.get("loved_titles") or []
        title_bias = feedback.get("title_bias") if isinstance(feedback.get("title_bias"), dict) else {}
        skipped_signature = ",".join(sorted(canonical_title_key(value) for value in skipped if canonical_title_key(value))[:4])
        loved_signature = ",".join(sorted(canonical_title_key(value) for value in loved if canonical_title_key(value))[:4])
        bias_signature = ",".join(sorted(canonical_title_key(key) for key in title_bias.keys() if canonical_title_key(key))[:4])
        feedback_signature = f"skip:{skipped_signature}|love:{loved_signature}|bias:{bias_signature}"
        favorites = data.get("favorite_titles") or []
        favorite_signature = f"fav{len(favorites)}"
        experiment_variant = normalize_text(data.get("experiment_variant") or data.get("experimentVariant") or "control")
        day_bucket = time.strftime("%Y-%m-%d")
        return "|".join(
            [
                user_id,
                meal_type,
                goal,
                calorie_target,
                force_exploration,
                feedback_signature,
                favorite_signature,
                experiment_variant,
                day_bucket,
            ]
        )

    @staticmethod
    def _normalize_goal(goal_value: Any) -> str:
        goal = normalize_text(goal_value or "maintain")
        return goal if goal in VALID_GOALS else "maintain"

    @staticmethod
    def _normalize_meal_type(meal_type_value: Any) -> str:
        meal_type = normalize_text(meal_type_value or "all")
        if meal_type in MEAL_SLOTS or meal_type == "all":
            return meal_type
        return "all"

    @staticmethod
    def _is_australian_user(demographics: dict[str, Any]) -> bool:
        if not demographics:
            return False
        raw_text = " ".join(
            str(demographics.get(key) or "")
            for key in ("country", "countryCode", "locale", "timezone", "region", "language")
        )
        normalized = normalize_text(raw_text)
        if "australia" in normalized or "australian" in normalized:
            return True
        return "au" in tokenize(raw_text)

    @staticmethod
    def _slot_weight_map(allocation: dict[str, float]) -> dict[str, float]:
        slot_weights = {}
        for meal in MEAL_SLOTS:
            slot_weights[meal] = max(0.0, to_float(allocation.get(meal), DEFAULT_MEAL_ALLOCATION[meal]))
        total = sum(slot_weights.values())
        if total <= 0:
            return DEFAULT_MEAL_ALLOCATION.copy()
        return {meal: slot_weights[meal] / total for meal in MEAL_SLOTS}

    @staticmethod
    def _normalize_title_set(values: Any) -> set[str]:
        output: set[str] = set()
        if values is None:
            return output
        iterable = values if isinstance(values, (list, tuple, set)) else [values]
        for value in iterable:
            normalized = canonical_title_key(value)
            if normalized:
                output.add(normalized)
        return output

    @staticmethod
    def _is_safe_positive_feedback_title(title: str) -> bool:
        normalized = canonical_title_key(title)
        if not normalized:
            return False

        if normalized in {
            "avocado",
            "miso soup",
            "oatmeal with milk",
            "orange juice",
            "tofu rice bowl",
            "turkey & american wrap",
            "vegetable stir fry",
        }:
            return True

        blocked_phrases = (
            "french toast",
            "gut shot",
            "hotcakes",
            "probiotic beverage",
            "smoothie cubes",
            "wellness probiotic beverage",
        )
        if any(phrase in normalized for phrase in blocked_phrases):
            return False

        blocked_tokens = {
            "bar",
            "bars",
            "brioche",
            "combo",
            "cookies",
            "cracker",
            "crackers",
            "cube",
            "cubes",
            "deluxe",
            "elbow",
            "elbows",
            "hotcakes",
            "kolache",
            "puffs",
            "shell",
            "shells",
            "snack",
            "snacks",
            "spud",
            "stromboli",
            "supermelt",
            "wafer",
            "wafers",
        }
        title_tokens = set(tokenize(normalized))
        if title_tokens.intersection(blocked_tokens):
            return False

        structured_meal_tokens = {
            "avocado",
            "bowl",
            "egg",
            "eggs",
            "juice",
            "oatmeal",
            "quinoa",
            "rice",
            "salad",
            "smoothie",
            "soup",
            "toast",
            "tofu",
            "wrap",
        }
        if len(title_tokens) <= 2 and not title_tokens.intersection(structured_meal_tokens):
            return False

        return True

    def _resolve_experiment_config(self, data: dict[str, Any], user_id: Any) -> dict[str, Any]:
        requested = normalize_text(data.get("experiment_variant") or data.get("experimentVariant") or "")
        if requested in EXPERIMENT_VARIANTS:
            return {"name": requested, **EXPERIMENT_VARIANTS[requested]}

        default_variant = normalize_text(os.getenv("ML_EXPERIMENT_DEFAULT_VARIANT", "control"))
        if default_variant in EXPERIMENT_VARIANTS:
            return {"name": default_variant, **EXPERIMENT_VARIANTS[default_variant]}

        # Keep experiments off by default. A stable override path still exists for controlled rollout.
        return {"name": "control", **EXPERIMENT_VARIANTS["control"]}

    def _should_expand_query(self, retrieval_metrics: dict[str, Any]) -> bool:
        candidate_count = int(retrieval_metrics.get("candidate_count", 0))
        unique_ratio = to_float(retrieval_metrics.get("unique_title_ratio"), 0.0)
        mean_top_distance = to_float(retrieval_metrics.get("mean_top_distance"), 0.0)
        return (
            candidate_count < int(QUERY_EXPANSION_MIN_CANDIDATES)
            or unique_ratio < float(QUERY_EXPANSION_MIN_UNIQUE_RATIO)
            or mean_top_distance > float(QUERY_EXPANSION_MAX_TOP_DISTANCE)
        )

    def _build_query_expansion_regex(
        self,
        profile: dict[str, Any],
        demographics: dict[str, Any],
        meal_type: str,
    ) -> str:
        query_terms = self.retriever.build_keywords(profile.get("top_foods", []), demographics, meal_type)
        tokens: list[str] = []
        for term in query_terms[: int(QUERY_EXPANSION_QUERY_LIMIT)]:
            normalized = normalize_text(term)
            if normalized:
                tokens.append(re.escape(normalized))
        return "|".join(tokens)

    def _compute_adaptive_tuning(
        self,
        retrieval_metrics: dict[str, Any],
        experiment_config: dict[str, Any],
    ) -> dict[str, Any]:
        control_config = EXPERIMENT_VARIANTS.get("control", {})
        base_mmr = to_float(experiment_config.get("mmr_lambda"), to_float(control_config.get("mmr_lambda"), 0.5))
        base_reuse_penalty = to_float(
            experiment_config.get("combo_reuse_penalty_base"),
            to_float(control_config.get("combo_reuse_penalty_base"), COMBO_REUSE_PENALTY_BASE),
        )
        unique_ratio = to_float(retrieval_metrics.get("unique_title_ratio"), 0.0)
        candidate_count = int(retrieval_metrics.get("candidate_count", 0))
        mean_top_distance = to_float(retrieval_metrics.get("mean_top_distance"), 0.0)

        diversity_gap = max(0.0, float(QUERY_EXPANSION_MIN_UNIQUE_RATIO) - unique_ratio)
        confidence_gap = max(0.0, mean_top_distance - float(QUERY_EXPANSION_MAX_TOP_DISTANCE))
        scarcity_gap = max(0.0, float(QUERY_EXPANSION_MIN_CANDIDATES - candidate_count) / max(1.0, float(QUERY_EXPANSION_MIN_CANDIDATES)))

        mmr_lambda = base_mmr - (diversity_gap * 0.45) + (confidence_gap * 0.20) + (scarcity_gap * 0.10)
        combo_reuse_penalty_base = base_reuse_penalty - (diversity_gap * 0.35) + (confidence_gap * 0.10)

        return {
            "mmr_lambda": round(float(np.clip(mmr_lambda, COMBO_MMR_LAMBDA_MIN, COMBO_MMR_LAMBDA_MAX)), 4),
            "combo_reuse_penalty_base": round(
                float(np.clip(combo_reuse_penalty_base, COMBO_REUSE_PENALTY_MIN, COMBO_REUSE_PENALTY_MAX)),
                4,
            ),
            "ranking_weights": experiment_config.get("ranking_weights") or {},
        }

    @staticmethod
    def _aggregate_model_metrics(metrics_by_slot: dict[str, dict[str, Any]], experiment_variant: str) -> dict[str, Any]:
        combo_diagnostics = [metrics.get("combo_diagnostics", {}) for metrics in metrics_by_slot.values() if isinstance(metrics, dict)]
        mapping_diagnostics = [metrics.get("mapping_diagnostics", {}) for metrics in metrics_by_slot.values() if isinstance(metrics, dict)]
        dashboards = [metrics.get("diversity_dashboard", {}) for metrics in metrics_by_slot.values() if isinstance(metrics, dict)]
        timing = [metrics.get("timing", {}) for metrics in metrics_by_slot.values() if isinstance(metrics, dict)]
        return {
            "metric_metadata": build_runtime_aggregate_metric_metadata(),
            "slots": metrics_by_slot,
            "overall_combo_diagnostics": merge_combo_diagnostics(combo_diagnostics),
            "overall_mapping_diagnostics": merge_mapping_diagnostics(mapping_diagnostics),
            "overall_diversity": merge_diversity_dashboards(dashboards),
            "overall_timing": merge_timing_metrics(timing),
            "experimentation": {"variant": experiment_variant},
            "ausnut_benchmark": _load_latest_ausnut_benchmark_summary(),
        }

    def _safe_queue_size(self) -> int:
        if self._async_queue is None:
            return 0
        try:
            return int(self._async_queue.qsize())
        except Exception:
            return 0

    def _async_backlog_state(self) -> dict[str, int]:
        queue_size = self._safe_queue_size()
        with self._async_inflight_lock:
            inflight = len(self._async_inflight_recipe_ids)
        return {
            "queue_size": queue_size,
            "inflight": inflight,
            "backlog": max(queue_size, inflight),
            "high_watermark": max(60, self.async_mapping_prefetch_per_slot * len(MEAL_SLOTS) * 2),
        }

    def _mapping_queue_status(self) -> dict[str, int]:
        backlog_state = self._async_backlog_state()
        return {
            "enabled": int(self.async_mapping_enabled and self._async_queue is not None),
            "queued": int(backlog_state["queue_size"]),
            "inflight": int(backlog_state["inflight"]),
            "capacity": int(self.async_mapping_queue_size if self._async_queue is not None else 0),
        }

    def _mapping_retry_allowed(self, recipe_id: str) -> bool:
        if not recipe_id:
            return False
        with self._async_miss_lock:
            payload = self._async_miss_state.get(recipe_id)
        if not payload:
            return True
        miss_count = int(payload.get("count", 0))
        next_retry_at = float(payload.get("next_retry_at", 0.0))
        if miss_count >= self.async_mapping_miss_max_retries:
            return False
        return time.time() >= next_retry_at

    def _mark_mapping_miss(self, recipe_id: str, source: str) -> None:
        if not recipe_id:
            return
        now = time.time()
        with self._async_miss_lock:
            payload = self._async_miss_state.get(recipe_id, {"count": 0, "next_retry_at": 0.0})
            miss_count = int(payload.get("count", 0)) + 1
            next_retry_at = now + float(self.async_mapping_miss_backoff_seconds)
            self._async_miss_state[recipe_id] = {"count": miss_count, "next_retry_at": next_retry_at}

    def _clear_mapping_miss(self, recipe_id: str) -> None:
        if not recipe_id:
            return
        with self._async_miss_lock:
            self._async_miss_state.pop(recipe_id, None)

    def _ensure_async_workers_started(self) -> None:
        if not self.async_mapping_enabled or self._async_queue is None or not self.has_fatsecret_credentials:
            return
        with self._async_worker_lock:
            if self._async_workers_started:
                return
            for worker_idx in range(self.async_mapping_worker_count):
                worker = Thread(
                    target=self._async_mapping_worker,
                    name=f"mapping-worker-{worker_idx + 1}",
                    daemon=True,
                )
                worker.start()
                self._async_workers.append(worker)
            self._async_workers_started = True

    def _save_mapping_snapshot(
        self,
        recipe_candidate: dict[str, Any],
        meal_type: str,
        resolved_payload: dict[str, Any],
        source: str,
    ) -> dict[str, Any] | None:
        recipe_id = self._recipe_id_from_candidate(recipe_candidate)
        if not recipe_id:
            return None
        if self._is_safety_recipe_id(recipe_id):
            return None

        mapped_candidate = resolved_payload.get("mapped_candidate") or {}
        snapshot = self._build_mapping_snapshot(
            recipe_candidate=recipe_candidate,
            fatsecret_candidate=mapped_candidate,
            source_query=str(resolved_payload.get("source_query") or ""),
            calorie_gap_ratio=to_float(resolved_payload.get("calorie_gap_ratio"), 1.0),
            serving_calorie_gap_ratio=to_float(resolved_payload.get("serving_calorie_gap_ratio"), 1.0),
            title_similarity=to_float(resolved_payload.get("title_similarity"), 0.0),
            acceptance_mode=str(resolved_payload.get("acceptance_mode") or "strict"),
        )
        if not snapshot:
            return None

        previous_snapshot = self.mapping_store.get(recipe_id)
        self.mapping_store.set(recipe_id, snapshot)
        self._clear_mapping_miss(recipe_id)
        if previous_snapshot != snapshot:
            self._clear_mapping_title_lookup_cache()
            self._response_cache_clear()
            print(
                f"**** Response Cache Cleared: recipe_id={recipe_id} meal_type={meal_type} source={source}"
            )
        return snapshot

    def _delete_mapping_snapshot(self, recipe_id: Any) -> None:
        key = str(recipe_id or "").strip()
        if not key:
            return
        self.mapping_store.delete(key)
        self._clear_mapping_title_lookup_cache()

    def _queue_background_mapping_candidate(self, recipe_candidate: dict[str, Any], meal_type: str) -> bool:
        if not self.async_mapping_enabled or self._async_queue is None or not self.has_fatsecret_credentials:
            return False

        recipe_id = self._recipe_id_from_candidate(recipe_candidate)
        if not recipe_id:
            return False
        if self._is_safety_recipe_id(recipe_id):
            return False
        if self.mapping_store.get(recipe_id):
            return False
        if not self._mapping_retry_allowed(recipe_id):
            return False
        backlog_state = self._async_backlog_state()
        if backlog_state["backlog"] >= backlog_state["high_watermark"]:
            return False
        if backlog_state["queue_size"] >= max(1, int(self.async_mapping_queue_size * 0.9)):
            return False

        with self._async_inflight_lock:
            if recipe_id in self._async_inflight_recipe_ids:
                return False
            self._async_inflight_recipe_ids.add(recipe_id)

        try:
            self._async_queue.put_nowait((recipe_candidate, meal_type))
            return True
        except Full:
            with self._async_inflight_lock:
                self._async_inflight_recipe_ids.discard(recipe_id)
            return False

    def _enqueue_background_mapping_candidates(
        self,
        meal_type: str,
        prioritized_candidates: list[dict[str, Any]],
        fallback_candidates: list[dict[str, Any]],
    ) -> int:
        if not self.async_mapping_enabled or self._async_queue is None:
            return 0
        backlog_state = self._async_backlog_state()
        queue_size = int(backlog_state["queue_size"])
        inflight = int(backlog_state["inflight"])
        backlog = int(backlog_state["backlog"])
        high_watermark = int(backlog_state["high_watermark"])
        if backlog >= high_watermark:
            print(
                f"**** Slot={meal_type} async-mapping throttled: "
                f"queue_size={queue_size} inflight={inflight} high_watermark={high_watermark}"
            )
            return 0

        # Prioritize foods shown to the user, then fill with other high-ranked local candidates.
        queue_budget = min(
            max(RECOMMENDED_ITEMS_PER_MEAL, int(self.async_mapping_prefetch_per_slot)),
            max(0, high_watermark - backlog),
        )
        if queue_budget <= 0:
            return 0
        queued = 0
        seen_recipe_ids: set[str] = set()
        for candidate in [*prioritized_candidates, *fallback_candidates]:
            recipe_id = self._recipe_id_from_candidate(candidate)
            if not recipe_id or recipe_id in seen_recipe_ids:
                continue
            seen_recipe_ids.add(recipe_id)
            if self.mapping_store.get(recipe_id):
                continue
            if self._queue_background_mapping_candidate(candidate, meal_type):
                queued += 1
            if queued >= queue_budget:
                break

        if queued > 0:
            print(
                f"**** Slot={meal_type} async-mapping queued={queued} "
                f"queue_size={self._safe_queue_size()}"
            )
        return queued

    def _async_mapping_worker(self) -> None:
        if self._async_queue is None:
            return

        while True:
            recipe_candidate, meal_type = self._async_queue.get()
            recipe_id = self._recipe_id_from_candidate(recipe_candidate)
            try:
                if not recipe_id:
                    continue
                if self.mapping_store.get(recipe_id):
                    continue
                if not self._mapping_retry_allowed(recipe_id):
                    continue

                resolved_payload = self._search_best_fatsecret_match(recipe_candidate, meal_type)
                if not resolved_payload:
                    self._mark_mapping_miss(recipe_id, "async")
                    continue

                self._save_mapping_snapshot(
                    recipe_candidate=recipe_candidate,
                    meal_type=meal_type,
                    resolved_payload=resolved_payload,
                    source="async",
                )
            except Exception as exc:
                pass
            finally:
                if recipe_id:
                    with self._async_inflight_lock:
                        self._async_inflight_recipe_ids.discard(recipe_id)
                self._async_queue.task_done()

    def _enrich_consumed_images(self, consumed_items: list[dict[str, Any]], max_lookups: int = 5) -> list[dict[str, Any]]:
        if not consumed_items:
            return []

        output: list[dict[str, Any]] = []
        lookups_used = 0
        for item in consumed_items:
            image = str(item.get("image") or "").strip()
            if not image and lookups_used < max_lookups:
                query = str(item.get("title") or item.get("food_name") or "").strip()
                if query:
                    hits = self.fs_client.search_foods(query, max_results=1)
                    if hits and isinstance(hits[0], dict):
                        first_hit = hits[0]
                        image = extract_image(first_hit) or str(first_hit.get("food_image") or "").strip()
                        # Optional deep lookup is disabled by default because it adds
                        # synchronous network latency to recommendation requests.
                        if not image and self.consumed_image_use_detail_lookup:
                            food_id = first_hit.get("food_id")
                            detail = self.fs_client.get_food(food_id) if food_id else None
                            detailed_food = (detail or {}).get("food", {}) if isinstance(detail, dict) else {}
                            image = extract_image(detailed_food) or extract_image(first_hit) or ""
                lookups_used += 1

            output.append({**item, "image": image})
        print(
            f"**** Consumed Images Enriched: items={len(consumed_items)} "
            f"lookups_used={lookups_used} max_lookups={max_lookups}"
        )
        return output

    def _load_consumption_snapshot(self, top_n: int = DEFAULT_TOP_CONSUMED_LIMIT) -> list[dict[str, Any]]:
        candidate_paths = [
            os.getenv("MOST_FOOD_CONSUMPTION_FILE", "").strip(),
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "most_food_consumption.json"),
            os.path.join(os.path.expanduser("~"), "Downloads", "most_food_consumption.json"),
        ]

        for path in candidate_paths:
            if not path:
                continue
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    payload = json.load(handle)
                if not isinstance(payload, list):
                    continue
                rows = []
                for row in payload[:top_n]:
                    if not isinstance(row, dict):
                        continue
                    title = str(row.get("food_name") or row.get("title") or "").strip()
                    if not title:
                        continue
                    count = int(to_float(row.get("number_appearance") or row.get("count"), 0.0))
                    rows.append(
                        {
                            "title": title,
                            "food_name": title,
                            "count": max(0, count),
                            "number_appearance": max(0, count),
                            "meal_type": "",
                            "image": "",
                        }
                    )
                return rows[:top_n]
            except Exception:
                continue
        return []

    @staticmethod
    def _title_similarity(source_title: str, target_title: str) -> float:
        source_text = normalize_text(canonicalize_title(source_title))
        target_text = normalize_text(canonicalize_title(target_title))
        raw_source_text = normalize_text(source_title)
        raw_target_text = normalize_text(target_title)
        token_aliases = {
            "browns": "brown",
            "cappy": "cappuccino",
            "eggs": "egg",
        }
        stopwords = {"a", "an", "and", "in", "of", "on", "the", "with"}
        latte_tea_tokens = {"chai", "matcha", "tea"}
        breakfast_toast_tokens = {"bagel", "biscuit", "brown", "croissant", "hash", "sandwich", "toast"}

        source_tokens = {token_aliases.get(token, token) for token in tokenize(source_text)}
        target_tokens = {token_aliases.get(token, token) for token in tokenize(target_text)}
        raw_source_tokens = {token_aliases.get(token, token) for token in tokenize(raw_source_text)}
        raw_target_tokens = {token_aliases.get(token, token) for token in tokenize(raw_target_text)}
        source_all_tokens = source_tokens.union(raw_source_tokens)
        target_all_tokens = target_tokens.union(raw_target_tokens)
        if not source_tokens or not target_tokens:
            return 0.0
        overlap = len(source_tokens.intersection(target_tokens))
        base = len(source_tokens.union(target_tokens)) or 1
        score = overlap / base

        source_content = {token for token in source_tokens if token not in stopwords}
        target_content = {token for token in target_tokens if token not in stopwords}
        shared_content = source_content.intersection(target_content)
        if len(shared_content) >= 2:
            score = max(
                score,
                len(shared_content) / max(3, min(len(source_content), len(target_content))),
            )

        if (
            "latte" in source_all_tokens
            and "latte" in target_all_tokens
            and not source_all_tokens.intersection(latte_tea_tokens)
            and not target_all_tokens.intersection(latte_tea_tokens)
        ):
            score = max(score, 0.5)
        if "cappuccino" in source_all_tokens and "cappuccino" in target_all_tokens:
            score = max(score, 0.5)
        if (
            "miso" in source_all_tokens
            and "miso" in target_all_tokens
            and "soup" in source_all_tokens
            and "soup" in target_all_tokens
        ):
            score = max(score, 0.6)
        if (
            "egg" in source_all_tokens
            and "egg" in target_all_tokens
            and "toast" in source_all_tokens
            and target_all_tokens.intersection(breakfast_toast_tokens)
        ):
            score = max(score, 0.55)

        if (
            any(marker in f"{source_text} {raw_source_text}" for marker in ("egg in a hole", "eggs in a hole"))
            and "egg" in target_all_tokens
            and target_all_tokens.intersection(breakfast_toast_tokens)
        ):
            score = max(score, 0.6)

        if (
            "egg" in source_all_tokens
            and "egg" in target_all_tokens
            and source_all_tokens.intersection({"cheddar", "cheese"})
            and target_all_tokens.intersection({"cheddar", "cheese"})
        ):
            score = max(score, 0.5)

        return min(1.0, score)

    @staticmethod
    def _serving_calorie_gap_ratio(source_candidate: dict[str, Any], mapped_candidate: dict[str, Any]) -> float:
        source_serving_calories = to_float(source_candidate.get("serving_calories"), 0.0)
        mapped_serving_calories = to_float(mapped_candidate.get("serving_calories"), 0.0)
        if source_serving_calories <= 0.0 or mapped_serving_calories <= 0.0:
            return 1.0
        return RecommendationService._calorie_gap_ratio(source_serving_calories, mapped_serving_calories)

    @staticmethod
    def _calorie_gap_ratio(left: Any, right: Any) -> float:
        left_value = max(1.0, to_float(left, 0.0))
        right_value = max(0.0, to_float(right, 0.0))
        return abs(left_value - right_value) / left_value

    @staticmethod
    def _candidate_per100(candidate: dict[str, Any]) -> dict[str, float]:
        per100 = candidate.get("per100") or {}
        calories = to_float(per100.get("calories"), 0.0)
        protein = to_float(per100.get("protein"), 0.0)
        carbs = to_float(per100.get("carbs"), 0.0)
        fats = to_float(per100.get("fats"), 0.0)
        if calories > 0.0:
            return {
                "calories": calories,
                "protein": protein,
                "carbs": carbs,
                "fats": fats,
            }

        metric_amount = to_float(candidate.get("metric_serving_amount"), 0.0)
        if metric_amount > 0.0:
            return normalize_to_per100(
                candidate.get("serving_calories"),
                candidate.get("serving_protein"),
                candidate.get("serving_carbs"),
                candidate.get("serving_fats"),
                metric_amount,
            )

        return {
            "calories": to_float(candidate.get("serving_calories"), 0.0),
            "protein": to_float(candidate.get("serving_protein"), 0.0),
            "carbs": to_float(candidate.get("serving_carbs"), 0.0),
            "fats": to_float(candidate.get("serving_fats"), 0.0),
        }

    def _mapping_acceptance_mode(
        self,
        calorie_gap: float,
        title_similarity: float,
        serving_calorie_gap: float | None = None,
        meal_type: str | None = None,
        category_hint: str | None = None,
    ) -> str:
        """
        Mapping acceptance policy:
        - strict: calorie gap <= strict threshold and title overlap clears a minimum floor,
        - relaxed_title_fallback: larger gap allowed when title similarity is strong,
        - reject: otherwise.
        """
        strict_title_floor = max(0.25, float(self.mapping_title_similarity_floor) * 0.6)
        if calorie_gap <= self.calorie_mismatch_threshold and title_similarity >= strict_title_floor:
            return "strict"
        if (
            calorie_gap <= self.relaxed_mapping_calorie_threshold
            and title_similarity >= self.mapping_title_similarity_floor
        ):
            return "relaxed_title_fallback"
        meal_key = normalize_text(meal_type)
        category_key = normalize_text(category_hint)
        if (
            serving_calorie_gap is not None
        ):
            if meal_key == "breakfast" and category_key == "drink":
                if serving_calorie_gap <= 0.18 and title_similarity >= 0.5:
                    return "relaxed_title_fallback"
                if serving_calorie_gap <= 0.35 and title_similarity >= 0.65:
                    return "relaxed_title_fallback"
                return "reject"
            if serving_calorie_gap > 0.22:
                return "reject"
            if meal_key == "breakfast":
                if title_similarity >= 0.6 and calorie_gap <= 0.5:
                    return "relaxed_title_fallback"
                return "reject"
            if title_similarity >= 0.5:
                return "relaxed_title_fallback"
            return "reject"
        return "reject"

    @staticmethod
    def _mapping_query_aliases(raw_title: str, meal_type: str, category_hint: str) -> list[str]:
        title_text = canonical_title_key(raw_title)
        raw_text = normalize_text(raw_title)
        meal_key = normalize_text(meal_type)
        category_key = normalize_text(category_hint)
        aliases: list[str] = []

        if any(marker in title_text for marker in ("egg in a hole", "eggs in a hole", "egg toast")):
            aliases.extend(
                [
                    "Egg Toast",
                    "Egg on Toast",
                    "Toast with Egg",
                ]
            )

        if any(marker in title_text for marker in ("hard boiled egg", "hard-boiled egg")) and (
            "cheddar" in title_text or "cheese" in title_text
        ):
            aliases.extend(
                [
                    "Hard Boiled Egg Cheese",
                    "Hard Boiled Egg Cheddar Cheese",
                    "Protein Pack Egg Cheese Nuts",
                ]
            )

        if "breakfast bowl" in title_text and ("egg" in title_text or "egg" in raw_text):
            aliases.extend(
                [
                    "Breakfast Bowl",
                    "Scrambled Egg Breakfast Bowl",
                    "Egg Breakfast Bowl",
                ]
            )

        if category_key == "side" and any(term in title_text for term in ("yogur", "yogurt", "yoghurt")) and any(
            term in title_text for term in ("fresa", "fresas", "strawberry", "strawberries")
        ):
            aliases.extend(
                [
                    "Strawberry Yogurt",
                    "Strawberries and Cream Yogurt",
                    "Yogurt with Strawberries",
                ]
            )

        if "iced vanilla latte" in title_text:
            aliases.extend(
                [
                    "Vanilla Iced Coffee",
                    "Vanilla Latte",
                    "Iced Coffee Drink",
                ]
            )

        if "iced latte" in title_text:
            aliases.extend(["Iced Latte", "Latte Drink", "Latte Coffee", "Iced Coffee Drink"])
            if "nescafe" in raw_text:
                aliases.extend(
                    [
                        "Ready to drink latte",
                        "Cold Brew Latte",
                        "Creamy Latte",
                        "Ready to Drink Coffee",
                        "Ready to Drink Latte Coffee",
                        "Nescafe Latte",
                    ]
                )
        elif category_key == "drink" and "latte" in title_text and meal_key == "breakfast":
            aliases.append("Latte Coffee")

        if category_key == "drink" and "cappuccino blast" in title_text:
            aliases.extend(
                [
                    "Oreo Cappy",
                    "Cappuccino",
                    "Cappuccino Drink",
                    "Cappuccino Blast",
                    "Oreo Cappuccino",
                    "Cookies Cappuccino",
                    "Frozen Cappuccino",
                ]
            )

        if category_key == "drink" and meal_key == "breakfast" and "iced coffee" in title_text:
            aliases.extend(["Coffee Drink", "Ready to Drink Coffee"])
            if "almond" in title_text:
                aliases.extend(["Almond Iced Coffee", "Almond Milk Iced Coffee"])

        if category_key == "main" and meal_key == "dinner" and "indian chicken" in title_text:
            aliases.extend(
                [
                    "Chicken Tikka Masala",
                    "Indian Style Chicken Tikka",
                    "India Style Chicken Tikka Masala",
                    "Chicken Curry",
                    "Butter Chicken",
                ]
            )

        return dedupe_strings(aliases, limit=6)

    @staticmethod
    def _candidate_title_key(candidate: dict[str, Any]) -> str:
        return canonical_title_key(
            candidate.get("canonical_title")
            or candidate.get("mapped_canonical_title")
            or candidate.get("title")
            or candidate.get("original_title")
        )

    @staticmethod
    def _food_id_from_candidate(candidate: dict[str, Any]) -> str:
        return str(candidate.get("food_id") or candidate.get("id") or "").strip()

    @staticmethod
    def _recipe_id_from_candidate(candidate: dict[str, Any]) -> str:
        return str(candidate.get("recipe_id") or "").strip()

    @staticmethod
    def _is_safety_recipe_id(recipe_id: str) -> bool:
        return str(recipe_id or "").strip().startswith("safety-")

    def _build_mapping_snapshot(
        self,
        recipe_candidate: dict[str, Any],
        fatsecret_candidate: dict[str, Any],
        source_query: str,
        calorie_gap_ratio: float,
        serving_calorie_gap_ratio: float,
        title_similarity: float,
        acceptance_mode: str,
    ) -> dict[str, Any] | None:
        food_id = self._food_id_from_candidate(fatsecret_candidate)
        recipe_id = self._recipe_id_from_candidate(recipe_candidate)
        if not food_id or not recipe_id:
            return None

        return {
            "recipe_id": recipe_id,
            "dataset_title": str(recipe_candidate.get("dataset_title") or recipe_candidate.get("title") or "").strip(),
            "food_id": food_id,
            "food_name": str(fatsecret_candidate.get("title") or "").strip(),
            "image": fatsecret_candidate.get("image") if title_similarity >= 0.5 else None,
            "food_type": fatsecret_candidate.get("food_type"),
            "food_url": fatsecret_candidate.get("food_url"),
            "brand_name": fatsecret_candidate.get("brand_name"),
            "per100": fatsecret_candidate.get("per100") or {},
            "serving_id": fatsecret_candidate.get("serving_id"),
            "serving_description": fatsecret_candidate.get("serving_description"),
            "metric_serving_amount": fatsecret_candidate.get("metric_serving_amount"),
            "metric_serving_unit": fatsecret_candidate.get("metric_serving_unit"),
            "number_of_units": fatsecret_candidate.get("number_of_units"),
            "measurement_description": fatsecret_candidate.get("measurement_description"),
            "serving_calories": round(to_float(fatsecret_candidate.get("serving_calories"), 0.0), 3),
            "serving_protein": round(to_float(fatsecret_candidate.get("serving_protein"), 0.0), 3),
            "serving_carbs": round(to_float(fatsecret_candidate.get("serving_carbs"), 0.0), 3),
            "serving_fats": round(to_float(fatsecret_candidate.get("serving_fats"), 0.0), 3),
            "allergens": fatsecret_candidate.get("allergens") or [],
            "preferences": fatsecret_candidate.get("preferences") or [],
            "food_sub_categories": fatsecret_candidate.get("food_sub_categories") or [],
            "mapped_query": source_query,
            "title_similarity": round(title_similarity, 6),
            "calorie_diff_ratio": round(calorie_gap_ratio, 6),
            "serving_calorie_diff_ratio": round(serving_calorie_gap_ratio, 6),
            "acceptance_mode": acceptance_mode,
            "mapped_at": _utc_now_iso(),
        }

    def _mapping_role_flip_rejected(
        self,
        recipe_candidate: dict[str, Any],
        mapped_candidate: dict[str, Any],
        title_similarity: float,
    ) -> bool:
        source_category = self._infer_combo_category(recipe_candidate)
        mapped_category = self._infer_combo_category(mapped_candidate)
        low_similarity = float(title_similarity) < float(self.mapping_title_similarity_floor)
        if (
            mapped_category == "drink"
            and source_category != "drink"
            and low_similarity
        ):
            return True

        source_title = canonicalize_title(
            recipe_candidate.get("dataset_title") or recipe_candidate.get("title") or recipe_candidate.get("original_title") or ""
        )
        mapped_title = canonicalize_title(
            mapped_candidate.get("mapped_title") or mapped_candidate.get("title") or mapped_candidate.get("original_title") or ""
        )
        raw_source_title = str(
            recipe_candidate.get("dataset_title") or recipe_candidate.get("title") or recipe_candidate.get("original_title") or ""
        ).strip()
        raw_mapped_title = str(
            mapped_candidate.get("mapped_title") or mapped_candidate.get("title") or mapped_candidate.get("original_title") or ""
        ).strip()
        source_text = normalize_text(source_title)
        source_tokens = set(tokenize(source_title))
        mapped_text = normalize_text(mapped_title)
        mapped_tokens = set(tokenize(mapped_title))
        raw_source_text = normalize_text(raw_source_title)
        raw_mapped_text = normalize_text(raw_mapped_title)
        raw_source_tokens = set(tokenize(raw_source_text))
        raw_mapped_tokens = set(tokenize(raw_mapped_text))
        source_all_tokens = source_tokens.union(raw_source_tokens)
        mapped_all_tokens = mapped_tokens.union(raw_mapped_tokens)
        mapped_text_all = f"{mapped_text} {raw_mapped_text}".strip()
        source_meal_like = bool(source_tokens.intersection(_MAPPING_MEAL_TITLE_MARKERS))
        side_identity_markers = (
            "salad",
            "slaw",
            "greens",
            "vegetable",
            "veggies",
            "broth",
            "soup",
            "bean",
            "beans",
            "lentil",
            "chickpea",
            "cucumber",
        )
        handheld_main_markers = (
            "sandwich",
            "sub",
            "wrap",
            "bagel",
            "biscuit",
            "croissant",
            "burger",
            "taco",
            "pizza",
            "mcgriddles",
            "sizzli",
        )
        source_breakfast_product_like = any(
            marker in source_text
            for marker in (
                "breakfast cereal",
                "cereal",
                "cracker",
                "crackers",
                "crispbread",
                "crispbreads",
                "granola",
                "muesli",
                "protein bar",
                "protein bars",
                "tosta",
                "tostas",
            )
        )
        source_supplement_like = any(normalize_text(term) in source_text for term in _MAPPING_SUPPLEMENT_TITLE_MARKERS)
        mapped_supplement_like = any(normalize_text(term) in mapped_text for term in _MAPPING_SUPPLEMENT_TITLE_MARKERS)
        source_side_identity_like = any(marker in source_text for marker in side_identity_markers)
        mapped_side_identity_like = any(marker in mapped_text for marker in side_identity_markers)
        source_handheld_like = any(marker in source_text for marker in handheld_main_markers)
        mapped_handheld_like = any(marker in mapped_text for marker in handheld_main_markers)
        source_has_egg_marker = bool(source_all_tokens.intersection({"egg", "eggs"}))
        mapped_has_egg_marker = bool(mapped_all_tokens.intersection({"egg", "eggs"}))
        source_has_cheese_marker = bool(source_all_tokens.intersection({"cheddar", "cheese"}))
        mapped_has_cheese_marker = bool(mapped_all_tokens.intersection({"cheddar", "cheese"}))
        source_yogurt_like = bool(source_all_tokens.intersection({"yogur", "yogurt", "yoghurt"}))
        mapped_yogurt_like = bool(mapped_all_tokens.intersection({"yogur", "yogurt", "yoghurt"}))
        source_latte_like = "latte" in source_all_tokens and not source_all_tokens.intersection({"chai", "matcha", "tea"})
        mapped_latte_like = "latte" in mapped_all_tokens and not mapped_all_tokens.intersection({"chai", "matcha", "tea"})
        source_cappuccino_like = "cappuccino" in source_all_tokens
        mapped_cappuccino_like = "cappuccino" in mapped_all_tokens
        coffee_identity_markers = {"coffee", "latte", "cappuccino", "espresso", "macchiato"}
        source_coffee_like = bool(source_all_tokens.intersection(coffee_identity_markers))
        mapped_coffee_like = bool(mapped_all_tokens.intersection(coffee_identity_markers))
        latte_flavor_markers = {"caramel", "hazelnut", "mocha", "pumpkin", "vainilla", "vanilla"}
        source_latte_flavors = source_all_tokens.intersection(latte_flavor_markers)
        mapped_latte_flavors = mapped_all_tokens.intersection(latte_flavor_markers)
        source_plant_milk_like = bool(source_all_tokens.intersection({"almond", "coconut", "oat", "soy", "soya"}))
        mapped_plant_milk_like = bool(mapped_all_tokens.intersection({"almond", "coconut", "oat", "soy", "soya"}))
        source_drink_mix_like = "drink mix" in source_text or bool(source_all_tokens.intersection({"mix", "mixes"}))
        mapped_drink_mix_like = "drink mix" in mapped_text_all or bool(mapped_all_tokens.intersection({"mix", "mixes"}))
        source_coffee_variant_like = any(
            marker in source_text for marker in ("nonfat", "decaffeinated", "decaf", "reduced fat")
        )
        mapped_coffee_variant_like = any(
            marker in mapped_text_all for marker in ("nonfat", "decaffeinated", "decaf", "reduced fat")
        )

        if source_has_egg_marker and not mapped_has_egg_marker:
            return True

        if source_has_cheese_marker and "hard boiled" in source_text and not mapped_has_cheese_marker:
            return True

        if "hard boiled" in source_text and any(marker in mapped_text for marker in ("soup", "soups")):
            return True

        if any(marker in source_text for marker in ("egg in a hole", "eggs in a hole")):
            if any(marker in mapped_text_all for marker in ("bacon", "bologna", "cheese", "ham", "jalapeno", "sausage")):
                return True
            if not any(marker in mapped_text_all for marker in ("toast", "sandwich", "hash brown", "hash browns")):
                return True

        if any(marker in source_text for marker in ("hard boiled egg", "hard-boiled egg")):
            if not all(marker in mapped_text for marker in ("hard", "boiled")):
                return True
            if any(marker in mapped_text for marker in ("macaroni", "pasta", "soup", "soups")):
                return True

        if source_latte_like and not mapped_latte_like and any(
            marker in mapped_text_all for marker in ("chai", "drink mix", "mix", "protein drink")
        ):
            return True

        if source_latte_like and mapped_latte_like and mapped_latte_flavors and not source_latte_flavors:
            return True

        if source_category == "drink" and source_latte_like and not mapped_latte_like:
            return True

        if source_category == "drink" and source_cappuccino_like and not mapped_cappuccino_like:
            return True

        if source_category == "drink" and source_coffee_like and mapped_coffee_like and mapped_latte_flavors and not source_latte_flavors:
            return True

        if source_category == "drink" and source_plant_milk_like and mapped_coffee_like and not mapped_plant_milk_like:
            return True

        if source_category == "drink" and source_coffee_like and mapped_coffee_variant_like and not source_coffee_variant_like:
            return True

        if source_category == "drink" and source_coffee_like and mapped_drink_mix_like and not source_drink_mix_like:
            return True

        if source_category == "side" and source_yogurt_like and not mapped_yogurt_like:
            return True

        if low_similarity and mapped_supplement_like and not source_supplement_like:
            return True

        if low_similarity and source_category == "main" and mapped_category != "main":
            return True

        if low_similarity and source_category in {"side", "drink"} and mapped_category == "main":
            return True

        if low_similarity and source_category == "side" and source_side_identity_like and not mapped_side_identity_like:
            return True

        if low_similarity and source_category == "main" and source_handheld_like and not mapped_handheld_like:
            return True

        if low_similarity and source_category == "drink" and any(
            marker in source_text
            for marker in (
                "gut shot",
                "probiotic beverage",
                "smoothie cubes",
                "wellness probiotic beverage",
                "shot",
            )
        ):
            return True

        if low_similarity and source_breakfast_product_like and mapped_category != source_category:
            return True

        if low_similarity and source_meal_like and mapped_supplement_like:
            return True

        return False

    def _hydrate_candidate_from_mapping(
        self,
        recipe_candidate: dict[str, Any],
        mapping: dict[str, Any],
        meal_type: str,
    ) -> dict[str, Any] | None:
        recipe_id = self._recipe_id_from_candidate(recipe_candidate)
        food_id = str(mapping.get("food_id") or "").strip()
        if not recipe_id or not food_id:
            return None

        serving_calories = to_float(mapping.get("serving_calories"), to_float(recipe_candidate.get("serving_calories"), 0.0))
        serving_protein = to_float(mapping.get("serving_protein"), to_float(recipe_candidate.get("serving_protein"), 0.0))
        serving_carbs = to_float(mapping.get("serving_carbs"), to_float(recipe_candidate.get("serving_carbs"), 0.0))
        serving_fats = to_float(mapping.get("serving_fats"), to_float(recipe_candidate.get("serving_fats"), 0.0))
        original_title = str(recipe_candidate.get("original_title") or recipe_candidate.get("title") or "").strip()
        mapped_title = str(mapping.get("food_name") or recipe_candidate.get("title") or "").strip()
        display_title = build_display_title(original_title, mapped_title)

        local_per100 = self._candidate_per100(recipe_candidate)
        mapped_per100 = self._candidate_per100(mapping)
        calorie_diff_ratio = self._calorie_gap_ratio(
            local_per100.get("calories"),
            mapped_per100.get("calories"),
        )
        merged = {
            **recipe_candidate,
            "id": str(recipe_candidate.get("id") or f"recipe-{recipe_id}"),
            "item_id": str(recipe_candidate.get("item_id") or recipe_candidate.get("id") or f"recipe-{recipe_id}"),
            "recipe_id": recipe_id,
            "food_id": food_id,
            "fatsecret_food_id": food_id,
            "meal_type": meal_type,
            "title": display_title or original_title,
            "original_title": original_title,
            "canonical_title": canonicalize_title(display_title or original_title),
            "mapped_title": build_display_title(mapped_title) or mapped_title,
            "mapped_canonical_title": canonicalize_title(
                str(mapping.get("mapped_canonical_title") or mapped_title or "").strip()
            ),
            "image": mapping.get("image") or recipe_candidate.get("image"),
            "food_type": mapping.get("food_type"),
            "food_url": mapping.get("food_url"),
            "brand_name": mapping.get("brand_name"),
            "per100": mapping.get("per100") or recipe_candidate.get("per100") or {},
            "serving_id": mapping.get("serving_id"),
            "serving_description": mapping.get("serving_description") or "1 serving",
            "metric_serving_amount": mapping.get("metric_serving_amount"),
            "metric_serving_unit": mapping.get("metric_serving_unit"),
            "number_of_units": mapping.get("number_of_units"),
            "measurement_description": mapping.get("measurement_description"),
            "serving_calories": round(serving_calories, 3),
            "serving_protein": round(serving_protein, 3),
            "serving_carbs": round(serving_carbs, 3),
            "serving_fats": round(serving_fats, 3),
            "allergens": mapping.get("allergens") or [],
            "preferences": mapping.get("preferences") or [],
            "food_sub_categories": mapping.get("food_sub_categories") or [],
            "source_keyword": recipe_candidate.get("source_keyword") or "",
            "mapped_query": mapping.get("mapped_query") or "",
            "calorie_diff_ratio": round(calorie_diff_ratio, 6),
            "serving_calorie_diff_ratio": round(
                to_float(
                    mapping.get("serving_calorie_diff_ratio"),
                    self._serving_calorie_gap_ratio(recipe_candidate, mapping),
                ),
                6,
            ),
            "title_similarity": round(to_float(mapping.get("title_similarity"), 0.0), 6),
            "mapping_acceptance_mode": str(mapping.get("acceptance_mode") or "strict"),
            "mapping_reason": None,
            "ml_tag": "FATSECRET_MAPPED",
        }
        return merged

    def _lookup_mapping_snapshot_by_title(self, candidate: dict[str, Any]) -> dict[str, Any] | None:
        raw_title = str(
            candidate.get("dataset_title") or candidate.get("title") or candidate.get("original_title") or ""
        ).strip()
        meal_type = str(candidate.get("meal_type") or "")
        category_hint = normalize_text(candidate.get("category") or self._infer_combo_category(candidate))

        title_keys = {
            canonical_title_key(candidate.get("canonical_title") or ""),
            canonical_title_key(candidate.get("mapped_canonical_title") or ""),
            canonical_title_key(raw_title),
            canonical_title_key(candidate.get("mapped_title") or candidate.get("food_name") or ""),
        }
        title_keys.update(
            canonical_title_key(alias)
            for alias in self._mapping_query_aliases(raw_title, meal_type, category_hint)
        )
        title_keys.discard("")
        if not title_keys:
            return None

        candidate_role = category_hint
        candidate_cache = self._candidate_service_cache(candidate)
        title_lookup_cache = candidate_cache.setdefault("title_mapping_lookup", {})
        title_lookup_key = f"{normalize_text(meal_type)}:{candidate_role}"
        if title_lookup_key in title_lookup_cache:
            cached_mapping = title_lookup_cache[title_lookup_key]
            return cached_mapping if isinstance(cached_mapping, dict) else None
        shared_lookup_key = f"{title_lookup_key}:{'|'.join(sorted(title_keys))}"
        shared_cached_mapping = self._mapping_title_lookup_cache_get(shared_lookup_key)
        if shared_cached_mapping is not None:
            title_lookup_cache[title_lookup_key] = shared_cached_mapping
            return shared_cached_mapping if isinstance(shared_cached_mapping, dict) else None

        matching_mappings: list[dict[str, Any]] = []
        find_by_title_keys = getattr(self.mapping_store, "find_by_title_keys", None)
        if callable(find_by_title_keys):
            matching_mappings = find_by_title_keys(title_keys)
        else:
            mappings = getattr(self.mapping_store, "_payload", {}).get("mappings", {})
            if isinstance(mappings, dict):
                for mapping in mappings.values():
                    if not isinstance(mapping, dict):
                        continue
                    mapping_title_keys = {
                        canonical_title_key(mapping.get("dataset_title") or ""),
                        canonical_title_key(mapping.get("mapped_title") or mapping.get("food_name") or ""),
                    }
                    mapping_title_keys.discard("")
                    if title_keys.intersection(mapping_title_keys):
                        matching_mappings.append(mapping)

        if not matching_mappings:
            title_lookup_cache[title_lookup_key] = False
            self._mapping_title_lookup_cache_set(shared_lookup_key, False)
            return None

        def _mapping_candidate(mapping: dict[str, Any]) -> dict[str, Any]:
            mapped_title = mapping.get("mapped_title") or mapping.get("food_name") or candidate.get("mapped_title")
            serving_calories = to_float(mapping.get("serving_calories"), to_float(candidate.get("serving_calories"), 0.0))
            return {
                **candidate,
                "category": candidate.get("category") or candidate_role,
                "mapped_title": mapped_title,
                "mapped_canonical_title": canonicalize_title(mapped_title),
                "food_name": mapping.get("food_name") or candidate.get("food_name"),
                "food_type": mapping.get("food_type") or candidate.get("food_type"),
                "brand_name": mapping.get("brand_name") or candidate.get("brand_name"),
                "source_keyword": candidate.get("source_keyword") or mapping.get("mapped_query") or "",
                "serving_calories": serving_calories,
                "calories": serving_calories or to_float(candidate.get("calories"), 0.0),
            }

        def _acceptance_rank(mapping: dict[str, Any]) -> int:
            acceptance_mode = normalize_text(mapping.get("acceptance_mode") or "strict")
            if acceptance_mode == "strict":
                return 0
            if acceptance_mode == "relaxed_title_fallback":
                return 1
            if acceptance_mode == "local_only":
                return 3
            return 2

        ranked_mappings: list[tuple[tuple[int, float, int, float, float], dict[str, Any]]] = []
        for mapping in matching_mappings:
            hydrated_like_candidate = _mapping_candidate(mapping)
            compatible = True
            if candidate_role in {"main", "side", "drink"}:
                compatible = self._is_role_compatible(hydrated_like_candidate, candidate_role)
            quality = self._role_quality_multiplier(hydrated_like_candidate, candidate_role, meal_type)
            ranked_mappings.append(
                (
                    (
                        0 if compatible else 1,
                        -quality,
                        _acceptance_rank(mapping),
                        -to_float(mapping.get("title_similarity"), 0.0),
                        to_float(mapping.get("calorie_diff_ratio"), 1.0),
                    ),
                    mapping,
                )
            )

        ranked_mappings.sort(key=lambda item: item[0])
        best_mapping = ranked_mappings[0][1]
        title_lookup_cache[title_lookup_key] = best_mapping
        self._mapping_title_lookup_cache_set(shared_lookup_key, best_mapping)
        return best_mapping

    def _resolve_visible_slot_candidates_with_mapping(
        self,
        selected_candidates: list[dict[str, Any]],
        ranked_candidates: list[dict[str, Any]],
        meal_type: str,
        lookup_state: dict[str, int],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        replacements: dict[str, dict[str, Any]] = {}
        for candidate in selected_candidates:
            recipe_id = self._recipe_id_from_candidate(candidate)
            if not recipe_id:
                continue
            if recipe_id in replacements:
                continue

            candidate_food_id = str(candidate.get("food_id") or "").strip()
            candidate_ml_tag = normalize_text(candidate.get("ml_tag") or "")
            candidate_fatsecret_food_id = str(candidate.get("fatsecret_food_id") or "").strip()
            is_local_candidate = (
                not candidate_fatsecret_food_id
                or candidate_ml_tag == "local_only"
                or candidate_food_id.startswith("local-")
            )
            if not is_local_candidate:
                continue

            snapshot = self.mapping_store.get(recipe_id)
            if not snapshot:
                snapshot = self._lookup_mapping_snapshot_by_title(candidate)
            if not snapshot:
                continue

            hydrated = self._hydrate_candidate_from_mapping(candidate, snapshot, meal_type)
            if not hydrated:
                continue

            replacements[recipe_id] = hydrated
            print(
                f"**** Visible Mapping Reuse: meal_type={meal_type} recipe_id={recipe_id} "
                f"food_id={hydrated.get('food_id')} title={hydrated.get('mapped_title') or hydrated.get('title')}"
            )

        if not replacements:
            return selected_candidates, ranked_candidates

        def _replace_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
            output: list[dict[str, Any]] = []
            for candidate in candidates:
                recipe_id = self._recipe_id_from_candidate(candidate)
                output.append(replacements.get(recipe_id, candidate))
            return output

        return _replace_candidates(selected_candidates), _replace_candidates(ranked_candidates)

    def _resolve_visible_recommended_items_with_mapping(
        self,
        recommended_items: list[dict[str, Any]],
        meal_type: str,
        slot_target: int,
        behavioral_insight: str,
        lookup_state: dict[str, int],
    ) -> list[dict[str, Any]]:
        if not self.has_fatsecret_credentials or not recommended_items:
            return recommended_items

        return [
            self._resolve_visible_serialized_item_with_mapping(
                item,
                meal_type,
                slot_target,
                behavioral_insight,
                lookup_state,
                "request_visible_serialized",
                False,
            )
            for item in recommended_items
        ]

    def _resolve_visible_serialized_item_with_mapping(
        self,
        item: dict[str, Any],
        meal_type: str,
        slot_target: int,
        behavioral_insight: str,
        lookup_state: dict[str, int],
        source: str,
        allow_live_search: bool = True,
    ) -> dict[str, Any]:
        recipe_id = self._recipe_id_from_candidate(item)
        fatsecret_food_id = str(item.get("fatsecret_food_id") or "").strip()
        item_ml_tag = normalize_text(item.get("ml_tag") or "")
        if not recipe_id or (fatsecret_food_id and item_ml_tag != "local_only"):
            return item

        cached_mapping = self.mapping_store.get(recipe_id)
        if cached_mapping:
            hydrated = self._hydrate_candidate_from_mapping(item, cached_mapping, meal_type)
            if hydrated:
                print(
                    f"**** Visible Serialized Cache Rescue: meal_type={meal_type} recipe_id={recipe_id} "
                    f"food_id={hydrated.get('food_id')} title={hydrated.get('mapped_title') or hydrated.get('title')}"
                )
                rescued = self._to_recommended_item(hydrated, meal_type, slot_target, behavioral_insight)
                if item.get("category") and not rescued.get("category"):
                    rescued["category"] = item.get("category")
                return rescued
            self._delete_mapping_snapshot(recipe_id)

        title_mapping = self._lookup_mapping_snapshot_by_title(item)
        if title_mapping:
            hydrated = self._hydrate_candidate_from_mapping(item, title_mapping, meal_type)
            if hydrated:
                print(
                    f"**** Visible Serialized Title Reuse: meal_type={meal_type} recipe_id={recipe_id} "
                    f"food_id={hydrated.get('food_id')} title={hydrated.get('mapped_title') or hydrated.get('title')}"
                )
                rescued = self._to_recommended_item(hydrated, meal_type, slot_target, behavioral_insight)
                if item.get("category") and not rescued.get("category"):
                    rescued["category"] = item.get("category")
                return rescued

        if not allow_live_search or not self.has_fatsecret_credentials:
            return item

        configured_sync_cap = max(0, int(lookup_state.get("slot_sync_lookup_cap", 0)))
        reserved_sync_cap = max(0, int(lookup_state.get("visible_sync_reserve", 0)))
        effective_sync_cap = configured_sync_cap if configured_sync_cap > 0 else reserved_sync_cap
        sync_lookups_used = max(0, int(lookup_state.get("slot_sync_lookups_used", 0)))
        remaining_lookup_attempts = max(0, int(lookup_state.get("remaining_lookup_attempts", 0)))

        if effective_sync_cap <= 0 or sync_lookups_used >= effective_sync_cap or remaining_lookup_attempts <= 0:
            return item

        lookup_state["slot_sync_lookups_used"] = sync_lookups_used + 1
        lookup_state["remaining_lookup_attempts"] = remaining_lookup_attempts - 1

        resolved_payload = self._search_best_fatsecret_match(item, meal_type)
        if not resolved_payload:
            return item

        if self._is_safety_recipe_id(recipe_id):
            snapshot = self._build_mapping_snapshot(
                recipe_candidate=item,
                fatsecret_candidate=resolved_payload.get("mapped_candidate") or {},
                source_query=str(resolved_payload.get("source_query") or ""),
                calorie_gap_ratio=to_float(resolved_payload.get("calorie_gap_ratio"), 1.0),
                serving_calorie_gap_ratio=to_float(resolved_payload.get("serving_calorie_gap_ratio"), 1.0),
                title_similarity=to_float(resolved_payload.get("title_similarity"), 0.0),
                acceptance_mode=str(resolved_payload.get("acceptance_mode") or "strict"),
            )
            if snapshot:
                previous_snapshot = self.mapping_store.get(recipe_id)
                self.mapping_store.set(recipe_id, snapshot)
                self._clear_mapping_miss(recipe_id)
                if previous_snapshot != snapshot:
                    self._clear_mapping_title_lookup_cache()
                    self._response_cache_clear()
                    print(
                        f"**** Response Cache Cleared: recipe_id={recipe_id} meal_type={meal_type} source={source}"
                    )
        else:
            lookup_state["slot_new_mappings_used"] = int(lookup_state.get("slot_new_mappings_used", 0)) + 1
            snapshot = self._save_mapping_snapshot(item, meal_type, resolved_payload, source)
        if not snapshot:
            return item

        hydrated = self._hydrate_candidate_from_mapping(item, snapshot, meal_type)
        if not hydrated:
            return item

        print(
            f"**** Visible Serialized Rescue: meal_type={meal_type} recipe_id={recipe_id} "
            f"food_id={hydrated.get('food_id')} title={hydrated.get('mapped_title') or hydrated.get('title')}"
        )
        rescued = self._to_recommended_item(hydrated, meal_type, slot_target, behavioral_insight)
        if item.get("category") and not rescued.get("category"):
            rescued["category"] = item.get("category")
        return rescued

    def _resolve_visible_combos_with_mapping(
        self,
        combos: list[dict[str, Any]],
        meal_type: str,
        slot_target: int,
        behavioral_insight: str,
        lookup_state: dict[str, int],
    ) -> list[dict[str, Any]]:
        if not combos:
            return []

        output: list[dict[str, Any]] = []
        resolved_item_cache: dict[str, dict[str, Any]] = {}
        for combo in combos:
            combo_items = combo.get("items")
            if not isinstance(combo_items, list) or not combo_items:
                output.append(combo)
                continue
            updated_combo = dict(combo)
            resolved_items: list[dict[str, Any]] = []
            for item in combo_items:
                cache_key = self._recipe_id_from_candidate(item) or str(
                    item.get("food_id") or item.get("item_id") or item.get("id") or ""
                ).strip()
                cached_item = resolved_item_cache.get(cache_key) if cache_key else None
                if cached_item is not None:
                    reused_item = dict(cached_item)
                    if item.get("category") and not reused_item.get("category"):
                        reused_item["category"] = item.get("category")
                    resolved_items.append(reused_item)
                    continue

                resolved_item = self._resolve_visible_serialized_item_with_mapping(
                    item,
                    meal_type,
                    slot_target,
                    behavioral_insight,
                    lookup_state,
                    "request_visible_combo",
                    False,
                )
                if cache_key:
                    resolved_item_cache[cache_key] = dict(resolved_item)
                resolved_items.append(resolved_item)

            updated_combo["items"] = resolved_items
            totals = self._combo_macro_totals(updated_combo["items"])
            updated_combo["total_calories"] = totals["calories"]
            updated_combo["total_protein"] = totals["protein"]
            updated_combo["total_carbs"] = totals["carbs"]
            updated_combo["total_fats"] = totals["fats"]
            updated_combo["title"] = "Combo: " + " + ".join(item.get("title") or "Item" for item in updated_combo["items"])
            output.append(updated_combo)
        return output

    def _build_local_only_candidate(
        self,
        recipe_candidate: dict[str, Any],
        meal_type: str,
        reason: str,
    ) -> dict[str, Any]:
        recipe_id = self._recipe_id_from_candidate(recipe_candidate)
        local_food_id = f"local-{recipe_id}" if recipe_id else str(recipe_candidate.get("id") or "")
        original_title = str(recipe_candidate.get("original_title") or recipe_candidate.get("title") or "").strip()
        display_title = build_display_title(original_title)
        return {
            **recipe_candidate,
            "id": str(recipe_candidate.get("id") or f"recipe-{recipe_id}"),
            "item_id": str(recipe_candidate.get("item_id") or recipe_candidate.get("id") or f"recipe-{recipe_id}"),
            "recipe_id": recipe_id,
            "food_id": local_food_id,
            "fatsecret_food_id": None,
            "meal_type": meal_type,
            "title": display_title or original_title,
            "original_title": original_title,
            "canonical_title": canonicalize_title(display_title or original_title),
            "mapped_title": None,
            "mapped_canonical_title": None,
            "food_type": recipe_candidate.get("food_type") or "LocalDataset",
            "source_keyword": recipe_candidate.get("source_keyword") or "local_dataset",
            "calorie_diff_ratio": 0.0,
            "title_similarity": 0.0,
            "ml_tag": "LOCAL_ONLY",
            "mapping_acceptance_mode": "local_only",
            "mapping_reason": reason,
        }

    def _search_best_fatsecret_match(
        self,
        recipe_candidate: dict[str, Any],
        meal_type: str,
    ) -> dict[str, Any] | None:
        raw_title = str(recipe_candidate.get("dataset_title") or recipe_candidate.get("title") or "").strip()
        if not raw_title:
            return None
        title = canonicalize_title(raw_title) or raw_title
        category_hint = self._infer_combo_category(recipe_candidate)
        source_category = normalize_text(category_hint)
        source_tokens = set(tokenize(title)).union(set(tokenize(raw_title)))
        latte_tea_tokens = {"chai", "matcha", "tea"}
        latte_flavor_markers = {"caramel", "hazelnut", "mocha", "pumpkin", "vainilla", "vanilla"}
        source_latte_like = "latte" in source_tokens and not source_tokens.intersection(latte_tea_tokens)
        source_has_latte_flavor = bool(source_tokens.intersection(latte_flavor_markers))
        recipe_category = str(recipe_candidate.get("recipe_category") or "").strip()
        canonical_recipe_category = canonicalize_title(recipe_category)

        queries = dedupe_strings(
            [
                title,
                raw_title,
                *self._mapping_query_aliases(raw_title, meal_type, category_hint),
                f"{title} {meal_type}",
                f"{title} healthy",
                recipe_category,
                canonical_recipe_category,
            ],
            limit=8 if source_category == "drink" else 5,
        )
        if not queries:
            return None

        local_per100 = self._candidate_per100(recipe_candidate)
        local_calories = to_float(local_per100.get("calories"), 0.0)
        detail_cache: dict[str, dict[str, Any] | None] = {}
        best_payload: dict[str, Any] | None = None
        best_score = -1.0

        search_plans: list[tuple[str | None, int]] = [(category_hint, 6)]
        if source_category == "drink":
            search_plans.append((None, 10))

        for search_category, max_results in search_plans:
            for query in queries:
                hits = self.fs_client.search_foods(query, max_results=max_results, category=search_category)
                for hit in hits[:max_results]:
                    food_id = str(hit.get("food_id") or "").strip()
                    if not food_id:
                        continue

                    if food_id not in detail_cache:
                        detail_cache[food_id] = self.fs_client.get_food(food_id)
                    detail = detail_cache[food_id]

                    mapped = map_food_hit_to_candidate(hit, detail, query, meal_type)
                    if not mapped:
                        continue

                    mapped_per100 = self._candidate_per100(mapped)
                    mapped_calories = to_float(mapped_per100.get("calories"), 0.0)
                    calorie_gap = self._calorie_gap_ratio(local_calories, mapped_calories)
                    serving_calorie_gap = self._serving_calorie_gap_ratio(recipe_candidate, mapped)
                    mapped_tokens = set(tokenize(str(mapped.get("title") or "")))
                    mapped_latte_like = "latte" in mapped_tokens and not mapped_tokens.intersection(latte_tea_tokens)
                    allow_drink_serving_gap = source_category == "drink" and source_latte_like and mapped_latte_like
                    effective_serving_gap = serving_calorie_gap if source_category != "drink" or allow_drink_serving_gap else None
                    similarity = self._title_similarity(title, str(mapped.get("title") or ""))
                    if self._mapping_role_flip_rejected(recipe_candidate, mapped, similarity):
                        continue
                    acceptance_mode = self._mapping_acceptance_mode(
                        calorie_gap,
                        similarity,
                        serving_calorie_gap,
                        meal_type,
                        source_category,
                    )
                    if acceptance_mode == "reject":
                        continue

                    effective_gap = calorie_gap
                    if effective_serving_gap is not None:
                        if source_category != "drink" or (allow_drink_serving_gap and not source_has_latte_flavor):
                            effective_gap = min(calorie_gap, effective_serving_gap)
                    normalized_gap = min(1.0, effective_gap / max(0.05, self.relaxed_mapping_calorie_threshold))
                    score = (1.0 - normalized_gap) * 0.55 + similarity * 0.45
                    if acceptance_mode == "relaxed_title_fallback":
                        score += 0.05
                    if score > best_score:
                        best_score = score
                        best_payload = {
                            "mapped_candidate": mapped,
                            "source_query": query,
                            "calorie_gap_ratio": calorie_gap,
                            "serving_calorie_gap_ratio": serving_calorie_gap,
                            "title_similarity": similarity,
                            "acceptance_mode": acceptance_mode,
                        }

        return best_payload

    def _resolve_candidate_with_mapping(
        self,
        recipe_candidate: dict[str, Any],
        meal_type: str,
        lookup_state: dict[str, int],
    ) -> dict[str, Any] | None:
        recipe_id = self._recipe_id_from_candidate(recipe_candidate)
        if not recipe_id:
            return None

        cached_mapping = self.mapping_store.get(recipe_id)
        if cached_mapping:
            hydrated = self._hydrate_candidate_from_mapping(recipe_candidate, cached_mapping, meal_type)
            if hydrated:
                calorie_gap = to_float(hydrated.get("calorie_diff_ratio"), 1.0)
                serving_calorie_gap = to_float(
                    hydrated.get("serving_calorie_diff_ratio"),
                    self._serving_calorie_gap_ratio(recipe_candidate, hydrated),
                )
                title_similarity = to_float(
                    hydrated.get("title_similarity"),
                    self._title_similarity(
                        str(recipe_candidate.get("dataset_title") or recipe_candidate.get("title") or ""),
                        str(hydrated.get("title") or ""),
                    ),
                )
                if self._mapping_role_flip_rejected(recipe_candidate, hydrated, title_similarity):
                    # Don't evict safety item mappings; visible rescue relies on finding them.
                    if not self._is_safety_recipe_id(recipe_id):
                        self._delete_mapping_snapshot(recipe_id)
                    hydrated = None
                if hydrated is None:
                    pass
                else:
                    source_category = normalize_text(self._infer_combo_category(recipe_candidate))
                    acceptance_mode = self._mapping_acceptance_mode(
                        calorie_gap,
                        title_similarity,
                        serving_calorie_gap,
                        meal_type,
                        source_category,
                    )
                    if (
                        acceptance_mode != "reject"
                        and normalize_text(meal_type) == "dinner"
                        and source_category == "main"
                        and title_similarity < 0.5
                    ):
                        title_mapping = self._lookup_mapping_snapshot_by_title(recipe_candidate)
                        if title_mapping and str(title_mapping.get("food_id") or "") != str(cached_mapping.get("food_id") or ""):
                            title_hydrated = self._hydrate_candidate_from_mapping(recipe_candidate, title_mapping, meal_type)
                            if title_hydrated:
                                title_gap = to_float(title_hydrated.get("calorie_diff_ratio"), 1.0)
                                title_serving_gap = to_float(
                                    title_hydrated.get("serving_calorie_diff_ratio"),
                                    self._serving_calorie_gap_ratio(recipe_candidate, title_hydrated),
                                )
                                title_similarity_override = to_float(
                                    title_hydrated.get("title_similarity"),
                                    self._title_similarity(
                                        str(recipe_candidate.get("dataset_title") or recipe_candidate.get("title") or ""),
                                        str(title_hydrated.get("title") or ""),
                                    ),
                                )
                                title_acceptance_mode = self._mapping_acceptance_mode(
                                    title_gap,
                                    title_similarity_override,
                                    title_serving_gap,
                                    meal_type,
                                    source_category,
                                )
                                if (
                                    title_acceptance_mode != "reject"
                                    and not self._mapping_role_flip_rejected(
                                        recipe_candidate,
                                        title_hydrated,
                                        title_similarity_override,
                                    )
                                    and title_similarity_override >= max(title_similarity + 0.2, 0.55)
                                    and title_gap <= max(calorie_gap + 0.08, 0.18)
                                ):
                                    hydrated = title_hydrated
                                    calorie_gap = title_gap
                                    serving_calorie_gap = title_serving_gap
                                    title_similarity = title_similarity_override
                                    acceptance_mode = title_acceptance_mode
                                    print(
                                        f"**** Mapping Cache Override: recipe_id={recipe_id} -> food_id={hydrated.get('food_id')} "
                                        f"gap={calorie_gap:.3f} serving_gap={serving_calorie_gap:.3f} "
                                        f"title_similarity={title_similarity:.3f} mode={acceptance_mode}"
                                    )
                    if acceptance_mode != "reject":
                        print(
                            f"**** Mapping Cache Hit: recipe_id={recipe_id} -> food_id={hydrated.get('food_id')} "
                            f"gap={calorie_gap:.3f} serving_gap={serving_calorie_gap:.3f} "
                            f"title_similarity={title_similarity:.3f} mode={acceptance_mode}"
                        )
                        return hydrated
            # NOTE: Only evict the mapping for non-safety recipe ids.
            # Safety item mappings are pre-seeded and the visible rescue path relies on
            # finding them in the store; evicting them on a fast-phase rejection causes
            # the visible rescue to fall through to a live FatSecret API call.
            if not self._is_safety_recipe_id(recipe_id):
                self._delete_mapping_snapshot(recipe_id)

        if self._is_safety_recipe_id(recipe_id):
            title_mapping = self._lookup_mapping_snapshot_by_title(recipe_candidate)
            if title_mapping:
                hydrated = self._hydrate_candidate_from_mapping(recipe_candidate, title_mapping, meal_type)
                if hydrated:
                    print(
                        f"**** Mapping Title Reuse: recipe_id={recipe_id} -> food_id={hydrated.get('food_id')} "
                        f"title={hydrated.get('mapped_title') or hydrated.get('title')}"
                    )
                    return hydrated
            return self._build_local_only_candidate(recipe_candidate, meal_type, "safety_candidate")

        # NOTE: Stale-while-revalidate: return local-only candidate, queue mapping in background.
        local_reason = "mapping_cache_miss"
        queued = False
        if self.has_fatsecret_credentials and self._mapping_retry_allowed(recipe_id):
            if lookup_state.get("remaining_new_mappings", 0) > 0 and lookup_state.get(
                "slot_new_mappings_used", 0
            ) < lookup_state.get("slot_new_mapping_cap", 0):
                queued = self._queue_background_mapping_candidate(recipe_candidate, meal_type)
                if queued:
                    lookup_state["remaining_new_mappings"] = max(0, lookup_state.get("remaining_new_mappings", 0) - 1)
                    lookup_state["slot_new_mappings_used"] = lookup_state.get("slot_new_mappings_used", 0) + 1
                    local_reason = "mapping_queued"
        if not self.has_fatsecret_credentials:
            local_reason = "fatsecret_disabled"
        elif not self._mapping_retry_allowed(recipe_id):
            local_reason = "mapping_retry_exhausted"

        return self._build_local_only_candidate(recipe_candidate, meal_type, local_reason)

    def _dedupe_candidates(self, candidates: list[dict[str, Any]], top_n: int) -> list[dict[str, Any]]:
        """
        Variety guard:
        - remove exact duplicates by food_id,
        - remove duplicates by (title, brand),
        - cap repeats per normalized title to avoid same-food spam.
        """
        selected: list[dict[str, Any]] = []
        seen_food_ids: set[str] = set()
        seen_brand_title: set[str] = set()
        title_counts: dict[str, int] = {}
        deferred_same_title: list[dict[str, Any]] = []

        for candidate in candidates:
            recipe_id = str(candidate.get("recipe_id") or "").strip()
            food_id = str(candidate.get("food_id") or "").strip()
            title_key = self._candidate_title_key(candidate)
            brand_key = normalize_text(candidate.get("brand_name") or "generic")
            brand_title_key = f"{title_key}|{brand_key}" if title_key else ""
            dedupe_key = food_id or recipe_id or brand_title_key or title_key
            if not dedupe_key:
                continue

            if food_id and food_id in seen_food_ids:
                continue
            if brand_title_key and brand_title_key in seen_brand_title:
                continue

            if title_key:
                current_title_count = int(title_counts.get(title_key, 0))
                if current_title_count >= self.max_duplicates_per_title_per_slot:
                    deferred_same_title.append(candidate)
                    continue
                title_counts[title_key] = current_title_count + 1

            if food_id:
                seen_food_ids.add(food_id)
            if brand_title_key:
                seen_brand_title.add(brand_title_key)
            selected.append(candidate)
            if len(selected) >= top_n:
                break

        # Second pass: if strict title cap was too aggressive, fill remaining slots
        # while still blocking exact food_id and (title, brand) duplicates.
        if len(selected) < top_n and deferred_same_title:
            for candidate in deferred_same_title:
                food_id = str(candidate.get("food_id") or "").strip()
                title_key = self._candidate_title_key(candidate)
                brand_key = normalize_text(candidate.get("brand_name") or "generic")
                brand_title_key = f"{title_key}|{brand_key}" if title_key else ""
                if food_id and food_id in seen_food_ids:
                    continue
                if brand_title_key and brand_title_key in seen_brand_title:
                    continue
                if food_id:
                    seen_food_ids.add(food_id)
                if brand_title_key:
                    seen_brand_title.add(brand_title_key)
                selected.append(candidate)
                if len(selected) >= top_n:
                    break

        return selected

    def _select_final_slot_candidates(self, candidates: list[dict[str, Any]], meal_type: str, top_n: int) -> list[dict[str, Any]]:
        if normalize_text(meal_type) == "breakfast":
            return self._select_breakfast_candidates(candidates, top_n)
        return self._dedupe_candidates(candidates, top_n)

    def _select_breakfast_candidates(self, candidates: list[dict[str, Any]], top_n: int) -> list[dict[str, Any]]:
        if top_n <= 0:
            return []

        breakfast_main_target = min(top_n, max(6, top_n // 2))
        breakfast_drink_cap = min(2, top_n)

        main_candidates = [candidate for candidate in candidates if self._infer_combo_category(candidate) == "main"]
        blocked_mains = [
            candidate for candidate in main_candidates if self._is_blocked_breakfast_main_candidate(candidate)
        ]

        aligned_mains = [
            candidate
            for candidate in main_candidates
            if candidate not in blocked_mains and self._is_breakfast_aligned_main_candidate(candidate)
        ]
        safety_aligned_mains = [
            candidate for candidate in aligned_mains if self._is_safety_recipe_id(self._recipe_id_from_candidate(candidate))
        ]
        regular_aligned_mains = [candidate for candidate in aligned_mains if candidate not in safety_aligned_mains]
        fallback_mains = [
            candidate
            for candidate in main_candidates
            if candidate not in blocked_mains and candidate not in aligned_mains
        ]
        if not aligned_mains and not fallback_mains:
            aligned_mains = [
                candidate for candidate in main_candidates if self._is_breakfast_aligned_main_candidate(candidate)
            ]
            safety_aligned_mains = [
                candidate for candidate in aligned_mains if self._is_safety_recipe_id(self._recipe_id_from_candidate(candidate))
            ]
            regular_aligned_mains = [candidate for candidate in aligned_mains if candidate not in safety_aligned_mains]
            fallback_mains = [candidate for candidate in main_candidates if candidate not in aligned_mains]
        drinks = [candidate for candidate in candidates if self._infer_combo_category(candidate) == "drink"]
        fallback_candidates = [
            candidate
            for candidate in candidates
            if candidate not in blocked_mains and candidate not in aligned_mains and candidate not in fallback_mains and candidate not in drinks
        ]

        prioritized = [
            *safety_aligned_mains,
            *regular_aligned_mains[: max(0, breakfast_main_target - len(safety_aligned_mains))],
            *fallback_mains,
            *drinks[:breakfast_drink_cap],
            *regular_aligned_mains[max(0, breakfast_main_target - len(safety_aligned_mains)) :],
            *fallback_candidates,
            *drinks[breakfast_drink_cap:],
        ]
        return self._dedupe_candidates(prioritized, top_n)

    def _to_recommended_item(
        self,
        candidate: dict[str, Any],
        meal_type: str,
        slot_target: int,
        behavioral_insight: str,
    ) -> dict[str, Any]:
        per100 = candidate.get("per100", {}) or {}

        serving_calories = max(0.0, to_float(candidate.get("serving_calories"), to_float(per100.get("calories"), 0.0)))
        serving_protein = max(0.0, to_float(candidate.get("serving_protein"), to_float(per100.get("protein"), 0.0)))
        serving_carbs = max(0.0, to_float(candidate.get("serving_carbs"), to_float(per100.get("carbs"), 0.0)))
        serving_fats = max(0.0, to_float(candidate.get("serving_fats"), to_float(per100.get("fats"), 0.0)))

        metric_serving_amount = to_float(candidate.get("metric_serving_amount"), 0.0)
        if metric_serving_amount <= 0.0:
            metric_serving_amount = 100.0

        recipe_id = str(candidate.get("recipe_id") or "").strip()
        item_id = str(candidate.get("item_id") or candidate.get("id") or f"recipe-{recipe_id}").strip()
        food_id = str(candidate.get("food_id") or "").strip() or item_id
        fatsecret_food_id = str(candidate.get("fatsecret_food_id") or "").strip() or None
        original_title = candidate.get("original_title") or candidate.get("title") or "Recommended Food"
        mapped_title = candidate.get("mapped_title") or None
        display_title = build_display_title(original_title, mapped_title)

        serving_fit_ratio = self._calorie_gap_ratio(slot_target, serving_calories) if slot_target > 0 else 0.0
        return {
            "id": item_id,
            "item_id": item_id,
            "recipe_id": recipe_id,
            "food_id": food_id,
            "fatsecret_food_id": fatsecret_food_id,
            "meal_type": meal_type,
            "slot_target": int(slot_target),
            "title": display_title or "Recommended Food",
            "original_title": original_title,
            "canonical_title": candidate.get("canonical_title") or canonicalize_title(display_title),
            "mapped_title": build_display_title(mapped_title) if mapped_title else None,
            "mapped_canonical_title": candidate.get("mapped_canonical_title") or canonicalize_title(mapped_title),
            "category": candidate.get("category"),
            "calories": round(serving_calories, 1),
            "protein": round(serving_protein, 1),
            "carbs": round(serving_carbs, 1),
            "fats": round(serving_fats, 1),
            "grams": int(round(metric_serving_amount)),
            "image": candidate.get("image"),
            "type": "food",
            "per100": {
                "calories": round(to_float(per100.get("calories"), serving_calories), 3),
                "protein": round(to_float(per100.get("protein"), serving_protein), 3),
                "carbs": round(to_float(per100.get("carbs"), serving_carbs), 3),
                "fats": round(to_float(per100.get("fats"), serving_fats), 3),
            },
            "serving_id": candidate.get("serving_id"),
            "serving_description": candidate.get("serving_description") or "1 serving",
            "metric_serving_amount": round(metric_serving_amount, 3),
            "metric_serving_unit": candidate.get("metric_serving_unit") or "g",
            "number_of_units": to_float(candidate.get("number_of_units"), 1.0),
            "measurement_description": candidate.get("measurement_description"),
            "serving_calories": round(serving_calories, 3),
            "serving_protein": round(serving_protein, 3),
            "serving_carbs": round(serving_carbs, 3),
            "serving_fats": round(serving_fats, 3),
            "food_type": candidate.get("food_type"),
            "food_url": candidate.get("food_url"),
            "brand_name": candidate.get("brand_name"),
            "allergens": candidate.get("allergens") or [],
            "preferences": candidate.get("preferences") or [],
            "food_sub_categories": candidate.get("food_sub_categories") or [],
            "source_keyword": candidate.get("source_keyword") or "",
            "score": round(to_float(candidate.get("score"), 0.0), 5),
            "knn_distance": round(to_float(candidate.get("knn_distance"), 0.0), 6),
            "adjusted_distance": round(to_float(candidate.get("adjusted_distance"), 0.0), 6),
            "consumed_recently": bool(candidate.get("consumed_recently")),
            "ml_tag": normalize_text(candidate.get("ml_tag") or "hybrid_faiss").upper() or "HYBRID_FAISS",
            "explanation": behavioral_insight,
            "dataset_serving_calories": round(to_float(candidate.get("dataset_serving_calories"), 0.0), 3),
            "calorie_diff_ratio": round(to_float(candidate.get("calorie_diff_ratio"), 0.0), 6),
            "title_similarity": round(to_float(candidate.get("title_similarity"), 0.0), 6),
            "mapping_acceptance_mode": str(candidate.get("mapping_acceptance_mode") or "strict"),
            "serving_fit_ratio": round(serving_fit_ratio, 6),
        }

    @staticmethod
    def _safety_to_local_candidates(meal_type: str, top_foods: list[str]) -> list[dict[str, Any]]:
        safety = build_safety_candidates(meal_type, top_foods)
        output: list[dict[str, Any]] = []
        for idx, item in enumerate(safety):
            per100 = item.get("per100", {}) or {}
            calories = to_float(per100.get("calories"), 0.0)
            protein = to_float(per100.get("protein"), 0.0)
            carbs = to_float(per100.get("carbs"), 0.0)
            fats = to_float(per100.get("fats"), 0.0)
            title_key = canonical_title_key(item.get("title") or "") or str(idx)
            recipe_id = f"safety-{meal_type}-{title_key}"
            category = normalize_text(item.get("category"))
            if category in {"main", "side", "drink"}:
                safety_source_keyword = f"{meal_type} {category} safety fallback"
                safety_recipe_category = category
                safety_keywords = f"{meal_type} {category} safety"
            else:
                safety_source_keyword = str(item.get("source_keyword") or "")
                safety_recipe_category = item.get("recipe_category") or None
                safety_keywords = str(item.get("keywords") or "")
            output.append(
                {
                    "id": recipe_id,
                    "item_id": recipe_id,
                    "recipe_id": recipe_id,
                    "title": item.get("title") or "Safety Food",
                    "dataset_title": item.get("title") or "Safety Food",
                    "image": item.get("image"),
                    "meal_type": meal_type,
                    "source_keyword": safety_source_keyword,
                    "safety_anchor": str(item.get("source_keyword") or ""),
                    "recipe_category": safety_recipe_category,
                    "keywords": safety_keywords,
                    "category": item.get("category") or None,
                    "serving_description": "1 serving",
                    "metric_serving_amount": 100.0,
                    "metric_serving_unit": "g",
                    "number_of_units": 1.0,
                    "measurement_description": "serving",
                    "serving_calories": calories,
                    "serving_protein": protein,
                    "serving_carbs": carbs,
                    "serving_fats": fats,
                    "dataset_serving_calories": calories,
                    "dataset_serving_protein": protein,
                    "dataset_serving_carbs": carbs,
                    "dataset_serving_fats": fats,
                    "per100": {
                        "calories": calories,
                        "protein": protein,
                        "carbs": carbs,
                        "fats": fats,
                    },
                    "knn_distance": 0.0,
                    "ml_tag": "SAFETY",
                    "food_id": recipe_id,
                }
            )
        return output

    @staticmethod
    def _beverage_safety_candidates(meal_type: str) -> list[dict[str, Any]]:
        drinks_by_meal = {
            "breakfast": [
                {"title": "Nescafe Iced Latte", "serving_calories": 110.0, "serving_amount": 240.0},
                {"title": "Berry Smoothie", "serving_calories": 180.0, "serving_amount": 300.0},
                {"title": "Chocolate Banana Flavour Almond Beverage", "serving_calories": 130.0, "serving_amount": 250.0},
                {"title": "Fresh Orange Juice", "serving_calories": 95.0, "serving_amount": 250.0},
            ],
            "lunch": [
                {"title": "Berry Smoothie", "serving_calories": 180.0, "serving_amount": 300.0},
                {"title": "Unsweetened Almond Beverage", "serving_calories": 45.0, "serving_amount": 250.0},
                {"title": "Fresh Orange Juice", "serving_calories": 95.0, "serving_amount": 250.0},
            ],
            "dinner": [
                {"title": "Coconut Water", "serving_calories": 60.0, "serving_amount": 330.0},
                {"title": "Berry Smoothie", "serving_calories": 180.0, "serving_amount": 300.0},
                {"title": "Unsweetened Almond Beverage", "serving_calories": 45.0, "serving_amount": 250.0},
            ],
        }

        normalized = normalize_text(meal_type)
        drinks = drinks_by_meal.get(normalized, [])
        output: list[dict[str, Any]] = []
        for idx, item in enumerate(drinks):
            title_key = canonical_title_key(item.get("title") or "") or str(idx)
            recipe_id = f"safety-drink-{meal_type}-{title_key}"
            serving_amount = to_float(item.get("serving_amount"), 250.0)
            serving_calories = to_float(item.get("serving_calories"), 0.0)
            output.append(
                {
                    "id": recipe_id,
                    "item_id": recipe_id,
                    "recipe_id": recipe_id,
                    "title": item["title"],
                    "dataset_title": item["title"],
                    "canonical_title": canonicalize_title(item["title"]),
                    "image": None,
                    "meal_type": meal_type,
                    "source_keyword": "beverage safety fallback",
                    "recipe_category": "beverage",
                    "keywords": "beverage drink",
                    "serving_description": f"1 serving ({int(round(serving_amount))} ml)",
                    "metric_serving_amount": serving_amount,
                    "metric_serving_unit": "ml",
                    "number_of_units": 1.0,
                    "measurement_description": "serving",
                    "serving_calories": serving_calories,
                    "serving_protein": 0.0,
                    "serving_carbs": 0.0,
                    "serving_fats": 0.0,
                    "dataset_serving_calories": serving_calories,
                    "dataset_serving_protein": 0.0,
                    "dataset_serving_carbs": 0.0,
                    "dataset_serving_fats": 0.0,
                    "per100": {
                        "calories": round((serving_calories / max(serving_amount, 1.0)) * 100.0, 3),
                        "protein": 0.0,
                        "carbs": 0.0,
                        "fats": 0.0,
                    },
                    "knn_distance": 0.0,
                    "ml_tag": "SAFETY",
                    "food_id": recipe_id,
                    "category": "drink",
                }
            )
        return output

    @staticmethod
    def _breakfast_side_safety_candidates(meal_type: str) -> list[dict[str, Any]]:
        if normalize_text(meal_type) != "breakfast":
            return []

        sides = [
            {"title": "Mixed Berries", "serving_calories": 60.0, "serving_amount": 100.0, "family": "fruit"},
            {"title": "Apple Slices", "serving_calories": 55.0, "serving_amount": 100.0, "family": "fruit"},
            {"title": "Walnut Crunch", "serving_calories": 110.0, "serving_amount": 20.0, "family": "nuts"},
            {"title": "Whole Grain Toast", "serving_calories": 95.0, "serving_amount": 40.0, "family": "bread"},
            {"title": "Granola Cup", "serving_calories": 120.0, "serving_amount": 35.0, "family": "grain"},
        ]
        output: list[dict[str, Any]] = []
        for idx, item in enumerate(sides):
            title_key = canonical_title_key(item.get("title") or "") or str(idx)
            recipe_id = f"safety-side-breakfast-{title_key}"
            serving_amount = to_float(item.get("serving_amount"), 100.0)
            serving_calories = to_float(item.get("serving_calories"), 0.0)
            output.append(
                {
                    "id": recipe_id,
                    "item_id": recipe_id,
                    "recipe_id": recipe_id,
                    "title": item["title"],
                    "dataset_title": item["title"],
                    "canonical_title": canonicalize_title(item["title"]),
                    "image": None,
                    "meal_type": meal_type,
                    "source_keyword": f"breakfast side safety {item['family']}",
                    "recipe_category": "side",
                    "keywords": f"side breakfast {item['family']}",
                    "serving_description": f"1 serving ({int(round(serving_amount))} g)",
                    "metric_serving_amount": serving_amount,
                    "metric_serving_unit": "g",
                    "number_of_units": 1.0,
                    "measurement_description": "serving",
                    "serving_calories": serving_calories,
                    "serving_protein": 0.0,
                    "serving_carbs": 0.0,
                    "serving_fats": 0.0,
                    "dataset_serving_calories": serving_calories,
                    "dataset_serving_protein": 0.0,
                    "dataset_serving_carbs": 0.0,
                    "dataset_serving_fats": 0.0,
                    "per100": {
                        "calories": round((serving_calories / max(serving_amount, 1.0)) * 100.0, 3),
                        "protein": 0.0,
                        "carbs": 0.0,
                        "fats": 0.0,
                    },
                    "knn_distance": 0.0,
                    "ml_tag": "SAFETY",
                    "food_id": recipe_id,
                    "category": "side",
                }
            )
        return output

    @staticmethod
    def _dinner_side_safety_candidates(meal_type: str) -> list[dict[str, Any]]:
        if normalize_text(meal_type) != "dinner":
            return []

        sides = [
            {"title": "Garden Salad", "serving_calories": 60.0, "serving_amount": 110.0, "family": "salad"},
            {"title": "Bean Salad", "serving_calories": 120.0, "serving_amount": 120.0, "family": "salad"},
            {"title": "Instant Miso Soup", "serving_calories": 45.0, "serving_amount": 240.0, "family": "soup"},
            {"title": "Tomato Soup", "serving_calories": 85.0, "serving_amount": 240.0, "family": "soup"},
            {"title": "Mediterranean Salad", "serving_calories": 110.0, "serving_amount": 120.0, "family": "salad"},
            {"title": "Roasted Broccoli", "serving_calories": 70.0, "serving_amount": 100.0, "family": "vegetable"},
        ]
        output: list[dict[str, Any]] = []
        for idx, item in enumerate(sides):
            title_key = canonical_title_key(item.get("title") or "") or str(idx)
            recipe_id = f"safety-side-{meal_type}-{title_key}"
            serving_amount = to_float(item.get("serving_amount"), 100.0)
            serving_calories = to_float(item.get("serving_calories"), 0.0)
            output.append(
                {
                    "id": recipe_id,
                    "item_id": recipe_id,
                    "recipe_id": recipe_id,
                    "title": item["title"],
                    "dataset_title": item["title"],
                    "canonical_title": canonicalize_title(item["title"]),
                    "image": None,
                    "meal_type": meal_type,
                    "source_keyword": f"dinner side safety {item['family']}",
                    "recipe_category": "side",
                    "keywords": f"side {item['family']} dinner",
                    "serving_description": f"1 serving ({int(round(serving_amount))} g)",
                    "metric_serving_amount": serving_amount,
                    "metric_serving_unit": "g",
                    "number_of_units": 1.0,
                    "measurement_description": "serving",
                    "serving_calories": serving_calories,
                    "serving_protein": 0.0,
                    "serving_carbs": 0.0,
                    "serving_fats": 0.0,
                    "dataset_serving_calories": serving_calories,
                    "dataset_serving_protein": 0.0,
                    "dataset_serving_carbs": 0.0,
                    "dataset_serving_fats": 0.0,
                    "per100": {
                        "calories": round((serving_calories / max(serving_amount, 1.0)) * 100.0, 3),
                        "protein": 0.0,
                        "carbs": 0.0,
                        "fats": 0.0,
                    },
                    "knn_distance": 0.0,
                    "ml_tag": "SAFETY",
                    "food_id": recipe_id,
                    "category": "side",
                    "side_family": item["family"],
                }
            )
        return output

    def _build_health_insight(
        self,
        candidates: list[dict[str, Any]],
        is_australian_user: bool,
    ) -> str:
        if not candidates:
            return ""
        health_scores = [
            to_float(item.get("health_score"), 0.0) for item in candidates if to_float(item.get("health_score"), 0.0) > 0
        ]
        max_hsr = max(health_scores) if health_scores else 0.0
        sugar_values = [
            to_float(item.get("serving_sugar"), 0.0) for item in candidates if to_float(item.get("serving_sugar"), 0.0) > 0
        ]
        low_sugar = bool(sugar_values) and max(sugar_values) <= float(SUGAR_LIMIT_PER_MEAL)
        local_count = sum(1 for item in candidates if bool(item.get("is_australian")))

        notes: list[str] = []
        if max_hsr >= float(MIN_HEALTH_SCORE):
            notes.append(f"meets the Health Star Rating of {max_hsr:.1f}")
        if low_sugar:
            notes.append(f"kept sugars under {float(SUGAR_LIMIT_PER_MEAL):.0f}g")
        if is_australian_user and local_count:
            notes.append("prioritized Australian items")

        if not notes:
            return ""
        return "We prioritized this meal because it " + " and ".join(notes) + "."

    @staticmethod
    def _role_title_key(candidate: dict[str, Any]) -> str:
        return canonical_title_key(candidate.get("canonical_title") or candidate.get("title") or "")

    @staticmethod
    def _candidate_service_cache(candidate: dict[str, Any]) -> dict[str, Any]:
        cache = candidate.get("_service_cache")
        if isinstance(cache, dict):
            return cache

        cache = {}
        candidate["_service_cache"] = cache
        return cache

    def _is_role_compatible(self, candidate: dict[str, Any], role: str) -> bool:
        role_key = normalize_text(role)
        if role_key not in {"main", "side", "drink"}:
            return True

        cache = self._candidate_service_cache(candidate)
        role_cache = cache.setdefault("role_compatibility", {})
        if role_key in role_cache:
            return bool(role_cache[role_key])

        compatible = is_candidate_role_compatible(candidate, role_key)
        role_cache[role_key] = compatible
        return compatible

    def _role_quality_multiplier(self, candidate: dict[str, Any], role: str, meal_type: str) -> float:
        role_key = normalize_text(role)
        meal_key = normalize_text(meal_type)
        if role_key not in {"main", "side", "drink"}:
            return 1.0

        cache = self._candidate_service_cache(candidate)
        quality_cache = cache.setdefault("role_quality", {})
        cache_key = f"{meal_key}:{role_key}"
        if cache_key in quality_cache:
            return float(quality_cache[cache_key])

        if role_key == "main":
            quality = float(main_role_quality_multiplier(candidate, meal_type))
        elif role_key == "side":
            quality = float(side_role_quality_multiplier(candidate, meal_type))
        else:
            quality = float(drink_role_quality_multiplier(candidate, meal_type))

        quality_cache[cache_key] = quality
        return quality

    @staticmethod
    def _candidate_serving_calories(candidate: dict[str, Any]) -> float:
        return float(to_float(candidate.get("serving_calories"), to_float(candidate.get("calories"), 0.0)))

    def _candidate_combo_score(
        self,
        candidate: dict[str, Any],
        category: str,
        meal_type: str,
        *,
        habitual: bool = False,
    ) -> float:
        category_key = normalize_text(category)
        meal_key = normalize_text(meal_type)
        if category_key not in {"main", "side", "drink"}:
            return 0.0

        if not is_candidate_role_compatible(candidate, category_key):
            return 0.0
        if (
            category_key == "side"
            and meal_key in {"lunch", "dinner"}
            and self._is_blocked_non_breakfast_side_candidate(candidate)
        ):
            return 0.0

        score = to_float(candidate.get("score"), 0.0)
        if habitual:
            score *= 1.1
        if category_key == "side":
            score *= side_role_quality_multiplier(candidate, meal_type)
        elif category_key == "drink":
            score *= drink_role_quality_multiplier(candidate, meal_type)

        return float(score)

    def _candidate_dinner_side_priority(self, candidate: dict[str, Any]) -> tuple[int, int]:
        text = normalize_text(candidate.get("title") or candidate.get("canonical_title") or "")
        preferred_terms = (
            "salad",
            "vegetable",
            "greens",
            "broccoli",
            "bean",
            "beans",
            "slaw",
            "soup",
            "carrot",
            "pea",
            "peas",
        )
        rice_terms = (
            "rice",
            "fried rice",
            "jasmine",
            "pilaf",
            "risotto",
        )
        preferred_hit = any(term in text for term in preferred_terms)
        rice_hit = any(term in text for term in rice_terms)
        return (1 if preferred_hit else 0, 1 if rice_hit and not preferred_hit else 0)

    def _role_compatible_candidates(self, candidates: list[dict[str, Any]], role: str) -> list[dict[str, Any]]:
        compatible: list[dict[str, Any]] = []
        seen: set[str] = set()
        for candidate in candidates:
            if self._infer_combo_category(candidate) != role:
                continue
            if not self._is_role_compatible(candidate, role):
                continue
            if normalize_text(role) == "side" and self._is_blocked_dinner_side_candidate(candidate):
                continue
            dedupe_key = self._role_title_key(candidate) or self._recipe_id_from_candidate(candidate)
            if dedupe_key and dedupe_key in seen:
                continue
            if dedupe_key:
                seen.add(dedupe_key)
            compatible.append(candidate)
        return compatible

    @staticmethod
    def _drink_title_text(candidate: dict[str, Any]) -> str:
        return normalize_text(
            " ".join(
                str(candidate.get(key) or "")
                for key in ("title", "canonical_title", "mapped_title", "mapped_canonical_title", "brand_name")
            )
        )

    @staticmethod
    def _is_plant_milk_drink_candidate(candidate: dict[str, Any]) -> bool:
        title_text = RecommendationService._drink_title_text(candidate)
        if not title_text:
            return False
        plant_milk_terms = (
            "almond beverage",
            "almond milk",
            "cashew beverage",
            "cashew milk",
            "coconut beverage",
            "coconut milk",
            "non-dairy beverage",
            "non dairy beverage",
            "oat beverage",
            "oat milk",
            "plant based beverage",
            "plant based milk",
            "plant-based beverage",
            "plant-based milk",
            "rice beverage",
            "rice milk",
            "soy plant based beverage",
            "soy plant-based beverage",
            "soy beverage",
            "soy milk",
            "soya beverage",
            "soya milk",
            "soymilk beverage",
        )
        if any(term in title_text for term in plant_milk_terms):
            return True
        nut_terms = ("almond", "cashew", "hazelnut", "macadamia", "walnut")
        soy_terms = ("soy", "soya")
        beverage_terms = ("beverage", "milk", "non-dairy", "non dairy", "plant-based", "plant based")
        return (
            (any(term in title_text for term in nut_terms) or any(term in title_text for term in soy_terms))
            and any(term in title_text for term in beverage_terms)
        )

    @staticmethod
    def _is_generic_plant_milk_drink_candidate(candidate: dict[str, Any]) -> bool:
        if not RecommendationService._is_plant_milk_drink_candidate(candidate):
            return False
        title_text = RecommendationService._drink_title_text(candidate)
        if not title_text:
            return False
        generic_terms = (
            "non-dairy",
            "original",
            "plain",
            "sweetened",
            "unsweetened",
            "unflavored",
            "unflavoured",
            "vanilla",
        )
        return any(term in title_text for term in generic_terms)

    @staticmethod
    def _is_nut_based_plant_milk_drink_candidate(candidate: dict[str, Any]) -> bool:
        if not RecommendationService._is_plant_milk_drink_candidate(candidate):
            return False
        title_text = RecommendationService._drink_title_text(candidate)
        if not title_text:
            return False
        nut_based_terms = (
            "almond beverage",
            "almond milk",
            "cashew beverage",
            "cashew milk",
            "hazelnut beverage",
            "hazelnut milk",
            "macadamia beverage",
            "macadamia milk",
            "nut-based drinks",
            "nut based drinks",
            "walnut beverage",
            "walnut milk",
        )
        if any(term in title_text for term in nut_based_terms):
            return True
        nut_terms = ("almond", "cashew", "hazelnut", "macadamia", "walnut")
        beverage_terms = ("beverage", "milk", "non-dairy", "non dairy", "plant-based", "plant based")
        return any(term in title_text for term in nut_terms) and any(term in title_text for term in beverage_terms)

    def _dinner_primary_plant_milk_priority(self, candidate: dict[str, Any]) -> tuple[float, ...]:
        return (
            1.0 if RecommendationService._is_nut_based_plant_milk_drink_candidate(candidate) else 0.0,
            1.0 if not RecommendationService._is_generic_plant_milk_drink_candidate(candidate) else 0.0,
            1.0 if not RecommendationService._is_safety_drink_candidate(candidate) else 0.0,
            drink_role_quality_multiplier(candidate, "dinner"),
            float(to_float(candidate.get("score"), 0.0)),
        )

    def _ordered_dinner_drink_candidates(self, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return sorted(
            [
                candidate
                for candidate in candidates
                if self._infer_combo_category(candidate) == "drink"
                and is_candidate_role_compatible(candidate, "drink")
            ],
            key=lambda candidate: (
                1 if "smoothie" in self._drink_title_text(candidate) else 0,
                1 if "coconut water" in self._drink_title_text(candidate) else 0,
                1 if self._is_safety_drink_candidate(candidate) else 0,
                1 if not self._is_generic_plant_milk_drink_candidate(candidate) else 0,
                1 if not self._is_plant_milk_drink_candidate(candidate) else 0,
                0
                if (
                    "juice" in self._drink_title_text(candidate)
                    and "vegetable juice" not in self._drink_title_text(candidate)
                    and "smoothie" not in self._drink_title_text(candidate)
                )
                else 1,
                drink_role_quality_multiplier(candidate, "dinner"),
                to_float(candidate.get("score"), 0.0),
            ),
            reverse=True,
        )

    def _select_primary_dinner_drink_candidates(
        self,
        candidates: list[dict[str, Any]],
        limit: int,
    ) -> list[dict[str, Any]]:
        if limit <= 0:
            return []

        prioritized = self._ordered_dinner_drink_candidates(candidates)
        if not prioritized:
            return []

        selected: list[dict[str, Any]] = []
        selected_keys: set[str] = set()
        plant_milk_selected = 0

        primary_plant_milk_candidates = [
            candidate
            for candidate in prioritized
            if self._is_plant_milk_drink_candidate(candidate) and not self._is_safety_drink_candidate(candidate)
        ]
        if primary_plant_milk_candidates:
            lead_candidate = max(primary_plant_milk_candidates, key=self._dinner_primary_plant_milk_priority)
            lead_key = self._candidate_merge_key(lead_candidate)
            if lead_key:
                selected_keys.add(lead_key)
            selected.append(lead_candidate)
            plant_milk_selected = 1

        for candidate in prioritized:
            candidate_key = self._candidate_merge_key(candidate)
            if candidate_key and candidate_key in selected_keys:
                continue
            if self._is_plant_milk_drink_candidate(candidate) and plant_milk_selected >= 1:
                continue
            if candidate_key:
                selected_keys.add(candidate_key)
            if self._is_plant_milk_drink_candidate(candidate):
                plant_milk_selected += 1
            selected.append(candidate)
            if len(selected) >= limit:
                break

        return selected[:limit]

    @staticmethod
    def _is_safety_drink_candidate(candidate: dict[str, Any]) -> bool:
        recipe_id = RecommendationService._recipe_id_from_candidate(candidate)
        if RecommendationService._is_safety_recipe_id(recipe_id):
            return True
        source_keyword = normalize_text(candidate.get("source_keyword"))
        return "beverage safety" in source_keyword or "safety fallback" in source_keyword

    def _select_diverse_dinner_drink_candidates(
        self,
        candidates: list[dict[str, Any]],
        limit: int,
    ) -> list[dict[str, Any]]:
        if limit <= 0:
            return []

        prioritized = self._ordered_dinner_drink_candidates(candidates)
        if not prioritized:
            return []

        selected: list[dict[str, Any]] = []
        selected_keys: set[str] = set()
        plant_milk_selected = 0

        for candidate in prioritized:
            candidate_key = self._candidate_merge_key(candidate)
            if candidate_key and candidate_key in selected_keys:
                continue
            if self._is_plant_milk_drink_candidate(candidate) and plant_milk_selected >= 1:
                continue
            if candidate_key:
                selected_keys.add(candidate_key)
            if self._is_plant_milk_drink_candidate(candidate):
                plant_milk_selected += 1
            selected.append(candidate)
            if len(selected) >= limit:
                return selected

        for candidate in prioritized:
            candidate_key = self._candidate_merge_key(candidate)
            if candidate_key and candidate_key in selected_keys:
                continue
            if candidate_key:
                selected_keys.add(candidate_key)
            selected.append(candidate)
            if len(selected) >= limit:
                break

        return selected

    @staticmethod
    def _drink_pool_floor(meal_type: str) -> int:
        normalized = normalize_text(meal_type)
        if normalized == "dinner":
            return 2
        if normalized == "lunch":
            return 2
        return 1

    @staticmethod
    def _drink_diversity_target(meal_type: str) -> int:
        normalized = normalize_text(meal_type)
        if normalized == "dinner":
            return 3
        if normalized == "lunch":
            return 2
        return 1

    @staticmethod
    def _is_alcoholic_drink_candidate(candidate: dict[str, Any]) -> bool:
        candidate_text = normalize_text(
            " ".join(
                str(candidate.get(key) or "")
                for key in (
                    "title",
                    "canonical_title",
                    "mapped_title",
                    "mapped_canonical_title",
                    "brand_name",
                    "source_keyword",
                    "recipe_category",
                    "keywords",
                )
            )
        )
        if not candidate_text:
            return False

        normalized_candidate_text = re.sub(r"[^a-z0-9]+", " ", candidate_text).strip()
        if not normalized_candidate_text:
            return False

        padded_candidate_text = f" {normalized_candidate_text} "
        alcohol_terms = (
            "alcoholic beverage",
            "alcoholic beverages",
            "beer",
            "bourbon",
            "champagne",
            "cider",
            "cocktail",
            "cocktails",
            "distilled beverage",
            "distilled beverages",
            "eau de vie",
            "gin",
            "hard liquor",
            "hard liquors",
            "lager",
            "liqueur",
            "liqueurs",
            "liquor",
            "prosecco",
            "rum",
            "spirit",
            "spirituosen",
            "tequila",
            "vodka",
            "vodkas",
            "whiskey",
            "whisky",
            "wine",
        )
        return any(
            f" {re.sub(r'[^a-z0-9]+', ' ', normalize_text(term)).strip()} " in padded_candidate_text
            for term in alcohol_terms
        )

    def _filter_drink_candidates_for_meal(self, meal_type: str, drink_candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        normalized = normalize_text(meal_type)
        if normalized not in {"lunch", "dinner"}:
            return drink_candidates

        preferred: list[dict[str, Any]] = []
        fallback: list[dict[str, Any]] = []
        safety_fallback: list[dict[str, Any]] = []
        last_resort: list[dict[str, Any]] = []
        for candidate in drink_candidates:
            title_text = normalize_text(
                " ".join(
                    str(candidate.get(key) or "")
                    for key in ("title", "canonical_title", "mapped_title", "mapped_canonical_title", "brand_name")
                )
            )
            if not is_candidate_role_compatible(candidate, "drink"):
                continue
            if RecommendationService._is_alcoholic_drink_candidate(candidate):
                continue
            coffee_like = any(
                term in title_text for term in ("coffee", "latte", "cappuccino", "espresso", "macchiato", "cold brew")
            )
            dessert_drink_like = any(
                term in title_text for term in ("cocoa", "hot chocolate", "chocolate drink")
            )
            quality = drink_role_quality_multiplier(candidate, meal_type)
            plant_milk_like = RecommendationService._is_plant_milk_drink_candidate(candidate)
            safety_candidate = RecommendationService._is_safety_drink_candidate(candidate)

            if normalized == "lunch" and coffee_like:
                continue
            if normalized in {"lunch", "dinner"} and dessert_drink_like:
                continue
            if quality >= (0.62 if normalized == "lunch" else 0.55):
                preferred.append(candidate)
                continue
            if safety_candidate and quality >= (0.28 if normalized == "lunch" else 0.24):
                safety_fallback.append(candidate)
                continue
            if not coffee_like and not plant_milk_like and quality >= 0.42:
                fallback.append(candidate)
                continue
            if not coffee_like and quality >= (0.28 if normalized == "lunch" else 0.2):
                last_resort.append(candidate)

        if preferred or fallback or safety_fallback:
            prioritized = merge_candidates(
                preferred,
                fallback,
                safety_fallback,
                max_items=max(12, len(preferred) + len(fallback) + len(safety_fallback)),
            )
            if normalized in {"lunch", "dinner"} and last_resort:
                prioritized_keys = {
                    RecommendationService._candidate_merge_key(candidate)
                    for candidate in prioritized
                    if RecommendationService._candidate_merge_key(candidate)
                }
                needed_last_resort = max(
                    0,
                    RecommendationService._drink_diversity_target(meal_type) - len(prioritized_keys),
                )
                if needed_last_resort > 0:
                    supplemental_last_resort = [
                        candidate
                        for candidate in sorted(
                            last_resort,
                            key=lambda candidate: (
                                drink_role_quality_multiplier(candidate, meal_type),
                                0 if RecommendationService._is_plant_milk_drink_candidate(candidate) else 1,
                                to_float(candidate.get("score"), 0.0),
                            ),
                            reverse=True,
                        )
                        if RecommendationService._candidate_merge_key(candidate) not in prioritized_keys
                    ]
                    prioritized = merge_candidates(
                        prioritized,
                        supplemental_last_resort[:needed_last_resort],
                        max_items=max(12, len(prioritized) + needed_last_resort),
                    )
            return prioritized
        return last_resort

    @staticmethod
    def _side_pool_floor(meal_type: str) -> int:
        normalized = normalize_text(meal_type)
        if normalized == "dinner":
            return 4
        if normalized == "lunch":
            return 3
        return 2

    @staticmethod
    def _main_pool_floor(meal_type: str) -> int:
        normalized = normalize_text(meal_type)
        if normalized == "breakfast":
            return 3
        if normalized in {"lunch", "dinner"}:
            return 2
        return 0

    @staticmethod
    def _should_skip_parallel_query_expansion(
        meal_type: str,
        merged_role_counts: dict[str, int],
        *,
        dedicated_parallel_path: bool,
        dinner_drink_target: int = 0,
    ) -> bool:
        if not dedicated_parallel_path:
            return False

        normalized = normalize_text(meal_type)
        minimum_parallel_query_expansion_drink_count = max(1, RecommendationService._drink_pool_floor(meal_type))
        main_count = int(merged_role_counts.get("main", 0))
        side_count = int(merged_role_counts.get("side", 0))
        drink_count = int(merged_role_counts.get("drink", 0))

        if normalized == "breakfast":
            return (
                main_count >= 8
                and side_count >= 8
                and drink_count >= max(3, minimum_parallel_query_expansion_drink_count)
            )
        if normalized == "lunch":
            return (
                main_count >= 5
                and side_count >= 8
                and drink_count >= 1
            )
        if normalized == "dinner":
            return (
                main_count >= 6
                and side_count >= 3
                and drink_count >= max(2, int(dinner_drink_target))
            )
        return False

    @staticmethod
    def _candidate_merge_key(candidate: dict[str, Any]) -> str:
        return str(
            candidate.get("id")
            or candidate.get("food_id")
            or candidate.get("recipe_id")
            or RecommendationService._role_title_key(candidate)
            or ""
        ).strip()

    @staticmethod
    def _primary_search_budgets(meal_type: str, candidate_pool_target: int, prefetch_target: int) -> dict[str, int]:
        normalized = normalize_text(meal_type)
        main_top_k = max(32, int(round(candidate_pool_target * 0.70)))
        side_top_k = max(18, int(round(candidate_pool_target * (0.38 if normalized == "dinner" else 0.32))))
        if normalized == "breakfast":
            drink_top_k = max(26, int(round(candidate_pool_target * 0.32)))
        elif normalized == "dinner":
            drink_top_k = max(24, int(round(candidate_pool_target * 0.30)))
        else:
            drink_top_k = max(12, int(round(candidate_pool_target * 0.20)))

        main_prefetch = max(main_top_k, int(round(prefetch_target * 0.72)))
        side_prefetch = max(side_top_k, int(round(prefetch_target * (0.26 if normalized == "breakfast" else 0.30))))
        if normalized == "breakfast":
            drink_prefetch = max(drink_top_k, int(round(prefetch_target * 0.24)))
        elif normalized == "dinner":
            drink_prefetch = max(drink_top_k, int(round(prefetch_target * 0.25)))
        else:
            drink_prefetch = max(drink_top_k, int(round(prefetch_target * 0.18)))

        return {
            "main_top_k": main_top_k,
            "side_top_k": side_top_k,
            "drink_top_k": drink_top_k,
            "main_prefetch": main_prefetch,
            "side_prefetch": side_prefetch,
            "drink_prefetch": drink_prefetch,
        }

    # NOTE: Primary role retrieval is independent across main/side/drink.
    # Running those DuckDB searches concurrently reduces cold-slot retrieval time
    # without changing ranking or mapping semantics.
    def _search_primary_role_candidates(
        self,
        *,
        meal_type: str,
        main_query_vec: np.ndarray,
        side_query_vec: np.ndarray,
        drink_query_vec: np.ndarray,
        side_query_str: str,
        drink_query_str: str,
        primary_search_budgets: dict[str, int],
        is_australian_user: bool,
        use_dedicated_search_connection: bool = False,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, bool]]:
        # Allow both slot-level and role-level parallelism simultaneously.
        # When slots run in parallel each slot already has its own dedicated connection;
        # inner role-parallel is still safe because each search call spawns fresh
        # ThreadPoolExecutor workers that each request their own dedicated connection.
        run_parallel_retrieval = self.parallel_primary_role_retrieval_enabled
        search_with_dedicated_connection = use_dedicated_search_connection or run_parallel_retrieval
        stagger_dedicated_drink_search = (
            run_parallel_retrieval
            and use_dedicated_search_connection
            and normalize_text(meal_type) == "lunch"
        )

        def _search_main_candidates() -> list[dict[str, Any]]:
            candidates = self.local_dataset.search(
                meal_type=meal_type,
                query_vector=main_query_vec,
                top_k=primary_search_budgets["main_top_k"],
                prefetch=primary_search_budgets["main_prefetch"],
                is_australian_user=is_australian_user,
                role_hint="main",
                dedicated_connection=search_with_dedicated_connection,
            )
            return self._filter_main_candidates_for_meal(meal_type, candidates)

        def _search_side_candidates() -> list[dict[str, Any]]:
            candidates = self.local_dataset.search(
                meal_type=meal_type,
                query_vector=side_query_vec,
                top_k=primary_search_budgets["side_top_k"],
                prefetch=primary_search_budgets["side_prefetch"],
                is_australian_user=is_australian_user,
                text_query=side_query_str,
                role_hint="side",
                dedicated_connection=search_with_dedicated_connection,
            )
            return self._filter_side_candidates_for_meal(meal_type, candidates)

        def _search_drink_candidates() -> list[dict[str, Any]]:
            return self.local_dataset.search(
                meal_type=meal_type,
                query_vector=drink_query_vec,
                top_k=primary_search_budgets["drink_top_k"],
                prefetch=primary_search_budgets["drink_prefetch"],
                is_australian_user=is_australian_user,
                text_query=drink_query_str,
                role_hint="drink",
                dedicated_connection=search_with_dedicated_connection,
            )

        if run_parallel_retrieval and stagger_dedicated_drink_search:
            # The all-slot cold path already fans out three slots at once. Stagger the
            # lunch drink retrieval behind main/side on that dedicated path so we reduce
            # peak concurrent DuckDB searches where the current cold route spends the
            # most retrieval time, without changing budgets or top-k semantics.
            main_future = self._primary_retrieval_executor.submit(_search_main_candidates)
            side_future = self._primary_retrieval_executor.submit(_search_side_candidates)
            main_candidates = main_future.result()
            side_candidates = side_future.result()
            drink_candidates = _search_drink_candidates()
        elif run_parallel_retrieval:
            main_future = self._primary_retrieval_executor.submit(_search_main_candidates)
            side_future = self._primary_retrieval_executor.submit(_search_side_candidates)
            drink_future = self._primary_retrieval_executor.submit(_search_drink_candidates)
            main_candidates = main_future.result()
            side_candidates = side_future.result()
            drink_candidates = drink_future.result()
        else:
            main_candidates = _search_main_candidates()
            side_candidates = _search_side_candidates()
            drink_candidates = _search_drink_candidates()

        return main_candidates, side_candidates, drink_candidates, {
            "parallel_used": run_parallel_retrieval,
            "dedicated_connection": search_with_dedicated_connection,
            "staggered_drink_search": stagger_dedicated_drink_search,
        }

    def _merge_balanced_primary_candidates(
        self,
        main_candidates: list[dict[str, Any]],
        side_candidates: list[dict[str, Any]],
        drink_candidates: list[dict[str, Any]],
        max_items: int,
    ) -> list[dict[str, Any]]:
        role_pools = {
            "main": list(main_candidates or []),
            "side": list(side_candidates or []),
            "drink": list(drink_candidates or []),
        }
        role_indices = {role: 0 for role in role_pools}
        merged: list[dict[str, Any]] = []
        seen: set[str] = set()
        weighted_role_order = ("main", "main", "main", "side", "drink")

        def _append_next(role: str) -> bool:
            candidates = role_pools[role]
            while role_indices[role] < len(candidates):
                candidate = candidates[role_indices[role]]
                role_indices[role] += 1
                dedupe_key = self._candidate_merge_key(candidate)
                if not dedupe_key or dedupe_key in seen:
                    continue
                seen.add(dedupe_key)
                merged.append(candidate)
                return True
            return False

        while len(merged) < max_items:
            progressed = False
            for role in weighted_role_order:
                if len(merged) >= max_items:
                    break
                if _append_next(role):
                    progressed = True
            if not progressed:
                break

        if len(merged) < max_items:
            for role in ("main", "side", "drink"):
                while len(merged) < max_items and _append_next(role):
                    continue

        return merged

    def _filter_main_candidates_for_meal(self, meal_type: str, main_candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        normalized = normalize_text(meal_type)
        if normalized not in {"breakfast", "lunch", "dinner"}:
            return main_candidates

        eligible_candidates = [
            candidate
            for candidate in main_candidates
            if not (
                normalized in {"lunch", "dinner"}
                and self._is_blocked_non_breakfast_main_candidate(candidate)
            )
        ]
        preferred: list[dict[str, Any]] = []
        fallback: list[dict[str, Any]] = []
        minimum_candidates = self._main_pool_floor(meal_type)

        for candidate in eligible_candidates:
            quality = main_role_quality_multiplier(candidate, meal_type)
            if normalized == "breakfast":
                if quality >= 0.68 or self._is_breakfast_aligned_main_candidate(candidate):
                    preferred.append(candidate)
                elif quality >= 0.52:
                    fallback.append(candidate)
                continue

            if normalized == "lunch":
                if quality >= 0.62:
                    preferred.append(candidate)
                elif quality >= 0.5:
                    fallback.append(candidate)
                continue

            if quality >= 0.66:
                preferred.append(candidate)
            elif quality >= 0.54:
                fallback.append(candidate)

        if len(preferred) >= minimum_candidates:
            return preferred
        if len(preferred) + len(fallback) >= minimum_candidates:
            return preferred + fallback
        return eligible_candidates

    @staticmethod
    def _is_blocked_non_breakfast_main_candidate(candidate: dict[str, Any]) -> bool:
        title_text = normalize_text(
            " ".join(
                str(candidate.get(key) or "")
                for key in (
                    "canonical_title",
                    "mapped_canonical_title",
                    "mapped_title",
                    "title",
                    "original_title",
                )
            )
        )
        if not title_text:
            return False

        blocked_titles = (
            "chili hot bowl",
            "chili burrito bowl",
        )
        return any(title in title_text for title in blocked_titles)

    @staticmethod
    def _is_breakfast_aligned_main_candidate(candidate: dict[str, Any]) -> bool:
        title_text = normalize_text(
            " ".join(
                str(candidate.get(key) or "")
                for key in ("canonical_title", "title", "original_title")
            )
        )
        context_text = normalize_text(
            " ".join(
                str(candidate.get(key) or "")
                for key in (
                    "recipe_category",
                    "food_type",
                    "serving_description",
                    "measurement_description",
                )
            )
        )
        text = " ".join(part for part in (title_text, context_text) if part)
        if not text:
            return False

        breakfast_dessert_main_markers = (
            "frozen custard",
            "gelato",
            "ice cream",
            "sundae",
            "waffle cone",
        )
        if any(marker in title_text for marker in breakfast_dessert_main_markers):
            return False
        if "scoop" in title_text and any(marker in title_text for marker in ("cone", "custard", "gelato", "ice cream")):
            return False

        breakfast_true_main_markers = (
            "acai",
            "avocado toast",
            "breakfast burrito",
            "egg",
            "eggs",
            "hash brown",
            "oat",
            "oatmeal",
            "omelet",
            "omelette",
            "parfait",
            "porridge",
            "quiche",
            "scramble",
            "skillet",
            "smoothie bowl",
            "toast",
        )
        breakfast_markers = (
            "breakfast",
            "brunch",
            "egg",
            "omelet",
            "omelette",
            "oat",
            "oatmeal",
            "porridge",
            "toast",
            "bagel",
            "muffin",
            "pancake",
            "waffle",
            "yogurt",
            "yoghurt",
            "parfait",
            "smoothie bowl",
        )
        breakfast_product_markers = (
            "breakfast cereal",
            "biscuit",
            "biscuits",
            "cereal",
            "cookie",
            "cookies",
            "cracker",
            "crackers",
            "crispbread",
            "crispbreads",
            "croissant",
            "croissants",
            "flapjack",
            "flapjacks",
            "granola",
            "macaroon",
            "macaroons",
            "muesli",
            "pastry",
            "pastries",
            "protein bar",
            "protein bars",
            "protein powder",
            "protein-powders",
            "tosta",
            "tostas",
            "viennoiseries",
        )
        breakfast_supplement_markers = (
            "bodybuilding",
            "bodybuilding-supplements",
            "compléments alimentaires",
            "complements alimentaires",
            "dietary-supplements",
            "protein powder",
            "protein powders",
            "protein-powders",
            "protéines en poudre",
            "supplement",
            "supplements",
            "whey",
        )
        breakfast_condiment_markers = (
            "apple butter",
            "caviar",
            "caviars",
            "chutney",
            "cod caviar",
            "curd",
            "curds",
            "fish eggs",
            "fruit curd",
            "fruit curds",
            "jam",
            "jelly",
            "marmalade",
            "preserve",
            "preserves",
            "roe",
            "salted spreads",
            "spread",
            "spreads",
            "sweet spreads",
        )
        snack_family_markers = (
            "chips",
            "crisps",
            "salty-snacks",
            "snack",
            "snacks",
            "sweet snack",
            "sweet snacks",
            "sweet-snacks",
        )
        non_breakfast_markers = (
            "brownie",
            "burger",
            "burrito",
            "club",
            "curry",
            "egg roll",
            "egg rolls",
            "fries",
            "lasagna",
            "pasta",
            "pho",
            "pizza",
            "ramen",
            "sandwich",
            "slider",
            "steak",
            "sushi",
            "taco",
            "wings",
        )
        if any(marker in text for marker in non_breakfast_markers):
            return False
        if any(marker in text for marker in breakfast_supplement_markers):
            return False
        if any(marker in text for marker in breakfast_condiment_markers) and not any(
            marker in title_text for marker in breakfast_true_main_markers
        ):
            return False
        if any(marker in text for marker in snack_family_markers):
            return False
        if any(marker in text for marker in breakfast_product_markers) and not any(
            marker in text for marker in breakfast_true_main_markers
        ):
            return False
        return any(marker in text for marker in breakfast_markers)

    @staticmethod
    def _is_blocked_breakfast_main_candidate(candidate: dict[str, Any]) -> bool:
        title_text = normalize_text(
            " ".join(
                str(candidate.get(key) or "")
                for key in ("canonical_title", "title", "original_title")
            )
        )
        context_text = normalize_text(
            " ".join(
                str(candidate.get(key) or "")
                for key in (
                    "source_keyword",
                    "recipe_category",
                    "keywords",
                    "ingredient_text",
                    "food_type",
                )
            )
        )
        text = " ".join(part for part in (title_text, context_text) if part)
        if not text or is_placeholder_title(title_text):
            return True

        blocked_markers = (
            "baking mix",
            "egg replacement",
            "egg replacer",
            "egg substitute",
            "ei ersatz",
            "ei-ersatz",
            "foie de canard",
            "foie gras",
            "foies gras",
            "seasoning mix",
            "spice mix",
        )
        return any(marker in text for marker in blocked_markers)

    def _meal_main_safety_candidates(self, meal_type: str, top_foods: list[str]) -> list[dict[str, Any]]:
        candidates = [
            candidate
            for candidate in self._safety_to_local_candidates(meal_type, top_foods)
            if normalize_text(candidate.get("category")) == "main"
        ]
        if normalize_text(meal_type) == "breakfast":
            aligned = [candidate for candidate in candidates if self._is_breakfast_aligned_main_candidate(candidate)]
            return aligned or candidates
        return candidates

    def _meal_side_safety_candidates(self, meal_type: str, top_foods: list[str]) -> list[dict[str, Any]]:
        base_candidates = [
            candidate
            for candidate in self._safety_to_local_candidates(meal_type, top_foods)
            if normalize_text(candidate.get("category")) == "side"
        ]
        if normalize_text(meal_type) != "breakfast":
            return base_candidates
        return merge_candidates(
            self._breakfast_side_safety_candidates(meal_type),
            base_candidates,
            max_items=12,
        )

    @staticmethod
    def _drink_supplement_regex(meal_type: str) -> str:
        normalized = normalize_text(meal_type)
        if normalized == "breakfast":
            keywords = [
                *COMBO_DRINK_KEYWORDS,
                "drink",
                "beverage",
                "smoothie",
                "milk",
                "latte",
                "cappuccino",
                "orange juice",
                "coconut water",
            ]
        elif normalized == "dinner":
            keywords = [
                "soya beverage sweetened",
                "soy beverage",
                "soymilk beverage",
                "almond beverage",
                "almond milk",
                "oat beverage",
                "oat milk",
            ]
        else:
            keywords = [
                "berry smoothie",
                "smoothie",
                "juice drink",
                "kombucha",
                "lemonade",
                "iced tea",
                "herbal tea",
                "coconut water",
                "orange juice",
                "vegetable juice",
                "sparkling water",
                "seltzer",
            ]
        keywords = dedupe_strings(keywords)
        return "|".join(normalize_text(term) for term in keywords if normalize_text(term))

    @staticmethod
    def _primary_drink_query_regex(meal_type: str) -> str:
        normalized = normalize_text(meal_type)
        if normalized == "breakfast":
            keywords = [
                "coffee",
                "iced coffee",
                "cold brew",
                "tea",
                "smoothie",
                "espresso",
                "iced espresso",
                "cappuccino",
                "matcha",
                "macchiato",
                "orange juice",
                "coconut water",
                "soy beverage",
                "soymilk beverage",
                "non dairy beverage",
                "non-dairy beverage",
                "plant-based beverage",
            ]
        elif normalized == "dinner":
            keywords = [
                "water",
                "juice",
                "berry smoothie",
                "smoothie",
                "sparkling water",
                "seltzer",
                "kombucha",
                "lemonade",
                "iced tea",
                "herbal tea",
                "coconut water",
                "orange juice",
                "vegetable juice",
                "almond beverage",
                "almond milk",
                "oat beverage",
                "oat milk",
                "soy beverage",
                "soy milk",
                "soymilk beverage",
                "plant-based milk",
                "plant-based beverage",
                "non dairy",
                "non-dairy",
            ]
        else:
            keywords = [
                "water",
                "juice",
                "smoothie",
                "sparkling water",
                "seltzer",
                "kombucha",
                "lemonade",
                "iced tea",
                "herbal tea",
                "coconut water",
                "orange juice",
                "vegetable juice",
                "soy",
                "soymilk",
                "non dairy",
                "non-dairy",
            ]
        keywords = dedupe_strings(keywords)
        return "|".join(normalize_text(term) for term in keywords if normalize_text(term))

    @staticmethod
    def _side_supplement_regex(meal_type: str) -> str:
        normalized = normalize_text(meal_type)
        if normalized == "breakfast":
            terms = dedupe_strings(
                [
                    "berries",
                    "fruit",
                    "fruit cup",
                    "banana",
                    "apple",
                    "granola",
                    "muesli",
                    "toast",
                    "bread",
                    "bagel",
                    "english muffin",
                    "nuts",
                    "seed",
                    "cracker",
                    "crispbread",
                ]
            )
            return "|".join(normalize_text(term) for term in terms if normalize_text(term))
        if normalized == "lunch":
            terms = dedupe_strings(
                [
                    "salad",
                    "side salad",
                    "vegetable",
                    "broccoli",
                    "greens",
                    "beans",
                    "fruit",
                    "apple",
                    "banana",
                    "yogurt",
                    "nuts",
                    "soup",
                ]
            )
            return "|".join(normalize_text(term) for term in terms if normalize_text(term))
        if normalized != "dinner":
            return ""
        terms = dedupe_strings(
            [
                "bean salad",
                "broth",
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
                "side salad",
                "slaw",
                "steamed vegetables",
                "tomato soup",
                "vegetable medley",
                "vegetable soup",
            ]
        )
        return "|".join(normalize_text(term) for term in terms if normalize_text(term))

    @staticmethod
    def _dinner_side_family(candidate: dict[str, Any]) -> str:
        explicit_family = normalize_text(candidate.get("side_family"))
        if explicit_family in {"soup", "salad", "vegetable"}:
            return explicit_family

        text = normalize_text(
            " ".join(
                str(candidate.get(key) or "")
                for key in ("title", "canonical_title", "mapped_title", "mapped_canonical_title", "source_keyword")
            )
        )
        if not text:
            return "other"
        if any(term in text for term in ("miso soup", "tomato soup", "vegetable soup", "soup", "broth")):
            return "soup"
        if any(term in text for term in ("garden salad", "bean salad", "lentil salad", "chickpea salad", "salad", "slaw", "coleslaw")):
            return "salad"
        if any(
            term in text
            for term in (
                "vegetable",
                "veggies",
                "broccoli",
                "cauliflower",
                "green beans",
                "beans",
                "asparagus",
                "carrot",
                "zucchini",
                "brussels sprouts",
                "medley",
            )
        ):
            return "vegetable"
        return "other"

    @staticmethod
    def _breakfast_side_family(candidate: dict[str, Any]) -> str:
        text = normalize_text(
            " ".join(
                str(candidate.get(key) or "")
                for key in ("title", "canonical_title", "mapped_title", "mapped_canonical_title", "source_keyword")
            )
        )
        if not text:
            return "other"
        if any(
            term in text
            for term in (
                "actifidus",
                "fromage",
                "fromage blanc",
                "iogurt",
                "joghurt",
                "jogurt",
                "kefir",
                "parfait",
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
            term in text
            for term in (
                "apple",
                "banana",
                "berries",
                "berry",
                "blaubeere",
                "blueberry",
                "erdbeer",
                "fruit",
                "framboise",
                "grape",
                "kirsche",
                "kiwi",
                "mango",
                "melon",
                "orange",
                "pineapple",
            )
        ):
            return "fruit"
        if any(term in text for term in ("almond", "cashew", "hazelnut", "nut", "pecan", "seed", "trail mix", "walnut")):
            return "nuts"
        if any(term in text for term in ("bagel", "bread", "cracker", "crispbread", "croissant", "english muffin", "toast", "tosta")):
            return "bread"
        if any(term in text for term in ("cereal", "granola", "muesli", "oat", "oatmeal", "porridge")):
            return "grain"
        return "other"

    def _select_diverse_breakfast_sides(self, candidates: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
        if len(candidates) <= 1:
            return candidates[:limit]

        family_priority = {
            "fruit": 0,
            "nuts": 1,
            "bread": 2,
            "grain": 3,
            "cultured_dairy": 4,
            "other": 5,
        }
        selected: list[dict[str, Any]] = []
        family_counts: dict[str, int] = {}
        remaining = list(candidates)

        while remaining and len(selected) < limit:
            best_index = 0
            best_rank: tuple[int, int, int] | None = None
            for index, candidate in enumerate(remaining):
                family = self._breakfast_side_family(candidate)
                rank = (
                    family_counts.get(family, 0),
                    family_priority.get(family, family_priority["other"]),
                    index,
                )
                if best_rank is None or rank < best_rank:
                    best_rank = rank
                    best_index = index

            chosen = remaining.pop(best_index)
            selected.append(chosen)
            family = self._breakfast_side_family(chosen)
            family_counts[family] = family_counts.get(family, 0) + 1

        return selected[:limit]

    def _needs_breakfast_side_diversity_support(self, candidates: list[dict[str, Any]]) -> bool:
        if not candidates:
            return True

        family_counts: dict[str, int] = {}
        for candidate in candidates:
            family = self._breakfast_side_family(candidate) or "other"
            family_counts[family] = family_counts.get(family, 0) + 1

        cultured_dairy_count = family_counts.get("cultured_dairy", 0)
        non_cultured_families = {
            family
            for family, count in family_counts.items()
            if count > 0 and family not in {"cultured_dairy", "other"}
        }
        return cultured_dairy_count >= 2 and len(non_cultured_families) < 2

    def _side_diversity_key(self, candidate: dict[str, Any], meal_type: str) -> str:
        if normalize_text(meal_type) == "breakfast":
            family = self._breakfast_side_family(candidate)
            if family and family != "other":
                return family
        return self._role_title_key(candidate)

    def _select_diverse_dinner_sides(self, candidates: list[dict[str, Any]], limit: int, meal_type: str) -> list[dict[str, Any]]:
        if normalize_text(meal_type) != "dinner" or len(candidates) <= 1:
            return candidates[:limit]

        selected: list[dict[str, Any]] = []
        family_counts = {"soup": 0, "salad": 0, "vegetable": 0, "other": 0}
        remaining = list(candidates)
        while remaining and len(selected) < limit:
            best_index = 0
            best_rank: tuple[int, int, int, int] | None = None
            for index, candidate in enumerate(remaining):
                family = self._dinner_side_family(candidate)
                family_count = family_counts.get(family, 0)
                family_priority = 0 if family in {"salad", "vegetable", "soup"} else 1
                soup_penalty = 1 if family == "soup" and (family_counts.get("salad", 0) == 0 or family_counts.get("vegetable", 0) == 0) else 0
                rank = (family_count, soup_penalty, family_priority, index)
                if best_rank is None or rank < best_rank:
                    best_rank = rank
                    best_index = index

            chosen = remaining.pop(best_index)
            selected.append(chosen)
            family = self._dinner_side_family(chosen)
            family_counts[family] = family_counts.get(family, 0) + 1

        return selected[:limit]

    @staticmethod
    def _side_query_keywords(meal_type: str) -> list[str]:
        normalized = normalize_text(meal_type)
        if normalized == "dinner":
            return [
                "salad",
                "vegetable",
                "greens",
                "broccoli",
                "beans",
                "slaw",
                "soup",
                "carrot",
            ]
        if normalized == "breakfast":
            return [
                "berries",
                "fruit",
                "granola",
                "muesli",
                "toast",
                "yogurt",
            ]
        return list(COMBO_SIDE_KEYWORDS)

    @staticmethod
    def _side_expansion_regex(meal_type: str) -> str:
        normalized = normalize_text(meal_type)
        if normalized != "dinner":
            return ""
        terms = dedupe_strings(
            [
                "salad",
                "slaw",
                "greens",
                "lettuce",
                "cabbage",
                "spinach",
                "kale",
                "broccoli slaw",
                "vegetable",
                "broccoli",
                "cauliflower",
                "asparagus",
                "green beans",
                "peas",
                "carrot",
                "zucchini",
                "brussels sprouts",
                "bean salad",
                "lentil salad",
                "chickpea salad",
                "cucumber salad",
                "garden salad",
                "side salad",
                "soup",
                "miso soup",
                "broth",
            ]
        )
        return "|".join(normalize_text(term) for term in terms if normalize_text(term))

    @staticmethod
    def _side_expansion_needed(meal_type: str, side_candidates: list[dict[str, Any]]) -> bool:
        if normalize_text(meal_type) != "dinner":
            return False
        if not side_candidates:
            return True

        preferred_terms = (
            "salad",
            "slaw",
            "greens",
            "lettuce",
            "cabbage",
            "spinach",
            "kale",
            "broccoli",
            "vegetable",
            "cauliflower",
            "asparagus",
            "green beans",
            "peas",
            "carrot",
            "zucchini",
            "brussels sprouts",
            "bean",
            "lentil",
            "chickpea",
            "cucumber",
            "soup",
            "broth",
        )
        rice_terms = ("rice", "fried rice", "jasmine", "pilaf", "risotto")
        sample = side_candidates[:8]
        preferred_hits = 0
        rice_hits = 0
        for candidate in sample:
            text = normalize_text(candidate.get("title") or candidate.get("canonical_title") or "")
            if any(term in text for term in preferred_terms):
                preferred_hits += 1
            if any(term in text for term in rice_terms):
                rice_hits += 1

        return preferred_hits < 2 or rice_hits >= max(2, preferred_hits)

    @staticmethod
    def _filter_side_candidates_for_meal(meal_type: str, side_candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if normalize_text(meal_type) not in {"lunch", "dinner"}:
            return side_candidates

        filtered: list[dict[str, Any]] = []
        for candidate in side_candidates:
            if RecommendationService._is_blocked_non_breakfast_side_candidate(candidate):
                continue
            filtered.append(candidate)
        return filtered

    @staticmethod
    def _is_blocked_non_breakfast_side_candidate(candidate: dict[str, Any]) -> bool:
        if RecommendationService._is_blocked_dinner_side_candidate(candidate):
            return True

        title_text = normalize_text(
            " ".join(
                str(candidate.get(key) or "")
                for key in ("title", "canonical_title", "mapped_title", "mapped_canonical_title")
            )
        )
        if not title_text:
            return False

        breakfast_side_context_terms = (
            "bean",
            "beans",
            "berry",
            "berries",
            "broth",
            "chickpea",
            "cucumber",
            "fruit",
            "greens",
            "lentil",
            "salad",
            "slaw",
            "soup",
            "vegetable",
        )
        breakfast_savory_terms = (
            "avocado toast",
            "breakfast",
            "french toast",
            "hash brown",
            "hotcakes",
            "omelet",
            "omelette",
            "pancake",
            "pancakes",
            "scramble",
            "scrapple",
            "skillet",
            "waffle",
            "waffles",
        )
        if any(term in title_text for term in breakfast_savory_terms) and not any(
            term in title_text for term in breakfast_side_context_terms
        ):
            return True
        if "egg" in title_text and not any(
            term in title_text
            for term in (
                "egg salad",
                "bean",
                "beans",
                "broth",
                "chickpea",
                "cucumber",
                "lentil",
                "salad",
                "slaw",
                "soup",
                "vegetable",
            )
        ):
            return True
        return False

    @staticmethod
    def _is_blocked_dinner_side_candidate(candidate: dict[str, Any]) -> bool:
        title_text = normalize_text(
            " ".join(
                str(candidate.get(key) or "")
                for key in ("title", "canonical_title", "mapped_title", "mapped_canonical_title")
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
        preferred_terms = (
            "salad",
            "slaw",
            "greens",
            "vegetable",
            "veggies",
            "broccoli",
            "soup",
            "broth",
            "beans",
            "stir fry",
        )
        generic_rice_terms = (
            "chicken fried rice",
            "fried rice",
            "fried rice with peas and carrots",
            "jasmine fried rice",
            "jasmine rice",
            "mom fried rice",
            "pilaf",
            "reis",
            "risotto",
            "rice",
            "rijst",
            "rundvlees rijst",
        )
        rice_context_terms = (
            "bean",
            "beans",
            "greens",
            "lentil",
            "salad",
            "slaw",
            "soup",
            "broth",
        )
        mainish_meal_terms = (
            "chicken",
            "beef",
            "huhn",
            "kip",
            "pollo",
            "pork",
            "poulet",
            "reis",
            "rijst",
            "rundvlees",
            "turkey",
            "salmon",
            "shrimp",
            "sausage",
            "steak",
            "mac and cheese",
            "grains",
            "grain",
        )
        strict_side_context_terms = (
            "salad",
            "slaw",
            "soup",
            "broth",
            "bean",
            "beans",
            "lentil",
            "chickpea",
            "miso",
            "vegetable soup",
            "tomato soup",
        )
        explicitly_blocked_titles = (
            "salade de riz au thon",
            "mexicana salaatti",
            "thunfisch-salat mexikanisch",
        )

        if any(term in title_text for term in explicitly_blocked_titles):
            return True

        if any(term in title_text for term in blocked_main_terms) and not any(term in title_text for term in preferred_terms):
            return True
        if any(term in title_text for term in mainish_meal_terms) and not any(term in title_text for term in strict_side_context_terms):
            return True
        if "soup" in title_text:
            protein_soup_terms = ("chicken", "beef", "pork", "turkey", "sausage", "steak")
            safe_soup_terms = ("miso", "broth", "vegetable", "tomato")
            if any(term in title_text for term in protein_soup_terms) and not any(term in title_text for term in safe_soup_terms):
                return True
        if any(term in title_text for term in generic_rice_terms) and not any(term in title_text for term in rice_context_terms):
            return True
        return False

    def _filter_mapped_candidates_for_meal(
        self,
        meal_type: str,
        candidates: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[str]]:
        normalized_meal = normalize_text(meal_type)
        if normalized_meal not in {"breakfast", "lunch", "dinner"}:
            return candidates, []

        filtered: list[dict[str, Any]] = []
        removed_titles: list[str] = []
        reserve_drink_candidates: list[tuple[int, dict[str, Any]]] = []
        minimum_drink_quality = 0.5 if normalized_meal == "breakfast" else 0.46
        relaxed_local_drink_floor = 0.32 if normalized_meal == "lunch" else 0.2 if normalized_meal == "dinner" else 0.0
        for index, candidate in enumerate(candidates):
            category = self._infer_combo_category(candidate)
            title_text = normalize_text(
                " ".join(
                    str(candidate.get(key) or "")
                    for key in ("title", "canonical_title", "mapped_title", "mapped_canonical_title", "brand_name")
                )
            )

            if category == "drink":
                drink_quality = drink_role_quality_multiplier(candidate, meal_type)
                safety_drink_candidate = self._is_safety_drink_candidate(candidate)
                dinner_coconut_water_candidate = (
                    normalized_meal == "dinner"
                    and safety_drink_candidate
                    and "coconut water" in title_text
                )
                if not is_candidate_role_compatible(candidate, "drink"):
                    removed_titles.append(str(candidate.get("title") or candidate.get("mapped_title") or "").strip())
                    continue
                if self._is_alcoholic_drink_candidate(candidate):
                    removed_titles.append(str(candidate.get("title") or candidate.get("mapped_title") or "").strip())
                    continue
                if drink_quality < minimum_drink_quality:
                    allow_relaxed_reserve = (
                        normalized_meal == "lunch"
                        or not safety_drink_candidate
                        or dinner_coconut_water_candidate
                    )
                    if (
                        normalized_meal in {"lunch", "dinner"}
                        and allow_relaxed_reserve
                        and drink_quality >= relaxed_local_drink_floor
                    ):
                        reserve_drink_candidates.append((index, candidate))
                        continue
                    removed_titles.append(str(candidate.get("title") or candidate.get("mapped_title") or "").strip())
                    continue
                filtered.append(candidate)
                continue

            if category == "main" and normalized_meal in {"lunch", "dinner"}:
                if self._is_blocked_non_breakfast_main_candidate(candidate):
                    removed_titles.append(str(candidate.get("title") or candidate.get("mapped_title") or "").strip())
                    continue
                main_quality = main_role_quality_multiplier(candidate, meal_type)
                if main_quality < (0.58 if normalized_meal == "lunch" else 0.62):
                    removed_titles.append(str(candidate.get("title") or candidate.get("mapped_title") or "").strip())
                    continue
                filtered.append(candidate)
                continue

            if category != "side" and not is_candidate_role_compatible(candidate, "side"):
                filtered.append(candidate)
                continue

            side_quality = side_role_quality_multiplier(candidate, meal_type)
            minimum_side_quality = 0.72 if normalized_meal == "breakfast" else 0.62 if normalized_meal == "lunch" else 0.75
            if side_quality < minimum_side_quality:
                removed_titles.append(str(candidate.get("title") or candidate.get("mapped_title") or "").strip())
                continue

            if normalized_meal in {"lunch", "dinner"} and self._is_blocked_non_breakfast_side_candidate(candidate):
                removed_titles.append(str(candidate.get("title") or candidate.get("mapped_title") or "").strip())
                continue
            if normalized_meal == "breakfast" and any(
                term in title_text
                for term in ("cheddar cheese", "fruit with cheddar cheese", "french toast", "hotcakes", "pancake", "pancakes", "waffle", "waffles")
            ):
                removed_titles.append(str(candidate.get("title") or candidate.get("mapped_title") or "").strip())
                continue
            filtered.append(candidate)

        if normalized_meal in {"lunch", "dinner"} and reserve_drink_candidates:
            compatible_drink_count = len(
                [
                    candidate
                    for candidate in filtered
                    if self._infer_combo_category(candidate) == "drink"
                    and is_candidate_role_compatible(candidate, "drink")
                ]
            )
            desired_drink_count = self._drink_pool_floor(meal_type)
            reserve_drink_keys = {
                self._role_title_key(candidate) or str(candidate.get("recipe_id") or candidate.get("id"))
                for _, candidate in reserve_drink_candidates
                if self._role_title_key(candidate) or str(candidate.get("recipe_id") or candidate.get("id"))
            }
            if normalized_meal == "dinner" and reserve_drink_keys:
                desired_drink_count = max(
                    desired_drink_count,
                    min(
                        self._drink_diversity_target(meal_type),
                        compatible_drink_count + len(reserve_drink_keys),
                    ),
                )
            needed_drinks = max(0, desired_drink_count - compatible_drink_count)
            if needed_drinks > 0:
                selected_reserves = sorted(
                    reserve_drink_candidates,
                    key=lambda item: (
                        drink_role_quality_multiplier(item[1], meal_type),
                        0 if self._is_plant_milk_drink_candidate(item[1]) else 1,
                        to_float(item[1].get("score"), 0.0),
                    ),
                    reverse=True,
                )[:needed_drinks]
                for _, candidate in sorted(selected_reserves, key=lambda item: item[0]):
                    filtered.append(candidate)

        return filtered, removed_titles

    def _expand_side_candidates(
        self,
        *,
        meal_type: str,
        query_vector: np.ndarray,
        side_candidates: list[dict[str, Any]],
        candidate_pool_target: int,
        prefetch_target: int,
        is_australian_user: bool,
        use_dedicated_search_connection: bool = False,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        expansion_regex = self._side_expansion_regex(meal_type)
        if not expansion_regex:
            return side_candidates, {
                "used": False,
                "queries": [],
                "added": 0,
                "candidate_count": len(side_candidates),
            }

        if not self._side_expansion_needed(meal_type, side_candidates):
            return side_candidates, {
                "used": False,
                "queries": [expansion_regex],
                "added": 0,
                "candidate_count": len(side_candidates),
            }

        exclude_recipe_ids = {
            str(candidate.get("recipe_id") or "").strip()
            for candidate in side_candidates
            if str(candidate.get("recipe_id") or "").strip()
        }
        expanded_candidates = self.local_dataset.search(
            meal_type=meal_type,
            query_vector=query_vector,
            top_k=max(8, int(candidate_pool_target / 6)),
            prefetch=max(16, int(prefetch_target / 5)),
            exclude_recipe_ids=exclude_recipe_ids,
            is_australian_user=is_australian_user,
            text_query=expansion_regex,
            role_hint="side",
            dedicated_connection=use_dedicated_search_connection,
        )

        if not expanded_candidates:
            return side_candidates, {
                "used": False,
                "queries": [expansion_regex],
                "added": 0,
                "candidate_count": len(side_candidates),
            }

        merged_candidates = merge_candidates(expanded_candidates, side_candidates, max_items=max(24, int(candidate_pool_target / 2)))
        added = max(0, len(merged_candidates) - len(side_candidates))
        print(
            f"**** Slot={meal_type} side expansion: before={len(side_candidates)} after={len(merged_candidates)} "
            f"added={added} query={expansion_regex}"
        )
        return merged_candidates, {
            "used": True,
            "queries": [expansion_regex],
            "added": added,
            "candidate_count": len(merged_candidates),
        }

    def _archive_runtime_slot_comparisons(
        self,
        *,
        user_id: Any,
        requested_meal_type: str,
        payload_by_slot: dict[str, Any],
        experiment_name: str,
        force_exploration: bool,
    ) -> None:
        if not self.runtime_comparison_log_enabled or not payload_by_slot:
            return

        rows: list[dict[str, Any]] = []
        timestamp = _utc_now_iso()
        for slot, payload in payload_by_slot.items():
            metrics = (payload or {}).get("model_metrics") or {}
            combo = metrics.get("combo_diagnostics") or {}
            diversity = metrics.get("diversity_dashboard") or {}
            mapping = metrics.get("mapping_diagnostics") or {}
            timing = metrics.get("timing") or {}
            experimentation = metrics.get("experimentation") or {}
            rows.append(
                {
                    "timestamp": timestamp,
                    "user_id": str(user_id or ""),
                    "requested_meal_type": requested_meal_type,
                    "slot": slot,
                    "experiment_variant": str(experiment_name or "control"),
                    "force_exploration": bool(force_exploration),
                    "slot_target": int((payload or {}).get("slot_target") or 0),
                    "recommended_item_count": int(len((payload or {}).get("recommended_items") or [])),
                    "combo_count": int(combo.get("combo_count", 0)),
                    "role_coverage_rate": round(to_float(combo.get("role_coverage_rate"), 0.0), 4),
                    "mean_combo_calorie_gap_ratio": round(
                        to_float(combo.get("mean_combo_calorie_gap_ratio"), 0.0),
                        4,
                    ),
                    "repeated_drink_rate": round(to_float(diversity.get("repeated_drink_rate"), 0.0), 4),
                    "repeated_side_rate": round(to_float(diversity.get("repeated_side_rate"), 0.0), 4),
                    "mapped_item_rate": round(to_float(mapping.get("mapped_item_rate"), 0.0), 4),
                    "local_only_item_rate": round(to_float(mapping.get("local_only_item_rate"), 0.0), 4),
                    "retrieval_ms": round(to_float(timing.get("retrieval_ms"), 0.0), 1),
                    "mapping_ms": round(to_float(timing.get("mapping_ms"), 0.0), 1),
                    "ranking_ms": round(to_float(timing.get("ranking_ms"), 0.0), 1),
                    "combo_assembly_ms": round(to_float(timing.get("combo_assembly_ms"), 0.0), 1),
                    "combo_pool_build_ms": round(to_float(timing.get("combo_pool_build_ms"), 0.0), 1),
                    "combo_candidate_generation_ms": round(to_float(timing.get("combo_candidate_generation_ms"), 0.0), 1),
                    "combo_diversity_ms": round(to_float(timing.get("combo_diversity_ms"), 0.0), 1),
                    "query_expansion_used": bool(experimentation.get("query_expansion_used")),
                    "drink_supplement_used": bool(experimentation.get("drink_supplement_used")),
                    "drink_supplement_added": int(experimentation.get("drink_supplement_added", 0) or 0),
                    "drink_compatible_before": int(experimentation.get("drink_compatible_before", 0) or 0),
                    "drink_compatible_after": int(experimentation.get("drink_compatible_after", 0) or 0),
                }
            )

        directory = os.path.dirname(self.runtime_comparison_log_path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        with self._runtime_comparison_log_lock:
            with open(self.runtime_comparison_log_path, "a", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, sort_keys=True) + "\n")

    def _infer_combo_category(self, candidate: dict[str, Any]) -> str:
        cache = self._candidate_service_cache(candidate)
        cached_category = cache.get("combo_category")
        if isinstance(cached_category, str) and cached_category:
            return cached_category

        category = infer_combo_category(candidate)
        cache["combo_category"] = category
        return category

    @staticmethod
    def _combo_unique_role_target(role: str, meal_type: str, max_combos: int) -> int:
        role_key = normalize_text(role)
        meal_key = normalize_text(meal_type)
        if role_key == "main" and meal_key in {"breakfast", "lunch"}:
            return 2
        if role_key == "drink":
            return RecommendationService._drink_diversity_target(meal_type)
        return min(3, max_combos)

    @staticmethod
    def _combo_macro_totals(items: list[dict[str, Any]]) -> dict[str, float]:
        total_calories = sum(to_float(item.get("calories"), 0.0) for item in items)
        total_protein = sum(to_float(item.get("protein"), 0.0) for item in items)
        total_carbs = sum(to_float(item.get("carbs"), 0.0) for item in items)
        total_fats = sum(to_float(item.get("fats"), 0.0) for item in items)
        return {
            "calories": round(total_calories, 1),
            "protein": round(total_protein, 1),
            "carbs": round(total_carbs, 1),
            "fats": round(total_fats, 1),
        }

    def _build_combo_payloads(
        self,
        meal_type: str,
        ranked_candidates: list[dict[str, Any]],
        slot_target: int,
        behavioral_insight: str,
        top_food_counts: dict[str, float] | None,
        combo_reuse_penalty_base: float,
        max_combos: int = 5,
    ) -> tuple[list[dict[str, Any]], dict[str, float]]:
        if not ranked_candidates:
            return [], {
                "combo_pool_build_ms": 0.0,
                "combo_candidate_generation_ms": 0.0,
                "combo_diversity_ms": 0.0,
            }

        combo_started_at = time.perf_counter()
        meal_key = normalize_text(meal_type)
        main_target = slot_target * float(COMBO_CATEGORY_TARGETS.get("main", 0.65))
        side_target = slot_target * float(COMBO_CATEGORY_TARGETS.get("side", 0.20))
        drink_target = slot_target * float(COMBO_CATEGORY_TARGETS.get("drink", 0.15))

        # NOTE: Identify habitual sides/drinks from history to bias combo assembly.
        habitual_titles: dict[str, set[str]] = {"main": set(), "side": set(), "drink": set()}
        if top_food_counts:
            for title, count in top_food_counts.items():
                if float(count) <= 0.0:
                    continue
                category = self._infer_combo_category({"title": title})
                habitual_titles[category].add(canonical_title_key(title))

        def candidate_combo_score(candidate: dict[str, Any], category: str) -> float:
            category_key = normalize_text(category)
            habitual = self._role_title_key(candidate) in habitual_titles.get(category_key, set())
            return self._candidate_combo_score(candidate, category_key, meal_type, habitual=habitual)

        def _candidate_calories(candidate: dict[str, Any]) -> float:
            # NOTE: Use per-serving calories; do not scale to target.
            return self._candidate_serving_calories(candidate)

        def _is_visible_local_only_candidate(candidate: dict[str, Any]) -> bool:
            ml_tag = normalize_text(candidate.get("ml_tag") or candidate.get("item_ml_tag") or "")
            food_id = str(candidate.get("food_id") or "").strip().lower()
            fatsecret_food_id = str(candidate.get("fatsecret_food_id") or "").strip()
            return ml_tag == "local_only" or food_id.startswith("local-") or (food_id and not fatsecret_food_id and ml_tag == "local_only")

        def _is_unresolved_visible_local_only_candidate(candidate: dict[str, Any]) -> bool:
            if not _is_visible_local_only_candidate(candidate):
                return False

            recipe_id = self._recipe_id_from_candidate(candidate)
            if recipe_id and self.mapping_store.get(recipe_id):
                return False

            return self._lookup_mapping_snapshot_by_title(candidate) is None

        def _prefer_mapped_pool(
            pool: list[dict[str, Any]],
            category: str,
            limit: int,
        ) -> list[dict[str, Any]]:
            hydration_ready_pool = [
                candidate for candidate in pool if not _is_unresolved_visible_local_only_candidate(candidate)
            ]
            minimum_mapped_candidates = max(1, min(limit, 2 if category in {"side", "drink"} else 3))
            if meal_key == "breakfast" and category == "main":
                minimum_mapped_candidates = 1
            if len(hydration_ready_pool) >= minimum_mapped_candidates:
                return hydration_ready_pool
            return pool

        def _is_breakfast_mapped_fallback_main_candidate(candidate: dict[str, Any]) -> bool:
            if _is_visible_local_only_candidate(candidate):
                return False
            if self._is_breakfast_aligned_main_candidate(candidate):
                return False
            if main_role_quality_multiplier(candidate, meal_type) < 0.52:
                return False

            text = normalize_text(
                " ".join(
                    str(candidate.get(key) or "")
                    for key in (
                        "canonical_title",
                        "title",
                        "original_title",
                        "mapped_title",
                        "recipe_category",
                        "food_type",
                    )
                )
            )
            if not text:
                return False
            if any(term in text for term in ("frozen custard", "gelato", "ice cream", "sundae", "waffle cone")):
                return False

            breakfast_fallback_markers = (
                "bagel",
                "breakfast",
                "burrito",
                "egg",
                "eggs",
                "omelet",
                "omelette",
                "sandwich",
                "scramble",
                "skillet",
                "toast",
                "wrap",
            )
            return any(marker in text for marker in breakfast_fallback_markers)

        def _dedupe_by_title(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
            seen: set[str] = set()
            output: list[dict[str, Any]] = []
            for item in items:
                key = self._role_title_key(item)
                if not key or key in seen:
                    continue
                seen.add(key)
                output.append(item)
            return output

        def _within_target(candidate: dict[str, Any], target: float, tolerance: float) -> bool:
            if target <= 0:
                return True
            calories = _candidate_calories(candidate)
            # Exception for low-calorie sides/drinks
            if calories <= 200.0:
                 return True
            if calories <= 0:
                return False
            return abs(calories - target) / max(1.0, target) <= tolerance

        def _dinner_side_priority(candidate: dict[str, Any]) -> tuple[int, int]:
            return self._candidate_dinner_side_priority(candidate)

        def pick_top(category: str, limit: int, target: float, tolerance: float) -> list[dict[str, Any]]:
            pool = [c for c in ranked_candidates if self._infer_combo_category(c) == category]
            pool = _dedupe_by_title(pool)
            compatible_pool = [c for c in pool if is_candidate_role_compatible(c, category)]
            if category == "main" and meal_key == "breakfast":
                aligned_pool = [c for c in compatible_pool if self._is_breakfast_aligned_main_candidate(c)]
                if aligned_pool:
                    mapped_aligned_pool = [c for c in aligned_pool if not _is_visible_local_only_candidate(c)]
                    breakfast_fallback_pool = _dedupe_by_title(
                        [
                            c
                            for c in compatible_pool
                            if c not in aligned_pool and _is_breakfast_mapped_fallback_main_candidate(c)
                        ]
                    )
                    if len(mapped_aligned_pool) < min(2, limit) and breakfast_fallback_pool:
                        supplemented_pool = _dedupe_by_title(
                            [*mapped_aligned_pool, *breakfast_fallback_pool, *aligned_pool]
                        )
                        pool = supplemented_pool
                        compatible_pool = supplemented_pool
                    else:
                        pool = aligned_pool
                        compatible_pool = aligned_pool
            if category == "side" and meal_key in {"lunch", "dinner"}:
                pool = [c for c in pool if not self._is_blocked_non_breakfast_side_candidate(c)]
                compatible_pool = [c for c in compatible_pool if not self._is_blocked_non_breakfast_side_candidate(c)]

            if category == "main":
                pool = compatible_pool
            else:
                pool = compatible_pool or pool

            if category == "drink":
                pool = self._filter_drink_candidates_for_meal(meal_type, pool)

            if not pool and category != "main":
                # NOTE: Fallback to any compatible candidate for side/drink so combo assembly still has coverage.
                fallback_pool = _dedupe_by_title(ranked_candidates)
                fallback_compatible = [c for c in fallback_pool if is_candidate_role_compatible(c, category)]
                if category == "side" and meal_key in {"lunch", "dinner"}:
                    fallback_pool = [c for c in fallback_pool if not self._is_blocked_non_breakfast_side_candidate(c)]
                    fallback_compatible = [c for c in fallback_compatible if not self._is_blocked_non_breakfast_side_candidate(c)]
                pool = fallback_compatible
                if category == "drink":
                    pool = self._filter_drink_candidates_for_meal(meal_type, pool)
                if category == "side" and not pool:
                    side_quality_floor = 0.72 if meal_key == "dinner" else 0.58
                    pool = [
                        c for c in fallback_pool
                        if side_role_quality_multiplier(c, meal_type) >= side_quality_floor
                    ]

            if not pool:
                return []
            # NOTE: Bias side/drink pools toward their calorie targets for better combo totals.
            in_band = [c for c in pool if _within_target(c, target, tolerance)]
            working = in_band or pool
            if category == "side":
                preferred_quality_floor = 0.84 if meal_key == "breakfast" else 0.82 if meal_key == "dinner" else 0.74
                preferred = [
                    c for c in working
                    if side_role_quality_multiplier(c, meal_type) >= preferred_quality_floor
                ]
                fallback = [
                    c for c in working
                    if side_role_quality_multiplier(c, meal_type) >= (0.72 if meal_key == "breakfast" else 0.7 if meal_key == "dinner" else 0.6)
                ]
                working = preferred or fallback or working
            if category == "main" and meal_key in {"lunch", "dinner"}:
                pre_quality_main_pool = list(working) if meal_key == "dinner" else []
                preferred_main_floor = 0.76 if meal_key == "dinner" else 0.74
                preferred = [
                    c for c in working
                    if main_role_quality_multiplier(c, meal_type) >= preferred_main_floor
                ]
                fallback = [
                    c for c in working
                    if main_role_quality_multiplier(c, meal_type) >= (0.62 if meal_key == "dinner" else 0.6)
                ]
                working = preferred or fallback or working
                if meal_key == "dinner":
                    strong_strict_dinner_mains = [
                        c
                        for c in pre_quality_main_pool
                        if not _is_visible_local_only_candidate(c)
                        and normalize_text(c.get("mapping_acceptance_mode")) == "strict"
                        and main_role_quality_multiplier(c, meal_type) >= 0.58
                        and to_float(c.get("calorie_diff_ratio"), 0.0) <= 0.12
                    ]
                    preferred_dinner_mains = [
                        c
                        for c in working
                        if not _is_visible_local_only_candidate(c)
                        and not (
                            normalize_text(c.get("mapping_acceptance_mode")) == "relaxed_title_fallback"
                            and to_float(c.get("calorie_diff_ratio"), 0.0) >= 0.5
                        )
                    ]
                    if len(strong_strict_dinner_mains) >= 2:
                        working = strong_strict_dinner_mains
                    elif len(preferred_dinner_mains) >= 2:
                        working = preferred_dinner_mains
            working = _prefer_mapped_pool(working, category, limit)
            if category == "main":
                scored = sorted(working, key=lambda c: candidate_combo_score(c, category), reverse=True)
            else:
                # Prefer higher-quality role fits before calorie tie-breaking for sides and drinks.
                scored = sorted(
                    working,
                    key=lambda c: (
                        -_dinner_side_priority(c)[0] if category == "side" and meal_key == "dinner" else 0,
                        _dinner_side_priority(c)[1] if category == "side" and meal_key == "dinner" else 0,
                        -side_role_quality_multiplier(c, meal_type) if category == "side" else 0.0,
                        -drink_role_quality_multiplier(c, meal_type) if category == "drink" else 0.0,
                        abs(_candidate_calories(c) - target),
                        -candidate_combo_score(c, category),
                    ),
                )
            if category == "side":
                if meal_key == "breakfast":
                    return self._select_diverse_breakfast_sides(scored, limit)
                if meal_key == "dinner":
                    return self._select_diverse_dinner_sides(scored, limit, meal_type)
            return scored[:limit]

        def _combo_deviation(total_calories: float) -> float:
            if slot_target <= 0:
                return 0.0
            return abs(total_calories - slot_target) / max(1.0, float(slot_target))

        category_tolerance = {"main": 0.50, "side": 0.50, "drink": 0.50}

        def _pool_summary(category: str, target: float, tolerance: float, limit: int = 5) -> dict[str, Any]:
            pool = [c for c in ranked_candidates if self._infer_combo_category(c) == category]
            deduped = _dedupe_by_title(pool)
            compatible = [c for c in deduped if is_candidate_role_compatible(c, category)]
            in_band = [c for c in compatible if _within_target(c, target, tolerance)]
            selected = pick_top(category, pool_limits[category], target, tolerance)
            sample_titles = [
                str(candidate.get("title") or candidate.get("canonical_title") or "")
                for candidate in selected[:limit]
                if str(candidate.get("title") or candidate.get("canonical_title") or "").strip()
            ]
            return {
                "raw": len(pool),
                "deduped": len(deduped),
                "compatible": len(compatible),
                "in_band": len(in_band),
                "selected": len(selected),
                "sample_titles": sample_titles,
            }

        combos: list[dict[str, Any]] = []
        pool_limits = {
            "main": 10 if meal_key == "breakfast" else 12,
            "side": 7 if meal_key == "breakfast" else 8,
            "drink": 4 if meal_key == "breakfast" else 6 if meal_key == "dinner" else 7,
        }
        main_pool_summary = _pool_summary("main", main_target, category_tolerance["main"])
        side_pool_summary = _pool_summary("side", side_target, category_tolerance["side"])
        drink_pool_summary = _pool_summary("drink", drink_target, category_tolerance["drink"])

        pool_started_at = time.perf_counter()
        mains = pick_top("main", pool_limits["main"], main_target, category_tolerance["main"])
        if meal_key == "breakfast" and len(mains) < 2:
            breakfast_main_recovery_pool = _dedupe_by_title(
                [
                    candidate
                    for candidate in ranked_candidates
                    if self._infer_combo_category(candidate) == "main"
                    and not _is_visible_local_only_candidate(candidate)
                    and (
                        self._is_breakfast_aligned_main_candidate(candidate)
                        or _is_breakfast_mapped_fallback_main_candidate(candidate)
                    )
                ]
            )
            breakfast_main_recovery_pool = sorted(
                breakfast_main_recovery_pool,
                key=lambda candidate: candidate_combo_score(candidate, "main"),
                reverse=True,
            )
            if len(breakfast_main_recovery_pool) >= 2:
                mains = breakfast_main_recovery_pool[: pool_limits["main"]]
                main_pool_summary["selected"] = len(mains)
                main_pool_summary["sample_titles"] = [
                    str(candidate.get("title") or candidate.get("canonical_title") or "")
                    for candidate in mains[:5]
                    if str(candidate.get("title") or candidate.get("canonical_title") or "").strip()
                ]
                print(
                    f"**** Combo Assembly Breakfast Main Recovery: meal_type={meal_type} "
                    f"selected={len(mains)} titles={main_pool_summary['sample_titles']}"
                )
        if not mains:
            main_fallback_pool = [candidate for candidate in ranked_candidates if self._infer_combo_category(candidate) == "main"]
            if meal_key == "breakfast":
                aligned_main_fallback = [
                    candidate for candidate in main_fallback_pool if self._is_breakfast_aligned_main_candidate(candidate)
                ]
                main_fallback_pool = aligned_main_fallback or main_fallback_pool
            if meal_key in {"lunch", "dinner"}:
                preferred_main_fallback = [
                    candidate for candidate in main_fallback_pool
                    if main_role_quality_multiplier(candidate, meal_type) >= (0.6 if meal_key == "lunch" else 0.62)
                ]
                main_fallback_pool = preferred_main_fallback or main_fallback_pool
            mains = _dedupe_by_title(main_fallback_pool)[: pool_limits["main"]]
            if not mains and meal_key == "breakfast":
                mains = ranked_candidates[: pool_limits["main"]]

        sides = pick_top("side", pool_limits["side"], side_target, category_tolerance["side"])
        if not sides:
            side_fallback_pool = [candidate for candidate in ranked_candidates if is_candidate_role_compatible(candidate, "side")]
            if meal_key in {"lunch", "dinner"}:
                side_fallback_pool = [
                    candidate
                    for candidate in side_fallback_pool
                    if not self._is_blocked_non_breakfast_side_candidate(candidate)
                ]
            minimum_side_quality = 0.72 if meal_key == "breakfast" else 0.62 if meal_key == "lunch" else 0.75
            preferred_side_fallback = [
                candidate for candidate in side_fallback_pool
                if side_role_quality_multiplier(candidate, meal_type) >= minimum_side_quality
            ]
            side_fallback_pool = preferred_side_fallback or side_fallback_pool
            sides = _dedupe_by_title(side_fallback_pool)[: pool_limits["side"]]

        drinks = pick_top("drink", pool_limits["drink"], drink_target, category_tolerance["drink"])
        if not drinks:
            drink_fallback_pool = self._filter_drink_candidates_for_meal(
                meal_type,
                [candidate for candidate in ranked_candidates if is_candidate_role_compatible(candidate, "drink")],
            )
            drinks = _dedupe_by_title(drink_fallback_pool)[: pool_limits["drink"]]
        pool_build_ms = _elapsed_ms(pool_started_at)

        if not mains or not sides or not drinks:
            attempted_combos = len(mains) * len(sides) * len(drinks)
            print(
                f"**** Combo Assembly: meal_type={meal_type} ranked={len(ranked_candidates)} "
                f"candidate_pool={attempted_combos} valid_pool=0 target_kcal={slot_target}"
            )
            print(
                f"**** Combo Assembly Pools: meal_type={meal_type} "
                f"main(raw={main_pool_summary['raw']},deduped={main_pool_summary['deduped']},compatible={main_pool_summary['compatible']},in_band={main_pool_summary['in_band']},selected={main_pool_summary['selected']}) "
                f"side(raw={side_pool_summary['raw']},deduped={side_pool_summary['deduped']},compatible={side_pool_summary['compatible']},in_band={side_pool_summary['in_band']},selected={side_pool_summary['selected']}) "
                f"drink(raw={drink_pool_summary['raw']},deduped={drink_pool_summary['deduped']},compatible={drink_pool_summary['compatible']},in_band={drink_pool_summary['in_band']},selected={drink_pool_summary['selected']}) best_gap_ratio=n/a"
            )
            print(
                f"**** Combo Assembly Timing: meal_type={meal_type} pool_build_ms={pool_build_ms} "
                f"candidate_generation_ms=0.0 duplicate_recipe_skips=0 attempted_combos={attempted_combos}"
            )
            print(
                f"**** Combo Assembly Empty Pool Detail: meal_type={meal_type} "
                f"main_samples={main_pool_summary['sample_titles'][:3]} "
                f"side_samples={side_pool_summary['sample_titles'][:3]} "
                f"drink_samples={drink_pool_summary['sample_titles'][:3]}"
            )
            return [], {
                "combo_pool_build_ms": round(pool_build_ms, 1),
                "combo_candidate_generation_ms": 0.0,
                "combo_diversity_ms": 0.0,
            }

        candidate_generation_started_at = time.perf_counter()
        attempted_combos = len(mains) * len(sides) * len(drinks)
        def _generate_valid_combos(max_deviation: float) -> tuple[list[dict[str, Any]], int]:
            duplicate_skips = 0
            generated: list[dict[str, Any]] = []
            for main in mains:
                if meal_key == "breakfast" and not (
                    self._is_breakfast_aligned_main_candidate(main)
                    or _is_breakfast_mapped_fallback_main_candidate(main)
                ):
                    continue
                main_recipe_id = str(main.get("recipe_id") or main.get("id") or "")
                main_cal = _candidate_calories(main)
                main_score = candidate_combo_score(main, "main")
                for side in sides:
                    side_recipe_id = str(side.get("recipe_id") or side.get("id") or "")
                    if side_recipe_id == main_recipe_id:
                        duplicate_skips += 1
                        continue

                    side_cal = _candidate_calories(side)
                    side_score = candidate_combo_score(side, "side")
                    side_quality = self._role_quality_multiplier(side, "side", meal_type)
                    subtotal = main_cal + side_cal
                    if subtotal > slot_target * (1.0 + float(COMBO_TARGET_TOLERANCE) + 0.15):
                        continue

                    for drink in drinks:
                        drink_recipe_id = str(drink.get("recipe_id") or drink.get("id") or "")
                        if drink_recipe_id in {main_recipe_id, side_recipe_id}:
                            duplicate_skips += 1
                            continue

                        drink_cal = _candidate_calories(drink)
                        total_cal = subtotal + drink_cal
                        deviation = _combo_deviation(total_cal)
                        if deviation > max_deviation:
                            continue

                        leftover = (side_target - side_cal) + (drink_target - drink_cal)
                        adjusted_main_target = max(1.0, main_target + leftover)
                        main_fit = 1.0 - min(1.0, abs(main_cal - adjusted_main_target) / adjusted_main_target)
                        total_fit = 1.0 - min(1.0, abs(total_cal - slot_target) / max(1.0, slot_target))

                        combo_score = (
                            main_score * 0.6
                            + side_score * 0.2
                            + candidate_combo_score(drink, "drink") * 0.2
                            + main_fit * 0.2
                            + total_fit * 0.2
                            + side_quality * 0.1
                        )

                        generated.append(
                            {
                                "main": main,
                                "side": side,
                                "drink": drink,
                                "combo_score": combo_score,
                                "total_calories": total_cal,
                            }
                        )
            return generated, duplicate_skips

        valid_combos, duplicate_recipe_skips = _generate_valid_combos(0.35)
        if not valid_combos and meal_key == "breakfast":
            # Keep breakfast ingredient filtering, but allow a wider calorie gap before giving up on combos.
            valid_combos, duplicate_recipe_skips = _generate_valid_combos(0.5)
            if valid_combos:
                print(
                    f"**** Combo Assembly Breakfast Retry: meal_type={meal_type} "
                    f"max_deviation=0.5 recovered_valid_pool={len(valid_combos)}"
                )
        combo_generation_ms = _elapsed_ms(candidate_generation_started_at)

        candidate_combos = valid_combos
        best_combo_gap_ratio = None
        if candidate_combos:
            best_combo_gap_ratio = round(
                min(_combo_deviation(to_float(combo.get("total_calories"), 0.0)) for combo in candidate_combos),
                4,
            )

        print(
            f"**** Combo Assembly: meal_type={meal_type} ranked={len(ranked_candidates)} "
            f"candidate_pool={attempted_combos} valid_pool={len(valid_combos)} target_kcal={slot_target}"
        )
        print(
            f"**** Combo Assembly Pools: meal_type={meal_type} "
            f"main(raw={main_pool_summary['raw']},deduped={main_pool_summary['deduped']},compatible={main_pool_summary['compatible']},in_band={main_pool_summary['in_band']},selected={main_pool_summary['selected']}) "
            f"side(raw={side_pool_summary['raw']},deduped={side_pool_summary['deduped']},compatible={side_pool_summary['compatible']},in_band={side_pool_summary['in_band']},selected={side_pool_summary['selected']}) "
            f"drink(raw={drink_pool_summary['raw']},deduped={drink_pool_summary['deduped']},compatible={drink_pool_summary['compatible']},in_band={drink_pool_summary['in_band']},selected={drink_pool_summary['selected']}) "
            f"best_gap_ratio={best_combo_gap_ratio if best_combo_gap_ratio is not None else 'n/a'}"
        )
        print(
            f"**** Combo Assembly Timing: meal_type={meal_type} pool_build_ms={pool_build_ms} "
            f"candidate_generation_ms={combo_generation_ms} duplicate_recipe_skips={duplicate_recipe_skips} "
            f"attempted_combos={attempted_combos}"
        )
        if not candidate_combos:
            print(
                f"**** Combo Assembly Empty Pool Detail: meal_type={meal_type} "
                f"main_samples={main_pool_summary['sample_titles'][:3]} "
                f"side_samples={side_pool_summary['sample_titles'][:3]} "
                f"drink_samples={drink_pool_summary['sample_titles'][:3]}"
            )
        
        # Sort by raw score initially
        diversity_started_at = time.perf_counter()
        valid_combos.sort(key=lambda x: x["combo_score"], reverse=True)
        
        # Diversity selection: penalize repeated use of same side/drink
        used_mains: dict[str, int] = {}
        used_sides: dict[str, int] = {}
        used_breakfast_side_families: dict[str, int] = {}
        used_drinks: dict[str, int] = {}
        unique_main_candidates = len(
            {
                self._role_title_key(combo["main"]) or str(combo["main"].get("recipe_id") or combo["main"].get("id"))
                for combo in valid_combos
                if combo.get("main")
            }
        )
        main_penalty_base = 1.0
        if meal_key == "breakfast":
            main_penalty_base = 0.82 if unique_main_candidates >= 2 else 0.9
        elif meal_key == "lunch":
            main_penalty_base = 0.78 if unique_main_candidates >= 2 else 0.9
        elif meal_key == "dinner":
            main_penalty_base = float(np.clip(combo_reuse_penalty_base, 0.42, 0.72))
            has_very_weak_relaxed_dinner_main = any(
                normalize_text(combo["main"].get("mapping_acceptance_mode")) == "relaxed_title_fallback"
                and to_float(combo["main"].get("calorie_diff_ratio"), 0.0) >= 0.5
                for combo in valid_combos
                if combo.get("main")
            )
            if has_very_weak_relaxed_dinner_main:
                main_penalty_base = max(main_penalty_base, 0.84)
        breakfast_side_family_penalty_base = float(np.clip(combo_reuse_penalty_base - 0.12, 0.38, combo_reuse_penalty_base))
        breakfast_side_title_penalty_base = float(np.clip(combo_reuse_penalty_base - 0.20, 0.34, combo_reuse_penalty_base))
        drink_penalty_base = float(np.clip(combo_reuse_penalty_base - 0.08, 0.35, combo_reuse_penalty_base))
        breakfast_unique_side_target = 0
        if meal_key == "breakfast":
            breakfast_unique_side_target = min(
                max_combos,
                len(
                    {
                        self._role_title_key(combo["side"])
                        or str(combo["side"].get("recipe_id") or combo["side"].get("id"))
                        for combo in valid_combos
                        if combo.get("side")
                    }
                ),
            )

        def _has_breakfast_non_cultured_side(limit_scan: int, *, require_unseen: bool) -> bool:
            if meal_key != "breakfast":
                return False
            for combo in valid_combos[:limit_scan]:
                side_family_id = self._breakfast_side_family(combo["side"]) or "other"
                if side_family_id in {"cultured_dairy", "other"}:
                    continue
                if require_unseen and used_breakfast_side_families.get(side_family_id, 0) > 0:
                    continue
                return True
            return False
        
        while len(combos) < max_combos and valid_combos:
            require_unique_breakfast_side = meal_key == "breakfast" and len(used_sides) < breakfast_unique_side_target
            scan_modes: list[tuple[int, bool, bool]] = []
            breakfast_cultured_dairy_used = used_breakfast_side_families.get("cultured_dairy", 0)
            if require_unique_breakfast_side and (
                _has_breakfast_non_cultured_side(len(valid_combos), require_unseen=True)
                or breakfast_cultured_dairy_used >= 2
            ):
                scan_modes.append((len(valid_combos), True, True))
            if require_unique_breakfast_side:
                scan_modes.append((len(valid_combos), True, False))
                if breakfast_cultured_dairy_used >= 1 and _has_breakfast_non_cultured_side(len(valid_combos), require_unseen=False):
                    scan_modes.append((len(valid_combos), False, True))
                scan_modes.append((min(len(valid_combos), 80), False, False))
            else:
                if meal_key == "breakfast" and breakfast_cultured_dairy_used >= 1 and _has_breakfast_non_cultured_side(len(valid_combos), require_unseen=False):
                    scan_modes.append((len(valid_combos), False, True))
                scan_modes.append((min(len(valid_combos), 80), False, False))

            best_idx = -1
            best_adj_score = -1.0
            for limit_scan, require_unique_side_scan, avoid_cultured_dairy_scan in scan_modes:
                best_idx = -1
                best_adj_score = -1.0
                for i in range(limit_scan):
                    c = valid_combos[i]
                    main_id = self._role_title_key(c["main"]) or str(c["main"].get("recipe_id") or c["main"].get("id"))
                    side_id = self._side_diversity_key(c["side"], meal_type) or str(c["side"].get("recipe_id") or c["side"].get("id"))
                    side_penalty = float(combo_reuse_penalty_base) ** used_sides.get(side_id, 0)
                    if meal_key == "breakfast":
                        side_id = self._role_title_key(c["side"]) or str(c["side"].get("recipe_id") or c["side"].get("id"))
                        if require_unique_side_scan and used_sides.get(side_id, 0) > 0:
                            continue
                        side_family_id = self._breakfast_side_family(c["side"]) or "other"
                        if avoid_cultured_dairy_scan and side_family_id == "cultured_dairy":
                            continue
                        side_family_reuse = used_breakfast_side_families.get(side_family_id, 0)
                        side_penalty = (
                            (breakfast_side_title_penalty_base ** used_sides.get(side_id, 0))
                            * (breakfast_side_family_penalty_base ** side_family_reuse)
                        )
                        if side_family_id == "cultured_dairy" and side_family_reuse >= 2:
                            side_penalty *= 0.35
                    drink_id = self._role_title_key(c["drink"]) or str(c["drink"].get("recipe_id") or c["drink"].get("id"))

                    # Penalty: degrade score if main/side/drink reuse narrows the visible combo set.
                    main_usage = used_mains.get(main_id, 0)
                    drink_usage = used_drinks.get(drink_id, 0)
                    usage_penalty = (main_penalty_base ** main_usage) * side_penalty * (drink_penalty_base ** drink_usage)

                    adj_score = to_float(c.get("combo_score"), 0.0) * usage_penalty

                    if adj_score > best_adj_score:
                        best_adj_score = adj_score
                        best_idx = i

                if best_idx >= 0:
                    break
            
            if best_idx >= 0:
                chosen = valid_combos.pop(best_idx)
                combos.append(chosen)
                
                m_id = self._role_title_key(chosen["main"]) or str(chosen["main"].get("recipe_id") or chosen["main"].get("id"))
                s_id = self._side_diversity_key(chosen["side"], meal_type) or str(chosen["side"].get("recipe_id") or chosen["side"].get("id"))
                if meal_key == "breakfast":
                    s_id = self._role_title_key(chosen["side"]) or str(chosen["side"].get("recipe_id") or chosen["side"].get("id"))
                    side_family_id = self._breakfast_side_family(chosen["side"]) or "other"
                    used_breakfast_side_families[side_family_id] = used_breakfast_side_families.get(side_family_id, 0) + 1
                d_id = self._role_title_key(chosen["drink"]) or str(chosen["drink"].get("recipe_id") or chosen["drink"].get("id"))
                used_mains[m_id] = used_mains.get(m_id, 0) + 1
                used_sides[s_id] = used_sides.get(s_id, 0) + 1
                used_drinks[d_id] = used_drinks.get(d_id, 0) + 1
            else:
                break

        diversity_ms = _elapsed_ms(diversity_started_at)

        combos.sort(key=lambda combo: combo.get("combo_score", 0.0), reverse=True)
        if not combos:
            return [], {
                "combo_pool_build_ms": round(pool_build_ms, 1),
                "combo_candidate_generation_ms": round(combo_generation_ms, 1),
                "combo_diversity_ms": round(diversity_ms, 1),
            }

        output: list[dict[str, Any]] = []
        seen_combo_keys: set[str] = set()
        used_titles: dict[str, set[str]] = {"main": set(), "side": set(), "drink": set()}
        unique_role_counts = {
            "main": len({self._role_title_key(combo["main"]) for combo in combos if self._role_title_key(combo["main"])}),
            "side": len({self._role_title_key(combo["side"]) for combo in combos if self._role_title_key(combo["side"])}),
            "drink": len({self._role_title_key(combo["drink"]) for combo in combos if self._role_title_key(combo["drink"])}),
        }
        enforce_unique_roles = {
            role: unique_role_counts.get(role, 0) >= self._combo_unique_role_target(role, meal_type, max_combos)
            for role in ("main", "side", "drink")
        }

        def _add_combo(
            combo: dict[str, Any],
            *,
            require_unique_main: bool = False,
            require_unique_side: bool = False,
            require_unique_drink: bool = False,
        ) -> bool:
            main = combo["main"]
            side = combo["side"]
            drink = combo["drink"]
            combo_key = "|".join(
                str(item.get("recipe_id") or item.get("id") or "")
                for item in (main, side, drink)
            )
            if combo_key in seen_combo_keys:
                return False

            side_title = self._role_title_key(side)
            drink_title = canonical_title_key(drink.get("canonical_title") or drink.get("title") or "")
            main_title = canonical_title_key(main.get("canonical_title") or main.get("title") or "")
            if (
                (require_unique_main and main_title in used_titles["main"])
                or (require_unique_side and side_title in used_titles["side"])
                or (require_unique_drink and drink_title in used_titles["drink"])
            ):
                return False

            seen_combo_keys.add(combo_key)
            if main_title:
                used_titles["main"].add(main_title)
            if side_title:
                used_titles["side"].add(side_title)
            if drink_title:
                used_titles["drink"].add(drink_title)

            combo_items = [
                {**self._to_recommended_item(main, meal_type, slot_target, behavioral_insight), "category": "main"},
                {**self._to_recommended_item(side, meal_type, slot_target, behavioral_insight), "category": "side"},
                {**self._to_recommended_item(drink, meal_type, slot_target, behavioral_insight), "category": "drink"},
            ]
            totals = self._combo_macro_totals(combo_items)
            title = "Combo: " + " + ".join(item.get("title") or "Item" for item in combo_items)

            output.append(
                {
                    "id": f"{meal_type}-combo-{len(output) + 1}",
                    "combo_id": f"{meal_type}-combo-{len(output) + 1}",
                    "meal_type": meal_type,
                    "title": title,
                    "items": combo_items,
                    "total_calories": totals["calories"],
                    "total_protein": totals["protein"],
                    "total_carbs": totals["carbs"],
                    "total_fats": totals["fats"],
                    "score": round(float(combo.get("combo_score", 0.0)), 5),
                    "explanation": behavioral_insight,
                    "behavioral_insight": behavioral_insight,
                    "ml_tag": "COMBO",
                    "slot_target": int(slot_target),
                }
            )
            return True

        # NOTE: First pass enforces unique main/side/drink for diversity.
        unique_first_pass_output = 0
        for combo in combos:
            if len(output) >= max_combos:
                break
            if _add_combo(
                combo,
                require_unique_main=enforce_unique_roles["main"],
                require_unique_side=enforce_unique_roles["side"],
                require_unique_drink=enforce_unique_roles["drink"],
            ):
                unique_first_pass_output += 1

        # NOTE: Breakfast backfill relaxes main/drink reuse before allowing exact side repeats.
        breakfast_side_fill_output = 0
        if meal_key == "breakfast" and len(output) < max_combos:
            for combo in combos:
                if len(output) >= max_combos:
                    break
                if _add_combo(combo, require_unique_side=True):
                    breakfast_side_fill_output += 1

        # NOTE: Final pass fills remaining slots without uniqueness requirement.
        if len(output) < max_combos:
            for combo in combos:
                if len(output) >= max_combos:
                    break
                _add_combo(combo)

        print(
            f"**** Combo Assembly Result: meal_type={meal_type} output={len(output)} "
            f"max_combos={max_combos} unique_first_pass={unique_first_pass_output} "
            f"breakfast_side_fill={breakfast_side_fill_output} "
            f"diversity_ms={diversity_ms} total_ms={_elapsed_ms(combo_started_at)}"
        )

        return output, {
            "combo_pool_build_ms": round(pool_build_ms, 1),
            "combo_candidate_generation_ms": round(combo_generation_ms, 1),
            "combo_diversity_ms": round(diversity_ms, 1),
        }

    def _get_or_build_profile(self, user_id: Any, history_df, meal_type: str) -> dict[str, Any]:
        cached = self._profile_cache_get(user_id, meal_type)
        if cached is not None:
            return cached
        profile = build_user_profile(history_df, meal_type=meal_type)
        self._profile_cache_set(user_id, meal_type, profile)
        return profile

    @staticmethod
    def _allocate_shared_slot_budget(total: int, slot_count: int) -> list[int]:
        normalized_total = max(0, int(total))
        normalized_slot_count = max(0, int(slot_count))
        if normalized_slot_count <= 0:
            return []

        allocations: list[int] = []
        remaining = normalized_total
        for index in range(normalized_slot_count):
            slots_left = normalized_slot_count - index
            if remaining <= 0:
                allocations.append(0)
                continue

            allocation = int((remaining + slots_left - 1) / slots_left)
            allocations.append(allocation)
            remaining = max(0, remaining - allocation)

        return allocations

    def _build_slot_lookup_plans(self, selected_meals: list[str]) -> dict[str, dict[str, int]]:
        slot_count = len(selected_meals)
        new_mapping_caps = self._allocate_shared_slot_budget(self.max_new_mapping_lookups, slot_count)
        lookup_attempt_caps = self._allocate_shared_slot_budget(self.max_mapping_lookup_attempts, slot_count)
        sync_lookup_caps = self._allocate_shared_slot_budget(self.sync_mapping_lookups_per_slot * slot_count, slot_count)
        visible_sync_reserve_cap = 4
        if self.parallel_slot_execution_enabled and slot_count > 1:
            # Parallel all-slot requests can otherwise spend up to 4 synchronous
            # visible rescues per slot on the critical path. Keep a small rescue
            # budget for first-paint quality, but cap it aggressively to protect
            # cold-start latency.
            visible_sync_reserve_cap = 1

        plans: dict[str, dict[str, int]] = {}
        for index, meal_type in enumerate(selected_meals):
            new_mapping_cap = new_mapping_caps[index] if index < len(new_mapping_caps) else 0
            lookup_attempt_cap = lookup_attempt_caps[index] if index < len(lookup_attempt_caps) else 0
            sync_lookup_cap = sync_lookup_caps[index] if index < len(sync_lookup_caps) else 0
            plans[meal_type] = {
                "remaining_new_mappings": new_mapping_cap,
                "remaining_lookup_attempts": lookup_attempt_cap,
                "remaining_sync_lookup_attempts": sync_lookup_cap,
                "slot_new_mapping_cap": new_mapping_cap,
                "slot_sync_lookup_cap": sync_lookup_cap,
                "visible_sync_reserve_cap": visible_sync_reserve_cap,
            }

        return plans

    def _summarize_lookup_state(self, payload_by_slot: dict[str, Any], selected_meals: list[str]) -> dict[str, int]:
        used_new_mappings = 0
        used_lookup_attempts = 0
        used_sync_lookups = 0

        for meal_type in selected_meals:
            payload = payload_by_slot.get(meal_type) or {}
            usage = payload.get("mapping_usage") or {}
            used_new_mappings += max(0, int(usage.get("slot_new_mappings_used", 0)))
            used_lookup_attempts += max(0, int(usage.get("slot_lookup_attempts_used", 0)))
            used_sync_lookups += max(0, int(usage.get("slot_sync_lookups_used", 0)))

        return {
            "remaining_new_mappings": max(0, int(self.max_new_mapping_lookups) - used_new_mappings),
            "remaining_lookup_attempts": max(0, int(self.max_mapping_lookup_attempts) - used_lookup_attempts),
            "remaining_sync_lookup_attempts": max(
                0,
                int(self.sync_mapping_lookups_per_slot * len(selected_meals)) - used_sync_lookups,
            ),
        }

    def _build_slot_payload(
        self,
        meal_type: str,
        slot_weights: dict[str, float],
        daily_calories: float,
        demographics: dict[str, Any],
        history_df,
        user_id: Any,
        lookup_state: dict[str, int],
        slot_new_mapping_cap: int,
        slot_sync_lookup_cap: int,
        is_australian_user: bool,
        experiment_config: dict[str, Any],
        feedback_context: dict[str, Any] | None = None,
        use_dedicated_search_connection: bool = False,
    ) -> dict[str, Any]:
        slot_started_at = time.perf_counter()
        slot_target = int(round(max(0.0, daily_calories) * slot_weights.get(meal_type, DEFAULT_MEAL_ALLOCATION[meal_type])))

        profile_started_at = time.perf_counter()
        profile = self._get_or_build_profile(user_id, history_df, meal_type)
        profile_ms = _elapsed_ms(profile_started_at)

        retrieval_started_at = time.perf_counter()
        candidate_pool_multiplier = max(0.75, to_float(experiment_config.get("candidate_pool_multiplier"), 1.0))
        candidate_pool_target = int(
            np.clip(
                round(LOCAL_CANDIDATE_POOL_PER_MEAL * candidate_pool_multiplier),
                LOCAL_CANDIDATE_POOL_EXPERIMENT_MIN,
                LOCAL_CANDIDATE_POOL_EXPERIMENT_MAX,
            )
        )
        effective_prefetch_pool = LOCAL_PREFETCH_POOL
        if use_dedicated_search_connection:
            # Parallel all-slot requests already multiply concurrent DuckDB readers.
            # Trim the prefetch fan-out for this path only so retrieval stays closer
            # to the tuned 80-candidate pool without the extra 160-row pressure per slot.
            effective_prefetch_pool = min(LOCAL_PREFETCH_POOL, 120)
        prefetch_target = max(int(round(effective_prefetch_pool * candidate_pool_multiplier)), candidate_pool_target)
        primary_search_budgets = self._primary_search_budgets(meal_type, candidate_pool_target, prefetch_target)

        main_target = float(slot_target) * float(COMBO_CATEGORY_TARGETS.get("main", 0.65))
        main_query_vec = build_query_vector(slot_target=main_target, user_vec=profile.get("user_vec"))
        side_target = float(slot_target) * float(COMBO_CATEGORY_TARGETS.get("side", 0.20))
        side_query_str = "|".join([normalize_text(kw) for kw in self._side_query_keywords(meal_type)])
        side_query_vec = build_query_vector(slot_target=side_target, user_vec=profile.get("user_vec"))

        drink_target = float(slot_target) * float(COMBO_CATEGORY_TARGETS.get("drink", 0.15))
        drink_query_str = self._primary_drink_query_regex(meal_type)
        drink_query_vec = build_query_vector(slot_target=drink_target, user_vec=profile.get("user_vec"))

        main_candidates, side_candidates, drink_candidates, primary_retrieval_info = self._search_primary_role_candidates(
            meal_type=meal_type,
            main_query_vec=main_query_vec,
            side_query_vec=side_query_vec,
            drink_query_vec=drink_query_vec,
            side_query_str=side_query_str,
            drink_query_str=drink_query_str,
            primary_search_budgets=primary_search_budgets,
            is_australian_user=is_australian_user,
            use_dedicated_search_connection=use_dedicated_search_connection,
        )

        side_expansion_info = {
            "used": False,
            "queries": [],
            "added": 0,
            "candidate_count": len(side_candidates),
        }
        side_candidates, side_expansion_info = self._expand_side_candidates(
            meal_type=meal_type,
            query_vector=side_query_vec,
            side_candidates=side_candidates,
            candidate_pool_target=candidate_pool_target,
            prefetch_target=prefetch_target,
            is_australian_user=is_australian_user,
            use_dedicated_search_connection=use_dedicated_search_connection,
        )
        side_candidates = self._filter_side_candidates_for_meal(meal_type, side_candidates)
        if normalize_text(meal_type) == "dinner":
            primary_dinner_drink_limit = min(2, self._drink_diversity_target(meal_type))
            drink_candidates = self._select_primary_dinner_drink_candidates(
                drink_candidates,
                primary_dinner_drink_limit,
            )
        primary_drink_candidate_count = len(drink_candidates)
        dinner_drink_target = self._drink_diversity_target(meal_type)
        dinner_drink_safety_fill_added = 0
        dinner_drink_prefer_safety_fill = False
        drink_expansion_info = {
            "used": False,
            "query": "",
            "added": 0,
            "candidate_count": primary_drink_candidate_count,
        }
        if normalize_text(meal_type) == "dinner":
            dinner_safety_drink_candidates = self._filter_drink_candidates_for_meal(
                meal_type,
                self._beverage_safety_candidates(meal_type),
            )
            has_local_nut_based_plant_milk = any(
                self._is_nut_based_plant_milk_drink_candidate(candidate) and not self._is_safety_drink_candidate(candidate)
                for candidate in drink_candidates
            )
            local_non_plant_drink_candidates = [
                candidate for candidate in drink_candidates if not self._is_plant_milk_drink_candidate(candidate)
            ]
            weak_local_plant_milk_candidates = [
                candidate
                for candidate in drink_candidates
                if self._is_plant_milk_drink_candidate(candidate)
                and not self._is_nut_based_plant_milk_drink_candidate(candidate)
            ]
            if local_non_plant_drink_candidates and weak_local_plant_milk_candidates:
                dinner_drink_prefer_safety_fill = True
                if len(local_non_plant_drink_candidates) != len(drink_candidates):
                    print(
                        f"**** Slot={meal_type} drink normalization: before={len(drink_candidates)} "
                        f"after={len(local_non_plant_drink_candidates)} reason=mixed_plant_milk_pool"
                    )
                drink_candidates = local_non_plant_drink_candidates
            elif drink_candidates and not has_local_nut_based_plant_milk and not local_non_plant_drink_candidates:
                dinner_drink_prefer_safety_fill = True
                strongest_plant_milk_candidate = max(
                    drink_candidates,
                    key=self._dinner_primary_plant_milk_priority,
                )
                normalized_drink_candidates = []
                if not self._is_generic_plant_milk_drink_candidate(strongest_plant_milk_candidate):
                    normalized_drink_candidates = [strongest_plant_milk_candidate]
                if len(normalized_drink_candidates) != len(drink_candidates):
                    print(
                        f"**** Slot={meal_type} drink normalization: before={len(drink_candidates)} "
                        f"after={len(normalized_drink_candidates)} reason=plant_milk_only_pool"
                    )
                drink_candidates = normalized_drink_candidates
            if has_local_nut_based_plant_milk and len(drink_candidates) < dinner_drink_target:
                drink_candidates = self._select_diverse_dinner_drink_candidates(
                    merge_candidates(
                        drink_candidates,
                        dinner_safety_drink_candidates,
                        max_items=max(12, len(drink_candidates) + len(dinner_safety_drink_candidates)),
                    ),
                    dinner_drink_target,
                )
        if normalize_text(meal_type) == "dinner" and len(drink_candidates) < dinner_drink_target:
            if len(drink_candidates) >= max(1, dinner_drink_target - 1) or dinner_drink_prefer_safety_fill:
                safety_completed_drink_candidates: list[dict[str, Any]] = []
                selected_drink_keys = {
                    candidate_key
                    for candidate_key in (
                        self._candidate_merge_key(candidate) for candidate in drink_candidates
                    )
                    if candidate_key
                }
                plant_milk_selected = any(
                    self._is_plant_milk_drink_candidate(candidate) for candidate in drink_candidates
                )
                for candidate in self._ordered_dinner_drink_candidates(dinner_safety_drink_candidates):
                    candidate_key = self._candidate_merge_key(candidate)
                    if candidate_key and candidate_key in selected_drink_keys:
                        continue
                    if self._is_plant_milk_drink_candidate(candidate) and plant_milk_selected:
                        continue
                    if candidate_key:
                        selected_drink_keys.add(candidate_key)
                    if self._is_plant_milk_drink_candidate(candidate):
                        plant_milk_selected = True
                    safety_completed_drink_candidates.append(candidate)
                    if len(drink_candidates) + len(safety_completed_drink_candidates) >= dinner_drink_target:
                        break

                if safety_completed_drink_candidates:
                    drink_candidates = merge_candidates(
                        drink_candidates,
                        safety_completed_drink_candidates,
                        max_items=max(12, dinner_drink_target + 6),
                    )
                    dinner_drink_safety_fill_added = len(safety_completed_drink_candidates)
                    print(
                        f"**** Slot={meal_type} drink safety-fill: before={primary_drink_candidate_count} "
                        f"after={len(drink_candidates)} added={dinner_drink_safety_fill_added}"
                    )
            else:
                expansion_query = self._drink_supplement_regex(meal_type)
                existing_drink_ids = {
                    recipe_id
                    for recipe_id in (
                        self._recipe_id_from_candidate(candidate)
                        for candidate in drink_candidates
                    )
                    if recipe_id
                }
                neutral_drink_query_vec = build_query_vector(slot_target=drink_target, user_vec=None)
                needed_drink_candidates = max(
                    0,
                    dinner_drink_target - len(drink_candidates),
                )
                expansion_top_k = 20
                expansion_prefetch = max(80, int(primary_search_budgets["drink_prefetch"]))
                if needed_drink_candidates <= 1:
                    expansion_top_k = max(9, int(round(candidate_pool_target * 0.10)))
                    expansion_prefetch = max(32, int(round(primary_search_budgets["drink_prefetch"] * 0.80)))
                else:
                    expansion_top_k = max(12, int(round(candidate_pool_target * 0.15)))
                    expansion_prefetch = max(48, int(round(primary_search_budgets["drink_prefetch"] * 1.2)))

                expanded_drink_candidates = self.local_dataset.search(
                    meal_type=meal_type,
                    query_vector=neutral_drink_query_vec,
                    top_k=expansion_top_k,
                    prefetch=expansion_prefetch,
                    exclude_recipe_ids=existing_drink_ids,
                    is_australian_user=is_australian_user,
                    text_query=expansion_query,
                    role_hint="drink",
                    dedicated_connection=use_dedicated_search_connection,
                )
                expanded_drink_candidates = self._filter_drink_candidates_for_meal(meal_type, expanded_drink_candidates)

                selected_plant_milk_candidates = [
                    candidate for candidate in drink_candidates if self._is_plant_milk_drink_candidate(candidate)
                ]
                replacement_applied = False
                if selected_plant_milk_candidates:
                    weakest_selected_plant_milk = min(
                        selected_plant_milk_candidates,
                        key=self._dinner_primary_plant_milk_priority,
                    )
                    replacement_candidates = [
                        candidate
                        for candidate in expanded_drink_candidates
                        if self._is_plant_milk_drink_candidate(candidate)
                        and self._dinner_primary_plant_milk_priority(candidate)
                        > self._dinner_primary_plant_milk_priority(weakest_selected_plant_milk)
                    ]
                    if replacement_candidates:
                        replacement_candidate = max(
                            enumerate(replacement_candidates),
                            key=lambda item: (self._dinner_primary_plant_milk_priority(item[1]), -item[0]),
                        )[1]
                        weakest_key = self._candidate_merge_key(weakest_selected_plant_milk)
                        replacement_key = self._candidate_merge_key(replacement_candidate)
                        drink_candidates = [
                            candidate
                            for candidate in drink_candidates
                            if self._candidate_merge_key(candidate) != weakest_key
                        ]
                        drink_candidates = merge_candidates(
                            drink_candidates,
                            [replacement_candidate],
                            max_items=max(12, dinner_drink_target + 6),
                        )
                        if replacement_key:
                            expanded_drink_candidates = [
                                candidate
                                for candidate in expanded_drink_candidates
                                if self._candidate_merge_key(candidate) != replacement_key
                            ]
                        replacement_applied = True

                expanded_drink_candidates = merge_candidates(
                    expanded_drink_candidates,
                    dinner_safety_drink_candidates,
                    max_items=max(12, len(expanded_drink_candidates) + len(dinner_safety_drink_candidates)),
                )
                expanded_drink_candidates = self._select_diverse_dinner_drink_candidates(
                    expanded_drink_candidates,
                    needed_drink_candidates,
                )
                if expanded_drink_candidates:
                    drink_candidates = merge_candidates(
                        drink_candidates,
                        expanded_drink_candidates,
                        max_items=max(12, dinner_drink_target + 6),
                    )

                added_drink_candidates = max(0, len(drink_candidates) - primary_drink_candidate_count)
                if added_drink_candidates > 0 or replacement_applied:
                    drink_expansion_info = {
                        "used": True,
                        "query": expansion_query,
                        "added": added_drink_candidates,
                        "candidate_count": len(drink_candidates),
                    }
                    print(
                        f"**** Slot={meal_type} drink expansion: before={primary_drink_candidate_count} "
                        f"after={len(drink_candidates)} added={added_drink_candidates} query={expansion_query}"
                    )

        local_candidates = self._merge_balanced_primary_candidates(
            main_candidates,
            side_candidates,
            drink_candidates,
            max_items=candidate_pool_target,
        )
        merged_role_counts = {
            "main": sum(1 for candidate in local_candidates if self._infer_combo_category(candidate) == "main"),
            "side": sum(1 for candidate in local_candidates if self._infer_combo_category(candidate) == "side"),
            "drink": sum(1 for candidate in local_candidates if self._infer_combo_category(candidate) == "drink"),
        }
        print(
            f"**** Slot={meal_type} primary retrieval budgets: "
            f"main={primary_search_budgets['main_top_k']}/{primary_search_budgets['main_prefetch']} "
            f"side={primary_search_budgets['side_top_k']}/{primary_search_budgets['side_prefetch']} "
            f"drink={primary_search_budgets['drink_top_k']}/{primary_search_budgets['drink_prefetch']} "
            f"parallel_roles={int(primary_retrieval_info['parallel_used'])} "
            f"staggered_drink={int(primary_retrieval_info.get('staggered_drink_search', False))} "
            f"merged_counts={merged_role_counts}"
        )
        initial_retrieval_metrics = compute_candidate_pool_metrics(local_candidates, top_n=20)
        expansion_regex = ""
        query_expansion_used = False
        skip_parallel_query_expansion = self._should_skip_parallel_query_expansion(
            meal_type,
            merged_role_counts,
            dedicated_parallel_path=bool(primary_retrieval_info.get("dedicated_connection")),
            dinner_drink_target=dinner_drink_target,
        )

        if skip_parallel_query_expansion:
            print(
                f"**** Slot={meal_type} query expansion skipped: "
                f"counts={merged_role_counts} parallel_slot_path=1"
            )
        elif self._should_expand_query(initial_retrieval_metrics):
            expansion_regex = self._build_query_expansion_regex(profile, demographics, meal_type)
            if expansion_regex:
                exclude_recipe_ids = {
                    str(candidate.get("recipe_id") or "").strip()
                    for candidate in local_candidates
                    if str(candidate.get("recipe_id") or "").strip()
                }
                expansion_top_k = max(20, int(candidate_pool_target / 2))
                expansion_prefetch = max(40, int(prefetch_target / 2))
                if normalize_text(meal_type) == "dinner":
                    expansion_top_k = max(18, int(round(candidate_pool_target * 0.30)))
                    expansion_prefetch = max(48, int(round(prefetch_target * 0.30)))
                expanded_candidates = self.local_dataset.search(
                    meal_type=meal_type,
                    query_vector=main_query_vec,
                    top_k=expansion_top_k,
                    prefetch=expansion_prefetch,
                    exclude_recipe_ids=exclude_recipe_ids,
                    is_australian_user=is_australian_user,
                    text_query=expansion_regex,
                    role_hint="main",
                    dedicated_connection=use_dedicated_search_connection,
                )
                expanded_candidates = self._filter_main_candidates_for_meal(meal_type, expanded_candidates)
                if expanded_candidates:
                    local_candidates = merge_candidates(local_candidates, expanded_candidates, max_items=candidate_pool_target)
                    query_expansion_used = True

        if not local_candidates:
            local_candidates = self._safety_to_local_candidates(meal_type, profile.get("top_foods", []))
        retrieval_ms = _elapsed_ms(retrieval_started_at)

        visible_sync_reserve = 0
        if max(0, int(slot_sync_lookup_cap)) <= 0:
            visible_sync_reserve = min(
                max(0, int(lookup_state.get("visible_sync_reserve_cap", 4))),
                max(0, int(slot_new_mapping_cap)),
                max(0, int(lookup_state.get("remaining_new_mappings", 0))),
                max(0, int(lookup_state.get("remaining_lookup_attempts", 0))),
            )

        slot_lookup_state = {
            "remaining_new_mappings": max(
                0,
                int(lookup_state.get("remaining_new_mappings", 0)) - int(visible_sync_reserve),
            ),
            "remaining_lookup_attempts": lookup_state.get("remaining_lookup_attempts", 0),
            "remaining_sync_lookup_attempts": lookup_state.get("remaining_sync_lookup_attempts", 0),
            "slot_new_mapping_cap": max(0, int(slot_new_mapping_cap)),
            "slot_new_mappings_used": 0,
            "slot_sync_lookup_cap": max(0, int(slot_sync_lookup_cap)),
            "slot_sync_lookups_used": 0,
            "visible_sync_reserve": int(visible_sync_reserve),
        }

        mapping_started_at = time.perf_counter()
        mapped_candidates: list[dict[str, Any]] = []
        seen_recipe_ids: set[str] = set()

        def _mapping_candidate_priority(candidate: dict[str, Any]) -> tuple[int, int, int, float, int, float, float, float]:
            recipe_id = self._recipe_id_from_candidate(candidate)
            category = self._infer_combo_category(candidate)
            meal_key = normalize_text(meal_type)
            role_quality = self._role_quality_multiplier(candidate, category, meal_type)

            title_tokens = tokenize(
                candidate.get("canonical_title")
                or candidate.get("mapped_canonical_title")
                or candidate.get("title")
                or candidate.get("original_title")
            )
            mapping_search_penalty = 0
            if meal_key == "breakfast" and category == "main":
                if self._is_blocked_breakfast_main_candidate(candidate):
                    mapping_search_penalty = 3
                elif self._is_breakfast_aligned_main_candidate(candidate):
                    if len(title_tokens) <= 2:
                        mapping_search_penalty = 0
                    elif len(title_tokens) <= 4:
                        mapping_search_penalty = 1
                    else:
                        mapping_search_penalty = 2
                else:
                    mapping_search_penalty = 2 if len(title_tokens) <= 4 else 3

            breakfast_alignment_penalty = (
                1
                if meal_key == "breakfast"
                and category == "main"
                and not self._is_breakfast_aligned_main_candidate(candidate)
                else 0
            )
            title_key = canonical_title_key(candidate.get("title") or candidate.get("original_title"))
            canonical_key = canonical_title_key(
                candidate.get("canonical_title")
                or candidate.get("mapped_canonical_title")
                or candidate.get("title")
                or candidate.get("original_title")
            )
            serving_calories = max(
                0.0,
                to_float(
                    candidate.get("serving_calories"),
                    candidate.get("dataset_serving_calories"),
                ),
            )
            category_target = max(0.0, float(slot_target) * float(COMBO_CATEGORY_TARGETS.get(category, 0.0)))
            calorie_gap = 1.0
            if serving_calories > 0.0 and category_target > 0.0:
                calorie_gap = abs(serving_calories - category_target) / max(100.0, category_target)
                if category in {"side", "drink"} and serving_calories <= 200.0:
                    calorie_gap = min(calorie_gap, 0.25)
            health_score = to_float(candidate.get("health_score"), 0.0)
            knn_distance = to_float(candidate.get("knn_distance"), 999.0)
            return (
                1 if self._is_safety_recipe_id(recipe_id) else 0,
                mapping_search_penalty,
                breakfast_alignment_penalty,
                -role_quality,
                0 if canonical_key and canonical_key != title_key else 1,
                calorie_gap,
                -health_score,
                knn_distance,
            )

        def _mapping_title_family_key(candidate: dict[str, Any]) -> str:
            return canonical_title_key(
                candidate.get("canonical_title")
                or candidate.get("mapped_canonical_title")
                or candidate.get("title")
                or candidate.get("original_title")
            )

        def _prioritize_mapping_candidates(source_candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
            category_buckets: dict[str, list[dict[str, Any]]] = {
                "main": [],
                "side": [],
                "drink": [],
                "other": [],
            }
            for candidate in source_candidates:
                category = self._infer_combo_category(candidate)
                bucket = category if category in category_buckets else "other"
                category_buckets[bucket].append(candidate)

            unique_buckets: dict[str, list[dict[str, Any]]] = {key: [] for key in category_buckets}
            repeat_buckets: dict[str, list[dict[str, Any]]] = {key: [] for key in category_buckets}
            for bucket, bucket_candidates in category_buckets.items():
                seen_title_keys: set[str] = set()
                for candidate in sorted(bucket_candidates, key=_mapping_candidate_priority):
                    title_family_key = _mapping_title_family_key(candidate)
                    if title_family_key and title_family_key in seen_title_keys:
                        repeat_buckets[bucket].append(candidate)
                        continue
                    if title_family_key:
                        seen_title_keys.add(title_family_key)
                    unique_buckets[bucket].append(candidate)

            prioritized: list[dict[str, Any]] = []

            def _append_weighted_rounds(bucket_map: dict[str, list[dict[str, Any]]]) -> None:
                category_order = ("main", "side", "drink", "main", "other")
                cursors = {key: 0 for key in bucket_map}
                total_candidates = sum(len(items) for items in bucket_map.values())
                appended = 0
                while appended < total_candidates:
                    progressed = False
                    for category in category_order:
                        cursor = cursors[category]
                        category_items = bucket_map[category]
                        if cursor >= len(category_items):
                            continue
                        prioritized.append(category_items[cursor])
                        cursors[category] = cursor + 1
                        appended += 1
                        progressed = True
                    if not progressed:
                        break

            _append_weighted_rounds(unique_buckets)
            _append_weighted_rounds(repeat_buckets)
            return prioritized

        prioritized_local_candidates = _prioritize_mapping_candidates(local_candidates)
        if prioritized_local_candidates:
            mapping_priority_preview: list[str] = []
            for candidate in prioritized_local_candidates[: max(6, int(slot_new_mapping_cap))]:
                preview_title = str(candidate.get("title") or candidate.get("canonical_title") or "").strip()
                if len(preview_title) > 48:
                    preview_title = f"{preview_title[:45]}..."
                mapping_priority_preview.append(f"{self._infer_combo_category(candidate)}:{preview_title}")
            print(f"**** Slot={meal_type} mapping-priority frontload={mapping_priority_preview}")

        def _append_mapped_candidates(source_candidates: list[dict[str, Any]], max_total: int) -> int:
            added = 0
            for candidate in source_candidates:
                recipe_id = self._recipe_id_from_candidate(candidate)
                if not recipe_id:
                    continue
                if recipe_id in seen_recipe_ids:
                    continue
                seen_recipe_ids.add(recipe_id)

                resolved = self._resolve_candidate_with_mapping(candidate, meal_type, slot_lookup_state)
                if not resolved:
                    continue

                mapped_candidates.append(resolved)
                added += 1
                if len(mapped_candidates) >= max_total:
                    break
            return added

        _append_mapped_candidates(prioritized_local_candidates, candidate_pool_target)

        drink_compatible_before = len(self._role_compatible_candidates(mapped_candidates, "drink"))
        side_compatible_before = len(self._role_compatible_candidates(mapped_candidates, "side"))
        strong_side_before = len(
            [
                candidate
                for candidate in self._role_compatible_candidates(mapped_candidates, "side")
                if self._role_quality_multiplier(candidate, "side", meal_type) >= (0.8 if normalize_text(meal_type) == "dinner" else 0.65)
            ]
        )
        drink_supplement_used = False
        drink_supplement_added = 0
        drink_supplement_regex = ""
        minimum_drink_pool = self._drink_pool_floor(meal_type)
        desired_ranked_drink_pool = minimum_drink_pool
        if normalize_text(meal_type) == "dinner":
            desired_ranked_drink_pool = max(
                desired_ranked_drink_pool,
                min(self._drink_diversity_target(meal_type), drink_compatible_before),
            )
        minimum_side_pool = self._side_pool_floor(meal_type)
        minimum_main_pool = self._main_pool_floor(meal_type)
        drink_supplement_attempted = False
        drink_supplement_source = "none"
        drink_supplement_candidate_count = 0
        drink_supplement_safety_added = 0
        side_supplement_used = False
        side_supplement_added = 0
        side_supplement_safety_added = 0
        side_supplement_regex = ""
        main_supplement_used = False
        main_supplement_added = 0

        def _eligible_main_count(candidates: list[dict[str, Any]]) -> int:
            mains = self._role_compatible_candidates(candidates, "main")
            if normalize_text(meal_type) == "breakfast":
                mains = [candidate for candidate in mains if self._is_breakfast_aligned_main_candidate(candidate)]
            return len(mains)

        def _drink_debug_samples(candidates: list[dict[str, Any]], limit: int = 5) -> list[dict[str, Any]]:
            samples: list[dict[str, Any]] = []
            for candidate in self._role_compatible_candidates(candidates, "drink")[:limit]:
                title = str(candidate.get("title") or candidate.get("canonical_title") or "").strip()
                if not title:
                    continue
                samples.append(
                    {
                        "title": title,
                        "inferred_category": self._infer_combo_category(candidate),
                        "explicit_category": str(candidate.get("category") or "").strip(),
                        "metric_serving_unit": str(candidate.get("metric_serving_unit") or "").strip(),
                        "serving_calories": round(to_float(candidate.get("serving_calories"), 0.0), 3),
                        "source_keyword": str(candidate.get("source_keyword") or "").strip(),
                        "recipe_category": str(candidate.get("recipe_category") or "").strip(),
                        "keywords": str(candidate.get("keywords") or "").strip(),
                        "mapped_query": str(candidate.get("mapped_query") or "").strip(),
                        "mapped_title": str(candidate.get("mapped_title") or "").strip(),
                    }
                )
            return samples

        mapped_drink_samples = _drink_debug_samples(mapped_candidates)

        def _apply_drink_supplement(max_total: int) -> int:
            nonlocal drink_supplement_used, drink_supplement_added, drink_supplement_regex, local_candidates
            nonlocal drink_supplement_attempted, drink_supplement_source, drink_supplement_candidate_count
            nonlocal drink_supplement_safety_added
            supplement_limit = max(max_total, len(mapped_candidates) + max(8, minimum_drink_pool * 4))
            drink_supplement_attempted = True
            drink_supplement_regex = self._drink_supplement_regex(meal_type)
            compatible_before_add = len(self._role_compatible_candidates(mapped_candidates, "drink"))
            needed_drinks = max(1, minimum_drink_pool - compatible_before_add)
            skipped_titles = feedback_context.get("skipped_titles", set()) if feedback_context else set()
            loved_titles = feedback_context.get("loved_titles", set()) if feedback_context else set()
            favorite_titles = feedback_context.get("favorite_titles", set()) if feedback_context else set()
            title_bias = feedback_context.get("title_bias", {}) if feedback_context else {}

            def _supplement_priority(candidate: dict[str, Any]) -> tuple[int, int, int, int, float, float, float]:
                title_key = canonical_title_key(
                    candidate.get("canonical_title")
                    or candidate.get("mapped_canonical_title")
                    or candidate.get("title")
                    or candidate.get("original_title")
                )
                recipe_id = self._recipe_id_from_candidate(candidate)
                is_safety_candidate = self._is_safety_drink_candidate(candidate)
                bias = float(title_bias.get(title_key, 0.0))
                loved = title_key in loved_titles or title_key in favorite_titles
                skipped = title_key in skipped_titles or bias <= -0.05
                return (
                    0 if is_safety_candidate else 1,
                    0 if skipped else 1,
                    self._role_quality_multiplier(candidate, "drink", meal_type),
                    1 if loved else 0,
                    0 if self._is_plant_milk_drink_candidate(candidate) else 1,
                    bias,
                    to_float(candidate.get("score"), 0.0),
                )

            lunch_safety_short_circuit = normalize_text(meal_type) == "lunch" and compatible_before_add >= 1 and needed_drinks == 1
            if lunch_safety_short_circuit:
                existing_drink_keys = {
                    self._role_title_key(candidate) or self._recipe_id_from_candidate(candidate)
                    for candidate in self._role_compatible_candidates(mapped_candidates, "drink")
                    if self._role_title_key(candidate) or self._recipe_id_from_candidate(candidate)
                }
                safety_candidates = self._filter_drink_candidates_for_meal(
                    meal_type,
                    self._beverage_safety_candidates(meal_type),
                )
                prioritized_safety_candidates = [
                    candidate
                    for candidate in sorted(
                        self._role_compatible_candidates(safety_candidates, "drink"),
                        key=_supplement_priority,
                        reverse=True,
                    )
                    if (self._role_title_key(candidate) or self._recipe_id_from_candidate(candidate)) not in existing_drink_keys
                ]
                if prioritized_safety_candidates:
                    selected_supplement_candidates = prioritized_safety_candidates[:1]
                    drink_supplement_candidate_count = len(prioritized_safety_candidates)
                    drink_supplement_used = True
                    drink_supplement_source = "safety_only"
                    drink_supplement_safety_added = len(selected_supplement_candidates)
                    local_candidates = merge_candidates(
                        local_candidates,
                        selected_supplement_candidates,
                        max_items=max(candidate_pool_target + 8, candidate_pool_target * 2),
                    )
                    added = _append_mapped_candidates(selected_supplement_candidates, supplement_limit)
                    drink_supplement_added += added
                    return added

            supplement_top_k = max(18, min(candidate_pool_target, 12 + (needed_drinks * 10)))
            supplement_prefetch = max(32, min(prefetch_target, supplement_top_k * 2))
            if use_dedicated_search_connection:
                if needed_drinks <= 1:
                    supplement_top_k = min(supplement_top_k, 18)
                    supplement_prefetch = min(supplement_prefetch, 36)
                elif needed_drinks == 2:
                    supplement_top_k = min(supplement_top_k, 24)
                    supplement_prefetch = min(supplement_prefetch, 48)
                else:
                    supplement_top_k = min(supplement_top_k, 28)
                    supplement_prefetch = min(supplement_prefetch, 56)
            supplement_candidates = self.local_dataset.search(
                meal_type=meal_type,
                query_vector=drink_query_vec,
                top_k=supplement_top_k,
                prefetch=supplement_prefetch,
                exclude_recipe_ids=set(seen_recipe_ids),
                is_australian_user=is_australian_user,
                text_query=drink_supplement_regex,
                role_hint="drink",
                dedicated_connection=use_dedicated_search_connection,
            )
            supplement_candidates = self._filter_drink_candidates_for_meal(meal_type, supplement_candidates)
            safety_candidates: list[dict[str, Any]] = []
            if len(supplement_candidates) < max(2, needed_drinks):
                safety_candidates = self._filter_drink_candidates_for_meal(
                    meal_type,
                    self._beverage_safety_candidates(meal_type),
                )
                supplement_candidates = merge_candidates(
                    supplement_candidates,
                    safety_candidates,
                    max_items=max(6, len(supplement_candidates) + 6),
                )
            if not supplement_candidates:
                drink_supplement_source = "none_found"
                return 0

            drink_supplement_candidate_count = len(supplement_candidates)
            drink_supplement_used = True
            drink_supplement_source = "retrieval_plus_safety" if safety_candidates else "retrieval_only"
            safety_candidate_ids = {
                self._recipe_id_from_candidate(candidate)
                for candidate in safety_candidates
                if self._recipe_id_from_candidate(candidate)
            }
            def _retrieval_supplement_priority(candidate: dict[str, Any]) -> tuple[int, int, int, int, float, float, float]:
                title_key = canonical_title_key(
                    candidate.get("canonical_title")
                    or candidate.get("mapped_canonical_title")
                    or candidate.get("title")
                    or candidate.get("original_title")
                )
                recipe_id = self._recipe_id_from_candidate(candidate)
                is_safety_candidate = recipe_id in safety_candidate_ids or self._is_safety_drink_candidate(candidate)
                bias = float(title_bias.get(title_key, 0.0))
                loved = title_key in loved_titles or title_key in favorite_titles
                skipped = title_key in skipped_titles or bias <= -0.05
                return (
                    0 if is_safety_candidate else 1,
                    0 if skipped else 1,
                    self._role_quality_multiplier(candidate, "drink", meal_type),
                    1 if loved else 0,
                    0 if self._is_plant_milk_drink_candidate(candidate) else 1,
                    bias,
                    to_float(candidate.get("score"), 0.0),
                )

            prioritized_supplement_candidates = self._role_compatible_candidates(
                merge_candidates(
                    supplement_candidates,
                    safety_candidates,
                    max_items=max(12, len(supplement_candidates) + len(safety_candidates)),
                ),
                "drink",
            )
            prioritized_supplement_candidates = sorted(
                prioritized_supplement_candidates,
                key=_retrieval_supplement_priority,
                reverse=True,
            )
            selected_supplement_candidates = prioritized_supplement_candidates[:needed_drinks]
            drink_supplement_safety_added = sum(
                1
                for candidate in selected_supplement_candidates
                if self._recipe_id_from_candidate(candidate) in safety_candidate_ids
            )
            local_candidates = merge_candidates(
                local_candidates,
                selected_supplement_candidates,
                max_items=max(candidate_pool_target + 12, candidate_pool_target * 2),
            )
            added = _append_mapped_candidates(selected_supplement_candidates, supplement_limit)
            compatible_after_add = len(self._role_compatible_candidates(mapped_candidates, "drink"))
            if compatible_after_add < minimum_drink_pool:
                remaining_gap = max(1, minimum_drink_pool - compatible_after_add)
                fallback_safety_candidates = self._filter_drink_candidates_for_meal(
                    meal_type,
                    self._beverage_safety_candidates(meal_type),
                )
                fallback_safety_candidates = sorted(
                    fallback_safety_candidates,
                    key=_retrieval_supplement_priority,
                    reverse=True,
                )[:remaining_gap]
                if fallback_safety_candidates:
                    drink_supplement_safety_added += len(fallback_safety_candidates)
                    drink_supplement_source = "retrieval_plus_safety"
                    local_candidates = merge_candidates(
                        local_candidates,
                        fallback_safety_candidates,
                        max_items=max(candidate_pool_target + 18, candidate_pool_target * 2),
                    )
                    added += _append_mapped_candidates(fallback_safety_candidates, supplement_limit)
            drink_supplement_added += added
            return added

        if drink_compatible_before < minimum_drink_pool:
            _apply_drink_supplement(candidate_pool_target + 12)

        if minimum_main_pool > 0:
            main_before = _eligible_main_count(mapped_candidates)
            if main_before < minimum_main_pool:
                safety_main_candidates = [
                    candidate
                    for candidate in self._meal_main_safety_candidates(meal_type, profile.get("top_foods", []))
                    if self._recipe_id_from_candidate(candidate) not in seen_recipe_ids
                ]
                if safety_main_candidates:
                    main_supplement_used = True
                    local_candidates = merge_candidates(
                        local_candidates,
                        safety_main_candidates,
                        max_items=max(candidate_pool_target + 12, candidate_pool_target * 2),
                    )
                    main_supplement_added += _append_mapped_candidates(
                        safety_main_candidates,
                        candidate_pool_target + 12,
                    )
                    main_after = _eligible_main_count(mapped_candidates)
                    print(
                        f"**** Slot={meal_type} main supplement: before={main_before} "
                        f"after={main_after} added={main_supplement_added}"
                    )

        drink_compatible_after = len(self._role_compatible_candidates(mapped_candidates, "drink"))
        post_mapping_drink_samples = _drink_debug_samples(mapped_candidates)

        def _apply_side_supplement(max_total: int) -> int:
            nonlocal side_supplement_used, side_supplement_added, side_supplement_safety_added, side_supplement_regex, local_candidates
            side_supplement_regex = self._side_supplement_regex(meal_type)
            side_candidates: list[dict[str, Any]] = []
            if normalize_text(meal_type) in {"breakfast", "lunch"}:
                safety_candidates = self._meal_side_safety_candidates(meal_type, profile.get("top_foods", []))
            else:
                safety_candidates = self._dinner_side_safety_candidates(meal_type)
            if safety_candidates:
                safety_candidates = [
                    candidate
                    for candidate in safety_candidates
                    if self._recipe_id_from_candidate(candidate) not in seen_recipe_ids
                ]
                safety_take = max(0, min(len(safety_candidates), max_total - len(mapped_candidates)))
                if safety_take:
                    if normalize_text(meal_type) == "breakfast":
                        prioritized_safety = self._select_diverse_breakfast_sides(safety_candidates, safety_take)
                    else:
                        prioritized_safety = self._select_diverse_dinner_sides(safety_candidates, safety_take, meal_type)
                    side_candidates = merge_candidates(side_candidates, prioritized_safety, max_items=safety_take)
                    side_supplement_safety_added += len(prioritized_safety)

            remaining_deficit = max(0, max_total - len(mapped_candidates) - len(side_candidates))
            supplement_candidates: list[dict[str, Any]] = []
            if remaining_deficit > 0 and side_supplement_regex:
                supplement_candidates = self.local_dataset.search(
                    meal_type=meal_type,
                    query_vector=side_query_vec,
                    top_k=max(3, remaining_deficit * 2),
                    prefetch=max(6, remaining_deficit * 3),
                    exclude_recipe_ids=set(seen_recipe_ids),
                    is_australian_user=is_australian_user,
                    text_query=side_supplement_regex,
                    role_hint="side",
                    dedicated_connection=use_dedicated_search_connection,
                )
                supplement_candidates = self._filter_side_candidates_for_meal(meal_type, supplement_candidates)
                if supplement_candidates:
                    if normalize_text(meal_type) == "breakfast":
                        supplement_candidates = self._select_diverse_breakfast_sides(
                            supplement_candidates,
                            max(remaining_deficit, min(len(supplement_candidates), remaining_deficit + 1)),
                        )
                    else:
                        supplement_candidates = self._select_diverse_dinner_sides(
                            supplement_candidates,
                            max(remaining_deficit, min(len(supplement_candidates), remaining_deficit + 1)),
                            meal_type,
                        )
                    side_candidates = merge_candidates(
                        side_candidates,
                        supplement_candidates,
                        max_items=max(remaining_deficit + len(side_candidates), len(side_candidates)),
                    )

            if not side_candidates:
                return 0

            side_supplement_used = True
            local_candidates = merge_candidates(
                local_candidates,
                side_candidates,
                max_items=max(candidate_pool_target + 12, candidate_pool_target * 2),
            )
            added = _append_mapped_candidates(side_candidates, max_total)
            side_supplement_added += added
            return added

        mapped_count = sum(1 for item in mapped_candidates if normalize_text(item.get("ml_tag")) != "local_only")
        local_only_count = len(mapped_candidates) - mapped_count
        print(
            f"**** Slot={meal_type} mapping summary: local={len(local_candidates)} "
            f"resolved={len(mapped_candidates)} mapped={mapped_count} local_only={local_only_count} "
            f"new_mappings_used={slot_lookup_state['slot_new_mappings_used']} "
            f"sync_lookups_used={slot_lookup_state['slot_sync_lookups_used']} "
            f"remaining_new_mappings={slot_lookup_state['remaining_new_mappings']} "
            f"remaining_sync_lookups={slot_lookup_state['remaining_sync_lookup_attempts']}"
        )
        if drink_supplement_used:
            print(
                f"**** Slot={meal_type} drink supplement: before={drink_compatible_before} "
                f"after={drink_compatible_after} added={drink_supplement_added} regex={drink_supplement_regex}"
            )

        side_compatible_after = len(self._role_compatible_candidates(mapped_candidates, "side"))

        mapped_candidates, removed_side_titles = self._filter_mapped_candidates_for_meal(meal_type, mapped_candidates)
        if removed_side_titles:
            print(
                f"**** Slot={meal_type} mapped-side cleanup: removed={len(removed_side_titles)} "
                f"samples={removed_side_titles[:6]}"
            )

        post_cleanup_side_candidates = self._role_compatible_candidates(mapped_candidates, "side")
        strong_side_after_cleanup = len(
            [
                candidate
                for candidate in post_cleanup_side_candidates
                if self._role_quality_multiplier(candidate, "side", meal_type) >= (0.8 if normalize_text(meal_type) == "dinner" else 0.65)
            ]
        )
        breakfast_side_diversity_support = (
            normalize_text(meal_type) == "breakfast"
            and self._needs_breakfast_side_diversity_support(post_cleanup_side_candidates)
        )
        if strong_side_after_cleanup < minimum_side_pool or breakfast_side_diversity_support:
            side_support_gap = max(0, minimum_side_pool - strong_side_after_cleanup)
            if breakfast_side_diversity_support:
                side_support_gap = max(side_support_gap, 4)
            side_target_total = len(mapped_candidates) + side_support_gap
            _apply_side_supplement(side_target_total)
            mapped_candidates, extra_removed_side_titles = self._filter_mapped_candidates_for_meal(meal_type, mapped_candidates)
            if extra_removed_side_titles:
                removed_side_titles.extend(extra_removed_side_titles)
                print(
                    f"**** Slot={meal_type} mapped-side cleanup: removed={len(extra_removed_side_titles)} "
                    f"samples={extra_removed_side_titles[:6]}"
                )

        mapping_ms = _elapsed_ms(mapping_started_at)
        side_compatible_after = len(self._role_compatible_candidates(mapped_candidates, "side"))
        strong_side_after = len(
            [
                candidate
                for candidate in self._role_compatible_candidates(mapped_candidates, "side")
                if self._role_quality_multiplier(candidate, "side", meal_type) >= (0.8 if normalize_text(meal_type) == "dinner" else 0.65)
            ]
        )
        if side_supplement_used:
            print(
                f"**** Slot={meal_type} side supplement: before={side_compatible_before} strong_before={strong_side_before} "
                f"post_cleanup={strong_side_after_cleanup} after={side_compatible_after} strong_after={strong_side_after} "
                f"added={side_supplement_added} safety_added={side_supplement_safety_added} "
                f"diversity_support={breakfast_side_diversity_support} regex={side_supplement_regex}"
            )

        retrieval_metrics = compute_candidate_pool_metrics(mapped_candidates or local_candidates, top_n=20)
        tuning = self._compute_adaptive_tuning(retrieval_metrics, experiment_config)

        slot_seed = (
            int(time.time() * 1000)
            ^ len(mapped_candidates)
            ^ len(local_candidates)
            ^ sum(ord(ch) for ch in meal_type)
        ) & 0xFFFFFFFF
        slot_rng = np.random.default_rng(slot_seed)
        skipped_titles = feedback_context.get("skipped_titles", set()) if feedback_context else set()
        loved_titles = feedback_context.get("loved_titles", set()) if feedback_context else set()
        favorite_titles = feedback_context.get("favorite_titles", set()) if feedback_context else set()
        title_bias = feedback_context.get("title_bias", {}) if feedback_context else {}

        ranking_started_at = time.perf_counter()
        ranked = rank_candidates(
            candidates=mapped_candidates,
            user_vec=profile.get("user_vec"),
            archetype_vectors=profile.get("archetype_vectors"),
            preference_tokens=profile.get("preference_tokens", set()),
            preference_token_weights=profile.get("preference_token_weights", {}),
            recent_titles=profile.get("recent_titles", set()),
            top_foods=profile.get("top_foods", []),
            top_food_counts=profile.get("top_food_counts", {}),
            meal_type=meal_type,
            meal_target=slot_target,
            top_n=80,
            serving_fit_tolerance=self.serving_fit_tolerance,
            stochastic_strength=self.stochastic_ranking_strength,
            rng=slot_rng,
            skipped_titles=skipped_titles,
            loved_titles=loved_titles,
            favorite_titles=favorite_titles,
            ranking_weights=tuning.get("ranking_weights"),
            mmr_lambda=tuning.get("mmr_lambda"),
            title_bias=title_bias,
        )

        def _ensure_breakfast_ranked_side_family_coverage(current_ranked: list[dict[str, Any]]) -> list[dict[str, Any]]:
            if normalize_text(meal_type) != "breakfast":
                return current_ranked

            desired_families = ("fruit", "nuts", "bread", "grain")
            ranked_side_families = {
                self._breakfast_side_family(candidate)
                for candidate in current_ranked
                if self._infer_combo_category(candidate) == "side"
                and self._is_role_compatible(candidate, "side")
                and self._role_quality_multiplier(candidate, "side", meal_type) >= 0.84
            }
            missing_families = [family for family in desired_families if family not in ranked_side_families]
            rescue_families = missing_families
            if len(ranked_side_families.intersection({"fruit", "nuts", "bread", "grain"})) < len(desired_families):
                rescue_families = list(dict.fromkeys(["bread", *missing_families]))
            if not rescue_families:
                return current_ranked

            seen_side_keys = {
                self._role_title_key(candidate) or str(candidate.get("recipe_id") or candidate.get("id"))
                for candidate in current_ranked
                if self._infer_combo_category(candidate) == "side"
            }
            rescued_candidates: list[dict[str, Any]] = []

            for family in rescue_families:
                family_candidates = [
                    candidate
                    for candidate in mapped_candidates
                    if self._breakfast_side_family(candidate) == family
                    and self._is_role_compatible(candidate, "side")
                    and self._role_quality_multiplier(candidate, "side", meal_type) >= 0.72
                    and (self._role_title_key(candidate) or str(candidate.get("recipe_id") or candidate.get("id"))) not in seen_side_keys
                ]
                if not family_candidates:
                    continue

                rescued_family_ranked = rank_candidates(
                    candidates=family_candidates,
                    user_vec=profile.get("user_vec"),
                    archetype_vectors=profile.get("archetype_vectors"),
                    preference_tokens=profile.get("preference_tokens", set()),
                    preference_token_weights=profile.get("preference_token_weights", {}),
                    recent_titles=profile.get("recent_titles", set()),
                    top_foods=profile.get("top_foods", []),
                    top_food_counts=profile.get("top_food_counts", {}),
                    meal_type=meal_type,
                    meal_target=slot_target,
                    top_n=max(1, len(family_candidates)),
                    serving_fit_tolerance=self.serving_fit_tolerance,
                    stochastic_strength=0.0,
                    rng=np.random.default_rng(slot_seed ^ (abs(hash(family)) & 0xFFFFFFFF)),
                    skipped_titles=skipped_titles,
                    loved_titles=loved_titles,
                    favorite_titles=favorite_titles,
                    ranking_weights=tuning.get("ranking_weights"),
                    mmr_lambda=1.0,
                    title_bias=title_bias,
                )
                if not rescued_family_ranked:
                    continue

                rescued_candidate = max(
                    rescued_family_ranked,
                    key=lambda candidate: to_float(candidate.get("score"), 0.0),
                )
                rescued_candidates.append(rescued_candidate)
                seen_side_keys.add(
                    self._role_title_key(rescued_candidate)
                    or str(rescued_candidate.get("recipe_id") or rescued_candidate.get("id"))
                )

            if not rescued_candidates:
                return current_ranked

            print(
                f"**** Slot={meal_type} breakfast side family rescue: "
                f"missing={missing_families} rescue_families={rescue_families} "
                f"added={[str(candidate.get('title') or '') for candidate in rescued_candidates]}"
            )
            return merge_candidates(
                current_ranked,
                rescued_candidates,
                max_items=max(len(current_ranked) + len(rescued_candidates), len(current_ranked)),
            )

        ranked = _ensure_breakfast_ranked_side_family_coverage(ranked)

        ranked_drink_compatible = len(self._role_compatible_candidates(ranked, "drink"))
        if ranked_drink_compatible < desired_ranked_drink_pool and not drink_supplement_used:
            added = _apply_drink_supplement(candidate_pool_target + 24)
            if added > 0:
                ranked = rank_candidates(
                    candidates=mapped_candidates,
                    user_vec=profile.get("user_vec"),
                    archetype_vectors=profile.get("archetype_vectors"),
                    preference_tokens=profile.get("preference_tokens", set()),
                    preference_token_weights=profile.get("preference_token_weights", {}),
                    recent_titles=profile.get("recent_titles", set()),
                    top_foods=profile.get("top_foods", []),
                    top_food_counts=profile.get("top_food_counts", {}),
                    meal_type=meal_type,
                    meal_target=slot_target,
                    top_n=80,
                    serving_fit_tolerance=self.serving_fit_tolerance,
                    stochastic_strength=self.stochastic_ranking_strength,
                    rng=slot_rng,
                    skipped_titles=skipped_titles,
                    loved_titles=loved_titles,
                    favorite_titles=favorite_titles,
                    ranking_weights=tuning.get("ranking_weights"),
                    mmr_lambda=tuning.get("mmr_lambda"),
                    title_bias=title_bias,
                )
                ranked = _ensure_breakfast_ranked_side_family_coverage(ranked)
                ranked_drink_compatible = len(self._role_compatible_candidates(ranked, "drink"))
                if drink_supplement_source == "none":
                    drink_supplement_source = "post_rank"

        if ranked_drink_compatible < desired_ranked_drink_pool:
            rescue_drinks = self._filter_drink_candidates_for_meal(
                meal_type,
                sorted(
                    self._role_compatible_candidates(mapped_candidates, "drink"),
                    key=lambda candidate: (
                        self._role_quality_multiplier(candidate, "drink", meal_type),
                        to_float(candidate.get("score"), 0.0),
                    ),
                    reverse=True,
                ),
            )
            if rescue_drinks:
                ranked = merge_candidates(
                    ranked,
                    rescue_drinks,
                    max_items=max(len(ranked) + desired_ranked_drink_pool, 96),
                )
                ranked_drink_compatible = len(self._role_compatible_candidates(ranked, "drink"))

        drink_compatible_after = ranked_drink_compatible
        ranked_drink_samples = _drink_debug_samples(ranked)
        ranking_ms = _elapsed_ms(ranking_started_at)
        print(
            f"**** Slot={meal_type} serving-fit filter: kept={len(ranked)} "
            f"tolerance={self.serving_fit_tolerance:.2f} target_kcal={slot_target}"
        )
        print(
            f"**** Slot={meal_type} stochastic ranking: seed={slot_seed} "
            f"strength={self.stochastic_ranking_strength:.4f}"
        )

        selected_candidates = self._select_final_slot_candidates(ranked, meal_type, RECOMMENDED_ITEMS_PER_MEAL)
        selected_candidates, ranked = self._resolve_visible_slot_candidates_with_mapping(
            selected_candidates,
            ranked,
            meal_type,
            slot_lookup_state,
        )
        async_queued = 0
        # Avoid feeding the async mapper twice on the same request.
        # When stale-while-revalidate mapping is active for this slot, visible local-only
        # candidates are already queued during request-time resolution misses.
        # Keep the broader post-slot fallback prefetch only for configurations that disable
        # request-time stale mapping queues.
        if slot_new_mapping_cap <= 0:
            async_queued = self._enqueue_background_mapping_candidates(
                meal_type=meal_type,
                prioritized_candidates=selected_candidates,
                fallback_candidates=[*ranked, *local_candidates],
            )
        most_consumed_slot = self._enrich_consumed_images(
            get_top_consumed_items(history_df, meal_type=meal_type, top_n=DEFAULT_SLOT_CONSUMED_LIMIT),
            max_lookups=self.slot_consumed_image_lookups,
        )
        health_insight = self._build_health_insight(selected_candidates, is_australian_user)
        behavioral_insight = build_behavioral_insight(
            meal_type=meal_type,
            slot_weights=slot_weights,
            top_foods=[str(item.get("title") or "") for item in most_consumed_slot],
            health_hint=health_insight,
        )

        if normalize_text(meal_type) == "breakfast":
            breakfast_serialized_combo_mains = [
                {
                    **self._to_recommended_item(candidate, meal_type, slot_target, behavioral_insight),
                    "category": "main",
                }
                for candidate in selected_candidates
                if self._infer_combo_category(candidate) == "main"
            ]
            breakfast_combo_main_rescues = [
                rescued
                for rescued in self._resolve_visible_recommended_items_with_mapping(
                    breakfast_serialized_combo_mains,
                    meal_type,
                    slot_target,
                    behavioral_insight,
                    slot_lookup_state,
                )
                if normalize_text(rescued.get("ml_tag") or "") != "local_only"
                and self._is_role_compatible(rescued, "main")
            ]

            if breakfast_combo_main_rescues:
                ranked_recipe_ids = {
                    self._recipe_id_from_candidate(candidate)
                    for candidate in ranked
                    if self._recipe_id_from_candidate(candidate)
                }
                rescue_by_recipe_id = {
                    self._recipe_id_from_candidate(candidate): candidate
                    for candidate in breakfast_combo_main_rescues
                    if self._recipe_id_from_candidate(candidate)
                }
                if rescue_by_recipe_id:
                    ranked = [
                        rescue_by_recipe_id.get(self._recipe_id_from_candidate(candidate), candidate)
                        for candidate in ranked
                    ]
                unmatched_breakfast_combo_main_rescues = [
                    candidate
                    for candidate in breakfast_combo_main_rescues
                    if self._recipe_id_from_candidate(candidate) not in ranked_recipe_ids
                ]
                if unmatched_breakfast_combo_main_rescues:
                    ranked = merge_candidates(
                        ranked,
                        unmatched_breakfast_combo_main_rescues,
                        max_items=max(
                            len(ranked) + len(unmatched_breakfast_combo_main_rescues),
                            len(ranked),
                        ),
                    )
                print(
                    f"**** Slot={meal_type} breakfast combo-main rescue: "
                    f"replaced={sum(1 for recipe_id in rescue_by_recipe_id if recipe_id in ranked_recipe_ids)} "
                    f"added={len(unmatched_breakfast_combo_main_rescues)} "
                    f"titles={[str(candidate.get('title') or '') for candidate in breakfast_combo_main_rescues]}"
                )

        combo_started_at = time.perf_counter()
        combos, combo_timing_breakdown = self._build_combo_payloads(
            meal_type=meal_type,
            ranked_candidates=ranked,
            slot_target=slot_target,
            behavioral_insight=behavioral_insight,
            top_food_counts=profile.get("top_food_counts", {}),
            combo_reuse_penalty_base=to_float(tuning.get("combo_reuse_penalty_base"), COMBO_REUSE_PENALTY_BASE),
            max_combos=5,
        )
        combos = self._resolve_visible_combos_with_mapping(
            combos,
            meal_type,
            slot_target,
            behavioral_insight,
            slot_lookup_state,
        )
        combo_assembly_ms = _elapsed_ms(combo_started_at)

        recommended_items = [
            self._to_recommended_item(candidate, meal_type, slot_target, behavioral_insight)
            for candidate in selected_candidates
        ]
        recommended_items = self._resolve_visible_recommended_items_with_mapping(
            recommended_items,
            meal_type,
            slot_target,
            behavioral_insight,
            slot_lookup_state,
        )

        metrics_started_at = time.perf_counter()
        proxy_metrics = compute_proxy_title_metrics(
            ranked_candidates=ranked,
            recommended_items=recommended_items,
            preferred_titles=[str(item.get("title") or "") for item in most_consumed_slot],
            candidate_cap=40,
        )
        offline_diagnostics = compute_offline_diagnostic_metrics(
            ranked_candidates=ranked,
            recommended_items=recommended_items,
            slot_target=slot_target,
            user_vec=profile.get("user_vec"),
        )
        combo_diagnostics = compute_combo_diagnostic_metrics(combos=combos, slot_target=slot_target)
        mapping_diagnostics = compute_mapping_diagnostics(recommended_items=recommended_items, combos=combos)
        diversity_dashboard = compute_diversity_dashboard(recommended_items, combos)
        metrics_ms = _elapsed_ms(metrics_started_at)
        timing = {
            "slot_total_ms": _elapsed_ms(slot_started_at),
            "profile_ms": profile_ms,
            "retrieval_ms": retrieval_ms,
            "mapping_ms": mapping_ms,
            "ranking_ms": ranking_ms,
            "combo_assembly_ms": combo_assembly_ms,
            "combo_pool_build_ms": round(to_float(combo_timing_breakdown.get("combo_pool_build_ms"), 0.0), 1),
            "combo_candidate_generation_ms": round(to_float(combo_timing_breakdown.get("combo_candidate_generation_ms"), 0.0), 1),
            "combo_diversity_ms": round(to_float(combo_timing_breakdown.get("combo_diversity_ms"), 0.0), 1),
            "metrics_ms": metrics_ms,
        }
        metrics = build_model_metrics(
            proxy_metrics=proxy_metrics,
            offline_diagnostics=offline_diagnostics,
            combo_diagnostics=combo_diagnostics,
            mapping_diagnostics=mapping_diagnostics,
            diversity_dashboard=diversity_dashboard,
            retrieval_metrics=retrieval_metrics,
            timing=timing,
            tuning=tuning,
            experimentation={
                "variant": experiment_config.get("name", "control"),
                "query_expansion_used": bool(query_expansion_used),
                "query_expansion_regex": expansion_regex,
                "side_expansion_used": bool(side_expansion_info.get("used")),
                "side_expansion_queries": side_expansion_info.get("queries") or [],
                "side_expansion_added": int(side_expansion_info.get("added", 0)),
                "side_expansion_candidate_count": int(side_expansion_info.get("candidate_count", 0)),
                "drink_expansion_used": bool(drink_expansion_info.get("used")),
                "drink_expansion_query": drink_expansion_info.get("query") or "",
                "drink_expansion_added": int(drink_expansion_info.get("added", 0)),
                "drink_expansion_candidate_count": int(drink_expansion_info.get("candidate_count", 0)),
                "drink_safety_fill_used": bool(dinner_drink_safety_fill_added > 0),
                "drink_safety_fill_added": int(dinner_drink_safety_fill_added),
                "candidate_pool_multiplier": candidate_pool_multiplier,
                "primary_role_parallel_retrieval_used": bool(primary_retrieval_info.get("parallel_used")),
                "primary_role_dedicated_connection": bool(primary_retrieval_info.get("dedicated_connection")),
                "primary_main_top_k": int(primary_search_budgets["main_top_k"]),
                "primary_side_top_k": int(primary_search_budgets["side_top_k"]),
                "primary_drink_top_k": int(primary_search_budgets["drink_top_k"]),
                "primary_main_prefetch": int(primary_search_budgets["main_prefetch"]),
                "primary_side_prefetch": int(primary_search_budgets["side_prefetch"]),
                "primary_drink_prefetch": int(primary_search_budgets["drink_prefetch"]),
                "primary_local_main_candidates": int(len(main_candidates)),
                "primary_local_side_candidates": int(len(side_candidates)),
                "primary_local_drink_candidates": int(primary_drink_candidate_count),
                "primary_merged_main_candidates": int(merged_role_counts["main"]),
                "primary_merged_side_candidates": int(merged_role_counts["side"]),
                "primary_merged_drink_candidates": int(merged_role_counts["drink"]),
                "minimum_drink_pool": int(minimum_drink_pool),
                "drink_supplement_attempted": bool(drink_supplement_attempted),
                "drink_supplement_used": bool(drink_supplement_used),
                "drink_supplement_source": drink_supplement_source,
                "drink_supplement_regex": drink_supplement_regex,
                "drink_supplement_added": int(drink_supplement_added),
                "drink_supplement_candidate_count": int(drink_supplement_candidate_count),
                "drink_supplement_safety_added": int(drink_supplement_safety_added),
                "drink_compatible_before": int(drink_compatible_before),
                "drink_compatible_after": int(drink_compatible_after),
                "mapped_drink_samples": mapped_drink_samples,
                "post_mapping_drink_samples": post_mapping_drink_samples,
                "ranked_drink_samples": ranked_drink_samples,
                "archetype_strategy": profile.get("archetype_strategy"),
                "archetype_cluster_count": profile.get("archetype_cluster_count", 0),
            },
        )
        print_metrics(meal_type, metrics)

        return {
            "slot": meal_type,
            "slot_target": slot_target,
            "recommended_items": recommended_items,
            "combos": combos,
            "behavioral_insight": behavioral_insight,
            "slot_weights": {slot: round(weight, 4) for slot, weight in slot_weights.items()},
            "daily_calorie_target": int(round(daily_calories)),
            "most_consumed_items": most_consumed_slot,
            "model_metrics": metrics,
            "background_mapping_queued": int(async_queued),
            "mapping_usage": {
                "slot_new_mapping_cap": int(slot_lookup_state.get("slot_new_mapping_cap", 0)),
                "slot_new_mappings_used": int(slot_lookup_state.get("slot_new_mappings_used", 0)),
                "slot_lookup_attempt_cap": int(lookup_state.get("remaining_lookup_attempts", 0)),
                "slot_lookup_attempts_used": int(
                    max(0, int(lookup_state.get("remaining_lookup_attempts", 0)) - int(slot_lookup_state.get("remaining_lookup_attempts", 0)))
                ),
                "slot_sync_lookup_cap": int(slot_lookup_state.get("slot_sync_lookup_cap", 0)),
                "slot_sync_lookups_used": int(slot_lookup_state.get("slot_sync_lookups_used", 0)),
            },
        }

    def recommend(self, data: dict[str, Any]) -> dict[str, Any]:
        data = data or {}
        demographics = data.get("demographics", {}) or {}
        goal = self._normalize_goal(demographics.get("goal") or "maintain")
        meal_type_req = self._normalize_meal_type(data.get("mealType") or data.get("slot") or "all")
        force_exploration = parse_force_exploration(data.get("force_exploration")) or parse_force_exploration(
            data.get("forceExploration")
        )
        is_australian_user = self._is_australian_user(demographics)
        user_id = data.get("userId")
        experiment_config = self._resolve_experiment_config(data, user_id)

        feedback_payload = data.get("feedback") or {}
        skipped_titles = self._normalize_title_set(feedback_payload.get("skipped_titles"))
        loved_titles = {
            title for title in self._normalize_title_set(feedback_payload.get("loved_titles"))
            if self._is_safe_positive_feedback_title(title)
        }
        favorite_titles = {
            title for title in self._normalize_title_set(data.get("favorite_titles"))
            if self._is_safe_positive_feedback_title(title)
        }
        title_bias = {
            canonical_title_key(key): float(value)
            for key, value in ((feedback_payload.get("title_bias") or {}) if isinstance(feedback_payload.get("title_bias"), dict) else {}).items()
            if canonical_title_key(key) and (float(value) <= 0.0 or self._is_safe_positive_feedback_title(key))
        }
        feedback_context = {
            "skipped_titles": skipped_titles,
            "loved_titles": loved_titles,
            "favorite_titles": favorite_titles,
            "title_bias": title_bias,
        }

        cache_key = self._build_cache_key(data, goal)
        if not force_exploration:
            while True:
                cached = self._response_cache_get(cache_key)
                if cached is not None:
                    return cached
                should_build, build_event = self._claim_response_build(cache_key)
                if should_build:
                    break
                build_event.wait()
        else:
            build_event = None

        try:
            selected_meals = list(MEAL_SLOTS) if meal_type_req == "all" else [meal_type_req]
            self._ensure_async_workers_started()
            cached_history = self._history_cache_get(user_id)
            cached_goal = self._goal_cache_get(user_id)
            if cached_history is None and cached_goal is None and user_id:
                cached_history, cached_goal = fetch_user_history_and_goal(user_id)
                self._history_cache_set(user_id, cached_history)
                self._goal_cache_set(user_id, cached_goal)
            elif cached_history is None:
                cached_history = fetch_user_meal_history(user_id)
                self._history_cache_set(user_id, cached_history)
            history_df = normalize_history_df(cached_history)

            slot_weights = self._slot_weight_map(compute_dynamic_meal_allocation(history_df))
            if cached_goal is None:
                cached_goal = fetch_active_daily_goal(user_id)
                self._goal_cache_set(user_id, cached_goal)

            daily_calories = resolve_daily_calories(
                calorie_override=data.get("calorieTarget"),
                active_goal_calories=cached_goal,
                demographics={**demographics, "goal": goal},
                goal=goal,
            )

            overall_consumed = self._enrich_consumed_images(
                get_top_consumed_items(history_df, meal_type=None, top_n=DEFAULT_TOP_CONSUMED_LIMIT),
                max_lookups=self.overall_consumed_image_lookups,
            )
            if not overall_consumed:
                overall_consumed = self._enrich_consumed_images(
                    self._load_consumption_snapshot(),
                    max_lookups=self.overall_consumed_image_lookups,
                )

            payload_by_slot: dict[str, Any] = {}
            metrics_by_slot: dict[str, Any] = {}
            consumed_by_slot: dict[str, Any] = {}

            slot_lookup_plans = self._build_slot_lookup_plans(selected_meals)
            run_slots_in_parallel = (
                self.parallel_slot_execution_enabled
                and meal_type_req == "all"
                and len(selected_meals) > 1
            )

            def _build_slot(meal_type: str) -> dict[str, Any]:
                slot_lookup_plan = slot_lookup_plans.get(meal_type) or {}
                return self._build_slot_payload(
                    meal_type=meal_type,
                    slot_weights=slot_weights,
                    daily_calories=daily_calories,
                    demographics={**demographics, "goal": goal},
                    history_df=history_df,
                    user_id=user_id,
                    lookup_state={
                        "remaining_new_mappings": int(slot_lookup_plan.get("remaining_new_mappings", 0)),
                        "remaining_lookup_attempts": int(slot_lookup_plan.get("remaining_lookup_attempts", 0)),
                        "remaining_sync_lookup_attempts": int(slot_lookup_plan.get("remaining_sync_lookup_attempts", 0)),
                        "visible_sync_reserve_cap": int(slot_lookup_plan.get("visible_sync_reserve_cap", 4)),
                    },
                    slot_new_mapping_cap=int(slot_lookup_plan.get("slot_new_mapping_cap", 0)),
                    slot_sync_lookup_cap=int(slot_lookup_plan.get("slot_sync_lookup_cap", 0)),
                    is_australian_user=is_australian_user,
                    experiment_config=experiment_config,
                    feedback_context=feedback_context,
                    use_dedicated_search_connection=run_slots_in_parallel,
                )

            print(
                f"**** Recommendation Slot Execution: mode={'parallel' if run_slots_in_parallel else 'sequential'} "
                f"slots={len(selected_meals)} "
                f"inner_parallel_roles={int(self.parallel_primary_role_retrieval_enabled)}"
            )

            if run_slots_in_parallel:
                futures_by_slot = {
                    meal_type: self._slot_build_executor.submit(_build_slot, meal_type)
                    for meal_type in selected_meals
                }
                for meal_type in selected_meals:
                    payload = futures_by_slot[meal_type].result()
                    payload_by_slot[meal_type] = payload
                    metrics_by_slot[meal_type] = payload.get("model_metrics", {})
                    consumed_by_slot[meal_type] = payload.get("most_consumed_items", [])
            else:
                for meal_type in selected_meals:
                    payload = _build_slot(meal_type)
                    payload_by_slot[meal_type] = payload
                    metrics_by_slot[meal_type] = payload.get("model_metrics", {})
                    consumed_by_slot[meal_type] = payload.get("most_consumed_items", [])

            lookup_state = self._summarize_lookup_state(payload_by_slot, selected_meals)

            self._archive_runtime_slot_comparisons(
                user_id=user_id,
                requested_meal_type=meal_type_req,
                payload_by_slot=payload_by_slot,
                experiment_name=str(experiment_config.get("name", "control")),
                force_exploration=bool(force_exploration),
            )

            aggregated_metrics = self._aggregate_model_metrics(metrics_by_slot, str(experiment_config.get("name", "control")))

            if meal_type_req == "all":
                total_options = sum(
                    len((payload_by_slot.get(meal_type) or {}).get("recommended_items", [])) for meal_type in MEAL_SLOTS
                )
                if total_options < MIN_FRONTEND_OPTIONS:
                    print(
                        f"**** Frontend Options Warning: total_options={total_options} < {MIN_FRONTEND_OPTIONS}. "
                        "Increase mapping coverage in food_mappings.json for faster and denser recommendations."
                    )

                response = {
                    "daily_calorie_target": int(round(daily_calories)),
                    "slot_weights": {slot: round(weight, 4) for slot, weight in slot_weights.items()},
                    "recommendationsByMeal": payload_by_slot,
                    "most_consumed_items": overall_consumed,
                    "most_consumed_by_meal": consumed_by_slot,
                    "model_metrics": aggregated_metrics,
                    "experiment_variant": experiment_config.get("name", "control"),
                    "mapping_lookup_summary": {
                        "remaining_new_mappings": lookup_state.get("remaining_new_mappings", 0),
                        "remaining_lookup_attempts": lookup_state.get("remaining_lookup_attempts", 0),
                        "remaining_sync_lookup_attempts": lookup_state.get("remaining_sync_lookup_attempts", 0),
                        "max_new_mappings": self.max_new_mapping_lookups,
                        "max_lookup_attempts": self.max_mapping_lookup_attempts,
                        "sync_lookups_per_slot": self.sync_mapping_lookups_per_slot,
                        "async_queue": self._mapping_queue_status(),
                    },
                }
                total_options = sum(
                    len((payload_by_slot.get(slot) or {}).get("recommended_items", []))
                    for slot in selected_meals
                )
                print(
                    f"**** Recommendation Ready: meal_type=all slots={len(selected_meals)} "
                    f"total_items={total_options} queue={self._mapping_queue_status()}"
                )
                return self._response_cache_set(cache_key, response)

            slot_payload = payload_by_slot[meal_type_req]
            response = {
                **slot_payload,
                "most_consumed_items": overall_consumed,
                "most_consumed_by_meal": consumed_by_slot,
                "model_metrics": slot_payload.get("model_metrics", {}),
                "experiment_variant": experiment_config.get("name", "control"),
                "mapping_lookup_summary": {
                    "remaining_new_mappings": lookup_state.get("remaining_new_mappings", 0),
                    "remaining_lookup_attempts": lookup_state.get("remaining_lookup_attempts", 0),
                    "remaining_sync_lookup_attempts": lookup_state.get("remaining_sync_lookup_attempts", 0),
                    "max_new_mappings": self.max_new_mapping_lookups,
                    "max_lookup_attempts": self.max_mapping_lookup_attempts,
                    "sync_lookups_per_slot": self.sync_mapping_lookups_per_slot,
                    "async_queue": self._mapping_queue_status(),
                },
            }
            slot_items = len((slot_payload or {}).get("recommended_items", []))
            print(
                f"**** Recommendation Ready: meal_type={meal_type_req} slots=1 "
                f"total_items={slot_items} queue={self._mapping_queue_status()}"
            )
            return self._response_cache_set(cache_key, response)
        finally:
            if not force_exploration and build_event is not None:
                self._release_response_build(cache_key, build_event)
