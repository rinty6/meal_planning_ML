from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock, get_ident
from typing import Any

from .utils import canonical_title_key


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class FoodMappingStore:
    _PERSIST_RETRY_ATTEMPTS = 6
    _PERSIST_RETRY_DELAY_SECONDS = 0.05

    def __init__(self, mapping_file_path: str):
        self.mapping_file_path = Path(mapping_file_path)
        self._lock = Lock()
        self._payload: dict[str, Any] = {"version": 1, "updated_at": _utc_now_iso(), "mappings": {}}
        self._title_index: dict[str, set[str]] = {}
        self._load()

    @staticmethod
    def _mapping_title_keys(mapping: dict[str, Any]) -> set[str]:
        if not isinstance(mapping, dict):
            return set()

        keys = {
            canonical_title_key(mapping.get("dataset_title") or ""),
            canonical_title_key(mapping.get("mapped_title") or mapping.get("food_name") or ""),
        }
        keys.discard("")
        return keys

    def _rebuild_title_index_locked(self) -> None:
        title_index: dict[str, set[str]] = {}
        mappings = self._payload.get("mappings", {})
        if isinstance(mappings, dict):
            for recipe_id, mapping in mappings.items():
                recipe_key = str(recipe_id or "").strip()
                if not recipe_key or not isinstance(mapping, dict):
                    continue
                for title_key in self._mapping_title_keys(mapping):
                    title_index.setdefault(title_key, set()).add(recipe_key)
        self._title_index = title_index

    def _remove_mapping_from_title_index_locked(self, recipe_id: str, mapping: dict[str, Any]) -> None:
        for title_key in self._mapping_title_keys(mapping):
            recipe_ids = self._title_index.get(title_key)
            if not recipe_ids:
                continue
            recipe_ids.discard(recipe_id)
            if not recipe_ids:
                self._title_index.pop(title_key, None)

    def _index_mapping_locked(self, recipe_id: str, mapping: dict[str, Any]) -> None:
        for title_key in self._mapping_title_keys(mapping):
            self._title_index.setdefault(title_key, set()).add(recipe_id)

    def _load(self) -> None:
        with self._lock:
            if not self.mapping_file_path.exists():
                self._ensure_parent_dir()
                self._persist()
                print(f"**** Food Mapping: created {self.mapping_file_path}.")
                return

            try:
                raw = json.loads(self.mapping_file_path.read_text(encoding="utf-8"))
                if isinstance(raw, dict) and isinstance(raw.get("mappings"), dict):
                    self._payload = raw
                elif isinstance(raw, dict):
                    # Backward compatibility: plain mapping dict.
                    self._payload = {"version": 1, "updated_at": _utc_now_iso(), "mappings": raw}
                else:
                    raise ValueError("Invalid mapping payload format.")
            except Exception as exc:
                print(f"**** Food Mapping: failed to read {self.mapping_file_path} ({exc}). Using empty mapping.")
                self._payload = {"version": 1, "updated_at": _utc_now_iso(), "mappings": {}}

            self._rebuild_title_index_locked()

            print(
                "**** Food Mapping Loaded:",
                f"items={len(self._payload.get('mappings', {}))}",
                f"path={self.mapping_file_path}",
            )

    def _ensure_parent_dir(self) -> None:
        self.mapping_file_path.parent.mkdir(parents=True, exist_ok=True)

    def _temp_mapping_path(self) -> Path:
        return self.mapping_file_path.with_name(
            f"{self.mapping_file_path.stem}.{os.getpid()}.{get_ident()}.tmp"
        )

    def _persist(self) -> bool:
        self._ensure_parent_dir()
        serialized_payload = json.dumps(self._payload, ensure_ascii=False, indent=2)
        last_error: OSError | None = None

        for attempt in range(self._PERSIST_RETRY_ATTEMPTS):
            temp_path = self._temp_mapping_path()
            try:
                temp_path.write_text(serialized_payload, encoding="utf-8")
                temp_path.replace(self.mapping_file_path)
                return True
            except OSError as exc:
                last_error = exc
                should_retry = getattr(exc, "winerror", None) == 5 and attempt < (self._PERSIST_RETRY_ATTEMPTS - 1)
                if not should_retry:
                    break
                time.sleep(self._PERSIST_RETRY_DELAY_SECONDS * (attempt + 1))
            finally:
                try:
                    temp_path.unlink(missing_ok=True)
                except OSError:
                    pass

        if getattr(last_error, "winerror", None) == 5:
            print(
                "**** Food Mapping Persist Warning:",
                f"path={self.mapping_file_path}",
                f"error={last_error}",
                "keeping in-memory mapping only.",
            )
            return False
        if last_error is not None:
            raise last_error
        return False

    def get(self, recipe_id: Any) -> dict[str, Any] | None:
        key = str(recipe_id or "").strip()
        if not key:
            return None
        with self._lock:
            mappings = self._payload.get("mappings", {})
            value = mappings.get(key) if isinstance(mappings, dict) else None
            return value if isinstance(value, dict) else None

    def set(self, recipe_id: Any, mapping: dict[str, Any]) -> None:
        key = str(recipe_id or "").strip()
        if not key:
            return
        if not isinstance(mapping, dict):
            return

        with self._lock:
            mappings = self._payload.setdefault("mappings", {})
            if not isinstance(mappings, dict):
                mappings = {}
                self._payload["mappings"] = mappings
            previous_mapping = mappings.get(key) if isinstance(mappings.get(key), dict) else None
            if previous_mapping is not None:
                self._remove_mapping_from_title_index_locked(key, previous_mapping)
            mappings[key] = mapping
            self._index_mapping_locked(key, mapping)
            self._payload["updated_at"] = _utc_now_iso()
            self._persist()

    def delete(self, recipe_id: Any) -> None:
        key = str(recipe_id or "").strip()
        if not key:
            return

        with self._lock:
            mappings = self._payload.get("mappings", {})
            if not isinstance(mappings, dict) or key not in mappings:
                return
            mapping = mappings.pop(key, None)
            if isinstance(mapping, dict):
                self._remove_mapping_from_title_index_locked(key, mapping)
            self._payload["updated_at"] = _utc_now_iso()
            self._persist()

    def find_by_title_keys(self, title_keys: set[str] | list[str] | tuple[str, ...]) -> list[dict[str, Any]]:
        normalized_keys = {str(title_key or "").strip() for title_key in (title_keys or []) if str(title_key or "").strip()}
        if not normalized_keys:
            return []

        with self._lock:
            mappings = self._payload.get("mappings", {})
            if not isinstance(mappings, dict) or not mappings:
                return []

            ordered_ids: list[str] = []
            seen_ids: set[str] = set()
            for title_key in normalized_keys:
                for recipe_id in self._title_index.get(title_key, ()):  # pragma: no branch - tiny hot-path helper
                    if recipe_id in seen_ids:
                        continue
                    mapping = mappings.get(recipe_id)
                    if not isinstance(mapping, dict):
                        continue
                    seen_ids.add(recipe_id)
                    ordered_ids.append(recipe_id)

            return [mappings[recipe_id] for recipe_id in ordered_ids if isinstance(mappings.get(recipe_id), dict)]
