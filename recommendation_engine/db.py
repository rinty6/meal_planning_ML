from __future__ import annotations


import os
from datetime import date
from threading import Lock
from typing import Any

import pandas as pd
from sqlalchemy import create_engine, text

from .constants import MAX_HISTORY_ROWS


_DB_ENGINE = None
_DB_ENGINE_URL = ""
_DB_ENGINE_LOCK = Lock()
_HISTORY_COLUMNS = [
        "meal_type",
        "food_name",
        "calories",
        "protein",
        "carbs",
        "fats",
        "image",
        "date",
        "created_at",
]
_HISTORY_QUERY = text(
        """
        SELECT meal_type, food_name, calories, protein, carbs, fats, image, date, created_at
        FROM meal_logs
        WHERE user_id = :uid
            AND meal_type IN ('breakfast', 'lunch', 'dinner')
        ORDER BY created_at DESC
        LIMIT :limit_rows
        """
)
_ACTIVE_GOAL_QUERY = text(
        """
        SELECT daily_calories
        FROM calorie_goals
        WHERE user_id = :uid
            AND start_date <= :today
            AND end_date >= :today
        ORDER BY created_at DESC
        LIMIT 1
        """
)


def get_db_engine():
    db_url = (os.getenv("DB_URL") or "").strip()
    if not db_url:
        raise RuntimeError("DB_URL is not configured.")

    global _DB_ENGINE
    global _DB_ENGINE_URL

    if _DB_ENGINE is not None and _DB_ENGINE_URL == db_url:
        return _DB_ENGINE

    with _DB_ENGINE_LOCK:
        if _DB_ENGINE is not None and _DB_ENGINE_URL == db_url:
            return _DB_ENGINE

        _DB_ENGINE = create_engine(db_url, pool_pre_ping=True, pool_recycle=1800)
        _DB_ENGINE_URL = db_url
        return _DB_ENGINE


def reset_db_engine_cache() -> None:
    global _DB_ENGINE
    global _DB_ENGINE_URL

    with _DB_ENGINE_LOCK:
        _DB_ENGINE = None
        _DB_ENGINE_URL = ""


def _empty_history_df() -> pd.DataFrame:
    return pd.DataFrame(columns=_HISTORY_COLUMNS)


def fetch_user_history_and_goal(user_id: Any, limit_rows: int = MAX_HISTORY_ROWS) -> tuple[pd.DataFrame, float | None]:
    if not user_id:
        return _empty_history_df(), None

    history = _empty_history_df()
    goal_value = None

    try:
        engine = get_db_engine()
        with engine.connect() as conn:
            try:
                history = pd.read_sql_query(
                    _HISTORY_QUERY,
                    conn,
                    params={"uid": user_id, "limit_rows": int(limit_rows)},
                )
            except Exception:
                history = _empty_history_df()

            try:
                payload = pd.read_sql_query(
                    _ACTIVE_GOAL_QUERY,
                    conn,
                    params={"uid": user_id, "today": date.today().isoformat()},
                )
                if not payload.empty:
                    value = payload.iloc[0].get("daily_calories")
                    goal_value = float(value) if value is not None else None
            except Exception:
                goal_value = None
    except Exception:
        return _empty_history_df(), None

    return history, goal_value


def fetch_user_meal_history(user_id: Any, limit_rows: int = MAX_HISTORY_ROWS) -> pd.DataFrame:
    if not user_id:
        return _empty_history_df()

    try:
        engine = get_db_engine()
        with engine.connect() as conn:
            return pd.read_sql_query(_HISTORY_QUERY, conn, params={"uid": user_id, "limit_rows": int(limit_rows)})
    except Exception:
        return _empty_history_df()


def fetch_active_daily_goal(user_id: Any) -> float | None:
    if not user_id:
        return None

    try:
        engine = get_db_engine()
        with engine.connect() as conn:
            payload = pd.read_sql_query(_ACTIVE_GOAL_QUERY, conn, params={"uid": user_id, "today": date.today().isoformat()})
        if payload.empty:
            return None
        value = payload.iloc[0].get("daily_calories")
        return float(value) if value is not None else None
    except Exception:
        return None

