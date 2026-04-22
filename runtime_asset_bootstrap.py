from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from pathlib import Path
from urllib.request import urlopen


def _off_db_path() -> Path:
    configured_path = os.getenv("LOCAL_FOOD_DB_PATH", "").strip()
    if configured_path:
        return Path(configured_path)
    return Path(__file__).resolve().parent / "dataset_process" / "off.db"


def _sha256_for_file(file_path: Path) -> str:
    hasher = hashlib.sha256()
    with file_path.open("rb") as source_handle:
        while True:
            chunk = source_handle.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def ensure_runtime_assets() -> None:
    target_path = _off_db_path()
    if target_path.exists():
        print(f"**** Runtime Assets: using local off.db at {target_path}")
        return

    download_url = os.getenv("OFF_DB_DOWNLOAD_URL", "").strip()
    if not download_url:
        print(
            "**** Runtime Assets: off.db is missing and OFF_DB_DOWNLOAD_URL is not set. "
            f"Expected file at {target_path}."
        )
        return

    # NOTE: Download the runtime DuckDB file before the recommendation service boots.
    target_path.parent.mkdir(parents=True, exist_ok=True)
    expected_sha256 = os.getenv("OFF_DB_DOWNLOAD_SHA256", "").strip().lower()
    timeout_seconds = int(str(os.getenv("OFF_DB_DOWNLOAD_TIMEOUT_SECONDS", "600")).strip() or "600")

    with tempfile.NamedTemporaryFile(delete=False, dir=str(target_path.parent), suffix=".tmp") as tmp_handle:
        temp_path = Path(tmp_handle.name)

    try:
        print(f"**** Runtime Assets: downloading off.db from {download_url}")
        with urlopen(download_url, timeout=timeout_seconds) as response, temp_path.open("wb") as output_handle:
            shutil.copyfileobj(response, output_handle, length=1024 * 1024)

        if expected_sha256:
            actual_sha256 = _sha256_for_file(temp_path)
            if actual_sha256 != expected_sha256:
                raise RuntimeError(
                    "Downloaded off.db failed SHA256 validation. "
                    f"Expected {expected_sha256}, got {actual_sha256}."
                )

        temp_path.replace(target_path)
        print(f"**** Runtime Assets: off.db downloaded to {target_path}")
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)