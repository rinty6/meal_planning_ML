# Meal Recommendation Service

This contains the Python recommendation service used by the meal planning backend. It serves recommendation generation, cache priming, and warmup status endpoints for personalized meal suggestions.

## Responsibilities

The service is responsible for:

- loading runtime food assets before the app boots
- building the recommendation engine from environment-based configuration
- serving recommendation responses for meal planning requests
- priming and reporting recommendation cache warmup state
- using local dataset artifacts, mapping files, and benchmark summaries to support ranking and filtering

## Runtime Surface

Current Flask routes:

- `POST /api/recommendation`
- `POST /recommend`
- `POST /api/prime`
- `POST /api/prime/status`

Important note for deployment:

- this service currently exposes only POST application routes
- if your hosting platform requires a GET health endpoint, you should either configure health checks accordingly or add a dedicated health route before relying on platform probing

## Project Layout

- `app.py`: Flask entrypoint and route definitions
- `recommendation_engine/`: ranking, mapping, database access, constants, and service logic
- `runtime_asset_bootstrap.py`: ensures `off.db` exists locally or downloads it when configured
- `dataset_process/`: dataset cleaning, audit, and benchmark utilities
- `tests/`: targeted regression checks and probes for recommendation behavior
- `Procfile`: production start command for Gunicorn-based hosting

## Python Stack

The checked-in requirements include:

- Flask
- Gunicorn
- pandas
- scikit-learn
- numpy
- python-dotenv
- psycopg2-binary
- SQLAlchemy
- DuckDB
- FAISS CPU
- pyarrow

## Environment Variables

Use `.env.example` as the source of truth for local and hosted configuration.

Core server variables:

- `PORT`: defaults to `5001`
- `HOST`: optional host binding, defaults to `0.0.0.0` in `app.py`
- `FLASK_DEBUG`: set to `1` only for local debugging
- `DB_URL`: PostgreSQL connection string used for user-context lookups and related runtime data

Runtime asset and dataset variables:

- `LOCAL_FOOD_DATASET_PATH`
- `LOCAL_FOOD_DB_PATH`
- `LOCAL_FOOD_DB_TABLE`
- `EATING_HISTORY_FILE`
- `FOOD_MAPPING_FILE`
- `AUSNUT_BENCHMARK_SUMMARY_PATH`
- `MOST_FOOD_CONSUMPTION_FILE` (optional)

Hosted `off.db` download variables:

- `OFF_DB_DOWNLOAD_URL`
- `OFF_DB_DOWNLOAD_SHA256` (optional integrity validation)
- `OFF_DB_DOWNLOAD_TIMEOUT_SECONDS` (optional)

External integration and tuning variables:

- `FATSECRET_CLIENT_ID`
- `FATSECRET_CLIENT_SECRET`
- `ML_RESPONSE_CACHE_SECONDS`
- `HISTORY_CACHE_SECONDS`
- `GOAL_CACHE_SECONDS`
- `PROFILE_CACHE_SECONDS`
- `PRIMARY_SEARCH_SHAPE_WARMUP_ENABLED`
- `PARALLEL_SLOT_EXECUTION_ENABLED`
- `PARALLEL_PRIMARY_ROLE_RETRIEVAL_ENABLED`

## Local Development

### Create A Virtual Environment

```bash
python -m venv .venv
```

### Activate And Install

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Configure Environment

Copy `.env.example` to `.env` and provide the required values.

### Run The Service

```bash
python app.py
```

## Production Run

The checked-in `Procfile` uses Gunicorn:

That is the expected production entrypoint for Railway-style hosting.

## Runtime Asset Bootstrap

`runtime_asset_bootstrap.py` runs before the recommendation service is created.

Behavior:

- if `LOCAL_FOOD_DB_PATH` already exists, the service uses the local `off.db`
- if the file is missing and `OFF_DB_DOWNLOAD_URL` is set, the service downloads `off.db` before startup
- if `OFF_DB_DOWNLOAD_SHA256` is set, the download is validated before it replaces the target file
- if neither a local file nor a download URL is available, the service logs the missing asset and continues boot with limited usefulness

For hosted environments with ephemeral filesystems, supplying `OFF_DB_DOWNLOAD_URL` is the safer deployment path.

## Backend Integration Contract

The sibling backend expects these ML endpoints:

- recommendation requests at `/api/recommendation`
- prime requests at `/api/prime`
- warmup status requests at `/api/prime/status`

Keep the backend environment aligned:

- `ML_SERVICE_URL` should point to this service's `/api/recommendation`
- `ML_SERVICE_PRIME_URL` should point to this service's `/api/prime`

## Railway Deployment Notes

Recommended deployment checklist:

1. deploy with the Gunicorn command from `Procfile`
2. set `PORT` from the host and keep `HOST` at `0.0.0.0` if you override it
3. provide `DB_URL` and any required FatSecret credentials
4. ensure `off.db` is available either in the image or through `OFF_DB_DOWNLOAD_URL`
5. validate `POST /api/prime`, `POST /api/prime/status`, and `POST /api/recommendation` from the backend host
6. monitor cold-start logs to confirm runtime assets loaded successfully

Recent deployment work in this codebase already points to a practical production target:

- Gunicorn on the assigned host port
- `off.db` downloaded from a hosted asset when not packaged locally
- DuckDB ready, mappings loaded, and warmup completed before live traffic

## Tests And Probes

The `tests/` folder contains targeted validation scripts and regression probes for recommendation behavior. In addition, many benchmark and validation harnesses under `../backend/scripts` call into this service as part of end-to-end recommendation checks.

Use those targeted probes when changing ranking, mapping, filtering, or cache behavior.
