from __future__ import annotations


import os

from dotenv import load_dotenv
from flask import Flask, jsonify, request

from recommendation_engine import RecommendationService
from runtime_asset_bootstrap import ensure_runtime_assets


load_dotenv()
# NOTE: Ensure required runtime data exists before building the recommendation service.
ensure_runtime_assets()
app = Flask(__name__)
recommendation_service = RecommendationService.from_env()


@app.route("/api/prime", methods=["POST"])
def prime():
    # NOTE: Prime caches on login to avoid slow first recommendations.
    try:
        data = request.json or {}
        payload = recommendation_service.prime_user_context(data)
        print("**** API /prime completed")
        return jsonify(payload)
    except Exception as exc:
        print("Prime Error:", exc)
        return jsonify({"error": str(exc)}), 500


@app.route("/api/prime/status", methods=["POST"])
def prime_status():
    try:
        data = request.json or {}
        payload = recommendation_service.get_prime_response_warmup_status(data)
        print("**** API /prime/status completed")
        return jsonify(payload)
    except Exception as exc:
        print("Prime Status Error:", exc)
        return jsonify({"error": str(exc)}), 500


@app.route("/api/runtime-metrics", methods=["POST"])
def runtime_metrics():
    # NOTE: Expose queue and process telemetry for Phase 10 runtime measurement only.
    try:
        payload = recommendation_service.get_runtime_metrics()
        print("**** API /runtime-metrics completed")
        return jsonify(payload)
    except Exception as exc:
        print("Runtime Metrics Error:", exc)
        return jsonify({"error": str(exc)}), 500


@app.route("/api/recommendation", methods=["POST"])
@app.route("/recommend", methods=["POST"])
def recommend():
    try:
        data = request.json or {}
        payload = recommendation_service.recommend(data)
        meal_type = (data or {}).get("mealType") or (data or {}).get("slot") or "all"
        print(f"**** API /recommend completed: meal_type={meal_type}")
        return jsonify(payload)
    except Exception as exc:
        print("Recommendation Error:", exc)
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    debug_flag = str(os.getenv("FLASK_DEBUG", "0")).strip().lower() in {"1", "true", "yes", "on"}
    host = str(os.getenv("HOST", "0.0.0.0")).strip() or "0.0.0.0"
    port = int(str(os.getenv("PORT", "5001")).strip() or "5001")
    # Disable debug reloader by default to avoid duplicate workers and duplicate background mapping jobs.
    app.run(host=host, port=port, debug=debug_flag)
