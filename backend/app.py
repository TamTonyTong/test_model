"""
TBRGS Flask Backend
Serves trained traffic models for traffic flow prediction.

Endpoints:
  POST /predict
  GET  /model-info
  GET  /health
"""

import os

import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS

from prediction_service import (
    DEFAULT_SITE_ID,
    LOOKBACK,
    MODEL_DIR,
    MODEL_REGISTRY,
    FLOW_CAP,
    SPEED_CAP,
    SPEED_LIMIT,
    build_feature_vector,
    flow_to_speed,
    get_model_path,
    get_road_condition,
    init_runtime,
    load_model_for_key,
    parse_bool,
    parse_model_key,
    travel_time,
)

app = Flask(__name__)
CORS(app)

runtime = init_runtime()
DEFAULT_MODEL_KEY = runtime["default_model_key"]
MODEL_PATH = runtime["model_path"]
MODEL_CACHE = runtime["model_cache"]
scaler = runtime["scaler"]
SITE_SCALERS = runtime["site_scalers"]
KNOWN_SITE_IDS = runtime["known_site_ids"]
SCALER_READY = runtime["scaler_ready"]


@app.route("/health", methods=["GET"])
def health():
    return jsonify(
        {
            "status": "ok",
            "model": MODEL_PATH,
            "model_key": DEFAULT_MODEL_KEY,
            "scaler_ready": SCALER_READY,
            "known_sites": len(KNOWN_SITE_IDS),
        }
    )


@app.route("/model-info", methods=["GET"])
def model_info():
    requested_model = request.args.get("model")
    try:
        info_model_key = parse_model_key(requested_model, DEFAULT_MODEL_KEY)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    info_model = MODEL_CACHE.get(info_model_key)
    if info_model is None:
        info_model, _ = load_model_for_key(info_model_key)
        MODEL_CACHE[info_model_key] = info_model

    info_model_path = get_model_path(info_model_key)
    layers = [{"name": l.name, "type": l.__class__.__name__} for l in info_model.layers]
    return jsonify(
        {
            "model_file": os.path.basename(info_model_path),
            "model_key": info_model_key,
            "model_label": MODEL_REGISTRY[info_model_key]["label"],
            "available_models": [
                {
                    "key": key,
                    "label": config["label"],
                    "file_name": config["file_name"],
                    "exists": os.path.exists(os.path.join(MODEL_DIR, config["file_name"])),
                }
                for key, config in MODEL_REGISTRY.items()
            ],
            "mode": "global-multi-site",
            "default_site_id": DEFAULT_SITE_ID,
            "known_sites": len(KNOWN_SITE_IDS),
            "lookback": LOOKBACK,
            "input_shape": str(info_model.input_shape),
            "output_shape": str(info_model.output_shape),
            "total_params": info_model.count_params(),
            "layers": layers,
            "scaler_ready": SCALER_READY,
            "flow_capacity": FLOW_CAP,
            "speed_capacity": SPEED_CAP,
            "speed_limit": SPEED_LIMIT,
        }
    )


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)

    try:
        requested_model_key = parse_model_key(data.get("model"), DEFAULT_MODEL_KEY)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    active_model = MODEL_CACHE.get(requested_model_key)
    if active_model is None:
        active_model, _ = load_model_for_key(requested_model_key)
        MODEL_CACHE[requested_model_key] = active_model

    flows = data.get("flows")
    if not isinstance(flows, list) or len(flows) != LOOKBACK:
        return jsonify({"error": f"'flows' must be a list of {LOOKBACK} values"}), 400
    try:
        flows = [float(v) for v in flows]
    except (TypeError, ValueError):
        return jsonify({"error": "All values in 'flows' must be numeric"}), 400

    if any(v < 0 for v in flows):
        return jsonify({"error": "All values in 'flows' must be non-negative"}), 400

    site_raw = data.get("site_id", data.get("scats_site", data.get("NB_SCATS_SITE")))
    if site_raw is not None:
        try:
            site_id = int(site_raw)
        except (TypeError, ValueError):
            return jsonify({"error": "'site_id' must be an integer"}), 400
    else:
        site_id = None

    try:
        interval_index = int(data.get("interval_index", 0))
        day_of_week = int(data.get("day_of_week", 0))
    except (TypeError, ValueError):
        return jsonify({"error": "'interval_index' and 'day_of_week' must be integers"}), 400

    if interval_index < 0 or interval_index > 95:
        return jsonify({"error": "'interval_index' must be between 0 and 95"}), 400
    if day_of_week < 0 or day_of_week > 6:
        return jsonify({"error": "'day_of_week' must be between 0 and 6"}), 400

    is_weekend = parse_bool(data.get("is_weekend"), default=(day_of_week >= 5))

    try:
        distance_km = float(data.get("distance_km", 1.0))
    except (TypeError, ValueError):
        return jsonify({"error": "'distance_km' must be numeric"}), 400

    try:
        num_ints = int(data.get("num_intersections", 1))
    except (TypeError, ValueError):
        return jsonify({"error": "'num_intersections' must be an integer"}), 400

    if distance_km <= 0:
        return jsonify({"error": "'distance_km' must be > 0"}), 400
    if num_ints < 0:
        return jsonify({"error": "'num_intersections' must be >= 0"}), 400

    if site_id is not None and site_id in SITE_SCALERS:
        flow_scaler = SITE_SCALERS[site_id]
        scaler_scope = "site"
    else:
        flow_scaler = scaler
        scaler_scope = "global"

    X = build_feature_vector(flows, interval_index, day_of_week, is_weekend, flow_scaler)
    X_input = X[np.newaxis, ...]

    pred_scaled = active_model.predict(X_input, verbose=0)[0][0]
    pred_flow_interval = float(flow_scaler.inverse_transform([[pred_scaled]])[0][0])
    pred_flow_interval = max(0.0, pred_flow_interval)
    pred_flow_hour = pred_flow_interval * 4

    speed, is_congested = flow_to_speed(pred_flow_hour)
    tt = travel_time(distance_km, speed, num_ints)
    condition = get_road_condition(pred_flow_hour, is_congested)

    return jsonify(
        {
            "predicted_flow_per_interval": round(pred_flow_interval, 2),
            "predicted_flow_per_hour": round(pred_flow_hour, 2),
            "speed_kmh": speed,
            "is_congested": is_congested,
            "travel_time_min": tt,
            "road_condition": condition,
            "site_id": site_id,
            "scaler_scope": scaler_scope,
            "model_key": requested_model_key,
            "model_label": MODEL_REGISTRY[requested_model_key]["label"],
            "input_flows": flows,
        }
    )


if __name__ == "__main__":
    print("\nTBRGS Prediction Server running on http://localhost:5000\n")
    app.run(debug=True, port=5000)
