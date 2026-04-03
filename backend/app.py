"""
TBRGS Flask Backend
Serves the trained CNN-LSTM model for traffic flow prediction.

Endpoints:
  POST /predict   - accepts 12 recent flow readings + datetime context, returns predicted flow & travel metrics
  GET  /model-info - returns model metadata
  GET  /health     - health check
"""

import os
import math
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

BACKEND_DIR    = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR    = os.path.dirname(BACKEND_DIR)
MODEL_DIR      = os.path.join(PROJECT_DIR, "model")
DATASET_PATH   = os.path.join(PROJECT_DIR, "public", "mapInfo", "VSDATA_202603_Summed.csv")
DEFAULT_SITE_ID = 4057
LOOKBACK       = 12        # number of 15-min intervals used as input
SPEED_LIMIT    = 60        # km/h
INTERSECTION_DELAY = 0.5  # minutes (30 seconds)

MODEL_REGISTRY = {
    "cnn_lstm": {
        "label": "CNN-LSTM",
        "file_name": "tbrgs_cnn_lstm_global_best.keras",
    },
    "gru": {
        "label": "GRU",
        "file_name": "tbrgs_gru_global_best.keras",
    },
    "lstm": {
        "label": "LSTM",
        "file_name": "tbrgs_lstm_global_best.keras",
    },
    "lstm_gru": {
        "label": "LSTM-GRU",
        "file_name": "tbrgs_lstm_gru_global_best.keras",
    },
}

MODEL_ALIASES = {
    "cnn-lstm": "cnn_lstm",
    "cnn_lstm": "cnn_lstm",
    "cnnlstm": "cnn_lstm",
    "gru": "gru",
    "lstm": "lstm",
    "lstm-gru": "lstm_gru",
    "lstm_gru": "lstm_gru",
    "lstmgru": "lstm_gru",
}


def normalize_model_key(raw_value):
    if raw_value is None:
        return "cnn_lstm"
    key = str(raw_value).strip().lower().replace(" ", "_")
    return MODEL_ALIASES.get(key, key)


def get_model_path(model_key):
    model_config = MODEL_REGISTRY[model_key]
    return os.path.join(MODEL_DIR, model_config["file_name"])


def load_model_for_key(model_key):
    model_path = get_model_path(model_key)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    loaded_model = tf.keras.models.load_model(model_path)
    return loaded_model, model_path


DEFAULT_MODEL_KEY = normalize_model_key(os.getenv("MODEL_NAME") or os.getenv("MODEL_KEY") or "cnn_lstm")
if DEFAULT_MODEL_KEY not in MODEL_REGISTRY:
    available_models = ", ".join(sorted(MODEL_REGISTRY))
    raise ValueError(f"Unknown default model '{DEFAULT_MODEL_KEY}'. Available models: {available_models}")

# Flow→speed quadratic params  (flow = a*speed² + b*speed)
A_QUAD = -1.4648375
B_QUAD = 93.75
# Capacity: d(flow)/d(speed)=0  → speed_cap = -B/(2A)
SPEED_CAP = -B_QUAD / (2 * A_QUAD)          # ≈ 32 km/h
FLOW_CAP  = A_QUAD * SPEED_CAP**2 + B_QUAD * SPEED_CAP   # ≈ 1500 veh/h
# Speed limit threshold: speed where flow quadratic = speed_limit (on the under-capacity branch)
# solve: A*v²+B*v = A*60²+B*60  → any flow produced by v>32 on green branch
FLOW_AT_SPEED_LIMIT = A_QUAD * SPEED_LIMIT**2 + B_QUAD * SPEED_LIMIT  # ≈ 430 veh/h

# Condition thresholds are derived from the same quadratic for consistency.
# Free Flow: speed > 50 km/h
# Moderate : 35–50 km/h
# Congested: < 35 km/h (or congested branch)
FLOW_AT_50_KMH = A_QUAD * 50**2 + B_QUAD * 50
FLOW_AT_35_KMH = A_QUAD * 35**2 + B_QUAD * 35

app = Flask(__name__)
CORS(app)

# Load model + fit scaler at startup
print("Loading model …")
MODEL_CACHE = {}
model, MODEL_PATH = load_model_for_key(DEFAULT_MODEL_KEY)
MODEL_CACHE[DEFAULT_MODEL_KEY] = model
print(f"  Model loaded from: {MODEL_PATH}")
print(f"  Active model key: {DEFAULT_MODEL_KEY}")
print(f"  Input shape: {model.input_shape}")

print("Refitting MinMaxScaler from dataset …")
SITE_SCALERS = {}
KNOWN_SITE_IDS = set()
try:
    df_raw = pd.read_csv(DATASET_PATH)

    site_col = next(
        (
            c for c in ["NB_SCATS_SITE", "SCATS Number", "SCATS_SITE", "VSDATA_SITE"]
            if c in df_raw.columns
        ),
        None,
    )
    if site_col is None:
        raise ValueError("No supported site id column found in dataset")

    # Melt the V00–V95 volume columns into long format
    v_cols = [c for c in df_raw.columns if str(c).startswith("V") and str(c)[1:].isdigit()]
    if not v_cols:
        raise ValueError("No V00-V95 flow columns found in dataset")

    df_long = df_raw.melt(
        id_vars=[site_col], value_vars=v_cols, var_name="interval", value_name="flow"
    )
    df_long["flow"] = pd.to_numeric(df_long["flow"], errors="coerce")
    df_long[site_col] = pd.to_numeric(df_long[site_col], errors="coerce")
    df_long.dropna(subset=["flow"], inplace=True)
    df_long.dropna(subset=[site_col], inplace=True)
    df_long = df_long[df_long["flow"] >= 0].copy()

    flow_values = df_long["flow"].values.reshape(-1, 1)
    if len(flow_values) == 0:
        raise ValueError("No valid flow rows found after cleaning")

    global_scaler = MinMaxScaler()
    global_scaler.fit(flow_values)

    # Build site-specific scalers so each site can be normalized similarly to training.
    for sid, grp in df_long.groupby(site_col):
        site_flows = grp["flow"].values.reshape(-1, 1)
        if len(site_flows) == 0:
            continue
        site_scaler = MinMaxScaler()
        site_scaler.fit(site_flows)
        sid_int = int(sid)
        SITE_SCALERS[sid_int] = site_scaler
        KNOWN_SITE_IDS.add(sid_int)

    scaler = global_scaler
    print(f"  Scaler fitted on {len(flow_values)} rows. Range: [{flow_values.min():.1f}, {flow_values.max():.1f}]")
    print(f"  Built {len(SITE_SCALERS)} site-specific scalers")
    SCALER_READY = True
except Exception as e:
    print(f"  WARNING: Could not load dataset – {e}")
    print("  Falling back to approximate scaler (0–1800 veh/h)")
    scaler = MinMaxScaler()
    scaler.fit(np.array([[0], [1800]]))
    SITE_SCALERS = {}
    KNOWN_SITE_IDS = set()
    SCALER_READY = False

# Helper: cyclical encoding
def cyclic(val, max_val):
    """Return (sin, cos) cyclical encoding."""
    angle = 2 * math.pi * val / max_val
    return math.sin(angle), math.cos(angle)


def parse_bool(value, default=False):
    """Parse common JSON/string boolean values robustly."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        txt = value.strip().lower()
        if txt in {"true", "1", "yes", "y", "on"}:
            return True
        if txt in {"false", "0", "no", "n", "off"}:
            return False
    return default


def parse_model_key(raw_value, default_key=DEFAULT_MODEL_KEY):
    if raw_value is None:
        return default_key

    model_key = normalize_model_key(raw_value)
    if model_key not in MODEL_REGISTRY:
        available_models = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(f"Unknown model '{raw_value}'. Available models: {available_models}")
    return model_key


def build_feature_vector(flow_seq, interval_index, day_of_week, is_weekend, flow_scaler):
    """
    Build a (LOOKBACK, n_features) array from raw flow sequence plus context.
    Features per timestep:
      0: scaled_flow
      1: sin_interval, 2: cos_interval   (96 intervals/day)
      3: sin_dow,      4: cos_dow        (0=Monday…6=Sunday)
      5: is_weekend
    """
    n_features = 6
    X = np.zeros((LOOKBACK, n_features))
    for i, flow in enumerate(flow_seq):
        scaled = flow_scaler.transform([[flow]])[0][0]
        # The interval for each step stepping back
        idx = (interval_index - (LOOKBACK - 1 - i)) % 96
        sin_i, cos_i = cyclic(idx, 96)
        sin_d, cos_d = cyclic(day_of_week, 7)
        X[i] = [scaled, sin_i, cos_i, sin_d, cos_d, float(is_weekend)]
    return X


def flow_to_speed(flow):
    """
    Convert predicted traffic flow (veh/h) to speed (km/h) using the
    quadratic fundamental diagram. Traffic is assumed under capacity.
    Returns (speed_kmh, is_congested).
    """
    # Solve A*v²+B*v = flow  → v = (-B ± sqrt(B²+4A*flow))/(2A)
    # With A < 0, the '+' root gives the LOWER speed and the '-' root gives the HIGHER speed.
    discriminant = B_QUAD**2 + 4 * A_QUAD * flow
    if discriminant < 0:
        # Flow exceeds max parabola – extremely congested, clamp
        speed = SPEED_CAP
        is_congested = True
    elif flow <= FLOW_AT_SPEED_LIMIT:
        # Under-capacity AND flow is low enough that speed ≥ speed limit → cap at limit
        speed = SPEED_LIMIT
        is_congested = False
    elif flow <= FLOW_CAP:
        # Under-capacity, speed below limit
        speed = (-B_QUAD - math.sqrt(discriminant)) / (2 * A_QUAD)
        is_congested = False
    else:
        # Congested branch (red): smaller root
        speed = (-B_QUAD + math.sqrt(discriminant)) / (2 * A_QUAD)
        is_congested = True
    speed = max(1.0, min(speed, SPEED_LIMIT))
    return round(speed, 2), is_congested


def travel_time(distance_km, speed_kmh, num_intersections=1):
    """
    Estimate travel time in minutes.
    travel_time = (distance / speed) * 60 + intersections * 0.5
    """
    drive_min = (distance_km / speed_kmh) * 60
    delay_min = num_intersections * INTERSECTION_DELAY
    return round(drive_min + delay_min, 2)


# ──────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model": MODEL_PATH,
        "model_key": DEFAULT_MODEL_KEY,
        "scaler_ready": SCALER_READY,
        "known_sites": len(KNOWN_SITE_IDS),
    })


@app.route("/model-info", methods=["GET"])
def model_info():
    requested_model = request.args.get("model")
    try:
        info_model_key = parse_model_key(requested_model)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    info_model = MODEL_CACHE.get(info_model_key)
    if info_model is None:
        info_model, _ = load_model_for_key(info_model_key)
        MODEL_CACHE[info_model_key] = info_model

    info_model_path = get_model_path(info_model_key)
    layers = [{"name": l.name, "type": l.__class__.__name__} for l in info_model.layers]
    return jsonify({
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
    })


@app.route("/predict", methods=["POST"])
def predict():
    """
    Request body (JSON):
    {
            "site_id": 4057,                 // optional but recommended for site-specific scaling
      "flows": [f0, f1, ..., f11],    // last 12 flow readings (veh/15min)
      "interval_index": 32,           // index of the NEXT interval (0–95, one per 15-min slot)
      "day_of_week": 1,               // 0=Monday … 6=Sunday
      "is_weekend": false,
      "distance_km": 1.0,             // optional, for travel time calc (default 1.0)
      "num_intersections": 1          // optional (default 1)
    }

    Response:
    {
      "predicted_flow_per_interval": float,   // veh/15min
      "predicted_flow_per_hour": float,       // veh/h  (×4)
      "speed_kmh": float,
      "is_congested": bool,
      "travel_time_min": float,
      "road_condition": "Free Flow" | "Moderate" | "Congested",
      "site_id": int | null,
      "scaler_scope": "site" | "global",
      "input_flows": [...]
    }
    """
    data = request.get_json(force=True)

    try:
        requested_model_key = parse_model_key(data.get("model"))
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
    # Model expects (batch, lookback, features)
    X_input = X[np.newaxis, ...]   # shape: (1, 12, 6)

    pred_scaled = active_model.predict(X_input, verbose=0)[0][0]
    pred_flow_interval = float(flow_scaler.inverse_transform([[pred_scaled]])[0][0])
    pred_flow_interval = max(0.0, pred_flow_interval)
    pred_flow_hour = pred_flow_interval * 4  # 15-min → per hour

    speed, is_congested = flow_to_speed(pred_flow_hour)
    tt = travel_time(distance_km, speed, num_ints)

    # Keep road condition coherent with the speed estimate from the same model.
    if is_congested or pred_flow_hour >= FLOW_AT_35_KMH:
        condition = "Congested"
    elif pred_flow_hour >= FLOW_AT_50_KMH:
        condition = "Moderate"
    else:
        condition = "Free Flow"

    return jsonify({
        "predicted_flow_per_interval": round(pred_flow_interval, 2),
        "predicted_flow_per_hour":     round(pred_flow_hour, 2),
        "speed_kmh":                   speed,
        "is_congested":                is_congested,
        "travel_time_min":             tt,
        "road_condition":              condition,
        "site_id":                     site_id,
        "scaler_scope":                scaler_scope,
        "model_key":                   requested_model_key,
        "model_label":                 MODEL_REGISTRY[requested_model_key]["label"],
        "input_flows":                 flows,
    })


if __name__ == "__main__":
    print("\n🚦 TBRGS Prediction Server running on http://localhost:5000\n")
    app.run(debug=True, port=5000)
