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

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
BACKEND_DIR    = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR    = os.path.dirname(BACKEND_DIR)
MODEL_PATH     = os.path.join(PROJECT_DIR, "model", "tbrgs_cnn_lstm_site_4057.keras")
DATASET_PATH   = os.path.join(PROJECT_DIR, "public", "mapInfo", "Scats Data October 2006.xls")
TARGET_SITE    = 4057
LOOKBACK       = 12        # number of 15-min intervals used as input
SPEED_LIMIT    = 60        # km/h
INTERSECTION_DELAY = 0.5  # minutes (30 seconds)

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

# ──────────────────────────────────────────────
# Load model + fit scaler at startup
# ──────────────────────────────────────────────
print("Loading model …")
model = tf.keras.models.load_model(MODEL_PATH)
print(f"  Model loaded from: {MODEL_PATH}")
print(f"  Input shape: {model.input_shape}")

print("Refitting MinMaxScaler from dataset …")
try:
    # Read the dataset (header on row 2, i.e. index 1)
    df_raw = pd.read_excel(DATASET_PATH, sheet_name="Data", header=1)

    # Keep only site 4057
    site_col = "SCATS Number" if "SCATS Number" in df_raw.columns else df_raw.columns[0]
    df_site = df_raw[df_raw[site_col] == TARGET_SITE].copy()

    # Melt the V00–V95 volume columns into long format
    v_cols = [c for c in df_raw.columns if str(c).startswith("V") and str(c)[1:].isdigit()]
    df_long = df_site.melt(id_vars=[site_col], value_vars=v_cols,
                           var_name="interval", value_name="flow")
    df_long["flow"] = pd.to_numeric(df_long["flow"], errors="coerce")
    df_long.dropna(subset=["flow"], inplace=True)

    flow_values = df_long["flow"].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaler.fit(flow_values)
    print(f"  Scaler fitted on {len(flow_values)} rows. Range: [{flow_values.min():.1f}, {flow_values.max():.1f}]")
    SCALER_READY = True
except Exception as e:
    print(f"  WARNING: Could not load dataset – {e}")
    print("  Falling back to approximate scaler (0–1800 veh/h)")
    scaler = MinMaxScaler()
    scaler.fit(np.array([[0], [1800]]))
    SCALER_READY = False


# ──────────────────────────────────────────────
# Helper: cyclical encoding
# ──────────────────────────────────────────────
def cyclic(val, max_val):
    """Return (sin, cos) cyclical encoding."""
    angle = 2 * math.pi * val / max_val
    return math.sin(angle), math.cos(angle)


def build_feature_vector(flow_seq, interval_index, day_of_week, is_weekend):
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
        scaled = scaler.transform([[flow]])[0][0]
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
    return jsonify({"status": "ok", "model": MODEL_PATH, "scaler_ready": SCALER_READY})


@app.route("/model-info", methods=["GET"])
def model_info():
    layers = [{"name": l.name, "type": l.__class__.__name__} for l in model.layers]
    return jsonify({
        "model_file": os.path.basename(MODEL_PATH),
        "site_id": TARGET_SITE,
        "lookback": LOOKBACK,
        "input_shape": str(model.input_shape),
        "output_shape": str(model.output_shape),
        "total_params": model.count_params(),
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
      "input_flows": [...]
    }
    """
    data = request.get_json(force=True)

    flows = data.get("flows")
    if not flows or len(flows) != LOOKBACK:
        return jsonify({"error": f"'flows' must be a list of {LOOKBACK} values"}), 400

    interval_index = int(data.get("interval_index", 0))
    day_of_week    = int(data.get("day_of_week", 0))
    is_weekend     = bool(data.get("is_weekend", day_of_week >= 5))
    distance_km    = float(data.get("distance_km", 1.0))
    num_ints       = int(data.get("num_intersections", 1))

    X = build_feature_vector(flows, interval_index, day_of_week, is_weekend)
    # Model expects (batch, lookback, features)
    X_input = X[np.newaxis, ...]   # shape: (1, 12, 6)

    pred_scaled = model.predict(X_input, verbose=0)[0][0]
    pred_flow_interval = float(scaler.inverse_transform([[pred_scaled]])[0][0])
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
        "input_flows":                 flows,
    })


if __name__ == "__main__":
    print("\n🚦 TBRGS Prediction Server running on http://localhost:5000\n")
    app.run(debug=True, port=5000)
