"""
Shared prediction/config service utilities for the Flask backend.
"""

import math
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BACKEND_DIR)
MODEL_DIR = os.path.join(PROJECT_DIR, "model")
DATASET_PATH = os.path.join(PROJECT_DIR, "public", "mapInfo", "VSDATA_202603_Summed.csv")
DEFAULT_SITE_ID = 4057
LOOKBACK = 12
SPEED_LIMIT = 60
INTERSECTION_DELAY = 0.5

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


# Flow->speed quadratic params (flow = a*speed^2 + b*speed)
A_QUAD = -1.4648375
B_QUAD = 93.75
SPEED_CAP = -B_QUAD / (2 * A_QUAD)
FLOW_CAP = A_QUAD * SPEED_CAP ** 2 + B_QUAD * SPEED_CAP
FLOW_AT_SPEED_LIMIT = A_QUAD * SPEED_LIMIT ** 2 + B_QUAD * SPEED_LIMIT
FLOW_AT_50_KMH = A_QUAD * 50 ** 2 + B_QUAD * 50
FLOW_AT_35_KMH = A_QUAD * 35 ** 2 + B_QUAD * 35


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


def parse_bool(value, default=False):
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


def parse_model_key(raw_value, default_key):
    if raw_value is None:
        return default_key

    model_key = normalize_model_key(raw_value)
    if model_key not in MODEL_REGISTRY:
        available_models = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(f"Unknown model '{raw_value}'. Available models: {available_models}")
    return model_key


def cyclic(val, max_val):
    angle = 2 * math.pi * val / max_val
    return math.sin(angle), math.cos(angle)


def build_feature_vector(flow_seq, interval_index, day_of_week, is_weekend, flow_scaler):
    n_features = 6
    X = np.zeros((LOOKBACK, n_features))
    for i, flow in enumerate(flow_seq):
        scaled = flow_scaler.transform([[flow]])[0][0]
        idx = (interval_index - (LOOKBACK - 1 - i)) % 96
        sin_i, cos_i = cyclic(idx, 96)
        sin_d, cos_d = cyclic(day_of_week, 7)
        X[i] = [scaled, sin_i, cos_i, sin_d, cos_d, float(is_weekend)]
    return X


def flow_to_speed(flow):
    discriminant = B_QUAD ** 2 + 4 * A_QUAD * flow
    if discriminant < 0:
        speed = SPEED_CAP
        is_congested = True
    elif flow <= FLOW_AT_SPEED_LIMIT:
        speed = SPEED_LIMIT
        is_congested = False
    elif flow <= FLOW_CAP:
        speed = (-B_QUAD - math.sqrt(discriminant)) / (2 * A_QUAD)
        is_congested = False
    else:
        speed = (-B_QUAD + math.sqrt(discriminant)) / (2 * A_QUAD)
        is_congested = True
    speed = max(1.0, min(speed, SPEED_LIMIT))
    return round(speed, 2), is_congested


def travel_time(distance_km, speed_kmh, num_intersections=1):
    drive_min = (distance_km / speed_kmh) * 60
    delay_min = num_intersections * INTERSECTION_DELAY
    return round(drive_min + delay_min, 2)


def get_road_condition(pred_flow_hour, is_congested):
    if is_congested or pred_flow_hour >= FLOW_AT_35_KMH:
        return "Congested"
    if pred_flow_hour >= FLOW_AT_50_KMH:
        return "Moderate"
    return "Free Flow"


def load_scalers(dataset_path):
    site_scalers = {}
    known_site_ids = set()

    df_raw = pd.read_csv(dataset_path)

    site_col = next(
        (
            c for c in ["NB_SCATS_SITE", "SCATS Number", "SCATS_SITE", "VSDATA_SITE"]
            if c in df_raw.columns
        ),
        None,
    )
    if site_col is None:
        raise ValueError("No supported site id column found in dataset")

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

    for sid, grp in df_long.groupby(site_col):
        site_flows = grp["flow"].values.reshape(-1, 1)
        if len(site_flows) == 0:
            continue
        site_scaler = MinMaxScaler()
        site_scaler.fit(site_flows)
        sid_int = int(sid)
        site_scalers[sid_int] = site_scaler
        known_site_ids.add(sid_int)

    return global_scaler, site_scalers, known_site_ids, flow_values


def init_runtime():
    default_model_key = normalize_model_key(os.getenv("MODEL_NAME") or os.getenv("MODEL_KEY") or "cnn_lstm")
    if default_model_key not in MODEL_REGISTRY:
        available_models = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(
            f"Unknown default model '{default_model_key}'. Available models: {available_models}"
        )

    print("Loading model ...")
    model_cache = {}
    model, model_path = load_model_for_key(default_model_key)
    model_cache[default_model_key] = model
    print(f"  Model loaded from: {model_path}")
    print(f"  Active model key: {default_model_key}")
    print(f"  Input shape: {model.input_shape}")

    print("Refitting MinMaxScaler from dataset ...")
    try:
        scaler, site_scalers, known_site_ids, flow_values = load_scalers(DATASET_PATH)
        print(
            "  Scaler fitted on "
            f"{len(flow_values)} rows. Range: [{flow_values.min():.1f}, {flow_values.max():.1f}]"
        )
        print(f"  Built {len(site_scalers)} site-specific scalers")
        scaler_ready = True
    except Exception as e:
        print(f"  WARNING: Could not load dataset - {e}")
        print("  Falling back to approximate scaler (0-1800 veh/h)")
        scaler = MinMaxScaler()
        scaler.fit(np.array([[0], [1800]]))
        site_scalers = {}
        known_site_ids = set()
        scaler_ready = False

    return {
        "default_model_key": default_model_key,
        "model_path": model_path,
        "model_cache": model_cache,
        "scaler": scaler,
        "site_scalers": site_scalers,
        "known_site_ids": known_site_ids,
        "scaler_ready": scaler_ready,
    }
