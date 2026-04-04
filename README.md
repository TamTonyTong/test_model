# TBRGS Backend Deployment And Usage Walkthrough

This guide explains how to deploy the trained model in the Flask backend and how to send valid prediction input.

## 1) What This Backend Does

The backend in backend/app.py loads:

- Model file: One of the trained global models (CNN-LSTM, GRU, LSTM, or LSTM-GRU)
- A MinMax scaler fitted from the SCATS dataset (if available)

Then it exposes three endpoints:

- GET /health
- GET /model-info
- POST /predict

The model predicts the next traffic flow value from a sequence of the latest 12 readings (15-minute intervals).

## 2) Data You Need To Feed The Model

For each prediction request, the backend expects JSON with these fields:

- site_id (optional but recommended): integer SCATS/site id for site-specific scaling
- flows (required): list of exactly 12 non-negative numbers
- interval_index (required): integer from 0 to 95 for the next 15-minute slot
- day_of_week (required): integer from 0 to 6 where 0=Monday, 6=Sunday
- is_weekend (optional): true or false; if omitted it is inferred from day_of_week
- distance_km (optional): route length in km, default 1.0
- num_intersections (optional): intersections count for delay estimate, default 1

Important notes:

- The 12 values in flows are the latest historical readings, ordered from oldest to newest.
- Each flow value should represent vehicles per 15-minute interval.
- interval_index is the time slot you want to predict for, not the first flow sample slot.

Example valid payload:

{
  "site_id": 4057,
  "flows": [42, 55, 68, 80, 91, 75, 63, 58, 72, 88, 95, 82],
  "interval_index": 32,
  "day_of_week": 1,
  "is_weekend": false,
  "distance_km": 1.2,
  "num_intersections": 2
}

## 3) Environment Setup

### Windows (PowerShell)

From project root:

1. Create virtual environment

    python -m venv .venv

2. Activate virtual environment

    .\.\.venv\\Scripts\\Activate.ps1

3. Install backend dependencies

    pip install -r backend/requirements.txt

If script execution is blocked, run PowerShell as admin once:

    Set-ExecutionPolicy RemoteSigned

### macOS / Linux (Bash/Zsh)

From project root:

1. Create virtual environment

    python3 -m venv .venv

2. Activate virtual environment

    source .venv/bin/activate

3. Install backend dependencies

    pip install -r backend/requirements.txt

## 4) Confirm Model Placement

Ensure at least one model file exists in the model/ directory:

- model/tbrgs_cnn_lstm_global_best.keras (CNN-LSTM) — default at startup
- model/tbrgs_gru_global_best.keras (GRU)
- model/tbrgs_lstm_global_best.keras (LSTM)
- model/tbrgs_lstm_gru_global_best.keras (LSTM-GRU)

The backend loads CNN-LSTM by default, but you can switch models via the /model-info endpoint or frontend selector.

## 5) Optional Dataset For Better Scaling

At startup, app.py tries to load the SCATS dataset from:

public/mapInfo/VSDATA_202603_Summed.csv

Behavior:

- If file exists and loads, a global scaler and site-specific scalers are fit from real traffic data
- If missing/unreadable, backend falls back to approximate scaler range [0, 1800]

The server still runs with fallback scaling, but real dataset scaling is strongly recommended for better prediction accuracy.

## 6) Start The Backend

From project root (with venv activated):

### Windows
    python backend/app.py

### macOS / Linux
    python3 backend/app.py

Expected startup output includes:

- Model loaded path and input shape
- Scaler status and number of sites
- Server URL: http://localhost:5000

## 7) Verify The Service

Open in browser or call with PowerShell:

- http://localhost:5000/health
- http://localhost:5000/model-info

Quick PowerShell checks:

Invoke-RestMethod -Uri "http://localhost:5000/health" -Method Get
Invoke-RestMethod -Uri "http://localhost:5000/model-info" -Method Get

## 8) Make A Prediction Request

PowerShell example:

$body = @{
  site_id = 4057
  flows = @(42,55,68,80,91,75,63,58,72,88,95,82)
  interval_index = 32
  day_of_week = 1
  is_weekend = $false
  distance_km = 1.2
  num_intersections = 2
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:5000/predict" -Method Post -ContentType "application/json" -Body $body

Typical response fields:

- predicted_flow_per_interval
- predicted_flow_per_hour
- speed_kmh
- is_congested
- travel_time_min
- road_condition
- input_flows

## 9) Connect Frontend And Backend

The React app already points to:

http://localhost:5000 and http://localhost:5000 (with fallback to localhost:5000)

Run frontend from project root in a second terminal (without venv, or in a separate shell):

    npm install
    npm start

Then open http://localhost:3000 and:

- Navigate to the Map page
- Select Origin and Destination traffic count locations
- Choose a model (CNN-LSTM, GRU, LSTM, LSTM-GRU)
- Click "Find Top 5 Paths" to get route options with traffic predictions
- View the Visualize page to inspect model training history and per-site evaluation metrics

## 10) Troubleshooting

Backend returns "'flows' must be a list of 12 values":

- Ensure flows has exactly 12 entries
- Ensure each entry is numeric and non-negative

Backend offline in frontend:

- Confirm backend is running on port 5000
- Confirm no firewall/proxy is blocking localhost

Model load error:

- Verify model path and file name
- Ensure TensorFlow version can read the .keras file

Slow first request:

- TensorFlow model warm-up can add startup latency

## 11) Input Reference Table

| Field | Type | Required | Range/Rule | Meaning |
|---|---|---|---|---|
| site_id | integer | No (recommended) | >= 0 | SCATS/site id used for site-specific scaler when available |
| flows | array[number] | Yes | Exactly 12 values, each >= 0 | Last 12 x 15-minute traffic readings |
| interval_index | integer | Yes | 0..95 | Next 15-minute slot to predict |
| day_of_week | integer | Yes | 0..6 | 0=Mon ... 6=Sun |
| is_weekend | boolean | No | true/false | Weekend flag; inferred if omitted |
| distance_km | number | No | > 0 recommended | Route distance for travel time |
| num_intersections | integer | No | >= 0 | Adds 0.5 min delay per intersection |

## 12) Minimal End-To-End Run Order

### Terminal 1 (Backend)

1. Activate venv:
   - Windows: `.\.\.venv\\Scripts\\Activate.ps1`
   - macOS/Linux: `source .venv/bin/activate`

2. Start backend:
   - `python backend/app.py` (Windows) or `python3 backend/app.py` (macOS/Linux)

3. Verify health:
   - Open http://127.0.0.1:5000/health in browser or terminal:
     - Windows: `Invoke-RestMethod -Uri "http://127.0.0.1:5000/health" -Method Get`
     - macOS/Linux: `curl http://127.0.0.1:5000/health`

### Terminal 2 (Frontend)

1. From project root (no venv needed):

    npm start

2. Open http://localhost:3000 in browser

3. Use the Map page to:
   - Select Origin and Destination locations
   - Choose model
   - Click "Find Top 5 Paths"

4. Check the Visualize page for model performance metrics
