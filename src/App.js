/**
 * App.js – TBRGS Traffic Flow Predictor GUI
 *
 * A single-page React app that communicates with the Flask backend at
 * http://localhost:5000 to predict traffic flow using the trained CNN-LSTM model.
 *
 * Sections:
 *   - Left column : Parameter settings + 12 flow inputs + Predict button
 *   - Right column: Results (metrics, condition badge, chart) + Model info
 */

import React, { useState, useEffect, useCallback } from 'react';
import './App.css';
import PredictionChart from './components/PredictionChart';

// ── Constants ──────────────────────────────────────────────
const API = 'http://localhost:5000';
const LOOKBACK = 12;
const DAYS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'];

// Example realistic traffic flows for SCATS site 4057 (morning peak-ish)
const SAMPLE_FLOWS = [42, 55, 68, 80, 91, 75, 63, 58, 72, 88, 95, 82];

// Generate interval labels: 00:00, 00:15, … 23:45
function makeIntervalLabels() {
  const labels = [];
  for (let h = 0; h < 24; h++) {
    for (let m = 0; m < 60; m += 15) {
      labels.push(`${String(h).padStart(2,'0')}:${String(m).padStart(2,'0')}`);
    }
  }
  return labels; // 96 labels
}
const INTERVAL_LABELS = makeIntervalLabels();

// ── Helper ─────────────────────────────────────────────────
function conditionClass(condition) {
  if (!condition) return '';
  return condition.toLowerCase().replace(' ', '-');
}

// ── Main Component ─────────────────────────────────────────
export default function App() {
  // Form state
  const [flows, setFlows] = useState(Array(LOOKBACK).fill(''));
  const [intervalIndex, setIntervalIndex] = useState(32); // default 08:00
  const [dayOfWeek, setDayOfWeek] = useState(0);          // default Monday
  const [distanceKm, setDistanceKm] = useState(1.0);
  const [numIntersections, setNumIntersections] = useState(1);

  // App state
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');
  const [modelInfo, setModelInfo] = useState(null);
  const [backendStatus, setBackendStatus] = useState('checking'); // 'ok' | 'error' | 'checking'

  // ── Check backend health on mount ──
  useEffect(() => {
    fetch(`${API}/health`)
      .then(r => r.json())
      .then(() => {
        setBackendStatus('ok');
        // Also fetch model info
        return fetch(`${API}/model-info`);
      })
      .then(r => r.json())
      .then(info => setModelInfo(info))
      .catch(() => setBackendStatus('error'));
  }, []);

  // ── Flow input handler ──
  const handleFlowChange = useCallback((idx, val) => {
    setFlows(prev => {
      const next = [...prev];
      next[idx] = val;
      return next;
    });
  }, []);

  // ── Fill with sample data ──
  const loadSample = () => {
    setFlows(SAMPLE_FLOWS.map(String));
    setIntervalIndex(32);
    setDayOfWeek(1); // Tuesday
    setError('');
    setResult(null);
  };

  // ── Clear all ──
  const clearAll = () => {
    setFlows(Array(LOOKBACK).fill(''));
    setResult(null);
    setError('');
  };

  // ── Validation ──
  const validate = () => {
    for (let i = 0; i < LOOKBACK; i++) {
      const v = parseFloat(flows[i]);
      if (isNaN(v) || v < 0) {
        return `Flow reading #${i + 1} is invalid. All 12 values must be non-negative numbers.`;
      }
    }
    return null;
  };

  // ── Predict ──
  const handlePredict = async () => {
    const err = validate();
    if (err) { setError(err); return; }
    setError('');
    setLoading(true);

    const isWeekend = dayOfWeek >= 5;
    const payload = {
      flows: flows.map(Number),
      interval_index: intervalIndex,
      day_of_week: dayOfWeek,
      is_weekend: isWeekend,
      distance_km: distanceKm,
      num_intersections: numIntersections,
    };

    try {
      const res = await fetch(`${API}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      if (!res.ok) {
        const errData = await res.json();
        throw new Error(errData.error || `HTTP ${res.status}`);
      }
      const data = await res.json();
      setResult(data);
    } catch (e) {
      setError(`Prediction failed: ${e.message}`);
    } finally {
      setLoading(false);
    }
  };

  // ── Derived ──
  const isWeekend = dayOfWeek >= 5;
  const nextInterval = INTERVAL_LABELS[(intervalIndex) % 96];
  const allFilled = flows.every(f => f !== '');

  // ── Render ──
  return (
    <div className="app">

      {/* ── Header ── */}
      <header className="app-header">
        <div className="header-icon">🚦</div>
        <div className="header-text">
          <h1>TBRGS — Traffic Flow Predictor</h1>
          <p>CNN-LSTM model · SCATS Site 4057 · Boroondara Network</p>
        </div>
        <div
          className="header-badge"
          style={{ borderColor: backendStatus === 'ok' ? 'rgba(16,185,129,0.4)' : 'rgba(239,68,68,0.4)',
                   color: backendStatus === 'ok' ? '#34d399' : '#f87171' }}
        >
          {backendStatus === 'checking' ? '⏳ Connecting…' :
           backendStatus === 'ok'       ? '✓ Backend Online' :
                                          '✗ Backend Offline'}
        </div>
      </header>

      <main className="app-main">

        {/* ════════════════════════════════ LEFT COLUMN ════════════════════════════════ */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>

          {/* ── Parameter Settings ── */}
          <div className="card">
            <div className="card-title">⚙ Parameter Settings</div>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '14px' }}>
              <div className="form-group" style={{ margin: 0 }}>
                <label className="form-label" htmlFor="interval-select">Predict for interval</label>
                <select
                  id="interval-select"
                  className="form-select"
                  value={intervalIndex}
                  onChange={e => setIntervalIndex(Number(e.target.value))}
                >
                  {INTERVAL_LABELS.map((lbl, i) => (
                    <option key={i} value={i}>{lbl}</option>
                  ))}
                </select>
              </div>

              <div className="form-group" style={{ margin: 0 }}>
                <label className="form-label" htmlFor="dow-select">Day of week</label>
                <select
                  id="dow-select"
                  className="form-select"
                  value={dayOfWeek}
                  onChange={e => setDayOfWeek(Number(e.target.value))}
                >
                  {DAYS.map((d, i) => (
                    <option key={i} value={i}>{d}</option>
                  ))}
                </select>
              </div>

              <div className="form-group" style={{ margin: 0 }}>
                <label className="form-label" htmlFor="distance-input">Segment distance (km)</label>
                <input
                  id="distance-input"
                  type="number"
                  className="form-input"
                  min="0.1" max="50" step="0.1"
                  value={distanceKm}
                  onChange={e => setDistanceKm(Number(e.target.value))}
                />
              </div>

              <div className="form-group" style={{ margin: 0 }}>
                <label className="form-label" htmlFor="intersections-input">Intersections on route</label>
                <input
                  id="intersections-input"
                  type="number"
                  className="form-input"
                  min="0" max="20" step="1"
                  value={numIntersections}
                  onChange={e => setNumIntersections(Number(e.target.value))}
                />
              </div>
            </div>

            {/* Context chips */}
            <div style={{ display: 'flex', gap: '8px', marginTop: '14px', flexWrap: 'wrap' }}>
              {[
                { label: `Predicting: ${nextInterval}`, color: '#3b82f6' },
                { label: isWeekend ? 'Weekend' : 'Weekday', color: isWeekend ? '#f59e0b' : '#10b981' },
                { label: `${numIntersections} × 30s delay`, color: '#8b5cf6' },
              ].map(chip => (
                <span key={chip.label} style={{
                  padding: '3px 10px',
                  borderRadius: '12px',
                  fontSize: '0.72rem',
                  fontWeight: 600,
                  border: `1px solid ${chip.color}55`,
                  color: chip.color,
                  background: `${chip.color}11`,
                }}>{chip.label}</span>
              ))}
            </div>
          </div>

          {/* ── Flow Inputs ── */}
          <div className="card">
            <div className="card-title">📊 Last 12 Traffic Flow Readings (veh/15 min)</div>

            <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: '14px' }}>
              Enter the 12 most recent 15-minute interval readings ending at <strong>{INTERVAL_LABELS[intervalIndex]}</strong>.
              These form the input sequence for the CNN-LSTM model (LOOKBACK = 12).
            </p>

            <div className="flow-grid">
              {Array.from({ length: LOOKBACK }, (_, i) => {
                const slotIdx = (intervalIndex - LOOKBACK + i + 96) % 96;
                return (
                  <div key={i} className="flow-cell">
                    <span className="flow-label">{INTERVAL_LABELS[slotIdx]}</span>
                    <input
                      id={`flow-input-${i}`}
                      type="number"
                      className="flow-input"
                      min="0"
                      max="2000"
                      placeholder="—"
                      value={flows[i]}
                      onChange={e => handleFlowChange(i, e.target.value)}
                      aria-label={`Flow at ${INTERVAL_LABELS[slotIdx]}`}
                    />
                  </div>
                );
              })}
            </div>

            {error && <div className="error-banner">⚠ {error}</div>}

            <div className="btn-row">
              <button
                id="predict-btn"
                className="btn btn-primary"
                onClick={handlePredict}
                disabled={loading || !allFilled || backendStatus !== 'ok'}
              >
                {loading ? <><div className="spinner" /> Predicting…</> : '🔮 Predict'}
              </button>
              <button id="sample-btn" className="btn btn-secondary" onClick={loadSample}>
                ✦ Sample Data
              </button>
              <button id="clear-btn" className="btn btn-secondary" onClick={clearAll}>
                ✕ Clear
              </button>
            </div>
          </div>
        </div>

        {/* ════════════════════════════════ RIGHT COLUMN ════════════════════════════════ */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>

          {/* ── Results ── */}
          <div className="card">
            <div className="card-title">🎯 Prediction Results</div>

            {!result ? (
              <div className="result-placeholder">
                <div className="result-placeholder-icon">🔮</div>
                <span>Fill in the 12 flow readings and click <strong>Predict</strong></span>
              </div>
            ) : (
              <>
                {/* Condition badge */}
                <div style={{ marginBottom: '16px' }}>
                  <span className={`condition-badge ${conditionClass(result.road_condition)}`}>
                    <span className="condition-dot" />
                    {result.road_condition}
                  </span>
                  <span style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginLeft: '8px' }}>
                    {result.is_congested ? '⚠ Road is over capacity' : 'Road is under capacity'}
                  </span>
                </div>

                {/* Metric cards */}
                <div className="metrics-grid">
                  <div className={`metric-card highlight ${conditionClass(result.road_condition)}`}>
                    <div className="metric-value">{Math.round(result.predicted_flow_per_interval)}</div>
                    <div className="metric-label">veh / 15 min (predicted)</div>
                  </div>
                  <div className={`metric-card ${conditionClass(result.road_condition)}`}>
                    <div className="metric-value">{Math.round(result.predicted_flow_per_hour)}</div>
                    <div className="metric-label">veh / hour</div>
                  </div>
                  <div className="metric-card">
                    <div className="metric-value" style={{ color: result.is_congested ? 'var(--accent-red)' : 'var(--accent-green)' }}>
                      {result.speed_kmh}
                    </div>
                    <div className="metric-label">km / h (est. speed)</div>
                  </div>
                  <div className="metric-card">
                    <div className="metric-value" style={{ color: '#a78bfa' }}>{result.travel_time_min}</div>
                    <div className="metric-label">min travel time ({distanceKm} km)</div>
                  </div>
                </div>

                {/* Chart */}
                <div className="chart-container">
                  <div className="chart-title">Traffic Flow Sequence (veh/15 min)</div>
                  <PredictionChart
                    flows={result.input_flows}
                    predicted={result.predicted_flow_per_interval}
                  />
                </div>
              </>
            )}
          </div>

          {/* ── Model Info ── */}
          <div className="card">
            <div className="card-title">🧠 Model Information</div>
            {!modelInfo ? (
              <div style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>
                {backendStatus === 'error'
                  ? '❌ Cannot reach backend. Start the Flask server first.'
                  : '⏳ Loading model info…'}
              </div>
            ) : (
              <div className="info-rows">
                {[
                  { key: 'Model file',     val: modelInfo.model_file },
                  { key: 'SCATS Site',     val: modelInfo.site_id },
                  { key: 'Lookback',       val: `${modelInfo.lookback} intervals (${modelInfo.lookback * 15} min)` },
                  { key: 'Input shape',    val: modelInfo.input_shape },
                  { key: 'Output shape',   val: modelInfo.output_shape },
                  { key: 'Total params',   val: modelInfo.total_params?.toLocaleString() },
                  { key: 'Scaler fitted',  val: modelInfo.scaler_ready ? '✓ Yes' : '✗ Approximated', isGreen: modelInfo.scaler_ready },
                  { key: 'Flow capacity',  val: `≈ ${Math.round(modelInfo.flow_capacity)} veh/h` },
                  { key: 'Speed capacity', val: `≈ ${Math.round(modelInfo.speed_capacity)} km/h` },
                ].map(({ key, val, isGreen }) => (
                  <div key={key} className="info-row">
                    <span className="info-key">{key}</span>
                    <span className={`info-val ${isGreen === true ? 'green' : isGreen === false ? 'red' : ''}`}>
                      {val}
                    </span>
                  </div>
                ))}
              </div>
            )}
          </div>

        </div>
      </main>

      <footer className="app-footer">
        TBRGS — Traffic-Based Route Guidance System &nbsp;·&nbsp; COS30019 Assignment 2B &nbsp;·&nbsp;
        CNN-LSTM · Boroondara Dataset · October 2006
      </footer>
    </div>
  );
}
