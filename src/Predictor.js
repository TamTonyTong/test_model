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

import React, { useState, useEffect, useCallback, useRef } from 'react';
import './styles/App.css';
import PredictionChart from './components/PredictionChart';

// Constants
const RUNTIME_HOST = typeof window !== 'undefined' ? window.location.hostname : 'localhost';
const RUNTIME_PROTOCOL = typeof window !== 'undefined' && window.location.protocol === 'https:'
  ? 'https:'
  : 'http:';
const API_BASES = Array.from(new Set([
  process.env.REACT_APP_API_BASE,
  'http://127.0.0.1:5000',
  `${RUNTIME_PROTOCOL}//${RUNTIME_HOST}:5000`,
  'http://localhost:5000',
].filter(Boolean)));
const HEALTH_CHECK_TIMEOUT_MS = 1500;
const LOOKBACK = 12;
const DAYS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'];
const MODELS = ['CNN-LSTM', 'LSTM', 'GRU', 'LSTM-GRU'];
const SAMPLE_DATASET_URL = '/mapInfo/VSDATA_202603_Summed.csv';

// Generate interval labels: 00:00, 00:15, … 23:45
function makeIntervalLabels() {
  const labels = [];
  for (let h = 0; h < 24; h++) {
    for (let m = 0; m < 60; m += 15) {
      labels.push(`${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}`);
    }
  }
  return labels; // 96 labels
}
const INTERVAL_LABELS = makeIntervalLabels();
const INTERVAL_COLUMNS = Array.from({ length: 96 }, (_, i) => `V${String(i).padStart(2, '0')}`);

// Helper
function conditionClass(condition) {
  if (!condition) return '';
  return condition.toLowerCase().replace(' ', '-');
}

function parseCsvLine(line) {
  const values = [];
  let current = '';
  let inQuotes = false;

  for (let i = 0; i < line.length; i++) {
    const char = line[i];

    if (char === '"') {
      if (inQuotes && line[i + 1] === '"') {
        current += '"';
        i += 1;
      } else {
        inQuotes = !inQuotes;
      }
      continue;
    }

    if (char === ',' && !inQuotes) {
      values.push(current);
      current = '';
      continue;
    }

    current += char;
  }

  values.push(current);
  return values;
}

function parseCsvDate(rawDate) {
  if (!rawDate) return null;
  const value = String(rawDate).trim();

  const isoMatch = value.match(/^(\d{4})-(\d{1,2})-(\d{1,2})$/);
  if (isoMatch) {
    const [, y, m, d] = isoMatch;
    return new Date(Date.UTC(Number(y), Number(m) - 1, Number(d)));
  }

  const slashMatch = value.match(/^(\d{1,2})\/(\d{1,2})\/(\d{4})$/);
  if (slashMatch) {
    const [, m, d, y] = slashMatch;
    return new Date(Date.UTC(Number(y), Number(m) - 1, Number(d)));
  }

  const parsed = new Date(value);
  return Number.isNaN(parsed.getTime()) ? null : parsed;
}

function getDowIndex(date) {
  return (date.getUTCDay() + 6) % 7; // JS: Sunday=0, app: Monday=0
}

function buildSampleDataset(csvText) {
  const lines = csvText.split(/\r?\n/).filter(line => line.trim().length > 0);
  if (lines.length < 2) {
    throw new Error('CSV file is empty or invalid.');
  }

  const header = parseCsvLine(lines[0]).map(col => col.trim());
  const siteIdx = header.indexOf('NB_SCATS_SITE');
  const dateIdx = header.indexOf('QT_INTERVAL_COUNT');
  if (siteIdx === -1 || dateIdx === -1) {
    throw new Error('CSV is missing required columns NB_SCATS_SITE or QT_INTERVAL_COUNT.');
  }

  const intervalIndices = INTERVAL_COLUMNS.map(col => header.indexOf(col));
  if (intervalIndices.some(idx => idx === -1)) {
    throw new Error('CSV is missing one or more V00-V95 columns.');
  }

  const rowsBySite = new Map();

  for (let i = 1; i < lines.length; i++) {
    const cols = parseCsvLine(lines[i]);
    if (cols.length <= dateIdx) continue;

    const site = Number(cols[siteIdx]);
    if (!Number.isInteger(site)) continue;

    const date = parseCsvDate(cols[dateIdx]);
    if (!date) continue;

    const flows = intervalIndices.map(idx => {
      const num = Number(cols[idx]);
      return Number.isFinite(num) ? num : null;
    });

    const row = {
      date,
      dayOfWeek: getDowIndex(date),
      flows,
    };

    if (!rowsBySite.has(site)) {
      rowsBySite.set(site, []);
    }
    rowsBySite.get(site).push(row);
  }

  for (const siteRows of rowsBySite.values()) {
    siteRows.sort((a, b) => a.date.getTime() - b.date.getTime());
  }

  return rowsBySite;
}

function findPreviousDayRow(rows, rowIdx) {
  if (rowIdx <= 0) return null;
  const current = rows[rowIdx];
  const prev = rows[rowIdx - 1];
  if (!current || !prev) return null;

  const oneDayMs = 24 * 60 * 60 * 1000;
  const diff = current.date.getTime() - prev.date.getTime();
  return diff === oneDayMs ? prev : null;
}

function extractLookbackSequence(currentRow, prevRow, intervalIdx) {
  const sequence = [];
  const startIdx = intervalIdx - (LOOKBACK - 1);

  for (let i = 0; i < LOOKBACK; i++) {
    const absoluteIdx = startIdx + i;
    const fromPrevDay = absoluteIdx < 0;
    const sourceRow = fromPrevDay ? prevRow : currentRow;
    const sourceIdx = fromPrevDay ? absoluteIdx + 96 : absoluteIdx;

    if (!sourceRow || sourceIdx < 0 || sourceIdx > 95) {
      return null;
    }

    const value = sourceRow.flows[sourceIdx];
    if (!Number.isFinite(value) || value < 0) {
      return null;
    }

    sequence.push(value);
  }

  return sequence;
}

// Calculate the next 15-minute interval index from current time (ceiling)
function getCurrentIntervalIndex() {
  const now = new Date();
  const hours = now.getHours();
  const minutes = now.getMinutes();
  const totalMinutes = hours * 60 + minutes;

  // Round up to next 15-minute interval
  const roundedMinutes = Math.ceil(totalMinutes / 15) * 15;
  const totalIntervalsFromMidnight = roundedMinutes / 15;

  // Return index (0-95, wrapping at 24 hours)
  return totalIntervalsFromMidnight % 96;
}

async function fetchWithTimeout(url, options = {}, timeoutMs = HEALTH_CHECK_TIMEOUT_MS) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(url, { ...options, signal: controller.signal });
  } finally {
    clearTimeout(timer);
  }
}

// Main Component
export default function Predictor() {
  const sampleDatasetPromiseRef = useRef(null);

  // Form state
  const [flows, setFlows] = useState(Array(LOOKBACK).fill(''));
  const [siteId, setSiteId] = useState('4057');
  const [intervalIndex, setIntervalIndex] = useState(getCurrentIntervalIndex()); // update to current interval
  const [dayOfWeek, setDayOfWeek] = useState(0);          // default Monday
  const [distanceKm, setDistanceKm] = useState(1.0);
  const [numIntersections, setNumIntersections] = useState(1);
  const [selectedModel, setSelectedModel] = useState('CNN-LSTM'); // default model selection

  // App state
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');
  const [modelInfo, setModelInfo] = useState(null);
  const [backendStatus, setBackendStatus] = useState('checking'); // 'ok' | 'error' | 'checking'

  const findReachableBackend = useCallback(async () => {
    for (const base of API_BASES) {
      try {
        const response = await fetchWithTimeout(`${base}/health`, { method: 'GET' });
        if (response.ok) {
          setBackendStatus('ok');
          return base;
        }
      } catch {
        // Try next host candidate.
      }
    }

    setBackendStatus('error');
    return null;
  }, []);

  // Check backend health on mount + periodic retry
  useEffect(() => {
    const boot = async () => {
      const reachableBase = await findReachableBackend();
      if (!reachableBase) {
        setModelInfo(null);
      }
    };

    boot();
    const intervalId = setInterval(boot, 15000);
    return () => clearInterval(intervalId);
  }, [findReachableBackend]);

  // Fetch model info when selected model changes
  useEffect(() => {
    const fetchModelInfo = async () => {
      const reachableBase = await findReachableBackend();
      if (!reachableBase) {
        setModelInfo(null);
        return;
      }

      try {
        const response = await fetch(`${reachableBase}/model-info?model=${encodeURIComponent(selectedModel)}`);
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`);
        }

        const info = await response.json();
        setModelInfo(info);

        if (info && Number.isInteger(info.default_site_id)) {
          setSiteId(String(info.default_site_id));
        }
      } catch {
        setModelInfo(null);
      }
    };

    fetchModelInfo();
  }, [selectedModel, findReachableBackend]);

  // Update interval index in real-time (every minute)
  useEffect(() => {
    setIntervalIndex(getCurrentIntervalIndex());
    const timer = setInterval(() => {
      setIntervalIndex(getCurrentIntervalIndex());
    }, 60000); // Update every 60 seconds (1 minute)
    return () => clearInterval(timer);
  }, []);

  // ── Flow input handler ──
  const handleFlowChange = useCallback((idx, val) => {
    setFlows(prev => {
      const next = [...prev];
      next[idx] = val;
      return next;
    });
  }, []);

  const getSampleDataset = useCallback(async () => {
    if (!sampleDatasetPromiseRef.current) {
      sampleDatasetPromiseRef.current = fetch(SAMPLE_DATASET_URL)
        .then(res => {
          if (!res.ok) {
            throw new Error(`HTTP ${res.status}`);
          }
          return res.text();
        })
        .then(buildSampleDataset)
        .catch(err => {
          sampleDatasetPromiseRef.current = null;
          throw err;
        });
    }
    return sampleDatasetPromiseRef.current;
  }, []);

  // ─ Fill with sample data
  const loadSample = async () => {
    setError('');
    setResult(null);

    const parsedSiteId = Number(siteId);
    if (!Number.isInteger(parsedSiteId) || parsedSiteId < 0) {
      setError('Enter a valid SCATS site id before loading sample data.');
      return;
    }

    try {
      const rowsBySite = await getSampleDataset();
      const siteRows = rowsBySite.get(parsedSiteId);
      if (!siteRows || siteRows.length === 0) {
        setError(`No sample rows found for SCATS site ${parsedSiteId} in CSV.`);
        return;
      }

      const matchingSequences = [];
      for (let i = 0; i < siteRows.length; i++) {
        const row = siteRows[i];
        if (row.dayOfWeek !== dayOfWeek) continue;

        const prevDayRow = intervalIndex < LOOKBACK - 1
          ? findPreviousDayRow(siteRows, i)
          : null;

        const sequence = extractLookbackSequence(row, prevDayRow, intervalIndex);
        if (sequence) {
          matchingSequences.push(sequence);
        }
      }

      if (matchingSequences.length === 0) {
        setError(
          `No valid 12-interval sequence found for site ${parsedSiteId}, ${DAYS[dayOfWeek]}, ${INTERVAL_LABELS[intervalIndex]}.`
        );
        return;
      }

      const randomIdx = Math.floor(Math.random() * matchingSequences.length);
      const selectedSequence = matchingSequences[randomIdx];
      setFlows(selectedSequence.map(v => String(v)));
    } catch (e) {
      setError(`Could not load sample data from CSV: ${e.message}`);
    }
  };

  // ── Clear all ──
  const clearAll = () => {
    setFlows(Array(LOOKBACK).fill(''));
    setResult(null);
    setError('');
  };

  // ── Validation ──
  const validate = () => {
    const parsedSiteId = Number(siteId);
    if (!Number.isInteger(parsedSiteId) || parsedSiteId < 0) {
      return 'Site ID must be a non-negative integer.';
    }

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
      site_id: Number(siteId),
      model: selectedModel,
      flows: flows.map(Number),
      interval_index: intervalIndex,
      day_of_week: dayOfWeek,
      is_weekend: isWeekend,
      distance_km: distanceKm,
      num_intersections: numIntersections,
    };

    try {
      const reachableBase = await findReachableBackend();
      if (!reachableBase) {
        throw new Error(`Backend is unreachable. Tried: ${API_BASES.join(', ')}`);
      }

      const res = await fetch(`${reachableBase}/predict`, {
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
                  disabled
                  title="Automatically updates to current time (ceiling to 15-minute interval)"
                >
                  {INTERVAL_LABELS.map((lbl, i) => (
                    <option key={i} value={i}>{lbl}</option>
                  ))}
                </select>
              </div>

              <div className="form-group" style={{ margin: 0 }}>
                <label className="form-label" htmlFor="site-id-input">SCATS site id</label>
                <input
                  id="site-id-input"
                  type="number"
                  className="form-input"
                  min="0"
                  step="1"
                  value={siteId}
                  onChange={e => setSiteId(e.target.value)}
                />
              </div>

              <div className="form-group" style={{ margin: 0 }}>
                <label className="form-label" htmlFor="model-select">Model</label>
                <select
                  id="model-select"
                  className="form-select"
                  value={selectedModel}
                  onChange={e => setSelectedModel(e.target.value)}
                >
                  {MODELS.map((model) => (
                    <option key={model} value={model}>{model}</option>
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
                  <span style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginLeft: '8px' }}>
                    Site: {result.site_id ?? Number(siteId)} ({result.scaler_scope || 'global'} scaler)
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
                  ? `❌ Cannot reach backend. Tried: ${API_BASES.join(' | ')}`
                  : '⏳ Loading model info…'}
              </div>
            ) : (
              <div className="info-rows">
                {[
                  { key: 'Model type', val: selectedModel },
                  { key: 'Model file', val: modelInfo.model_file?.replace(/_site_\d+/i, '') ?? selectedModel },
                  { key: 'Coverage', val: 'All SCATS Sites' },
                  { key: 'Lookback', val: `${modelInfo.lookback} intervals (${modelInfo.lookback * 15} min)` },
                  { key: 'Input shape', val: modelInfo.input_shape },
                  { key: 'Output shape', val: modelInfo.output_shape },
                  { key: 'Total params', val: modelInfo.total_params?.toLocaleString() },
                  { key: 'Scaler fitted', val: modelInfo.scaler_ready ? '✓ Yes' : '✗ Approximated', isGreen: modelInfo.scaler_ready },
                  { key: 'Flow capacity', val: `≈ ${Math.round(modelInfo.flow_capacity)} veh/h` },
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
