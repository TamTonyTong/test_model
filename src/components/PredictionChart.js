/**
 * PredictionChart.js
 * Renders an inline SVG chart showing the last 12 traffic flow readings
 * plus a predicted next value.
 */

import React from 'react';

const WIDTH = 520;
const HEIGHT = 160;
const PAD = { top: 16, right: 20, bottom: 32, left: 44 };
const INNER_W = WIDTH - PAD.left - PAD.right;
const INNER_H = HEIGHT - PAD.top - PAD.bottom;

function PredictionChart({ flows, predicted }) {
  const allValues = [...flows, predicted];
  const maxVal = Math.max(...allValues, 1);
  const minVal = 0;
  const range = maxVal - minVal || 1;

  // Map value to SVG y coordinate (invert: high value = top)
  const toY = (v) => PAD.top + INNER_H - ((v - minVal) / range) * INNER_H;

  // Map index to SVG x coordinate
  const totalPoints = flows.length + 1; // 12 historical + 1 predicted
  const toX = (i) => PAD.left + (i / (totalPoints - 1)) * INNER_W;

  // Build polyline points for historical data (indices 0..11)
  const histPoints = flows.map((v, i) => `${toX(i)},${toY(v)}`).join(' ');
  // Predicted point is at index 12
  const predX = toX(totalPoints - 1);
  const predY = toY(predicted);
  // Dashed connector from last historical to prediction
  const lastHistX = toX(flows.length - 1);
  const lastHistY = toY(flows[flows.length - 1]);

  // Y-axis ticks
  const ticks = [0, 0.25, 0.5, 0.75, 1].map((t) => ({
    y: toY(minVal + t * range),
    label: Math.round(minVal + t * range),
  }));

  // X-axis labels  (every 3 steps → T-11, T-8, T-5, T-2, T+1)
  const xLabels = [
    { i: 0,  label: `T-${flows.length - 1}` },
    { i: 3,  label: `T-${flows.length - 4}` },
    { i: 6,  label: `T-${flows.length - 7}` },
    { i: 9,  label: `T-2` },
    { i: 11, label: `T` },
    { i: 12, label: `T+1` },
  ];

  return (
    <svg
      viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
      style={{ width: '100%', height: 'auto', display: 'block' }}
      role="img"
      aria-label="Traffic flow prediction chart"
    >
      {/* Grid lines */}
      {ticks.map((t, idx) => (
        <line
          key={idx}
          x1={PAD.left} y1={t.y} x2={PAD.left + INNER_W} y2={t.y}
          stroke="#1f2d45" strokeWidth="1"
        />
      ))}

      {/* Y-axis labels */}
      {ticks.map((t, idx) => (
        <text
          key={idx}
          x={PAD.left - 6} y={t.y + 4}
          textAnchor="end"
          fontSize="9"
          fill="#4b5563"
          fontFamily="JetBrains Mono, monospace"
        >
          {t.label}
        </text>
      ))}

      {/* Gradient fill under historical line */}
      <defs>
        <linearGradient id="histGrad" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="#3b82f6" stopOpacity="0.25" />
          <stop offset="100%" stopColor="#3b82f6" stopOpacity="0.01" />
        </linearGradient>
      </defs>
      {/* Area fill */}
      <polygon
        points={`${toX(0)},${toY(minVal)} ${histPoints} ${toX(flows.length - 1)},${toY(minVal)}`}
        fill="url(#histGrad)"
      />

      {/* Historical line */}
      <polyline
        points={histPoints}
        fill="none"
        stroke="#3b82f6"
        strokeWidth="2"
        strokeLinejoin="round"
        strokeLinecap="round"
      />

      {/* Dashed connector to prediction */}
      <line
        x1={lastHistX} y1={lastHistY}
        x2={predX} y2={predY}
        stroke="#10b981"
        strokeWidth="2"
        strokeDasharray="5,3"
      />

      {/* Predicted point */}
      <circle cx={predX} cy={predY} r="6" fill="#10b981" stroke="#0a0e1a" strokeWidth="2" />
      <circle cx={predX} cy={predY} r="10" fill="none" stroke="#10b981" strokeOpacity="0.3" strokeWidth="1.5" />

      {/* X-axis labels */}
      {xLabels.map(({ i, label }) => (
        <text
          key={i}
          x={toX(i)}
          y={PAD.top + INNER_H + 20}
          textAnchor="middle"
          fontSize="8.5"
          fill={i === 12 ? '#10b981' : '#4b5563'}
          fontFamily="JetBrains Mono, monospace"
          fontWeight={i === 12 ? '600' : '400'}
        >
          {label}
        </text>
      ))}

      {/* Prediction value label */}
      <text
        x={predX}
        y={predY - 14}
        textAnchor="middle"
        fontSize="10"
        fill="#10b981"
        fontFamily="JetBrains Mono, monospace"
        fontWeight="600"
      >
        {Math.round(predicted)}
      </text>

      {/* Legend */}
      <circle cx={PAD.left + 4} cy={PAD.top - 4} r="4" fill="#3b82f6" />
      <text x={PAD.left + 12} y={PAD.top} fontSize="9" fill="#94a3b8" fontFamily="Inter, sans-serif">Historical</text>
      <circle cx={PAD.left + 75} cy={PAD.top - 4} r="4" fill="#10b981" />
      <text x={PAD.left + 83} y={PAD.top} fontSize="9" fill="#94a3b8" fontFamily="Inter, sans-serif">Predicted (T+1)</text>
    </svg>
  );
}

export default PredictionChart;
