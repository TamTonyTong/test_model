import React, { useEffect, useMemo, useState } from "react";
import Papa from "papaparse";
import { yenKShortest } from "./algorithm/yen";
import { SAMPLE_FLOWS } from "./utils/trafficDefaults";
import {
  buildRoadBasedNeighborMap,
  dist,
  getNearest,
  projectSites
} from "./utils/graphUtils";

const API_PREDICT = "http://127.0.0.1:5000/predict";

const MAX_NODES = 5000;
const MAP_WIDTH = 1000;
const MAP_HEIGHT = 620;
const MAP_PADDING = 40;
const PATH_COLORS = ["#ef4444", "#0ea5e9", "#f59e0b", "#22c55e", "#8b5cf6"];

// giả định capacity
// const FLOW_CAPACITY = 1800;
 
export default function MapPage() {

  const [sites, setSites] = useState([]);
  const [origin, setOrigin] = useState(null);
  const [destinations, setDestinations] = useState([]);
  const [paths, setPaths] = useState([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    fetch("/mapInfo/Traffic_Count_Locations_with_LONG_LAT.csv")
      .then(res => res.text())
      .then(csv => {
        Papa.parse(csv, {
          header: true,
          skipEmptyLines: true,
          complete: (results) => {

            const parsed = results.data.map(row => ({
              id: Number(row["OBJECTID"]),
              scats:    Number(row["SCATS_SITE"]) || "No SCATS Number",
              lat: parseFloat(row["Y"]),
              lng: parseFloat(row["X"]),
              location: row["SITE_DESC"],
              roadNumber: String(row["ROAD_NBR"] || "").trim(),
              declaredRoad: String(row["DECLARED_R"] || "").trim(),
              localRoad: String(row["LOCAL_ROAD"] || "").trim()
            }))
              .filter(x => !isNaN(x.lat) && !isNaN(x.lng))
              .slice(0, MAX_NODES);

            setSites(parsed);
          }
        });
      });
  }, []);

  // Get flow (ML)
  const flowCache = {};

  const getFlow = async (nodeId) => {

    if (flowCache[nodeId]) return flowCache[nodeId];

    try {
      const res = await fetch(API_PREDICT, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          flows: SAMPLE_FLOWS.map(v => v + (nodeId % 10) * 5),
          interval_index: 32,
          day_of_week: 1,
          is_weekend: false
        })
      });

      const data = await res.json();

      flowCache[nodeId] = data.predicted_flow_per_hour;

      return flowCache[nodeId];

    } catch {
      return 1000;
    }
  };

  const projectedSites = useMemo(() => {
    return projectSites(sites, MAP_WIDTH, MAP_HEIGHT, MAP_PADDING);
  }, [sites]);

  const siteById = useMemo(() => {
    const m = new Map();
    for (const s of projectedSites) m.set(s.id, s);
    return m;
  }, [projectedSites]);

  const rawSiteById = useMemo(() => {
    const m = new Map();
    for (const s of sites) m.set(s.id, s);
    return m;
  }, [sites]);

  const pathPolylines = useMemo(() => {
    return paths
      .map(path => path.map(id => siteById.get(id)).filter(Boolean))
      .filter(points => points.length > 1);
  }, [paths, siteById]);

  const previewEdges = useMemo(() => {
    if (sites.length === 0) return [];

    const neighborsById = buildRoadBasedNeighborMap(sites);
    const undirectedEdgeSet = new Set();

    for (const a of sites) {
      const roadNeighbors = Array.from(neighborsById.get(a.id) || [])
        .map(id => rawSiteById.get(id))
        .filter(Boolean);

      const neighbors = roadNeighbors.length > 0 ? roadNeighbors : getNearest(sites, a, 2);

      for (const n of neighbors) {
        const edgeKey = a.id < n.id ? `${a.id}-${n.id}` : `${n.id}-${a.id}`;
        undirectedEdgeSet.add(edgeKey);
      }
    }

    return Array.from(undirectedEdgeSet)
      .map(edgeKey => {
        const [fromId, toId] = edgeKey.split("-").map(Number);
        const from = siteById.get(fromId);
        const to = siteById.get(toId);
        if (!from || !to) return null;
        return { key: edgeKey, from, to };
      })
      .filter(Boolean);
  }, [sites, rawSiteById, siteById]);

  // calculate travel time from flow and distance
  const computeTravelTime = (flow, distanceKm, intersections = 1) => {

    // Solve Quadractic Equation: a*s^2 + b*s + c = 0
    const a = -1.4648375;
    const b = 93.75;
    const c = -flow;

    const delta = b * b - 4 * a * c;

    if (delta < 0) return 9999; // Error

    const sqrtDelta = Math.sqrt(delta);

    // 2 variables
    const s1 = (-b + sqrtDelta) / (2 * a);
    const s2 = (-b - sqrtDelta) / (2 * a);

    // Choose usable variable (speed > 0 and <= 60)
    let speed = Math.max(s1, s2);

    if (speed > 60) speed = 60;
    if (speed < 5) speed = 5;

    // calculate travel time
    const travelTime =
      (distanceKm / speed) * 60 + intersections * 0.5;

    return travelTime;
  };

  // Graph
  const buildGraph = async () => {

    const nodes = sites.map(s => ({
      id: s.id,
      x: s.lng,
      y: s.lat
    }));

    const edges = [];
    const edgeKeySet = new Set();
    const neighborsById = buildRoadBasedNeighborMap(sites);

    for (let a of sites) {
      // Call flow model once per source node (cached by nodeId).
      const flowA = await getFlow(a.id);
      const roadNeighbors = Array.from(neighborsById.get(a.id) || [])
        .map(id => rawSiteById.get(id))
        .filter(Boolean);

      const neighbors = roadNeighbors.length > 0 ? roadNeighbors : getNearest(sites, a, 2);

      for (let n of neighbors) {
        const flowN = await getFlow(n.id);
        const d = dist(a, n);
        const distanceKm = d * 111;

        const forwardKey = `${a.id}->${n.id}`;
        const backwardKey = `${n.id}->${a.id}`;

        // convert here
        // const travelTime = computeTravelTime(flowA, distanceKm, 1);

        if (!edgeKeySet.has(forwardKey)) {
          edgeKeySet.add(forwardKey);
          edges.push({
            from: a.id,
            to: n.id,
            cost: computeTravelTime(flowA, distanceKm, 1)
          });
        }

        if (!edgeKeySet.has(backwardKey)) {
          edgeKeySet.add(backwardKey);
          edges.push({
            from: n.id,
            to: a.id,
            cost: computeTravelTime(flowN, distanceKm, 1)
          });
        }
      }
    }

    return { nodes, edges };
  };

  // use algorithm
  const handleSolve = async () => {

    if (!origin || destinations.length === 0) return;

    setLoading(true);

    const { nodes, edges } = await buildGraph();

    const resultPaths = yenKShortest(
      nodes,
      edges,
      origin,
      destinations[0],
      2
    );

    if (!resultPaths || resultPaths.length === 0) {
      alert("No path found");
      setLoading(false);
      return;
    }

    setPaths(resultPaths);
    setLoading(false);
  };

  const handleNodeClick = (siteId) => {
    if (!origin) {
      setOrigin(siteId);
      return;
    }

    if (origin === siteId) {
      setOrigin(null);
      return;
    }

    setDestinations([siteId]);
  };

  return (
    <div style={{ height: "100vh", width: "100%" }}>

      <div style={{
        display: 'grid',
        gridTemplateColumns: '1fr 1fr',
        gap: '15px',
        padding: '15px',
        background: '#f9fafb',
        borderRadius: '8px',
        marginBottom: '15px'
      }}>

        {/* <div>
          <b>Origin</b>
          <br></br>
          <select value={origin ?? ""} onChange={e => setOrigin(Number(e.target.value))}>
            <option value="">Select</option>
            {(sites || []).map(s => (
              <option key={s.id} value={s.id}>{s.id} - {s.location}</option>
            ))}
          </select>
        </div> */}
        
        <div>
          <label htmlFor="origin-select" style={{ display: 'block', fontWeight: '600', marginBottom: '6px', color: '#374151' }}>
          Origin {origin && <span style={{ color: '#10b981' }}>✓</span>}
          </label>
          <select 
            id="origin-select"
            value={origin ?? ""} 
            onChange={e => setOrigin(Number(e.target.value))}
            style={{
              width: '100%',
              padding: '10px',
              border: origin ? '2px solid #10b981' : '1px solid #d1d5db',
              borderRadius: '6px',
              fontSize: '13px',
              fontFamily: 'inherit'
            }}
          >
            <option value="">Select Origin</option>
            {(sites || []).map(s => (
              <option key={s.id} value={s.id} disabled={s.id === destinations[0]}>
                {s.scats} - {s.id} - {s.location}
              </option>
            ))}
          </select>
        </div>

        {/* <div>
          <b>Destination</b>
          <br></br>
          <select value={destinations[0] ?? ""} onChange={e => setDestinations([Number(e.target.value)])}>
            <option value="">Select</option>
            {(sites || []).map(s => (
              <option key={s.id} value={s.id}>{s.id} - {s.location}</option>
            ))}
          </select>
        </div> */}
        
        <div>
          <label htmlFor="dest-select" style={{ display: 'block', fontWeight: '600', marginBottom: '6px', color: '#374151' }}>
          Destination {destinations[0] && <span style={{ color: '#10b981' }}>✓</span>}
          </label>
          <select 
            id="dest-select"
            value={destinations[0] ?? ""} 
            onChange={e => setDestinations([Number(e.target.value)])}
            style={{
              width: '100%',
              padding: '10px',
              border: destinations[0] ? '2px solid #10b981' : '1px solid #d1d5db',
              borderRadius: '6px',
              fontSize: '13px',
              fontFamily: 'inherit'
            }}
          >
            <option value="">Select Destination</option>
            {(sites || []).map(s => (
              <option key={s.id} value={s.id} disabled={s.id === origin}>
                {s.scats} - {s.id} - {s.location}
              </option>
            ))}
          </select>
        </div>
        
        {origin && destinations[0] ? (
          <div style={{ padding: '10px', background: '#ecfdf5', color: '#047857', borderRadius: '4px', marginBottom: '10px', fontSize: '13px' }}>
          Ready to find routes from {origin} to {destinations[0]}
          </div>
        ) : (
          <div style={{ padding: '10px', background: '#fef3c7', color: '#92400e', borderRadius: '4px', marginBottom: '10px', fontSize: '13px' }}>
          Please select both Origin and Destination
          </div>
        )}

        {/* <button onClick={handleSolve} disabled={loading}>
          {loading ? "Calculating..." : "Find top 5 paths"}
        </button> */}
        
        <button 
          onClick={handleSolve} 
          disabled={loading || !origin || !destinations[0]}
          style={{
            gridColumn: '1 / -1',
            padding: '12px 16px',
            fontSize: '14px',
            fontWeight: '600',
            border: 'none',
            borderRadius: '6px',
            backgroundColor: loading || !origin || !destinations[0] ? '#d1d5db' : '#0ea5e9',
            color: 'white',
            cursor: loading || !origin || !destinations[0] ? 'not-allowed' : 'pointer',
            transition: 'background-color 0.2s ease',
            opacity: loading || !origin || !destinations[0] ? 0.7 : 1
          }}
        >
          {loading ? "Calculating..." : "Find top 2 paths"}
        </button>
      </div>

      <div
        style={{
          width: "100%",
          height: "calc(100% - 180px)",
          minHeight: "520px",
          borderRadius: "12px",
          border: "1px solid #d1d5db",
          overflow: "hidden",
          background: "linear-gradient(180deg, #f8fafc 0%, #eef2ff 100%)"
        }}
      >
        <svg
          viewBox={`0 0 ${MAP_WIDTH} ${MAP_HEIGHT}`}
          style={{ width: "100%", height: "100%", display: "block" }}
          role="img"
          aria-label="Traffic network map"
        >
          <rect x="0" y="0" width={MAP_WIDTH} height={MAP_HEIGHT} fill="rgba(255,255,255,0.5)" />

          {previewEdges.map(edge => (
            <line
              key={`edge-${edge.key}`}
              x1={edge.from.x}
              y1={edge.from.y}
              x2={edge.to.x}
              y2={edge.to.y}
              stroke="#94a3b8"
              strokeWidth="1.2"
              opacity="0.65"
            />
          ))}

          {pathPolylines.map((points, idx) => (
            <polyline
              key={`route-${idx}`}
              points={points.map(p => `${p.x},${p.y}`).join(" ")}
              fill="none"
              stroke={PATH_COLORS[idx % PATH_COLORS.length]}
              strokeWidth="4"
              strokeLinecap="round"
              strokeLinejoin="round"
              opacity="0.92"
            />
          ))}

          {projectedSites.map(site => {
            const isOrigin = site.id === origin;
            const isDestination = site.id === destinations[0];
            const radius = isOrigin || isDestination ? 7 : 3;
            const fill = isOrigin ? "#16a34a" : isDestination ? "#dc2626" : "#334155";

            return (
              <g key={site.id}>
                <circle
                  cx={site.x}
                  cy={site.y}
                  r={radius}
                  fill={fill}
                  stroke="white"
                  strokeWidth={isOrigin || isDestination ? 2 : 1}
                  style={{ cursor: "pointer" }}
                  onClick={() => handleNodeClick(site.id)}
                >
                  <title>{`${site.id} - ${site.location}`}</title>
                </circle>
                {(isOrigin || isDestination) && (
                  <text
                    x={site.x + 8}
                    y={site.y - 8}
                    fill="#0f172a"
                    fontSize="12"
                    fontWeight="700"
                    paintOrder="stroke"
                    stroke="#ffffff"
                    strokeWidth="2"
                  >
                    {site.id}
                  </text>
                )}
              </g>
            );
          })}
        </svg>
      </div>

    </div>
  );
}