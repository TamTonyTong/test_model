import React, { useEffect, useMemo, useRef, useState } from "react";
import Papa from "papaparse";
import {
  CircleMarker,
  MapContainer,
  Pane,
  Polyline,
  TileLayer,
  Tooltip,
  ZoomControl,
  useMap
} from "react-leaflet";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import { yenKShortest } from "./algorithm/yen";
import { SAMPLE_FLOWS } from "./utils/trafficDefaults";
import {
  buildRoadBasedNeighborMap,
  dist,
  getNearest
} from "./utils/graphUtils";
import "./styles/Map.css";

const API_PREDICT = "http://127.0.0.1:5000/predict";

const MAX_NODES = 5000;
const FLOW_FETCH_CONCURRENCY = 18;
const PATH_COLORS = ["#ef4444", "#0ea5e9", "#f59e0b", "#22c55e", "#8b5cf6"];
const DEFAULT_CENTER = [-37.8136, 144.9631];
const DEFAULT_ZOOM = 12;

// giả định capacity
// const FLOW_CAPACITY = 1800;

export default function MapPage() {
  const [sites, setSites] = useState([]);
  const [origin, setOrigin] = useState(null);
  const [destinations, setDestinations] = useState([]);
  const [paths, setPaths] = useState([]);
  const [pathDetails, setPathDetails] = useState([]);  // {path, totalTime}
  const [loading, setLoading] = useState(false);
  const mapRef = useRef(null);
  const flowCacheRef = useRef(new Map());

  useEffect(() => {
    fetch("/mapInfo/Traffic_Count_Locations_FILTERED.csv")
      .then(res => res.text())
      .then(csv => {
        Papa.parse(csv, {
          header: true,
          skipEmptyLines: true,
          complete: (results) => {

            const parsed = results.data.map(row => ({
              id: Number(row["OBJECTID"]),
              scats: Number.isFinite(Number(row["TFM_ID"])) ? Number(row["TFM_ID"]) : null,
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

  const getFlow = async (nodeId) => {
    if (flowCacheRef.current.has(nodeId)) {
      return flowCacheRef.current.get(nodeId);
    }

    const site = rawSiteById.get(nodeId);
    const scatsSiteId = Number.isInteger(site?.scats) ? site.scats : undefined;

    try {
      const payload = {
        flows: SAMPLE_FLOWS.map(v => v + ((scatsSiteId ?? nodeId) % 10) * 5),
        interval_index: 32,
        day_of_week: 1,
        is_weekend: false
      };

      if (scatsSiteId !== undefined) {
        payload.site_id = scatsSiteId;
      }

      const res = await fetch(API_PREDICT, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });

      const data = await res.json();
      const value = Number(data.predicted_flow_per_hour);
      const safeValue = Number.isFinite(value) ? value : 1000;

      flowCacheRef.current.set(nodeId, safeValue);

      return safeValue;

    } catch {
      return 1000;
    }
  };

  const prefetchFlows = async (nodeIds) => {
    const uniqueIds = Array.from(new Set(nodeIds)).filter(id => !flowCacheRef.current.has(id));

    if (uniqueIds.length === 0) return;

    let cursor = 0;
    const workerCount = Math.min(FLOW_FETCH_CONCURRENCY, uniqueIds.length);

    const workers = Array.from({ length: workerCount }, () => (async () => {
      while (cursor < uniqueIds.length) {
        const index = cursor;
        cursor += 1;
        const nodeId = uniqueIds[index];
        await getFlow(nodeId);
      }
    })());

    await Promise.all(workers);
  };

  const rawSiteById = useMemo(() => {
    const m = new Map();
    for (const s of sites) m.set(s.id, s);
    return m;
  }, [sites]);

  const edgesById = useMemo(() => {
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
        const from = rawSiteById.get(fromId);
        const to = rawSiteById.get(toId);

        if (!from || !to) return null;

        return {
          key: edgeKey,
          points: [
            [from.lat, from.lng],
            [to.lat, to.lng]
          ]
        };
      })
      .filter(Boolean);
  }, [sites, rawSiteById]);

  const pathPolylines = useMemo(() => {
    return paths
      .map(path => path.map(id => rawSiteById.get(id)).filter(Boolean))
      .map(points => points.map(p => [p.lat, p.lng]))
      .filter(points => points.length > 1);
  }, [paths, rawSiteById]);

  // Extract all unique SCATS sites from multiple paths
  const extractUniqueSitesFromPaths = (pathsList) => {
    const uniqueNodeIds = new Set();
    for (const path of pathsList) {
      for (const nodeId of path) {
        uniqueNodeIds.add(nodeId);
      }
    }
    return Array.from(uniqueNodeIds);
  };

  // Recalculate travel times for paths using predicted flows
  const recalculatePathsWithPredictedFlows = async (pathsList) => {
    // Step 3: Get unique SCATS sites
    const uniqueNodeIds = extractUniqueSitesFromPaths(pathsList);

    // Step 4: Predict flows for these sites
    await prefetchFlows(uniqueNodeIds);

    // Step 5: Calculate total travel time for each path
    const recalculatedPaths = [];
    for (const path of pathsList) {
      let totalTime = 0;
      for (let i = 0; i < path.length - 1; i++) {
        const fromNodeId = path[i];
        const toNodeId = path[i + 1];
        const from = rawSiteById.get(fromNodeId);
        const to = rawSiteById.get(toNodeId);

        if (!from || !to) continue;

        const distanceKm = dist(from, to) * 111;
        const flow = flowCacheRef.current.get(fromNodeId) ?? 1000;
        const segmentTime = computeTravelTime(flow, distanceKm, 1);
        totalTime += segmentTime;
      }
      recalculatedPaths.push({ path, totalTime });
    }

    // Step 6: Sort by time (least to most consuming)
    recalculatedPaths.sort((a, b) => a.totalTime - b.totalTime);

    return recalculatedPaths;
  };

  const mapBounds = useMemo(() => {
    if (sites.length === 0) return null;

    return L.latLngBounds(sites.map(site => [site.lat, site.lng]));
  }, [sites]);

  const selectedPathDuration = useMemo(() => {
    if (pathDetails.length === 0) return null;
    return pathDetails[0].totalTime.toFixed(1);
  }, [pathDetails]);

  // Format travel time from minutes to days/hours/minutes
  const formatTravelTime = (minutes) => {
    if (minutes >= 1440) {
      const days = (minutes / 1440).toFixed(1);
      return `${days} day${days !== '1.0' ? 's' : ''}`;
    } else if (minutes >= 60) {
      const hours = (minutes / 60).toFixed(1);
      return `${hours} hour${hours !== '1.0' ? 's' : ''}`;
    } else {
      return `${minutes.toFixed(1)} min`;
    }
  };

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
    const draftEdges = [];
    const flowNodeIds = [];
    const neighborsById = buildRoadBasedNeighborMap(sites);

    for (let a of sites) {
      const roadNeighbors = Array.from(neighborsById.get(a.id) || [])
        .map(id => rawSiteById.get(id))
        .filter(Boolean);

      const neighbors = roadNeighbors.length > 0 ? roadNeighbors : getNearest(sites, a, 2);

      for (let n of neighbors) {
        const d = dist(a, n);
        const distanceKm = d * 111;

        const forwardKey = `${a.id}->${n.id}`;
        const backwardKey = `${n.id}->${a.id}`;

        if (!edgeKeySet.has(forwardKey)) {
          edgeKeySet.add(forwardKey);
          draftEdges.push({
            from: a.id,
            to: n.id,
            distanceKm
          });
          flowNodeIds.push(a.id);
        }

        if (!edgeKeySet.has(backwardKey)) {
          edgeKeySet.add(backwardKey);
          draftEdges.push({
            from: n.id,
            to: a.id,
            distanceKm
          });
          flowNodeIds.push(n.id);
        }
      }
    }

    await prefetchFlows(flowNodeIds);

    for (const edge of draftEdges) {
      const flow = flowCacheRef.current.get(edge.from) ?? 1000;
      edges.push({
        from: edge.from,
        to: edge.to,
        cost: computeTravelTime(flow, edge.distanceKm, 1)
      });
    }

    return { nodes, edges };
  };

  // Orchestrate the complete flow:
  // 1. Build graph with static cost
  // 2. Use Yen algorithm to get top 5 candidate paths
  // 3-7. Recalculate paths with predicted flows and sort
  const handleSolve = async () => {
    if (!origin || destinations.length === 0) return;

    setLoading(true);
    setPaths([]);
    setPathDetails([]);

    try {
      // Step 1: Build graph with static cost (distance-based)
      const { nodes, edges } = await buildGraph();

      // Step 2: Use Yen algorithm to take top 5 candidate paths
      const candidatePaths = yenKShortest(
        nodes,
        edges,
        origin,
        destinations[0],
        5
      );

      if (!candidatePaths || candidatePaths.length === 0) {
        alert("No path found");
        setLoading(false);
        return;
      }

      // Steps 3-6: Recalculate paths with predicted flows and sort
      const recalculatedPaths = await recalculatePathsWithPredictedFlows(candidatePaths);

      // Update state with sorted paths
      setPaths(recalculatedPaths.map(pd => pd.path));
      setPathDetails(recalculatedPaths);

    } catch (error) {
      console.error("Error during path calculation:", error);
      alert("Error calculating paths");
    } finally {
      setLoading(false);
    }
  };

  const handleNodeClick = (siteId) => {
    if (!origin) {
      setOrigin(siteId);
      setPaths([]);
      setPathDetails([]);
      return;
    }

    if (origin === siteId) {
      setOrigin(null);
      setPaths([]);
      setPathDetails([]);
      return;
    }

    setDestinations([siteId]);
    setPaths([]);
    setPathDetails([]);
  };

  const handleResetView = () => {
    if (!mapRef.current || !mapBounds) return;
    mapRef.current.fitBounds(mapBounds, { padding: [30, 30] });
  };

  const ResetBoundsOnData = ({ bounds }) => {
    const map = useMap();

    useEffect(() => {
      if (!bounds) return;
      map.fitBounds(bounds, { padding: [30, 30] });
    }, [map, bounds]);

    return null;
  };

  return (
    <div className="map-page">
      <section className="map-controls">
        <div className="map-controls-grid">
          <div>
            <label className="map-label" htmlFor="origin-select">
              Origin {origin && <span className="map-ok">selected</span>}
            </label>
            <select
              className="map-select"
              id="origin-select"
              value={origin ?? ""}
              onChange={e => {
                setOrigin(Number(e.target.value));
                setPaths([]);
                setPathDetails([]);
              }}
            >
              <option value="">Select Origin</option>
              {(sites || []).map(s => (
                <option key={s.id} value={s.id} disabled={s.id === destinations[0]}>
                  {(s.scats ?? "N/A")} - {s.id} - {s.location}
                </option>
              ))}
            </select>
          </div>

          <div>
            <label className="map-label" htmlFor="dest-select">
              Destination {destinations[0] && <span className="map-ok">selected</span>}
            </label>
            <select
              className="map-select"
              id="dest-select"
              value={destinations[0] ?? ""}
              onChange={e => {
                setDestinations([Number(e.target.value)]);
                setPaths([]);
                setPathDetails([]);
              }}
            >
              <option value="">Select Destination</option>
              {(sites || []).map(s => (
                <option key={s.id} value={s.id} disabled={s.id === origin}>
                  {(s.scats ?? "N/A")} - {s.id} - {s.location}
                </option>
              ))}
            </select>
          </div>
        </div>

        <div className="map-actions">
          <p className={`map-status ${origin && destinations[0] ? "ready" : "pending"}`}>
            {origin && destinations[0]
              ? `Ready to find routes from ${origin} to ${destinations[0]}`
              : "Select both origin and destination"}
          </p>

          <div className="map-action-buttons">
            <button
              className="map-btn map-btn-secondary"
              onClick={handleResetView}
              disabled={!mapBounds}
            >
              Reset View
            </button>

            <button
              className="map-btn map-btn-primary"
              onClick={handleSolve}
              disabled={loading || !origin || !destinations[0]}
            >
              {loading ? "Calculating..." : "Find Top 5 Paths"}
            </button>
          </div>
        </div>
      </section>

      <section className="map-canvas-wrap">
        <div className="map-summary">
          <div className="map-summary-metrics">
            <span>
              Nodes: <b>{sites.length}</b>
            </span>
            <span>
              Candidate edges: <b>{edgesById.length}</b>
            </span>
            <span>
              Best route (est): <b>{selectedPathDuration ? formatTravelTime(parseFloat(selectedPathDuration)) : "-"}</b>
            </span>
          </div>
          {pathDetails.length > 0 && (
            <div className="path-times-summary">
              <div className="path-times-header">Route Times (sorted):</div>
              <div className="path-times-list">
                {pathDetails.map((pathDetail, idx) => (
                  <div key={idx} className="path-time-item">
                    <span className="path-route-label">
                      <i
                        className="path-color-chip"
                        style={{ backgroundColor: PATH_COLORS[idx % PATH_COLORS.length] }}
                      />
                      <span className="path-number">Route {idx + 1}:</span>
                    </span>
                    <span className="path-time">{formatTravelTime(pathDetail.totalTime)}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        <MapContainer
          center={DEFAULT_CENTER}
          zoom={DEFAULT_ZOOM}
          className="leaflet-map"
          zoomControl={false}
          whenReady={(event) => {
            mapRef.current = event.target;
          }}
        >
          <ResetBoundsOnData bounds={mapBounds} />
          <ZoomControl position="bottomright" />

          <TileLayer
            url="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png"
            attribution="&copy; OpenStreetMap contributors &copy; CARTO"
          />

          <Pane name="edges" style={{ zIndex: 350 }}>
            {edgesById.map(edge => (
              <Polyline
                key={`edge-${edge.key}`}
                positions={edge.points}
                pathOptions={{ color: "#64748b", weight: 1.2, opacity: 0.45 }}
              />
            ))}
          </Pane>

          <Pane name="paths" style={{ zIndex: 500 }}>
            {pathPolylines.map((points, idx) => (
              <Polyline
                key={`route-${idx}`}
                positions={points}
                pathOptions={{
                  color: PATH_COLORS[idx % PATH_COLORS.length],
                  weight: 6,
                  opacity: 0.95,
                  lineCap: "round",
                  lineJoin: "round"
                }}
              />
            ))}
          </Pane>

          <Pane name="nodes" style={{ zIndex: 650 }}>
            {sites.map(site => {
              const isOrigin = site.id === origin;
              const isDestination = site.id === destinations[0];

              return (
                <CircleMarker
                  key={site.id}
                  center={[site.lat, site.lng]}
                  radius={isOrigin || isDestination ? 9 : 4}
                  pathOptions={{
                    fillColor: isOrigin ? "#10b981" : isDestination ? "#ef4444" : "#0f172a",
                    color: "#ffffff",
                    weight: isOrigin || isDestination ? 2 : 1,
                    fillOpacity: 0.92
                  }}
                  eventHandlers={{
                    click: () => handleNodeClick(site.id)
                  }}
                >
                  {(isOrigin || isDestination) && (
                    <Tooltip direction="top" offset={[0, -8]} permanent>
                      {isOrigin ? "Origin" : "Destination"}: {site.id}
                    </Tooltip>
                  )}
                  <Tooltip direction="top">
                    {(site.scats ?? "N/A")} - {site.id} - {site.location}
                  </Tooltip>
                </CircleMarker>
              );
            })}
          </Pane>
        </MapContainer>

        <div className="map-legend">
          <span><i className="dot origin" /> Origin</span>
          <span><i className="dot destination" /> Destination</span>
          <span><i className="dot node" /> Node</span>
          <span><i className="line route" /> Suggested Route</span>
        </div>
      </section>
    </div>
  );
}