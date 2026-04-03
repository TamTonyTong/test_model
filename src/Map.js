import React, { useEffect, useMemo, useRef, useState } from "react";
import Papa from "papaparse";
import {
    CircleMarker,
    MapContainer,
    Pane,
    Polyline,
    TileLayer,
    Tooltip,
    ZoomControl
} from "react-leaflet";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import { yenKShortest } from "./algorithm/yen";
import { SAMPLE_FLOWS } from "./utils/trafficDefaults";
import {
    buildRoadBasedNeighborMap,
    getNearest
} from "./utils/graphUtils";
import {
    buildGraphForYen,
    formatTravelTime,
    recalculatePathsWithPredictedFlows
} from "./utils/mapRoutingUtils";
import "./styles/Map.css";

const API_PREDICT = "http://127.0.0.1:5000/predict";
const MODEL_OPTIONS = [
    { value: "CNN-LSTM", label: "CNN-LSTM" },
    { value: "GRU", label: "GRU" },
    { value: "LSTM", label: "LSTM" },
    { value: "LSTM-GRU", label: "LSTM-GRU" },
];

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
    const [originInput, setOriginInput] = useState("");
    const [destinationInput, setDestinationInput] = useState("");
    const [selectedModel, setSelectedModel] = useState("CNN-LSTM");
    const [originOpen, setOriginOpen] = useState(false);
    const [destinationOpen, setDestinationOpen] = useState(false);
    const [paths, setPaths] = useState([]);
    const [pathDetails, setPathDetails] = useState([]);  // {path, totalTime}
    const [backendDisconnected, setBackendDisconnected] = useState(false);
    const [loading, setLoading] = useState(false);
    const mapRef = useRef(null);
    const flowCacheRef = useRef(new Map());
    const backendWarningLoggedRef = useRef(false);

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

    useEffect(() => {
        flowCacheRef.current.clear();
        setPaths([]);
        setPathDetails([]);
    }, [selectedModel]);

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
                is_weekend: false,
                model: selectedModel
            };

            if (scatsSiteId !== undefined) {
                payload.site_id = scatsSiteId;
            }

            const res = await fetch(API_PREDICT, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            });

            if (!res.ok) {
                throw new Error(`Predict API returned ${res.status}`);
            }

            const data = await res.json();
            const rawFlow = data?.predicted_flow_per_hour;
            const parsedFlow = typeof rawFlow === "number"
                ? rawFlow
                : (typeof rawFlow === "string" && rawFlow.trim() !== "" ? Number(rawFlow) : NaN);
            const safeValue = Number.isFinite(parsedFlow) ? parsedFlow : 1000;

            flowCacheRef.current.set(nodeId, safeValue);

            return safeValue;

        } catch (error) {
            if (!backendWarningLoggedRef.current) {
                console.warn("Prediction backend is not connected. Falling back to default flow 1000.", error);
                backendWarningLoggedRef.current = true;
            }

            setBackendDisconnected(true);
            flowCacheRef.current.set(nodeId, 1000);
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

    const resolveFromInput = (value, excludedId) => {
        const txt = value.trim();
        if (!txt) return null;

        const asNumber = Number(txt);
        if (Number.isInteger(asNumber) && rawSiteById.has(asNumber) && asNumber !== excludedId) {
            return rawSiteById.get(asNumber);
        }

        const lower = txt.toLowerCase();
        return sites.find(site =>
            site.id !== excludedId && (
                site.location.toLowerCase() === lower ||
                `${site.id} - ${site.location}`.toLowerCase() === lower
            )
        ) || null;
    };

    const originSuggestions = useMemo(() => {
        const q = originInput.trim().toLowerCase();
        return sites
            .filter(site => site.id !== destinations[0])
            .filter(site => {
                if (!q) return true;
                return String(site.id).includes(q) || site.location.toLowerCase().includes(q);
            })
            .slice(0, 20);
    }, [sites, originInput, destinations]);

    const destinationSuggestions = useMemo(() => {
        const q = destinationInput.trim().toLowerCase();
        return sites
            .filter(site => site.id !== origin)
            .filter(site => {
                if (!q) return true;
                return String(site.id).includes(q) || site.location.toLowerCase().includes(q);
            })
            .slice(0, 20);
    }, [sites, destinationInput, origin]);

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

    const mapBounds = useMemo(() => {
        if (sites.length === 0) return null;

        return L.latLngBounds(sites.map(site => [site.lat, site.lng]));
    }, [sites]);

    const selectedPathDuration = useMemo(() => {
        if (pathDetails.length === 0) return null;
        return pathDetails[0].totalTime.toFixed(1);
    }, [pathDetails]);

    // Orchestrate the complete flow:
    // 1. Build graph with static cost
    // 2. Use Yen algorithm to get top 5 candidate paths
    // 3-7. Recalculate paths with predicted flows and sort
    // replace handleSolve with this debug version

    const handleSolve = async () => {
        if (!origin || destinations.length === 0) return;

        setBackendDisconnected(false);
        backendWarningLoggedRef.current = false;
        setLoading(true);
        setPaths([]);
        setPathDetails([]);

        try {
            console.log("========== ROUTE SOLVE START ==========");
            console.log("Origin:", origin);
            console.log("Destination:", destinations[0]);

            // Step 1: Build graph
            const { nodes, edges } = buildGraphForYen(sites, rawSiteById);

            console.log("Graph stats:");
            console.log("- nodes:", nodes.length);
            console.log("- edges:", edges.length);
            console.log("- sample edges:", edges.slice(0, 10));

            // sanity check edge costs
            const suspiciousEdges = edges.filter(
                e => !Number.isFinite(e.cost) || e.cost <= 0 || e.cost > 50
            );

            if (suspiciousEdges.length > 0) {
                console.warn("Suspicious edges found:");
                console.table(suspiciousEdges.slice(0, 20));
            }

            // Step 2: Yen shortest paths
            const candidatePaths = yenKShortest(
                nodes,
                edges,
                origin,
                destinations[0],
                5
            );

            console.log("========== CANDIDATE PATHS ==========");
            console.log(candidatePaths);

            if (!candidatePaths || candidatePaths.length === 0) {
                alert("No path found");
                return;
            }

            // inspect each path before recalculation
            candidatePaths.forEach((path, idx) => {
                const uniqueCount = new Set(path).size;
                const hasCycle = uniqueCount !== path.length;

                console.group(`Path ${idx + 1}`);
                console.log("Nodes:", path);
                console.log("Path length:", path.length);
                console.log("Unique nodes:", uniqueCount);
                console.log("Has cycle:", hasCycle);

                if (hasCycle) {
                    const duplicates = path.filter(
                        (node, i) => path.indexOf(node) !== i
                    );
                    console.warn("Repeated nodes:", [...new Set(duplicates)]);
                }

                // estimate geometric distance manually
                let totalDistance = 0;

                for (let i = 0; i < path.length - 1; i++) {
                    const from = rawSiteById.get(path[i]);
                    const to = rawSiteById.get(path[i + 1]);

                    if (!from || !to) continue;

                    const latDiff = from.lat - to.lat;
                    const lngDiff = from.lng - to.lng;

                    const degreeDist = Math.sqrt(latDiff * latDiff + lngDiff * lngDiff);
                    const kmDist = degreeDist * 111;

                    totalDistance += kmDist;

                    if (kmDist > 20) {
                        console.warn(
                            `Large segment in path ${idx + 1}:`,
                            {
                                from: from.id,
                                to: to.id,
                                fromLatLng: [from.lat, from.lng],
                                toLatLng: [to.lat, to.lng],
                                degreeDist,
                                kmDist
                            }
                        );
                    }
                }

                console.log("Approx total geometric distance (km):", totalDistance);
                console.groupEnd();
            });

            // remove cyclic / absurd paths before recalculating
            const filteredPaths = candidatePaths.filter(path => {
                const noCycle = path.length === new Set(path).size;
                const reasonableLength = path.length < 150;
                return noCycle && reasonableLength;
            });

            console.log("Filtered paths count:", filteredPaths.length);

            if (filteredPaths.length === 0) {
                console.error("All candidate paths were invalid.");
                alert("No valid route found");
                return;
            }

            // DEBUG version of recalculation
            const recalculatedPaths = await recalculatePathsWithPredictedFlows(
                filteredPaths,
                {
                    rawSiteById,
                    flowCache: flowCacheRef.current,
                    prefetchFlows,
                    debug: true
                }
            );

            console.log("========== FINAL PATH DETAILS ==========");
            recalculatedPaths.forEach((pd, idx) => {
                console.group(`Final Path ${idx + 1}`);
                console.log("Path:", pd.path);
                console.log("Node count:", pd.path.length);
                console.log("Total time:", pd.totalTime, "minutes");
                console.log("Formatted:", formatTravelTime(pd.totalTime));
                console.groupEnd();
            });

            setPaths(recalculatedPaths.map(pd => pd.path));
            setPathDetails(recalculatedPaths);

        } catch (error) {
            console.error("========== ROUTE ERROR ==========");
            console.error(error);
            alert("Error calculating paths");
        } finally {
            console.log("========== ROUTE SOLVE END ==========");
            setLoading(false);
        }
    };

    const handleNodeClick = (siteId) => {
        if (!origin) {
            setOrigin(siteId);
            setOriginInput(String(siteId));
            setPaths([]);
            setPathDetails([]);
            return;
        }

        if (origin === siteId) {
            setOrigin(null);
            setOriginInput("");
            setPaths([]);
            setPathDetails([]);
            return;
        }

        setDestinations([siteId]);
        setDestinationInput(String(siteId));
        setPaths([]);
        setPathDetails([]);
    };

    const handleResetView = () => {
        if (!mapRef.current || !mapBounds) return;
        mapRef.current.fitBounds(mapBounds, { padding: [30, 30] });
    };

    const handleOriginInputChange = (value) => {
        setOriginInput(value);
        setPaths([]);
        setPathDetails([]);

        if (value.trim() === "") {
            setOrigin(null);
            return;
        }

        const resolved = resolveFromInput(value, destinations[0]);
        if (resolved) {
            setOrigin(resolved.id);
        } else {
            setOrigin(null);
        }
    };

    const handleDestinationInputChange = (value) => {
        setDestinationInput(value);
        setPaths([]);
        setPathDetails([]);

        if (value.trim() === "") {
            setDestinations([]);
            return;
        }

        const resolved = resolveFromInput(value, origin);
        if (resolved) {
            setDestinations([resolved.id]);
        } else {
            setDestinations([]);
        }
    };

    return (
        <div className="map-page">
            <section className="map-controls">
                <div className="map-controls-grid">
                    <div>
                        <label className="map-label" htmlFor="origin-input">
                            Origin {origin && <span className="map-ok">selected</span>}
                        </label>
                        <div className="map-combobox">
                            <input
                                className="map-select"
                                id="origin-input"
                                type="text"
                                placeholder="Type OBJECTID or location"
                                value={originInput}
                                onFocus={() => setOriginOpen(true)}
                                onBlur={() => setTimeout(() => setOriginOpen(false), 120)}
                                onChange={e => handleOriginInputChange(e.target.value)}
                            />
                            {originOpen && originSuggestions.length > 0 && (
                                <div className="map-suggest-list" role="listbox">
                                    {originSuggestions.map(site => (
                                        <button
                                            key={`origin-${site.id}`}
                                            type="button"
                                            className="map-suggest-item"
                                            onMouseDown={() => {
                                                setOrigin(site.id);
                                                setOriginInput(String(site.id));
                                                setPaths([]);
                                                setPathDetails([]);
                                                setOriginOpen(false);
                                            }}
                                        >
                                            <span className="map-suggest-id">{site.id}</span>
                                            <span className="map-suggest-name">{site.location}</span>
                                        </button>
                                    ))}
                                </div>
                            )}
                        </div>
                    </div>

                    <div>
                        <label className="map-label" htmlFor="dest-input">
                            Destination {destinations[0] && <span className="map-ok">selected</span>}
                        </label>
                        <div className="map-combobox">
                            <input
                                className="map-select"
                                id="dest-input"
                                type="text"
                                placeholder="Type OBJECTID or location"
                                value={destinationInput}
                                onFocus={() => setDestinationOpen(true)}
                                onBlur={() => setTimeout(() => setDestinationOpen(false), 120)}
                                onChange={e => handleDestinationInputChange(e.target.value)}
                            />
                            {destinationOpen && destinationSuggestions.length > 0 && (
                                <div className="map-suggest-list" role="listbox">
                                    {destinationSuggestions.map(site => (
                                        <button
                                            key={`destination-${site.id}`}
                                            type="button"
                                            className="map-suggest-item"
                                            onMouseDown={() => {
                                                setDestinations([site.id]);
                                                setDestinationInput(String(site.id));
                                                setPaths([]);
                                                setPathDetails([]);
                                                setDestinationOpen(false);
                                            }}
                                        >
                                            <span className="map-suggest-id">{site.id}</span>
                                            <span className="map-suggest-name">{site.location}</span>
                                        </button>
                                    ))}
                                </div>
                            )}
                        </div>
                    </div>

                    <div>
                        <label className="map-label" htmlFor="model-select">
                            Model {selectedModel && <span className="map-ok">active</span>}
                        </label>
                        <select
                            className="map-select"
                            id="model-select"
                            value={selectedModel}
                            onChange={e => setSelectedModel(e.target.value)}
                        >
                            {MODEL_OPTIONS.map(option => (
                                <option key={option.value} value={option.value}>
                                    {option.label}
                                </option>
                            ))}
                        </select>
                    </div>
                </div>

                <div className="map-actions">
                    <div className="map-status-wrap">
                        <p className={`map-status ${origin && destinations[0] ? "ready" : "pending"}`}>
                            {origin && destinations[0]
                                ? `Ready to find routes from ${origin} to ${destinations[0]}`
                                : "Select both origin and destination"}
                        </p>
                        {backendDisconnected && (
                            <p className="map-warning" role="alert">
                                Backend is not connected, the result is just based on search algorithm without traffic flow predicted.
                            </p>
                        )}
                    </div>

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
                        <span className="map-summary-metric map-summary-metric-nodes">
                            Nodes: <b>{sites.length}</b>
                        </span>
                        <span className="map-summary-metric map-summary-metric-edges">
                            Candidate edges: <b>{edgesById.length}</b>
                        </span>
                        <span className="map-summary-metric map-summary-metric-route">
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
                            const tooltipKey = isOrigin ? `origin-${site.id}` : isDestination ? `destination-${site.id}` : `site-${site.id}`;
                            const tooltipText = isOrigin || isDestination
                                ? `${isOrigin ? "Origin" : "Destination"}: ${site.id}`
                                : `${site.id} - ${site.location}`;

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
                                    <Tooltip
                                        key={tooltipKey}
                                        direction="top"
                                        offset={[0, -8]}
                                        permanent={isOrigin || isDestination}
                                    >
                                        {tooltipText}
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