import React, { useEffect, useState } from "react";
import { MapContainer, TileLayer, Marker, Popup, Polyline } from "react-leaflet";
import Papa from "papaparse";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import { yenKShortest } from "./algorithm/yen";

const API_PREDICT = "http://127.0.0.1:5000/predict";

const MAX_NODES = 150;
const SAMPLE_FLOWS = [42, 55, 68, 80, 91, 75, 63, 58, 72, 88, 95, 82]; //sample flows for traffic

// giả định capacity
// const FLOW_CAPACITY = 1800;
 
const originIcon = new L.Icon({
  iconUrl: "https://maps.google.com/mapfiles/ms/icons/green-dot.png",
  iconSize: [32, 32]
});

const destIcon = new L.Icon({
  iconUrl: "https://maps.google.com/mapfiles/ms/icons/red-dot.png",
  iconSize: [32, 32]
});

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
              location: row["SITE_DESC"]
            }))
              .filter(x => !isNaN(x.lat) && !isNaN(x.lng))
              .slice(0, MAX_NODES);

            setSites(parsed);
          }
        });
      });
  }, []);

  // distance between 2 points (degrees, not km)
  const dist = (a, b) => {
    const dx = a.lat - b.lat;
    const dy = a.lng - b.lng;
    return Math.sqrt(dx * dx + dy * dy);
  };

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

  // Nearest Node
  const getNearest = (a, k = 3) => {
    let nearest = [];

    sites.forEach(b => {
      if (a.id === b.id) return;

      const d = dist(a, b);

      if (nearest.length < k) {
        nearest.push({ node: b, d });
        nearest.sort((x, y) => x.d - y.d);
      } else if (d < nearest[nearest.length - 1].d) {
        nearest.pop();
        nearest.push({ node: b, d });
        nearest.sort((x, y) => x.d - y.d);
      }
    });

    return nearest.map(n => n.node);
  };

  // Graph
  const buildGraph = async () => {

    const nodes = sites.map(s => ({
      id: s.id,
      x: s.lng,
      y: s.lat
    }));

    const edges = [];

    for (let a of sites) {
      // Call flow model once per source node (cached by nodeId).
      const flowA = await getFlow(a.id);
      const neighbors = getNearest(a, 3);

      for (let n of neighbors) {
        const flowN = await getFlow(n.id);
        const d = dist(a, n);
        const distanceKm = d * 111;

        // convert here
        // const travelTime = computeTravelTime(flowA, distanceKm, 1);

        edges.push({
          from: a.id,
          to: n.id,
          cost: computeTravelTime(flowA, distanceKm, 1)
        });

        edges.push({
          from: n.id,
          to: a.id,
          cost: computeTravelTime(flowN, distanceKm, 1)
        });
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
      5
    );

    if (!resultPaths || resultPaths.length === 0) {
      alert("No path found");
      setLoading(false);
      return;
    }

    setPaths(resultPaths);
    setLoading(false);
  };

  const getLatLng = (id) => {
    const s = sites.find(x => x.id === id);
    return s ? [s.lat, s.lng] : [0, 0];
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
          {loading ? "Calculating..." : "Find top 5 paths"}
        </button>
      </div>

      <MapContainer center={[-37.81, 145.07]} zoom={13} style={{ height: "100%", width: "100%" }}>

        <TileLayer
          attribution="OpenStreetMap"
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />

        {(sites || []).map((s, i) => {

          let icon = undefined;
          if (s.id === origin) icon = originIcon;
          if (destinations.includes(s.id)) icon = destIcon;

          return (
            <Marker key={i} position={[s.lat, s.lng]} icon={icon || new L.Icon.Default()}>
              <Popup>
                {s.id} <br /> {s.location}
              </Popup>
            </Marker>
          );
        })}

        {paths.map((p, i) => (
          <Polyline key={i} positions={p.map(id => getLatLng(id))} color="red" />
        ))}

      </MapContainer>

    </div>
  );
}