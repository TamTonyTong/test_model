import { buildRoadBasedNeighborMap, dist, getNearest } from "./graphUtils";

export function formatTravelTime(minutes) {
  if (minutes >= 1440) {
    const days = (minutes / 1440).toFixed(1);
    return `${days} day${days !== "1.0" ? "s" : ""}`;
  }

  if (minutes >= 60) {
    const hours = (minutes / 60).toFixed(1);
    return `${hours} hour${hours !== "1.0" ? "s" : ""}`;
  }

  return `${minutes.toFixed(1)} min`;
}

export function computeTravelTime(flow, distanceKm, intersections = 1) {
  const a = -1.4648375;
  const b = 93.75;
  const c = -flow;

  const delta = b * b - 4 * a * c;
  if (delta < 0) return 9999;

  const sqrtDelta = Math.sqrt(delta);
  const s1 = (-b + sqrtDelta) / (2 * a);
  const s2 = (-b - sqrtDelta) / (2 * a);

  let speed = Math.max(s1, s2);
  if (speed > 60) speed = 60;
  if (speed < 5) speed = 5;

  return (distanceKm / speed) * 60 + intersections * 0.5;
}

export function extractUniqueNodeIdsFromPaths(pathsList) {
  const uniqueNodeIds = new Set();
  for (const path of pathsList) {
    for (const nodeId of path) {
      uniqueNodeIds.add(nodeId);
    }
  }
  return Array.from(uniqueNodeIds);
}

export const recalculatePathsWithPredictedFlows = async (
  pathsList,
  {
    rawSiteById,
    flowCache,
    prefetchFlows,
    debug = false
  }
) => {
  const uniqueNodeIds = [...new Set(pathsList.flat())];

  if (debug) {
    console.log("Prefetching flows for", uniqueNodeIds.length, "nodes");
  }

  await prefetchFlows(uniqueNodeIds);

  const results = [];

  for (const [pathIndex, path] of pathsList.entries()) {
    let totalTime = 0;
    let totalDistance = 0;

    if (debug) {
      console.group(`Recalculate path ${pathIndex + 1}`);
    }

    for (let i = 0; i < path.length - 1; i++) {
      const from = rawSiteById.get(path[i]);
      const to = rawSiteById.get(path[i + 1]);

      if (!from || !to) continue;

      const latDiff = from.lat - to.lat;
      const lngDiff = from.lng - to.lng;

      const distanceKm =
        Math.sqrt(latDiff * latDiff + lngDiff * lngDiff) * 111;

      totalDistance += distanceKm;

      const flow = flowCache.get(from.id) ?? 1000;

      // same quadratic as current code
      const a = -1.4648375;
      const b = 93.75;
      const c = -flow;

      const delta = b * b - 4 * a * c;

      let speed = 30;

      if (delta >= 0) {
        const sqrtDelta = Math.sqrt(delta);
        const s1 = (-b + sqrtDelta) / (2 * a);
        const s2 = (-b - sqrtDelta) / (2 * a);

        speed = Math.max(s1, s2);

        if (speed > 60) speed = 60;
        if (speed < 5) speed = 5;
      }

      const segmentTime = (distanceKm / speed) * 60 + 0.5;

      totalTime += segmentTime;

      if (debug) {
        console.log({
          from: from.id,
          to: to.id,
          distanceKm: Number(distanceKm.toFixed(3)),
          flow,
          speed: Number(speed.toFixed(2)),
          segmentTime: Number(segmentTime.toFixed(2))
        });
      }

      // detect clearly broken values
      if (distanceKm > 30 || segmentTime > 60) {
        console.warn("Suspicious segment:", {
          from: from.id,
          to: to.id,
          distanceKm,
          flow,
          speed,
          segmentTime
        });
      }
    }

    if (debug) {
      console.log("Total distance:", totalDistance, "km");
      console.log("Total time:", totalTime, "minutes");
      console.groupEnd();
    }

    results.push({
      path,
      totalTime,
      totalDistance
    });
  }

  results.sort((a, b) => a.totalTime - b.totalTime);

  return results;
};

export function buildGraphForYen(sites, rawSiteById) {
  const nodes = sites.map(s => ({
    id: s.id,
    x: s.lng,
    y: s.lat
  }));

  const edges = [];
  const edgeKeySet = new Set();
  const draftEdges = [];
  const neighborsById = buildRoadBasedNeighborMap(sites);

  for (const a of sites) {
    const roadNeighbors = Array.from(neighborsById.get(a.id) || [])
      .map(id => rawSiteById.get(id))
      .filter(Boolean);

    const neighbors = roadNeighbors.length > 0 ? roadNeighbors : getNearest(sites, a, 2);

    for (const n of neighbors) {
      const distanceKm = dist(a, n) * 111;
      const forwardKey = `${a.id}->${n.id}`;
      const backwardKey = `${n.id}->${a.id}`;

      if (!edgeKeySet.has(forwardKey)) {
        edgeKeySet.add(forwardKey);
        draftEdges.push({ from: a.id, to: n.id, distanceKm });
      }

      if (!edgeKeySet.has(backwardKey)) {
        edgeKeySet.add(backwardKey);
        draftEdges.push({ from: n.id, to: a.id, distanceKm });
      }
    }
  }

  for (const edge of draftEdges) {
    edges.push({
      from: edge.from,
      to: edge.to,
      cost: edge.distanceKm
    });
  }

  return { nodes, edges };
}
