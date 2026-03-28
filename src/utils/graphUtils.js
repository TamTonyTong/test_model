const MISSING_ROAD_VALUES = new Set(["", "MISSING DATA", "9999", "N/A", "NA"]);

export function dist(a, b) {
  const dx = a.lat - b.lat;
  const dy = a.lng - b.lng;
  return Math.sqrt(dx * dx + dy * dy);
}

export function getNearest(sites, a, k = 0) {
  const nearest = [];

  for (const b of sites) {
    if (a.id === b.id) continue;

    const d = dist(a, b);

    if (nearest.length < k) {
      nearest.push({ node: b, d });
      nearest.sort((x, y) => x.d - y.d);
    } else if (k > 0 && d < nearest[nearest.length - 1].d) {
      nearest.pop();
      nearest.push({ node: b, d });
      nearest.sort((x, y) => x.d - y.d);
    }
  }

  return nearest.map(n => n.node);
}

export function normalizeRoadName(name) {
  const cleaned = String(name || "").trim().toUpperCase();
  return MISSING_ROAD_VALUES.has(cleaned) ? "" : cleaned;
}

export function buildRoadBasedNeighborMap(sites) {
  const byRoad = new Map();

  for (const site of sites) {
    const roadNames = [
      normalizeRoadName(site.declaredRoad),
      normalizeRoadName(site.localRoad)
    ].filter(Boolean);

    for (const road of roadNames) {
      if (!byRoad.has(road)) byRoad.set(road, []);
      byRoad.get(road).push(site);
    }
  }

  const neighborsById = new Map();
  for (const site of sites) neighborsById.set(site.id, new Set());

  for (const roadSites of byRoad.values()) {
    if (roadSites.length < 2) continue;

    const latRange = Math.max(...roadSites.map(s => s.lat)) - Math.min(...roadSites.map(s => s.lat));
    const lngRange = Math.max(...roadSites.map(s => s.lng)) - Math.min(...roadSites.map(s => s.lng));

    const ordered = [...roadSites].sort((a, b) => {
      return lngRange > latRange ? a.lng - b.lng : a.lat - b.lat;
    });

    for (let i = 0; i < ordered.length - 1; i++) {
      const from = ordered[i].id;
      const to = ordered[i + 1].id;
      neighborsById.get(from).add(to);
      neighborsById.get(to).add(from);
    }
  }

  return neighborsById;
}

export function projectSites(sites, mapWidth, mapHeight, mapPadding) {
  if (sites.length === 0) return [];

  const minLat = Math.min(...sites.map(s => s.lat));
  const maxLat = Math.max(...sites.map(s => s.lat));
  const minLng = Math.min(...sites.map(s => s.lng));
  const maxLng = Math.max(...sites.map(s => s.lng));

  const latRange = Math.max(maxLat - minLat, 1e-9);
  const lngRange = Math.max(maxLng - minLng, 1e-9);
  const drawWidth = mapWidth - mapPadding * 2;
  const drawHeight = mapHeight - mapPadding * 2;

  return sites.map(site => {
    const x = mapPadding + ((site.lng - minLng) / lngRange) * drawWidth;
    const y = mapHeight - mapPadding - ((site.lat - minLat) / latRange) * drawHeight;
    return { ...site, x, y };
  });
}
