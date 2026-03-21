// Yen's Algorithm (Top-K shortest paths)
// reuse astar()

import { astar } from "./Astar";

// helper: calculate path cost
function pathCost(path, edges) {
  let cost = 0;

  for (let i = 0; i < path.length - 1; i++) {
    const e = edges.find(
      x => x.from === path[i] && x.to === path[i + 1]
    );
    if (!e) return Infinity;
    cost += e.cost;
  }

  return cost;
}

// clone edges and remove edge
function removeEdge(edges, from, to) {
  return edges.filter(e => !(e.from === from && e.to === to));
}

// main Yen
export function yenKShortest(nodes, edges, start, goal, K = 5) {

  const A = []; // results
  const B = []; // candidates

  // 1. First shortest path 
  const firstPath = astar(nodes, edges, start, goal);
  if (!firstPath) return [];

  A.push(firstPath);

  for (let k = 1; k < K; k++) {

    const prevPath = A[k - 1];

    for (let i = 0; i < prevPath.length - 1; i++) {

      const spurNode = prevPath[i];
      const rootPath = prevPath.slice(0, i + 1);

      let newEdges = [...edges];

      // remove edges with same prefix
      for (let p of A) {
        if (p.length > i && rootPath.every((v, idx) => v === p[idx])) {
          newEdges = removeEdge(newEdges, p[i], p[i + 1]);
        }
      }

      // find spur path
      const spurPath = astar(nodes, newEdges, spurNode, goal);

      if (spurPath) {

        const totalPath = [
          ...rootPath.slice(0, -1),
          ...spurPath
        ];

        const cost = pathCost(totalPath, edges);

        B.push({ path: totalPath, cost });
      }
    }

    if (B.length === 0) break;

    // sort candidates
    B.sort((a, b) => a.cost - b.cost);

    const best = B.shift();
    A.push(best.path);
  }

  return A;
}