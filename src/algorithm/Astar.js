// A* Algorithm (JS)

class PriorityQueue {
  constructor() {
    this.items = [];
  }

  enqueue(node, priority) {
    this.items.push({ node, priority });
    this.items.sort((a, b) => a.priority - b.priority);
  }

  dequeue() {
    return this.items.shift().node;
  }

  isEmpty() {
    return this.items.length === 0;
  }
}

// heuristic: khoảng cách thẳng tới goal
function heuristic(a, b) {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  return Math.sqrt(dx * dx + dy * dy);
}

export function astar(nodes, edges, startId, goalId) {

  // convert nodes → map
  const nodeMap = {};
  nodes.forEach(n => {
    nodeMap[n.id] = n;
  });

  // adjacency list
  const adj = {};
  edges.forEach(e => {
    if (!adj[e.from]) adj[e.from] = [];
    adj[e.from].push(e);
  });

  const frontier = new PriorityQueue();
  frontier.enqueue(
    { id: startId, cost: 0, path: [startId] },
    0
  );

  const explored = new Set();

  while (!frontier.isEmpty()) {
    const current = frontier.dequeue();

    if (current.id === goalId) {
      return current.path;
    }

    explored.add(current.id);

    const neighbors = adj[current.id] || [];

    for (let edge of neighbors) {

      if (explored.has(edge.to)) continue;

      const newCost = current.cost + edge.cost;

      const h = heuristic(nodeMap[edge.to], nodeMap[goalId]);

      frontier.enqueue(
        {
          id: edge.to,
          cost: newCost,
          path: [...current.path, edge.to]
        },
        newCost + h
      );
    }
  }

  return null; // không tìm được đường
}