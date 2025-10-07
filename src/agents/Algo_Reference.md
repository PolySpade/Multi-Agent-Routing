# Algorithm Reference Documentation
## MAS-FRO Pathfinding Algorithms

This document provides comprehensive reference implementations for pathfinding algorithms used in the Multi-Agent System for Flood Route Optimization (MAS-FRO).

---

### A* (A-star) Pathfinding Algorithm

**Purpose:** Finds the shortest path between nodes in a weighted graph using both actual cost and heuristic estimates. Optimal for pathfinding when a good heuristic is available.

**Description:** A* combines the benefits of Dijkstra's algorithm (guaranteed shortest path) with greedy best-first search (efficiency through heuristics). It uses f(n) = g(n) + h(n) where g(n) is the actual cost from start to node n, and h(n) is the heuristic estimate from n to goal.

**Pseudocode:**
```
function A_Star(start, goal, graph, heuristic):
    openSet = priority queue containing start
    gScore = map with default value of Infinity
    gScore[start] = 0
    fScore = map with default value of Infinity
    fScore[start] = heuristic(start)
    cameFrom = empty map
    
    while openSet is not empty:
        current = node in openSet with lowest fScore value
        
        if current == goal:
            return reconstruct_path(cameFrom, current)
        
        openSet.remove(current)
        
        for neighbor in graph.neighbors(current):
            tentative_gScore = gScore[current] + distance(current, neighbor)
            
            if tentative_gScore < gScore[neighbor]:
                cameFrom[neighbor] = current
                gScore[neighbor] = tentative_gScore
                fScore[neighbor] = gScore[neighbor] + heuristic(neighbor)
                if neighbor not in openSet:
                    openSet.add(neighbor)
    
    return failure
```

**Python Implementation:**
```python
import heapq
from typing import Dict, List, Tuple, Optional, Callable

def a_star(start: int, goal: int, graph: Dict[int, List[Tuple[int, float]]], 
           heuristic: Callable[[int], float]) -> Optional[List[int]]:
    """
    Performs the A* pathfinding algorithm.
    
    Args:
        start: Starting node ID
        goal: Goal node ID
        graph: Dictionary mapping node -> list of (neighbor, cost) tuples
        heuristic: Function mapping node -> estimated cost to goal
    
    Returns:
        List of node IDs representing the shortest path, or None if no path exists
    """
    # Priority queue: (f_score, g_score, node, path)
    open_set = []
    heapq.heappush(open_set, (heuristic(start), 0, start, [start]))
    
    # Track visited nodes to avoid cycles
    visited = set()
    
    while open_set:
        f_score, g_score, current, path = heapq.heappop(open_set)
        
        # Goal reached
        if current == goal:
            return path
        
        # Skip if already visited
        if current in visited:
            continue
        visited.add(current)
        
        # Explore neighbors
        for neighbor, cost in graph.get(current, []):
            if neighbor not in visited:
                new_g_score = g_score + cost
                new_f_score = new_g_score + heuristic(neighbor)
                heapq.heappush(open_set, 
                             (new_f_score, new_g_score, neighbor, path + [neighbor]))
    
    return None  # No path found
```

**Time Complexity:** O((V + E) log V) where V is vertices and E is edges, using a binary heap

**Space Complexity:** O(V) for storing the visited set and priority queue

**Typical Applications:**
- GPS navigation systems
- Video game pathfinding
- Robotics path planning
- Network routing protocols
- Emergency evacuation routing (MAS-FRO)

**References:**
- Hart, P. E., Nilsson, N. J., & Raphael, B. (1968). "A Formal Basis for the Heuristic Determination of Minimum Cost Paths". IEEE Transactions on Systems Science and Cybernetics.
- Russell, S., & Norvig, P. (2010). *Artificial Intelligence: A Modern Approach* (3rd ed.). Pearson.
- [Red Blob Games: A* Pathfinding Algorithm](https://www.redblobgames.com/pathfinding/a-star/introduction.html)

---

### Breadth-First Search (BFS)

**Purpose:** Finds the shortest path between nodes in an unweighted graph by exploring all nodes at the present depth before moving to nodes at the next depth level.

**Description:** BFS explores a graph level by level, guaranteeing the shortest path in terms of the number of edges. It uses a queue data structure to maintain the order of exploration, making it ideal for finding the shortest path in unweighted graphs.

**Pseudocode:**
```
function BFS(start, goal, graph):
    queue = empty queue
    queue.enqueue(start)
    visited = set containing start
    parent = empty map
    
    while queue is not empty:
        current = queue.dequeue()
        
        if current == goal:
            return reconstruct_path(parent, start, goal)
        
        for neighbor in graph.neighbors(current):
            if neighbor not in visited:
                visited.add(neighbor)
                parent[neighbor] = current
                queue.enqueue(neighbor)
    
    return failure
```

**Python Implementation:**
```python
from collections import deque
from typing import Dict, List, Optional

def breadth_first_search(start: int, goal: int, graph: Dict[int, List[int]]) -> Optional[List[int]]:
    """
    Performs Breadth-First Search (BFS) to find the shortest path in an unweighted graph.
    
    Args:
        start: Starting node ID
        goal: Goal node ID
        graph: Dictionary mapping node -> list of neighbor nodes
    
    Returns:
        List of node IDs representing the shortest path, or None if no path exists
    """
    # Queue stores (current_node, path_to_node)
    queue = deque([(start, [start])])
    visited = {start}
    
    while queue:
        current, path = queue.popleft()
        
        # Goal reached
        if current == goal:
            return path
        
        # Explore neighbors
        for neighbor in graph.get(current, []):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))
    
    return None  # No path found
```

**Time Complexity:** O(V + E) where V is the number of vertices and E is the number of edges

**Space Complexity:** O(V) for storing the visited set and queue

**Typical Applications:**
- Shortest path in unweighted graphs
- Web crawling
- Social network analysis (finding degrees of separation)
- Finding connected components
- Broadcasting in networks
- Puzzle solving (finding minimum moves)

**References:**
- Moore, E. F. (1959). "The shortest path through a maze". Proceedings of the International Symposium on the Theory of Switching.
- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2009). *Introduction to Algorithms* (3rd ed.). MIT Press.
- [GeeksforGeeks BFS in Python](https://www.geeksforgeeks.org/breadth-first-search-or-bfs-for-a-graph/)

---

### Depth-First Search (DFS)

**Purpose:** Explores a graph by going as deep as possible along each branch before backtracking, useful for exhaustive searching and topological ordering.

**Description:** DFS explores a graph by following paths to their conclusion before backtracking. It uses a stack (explicitly or via recursion) to track the exploration order. While it doesn't guarantee the shortest path, it's memory-efficient and useful for many graph problems.

**Pseudocode:**
```
function DFS(start, goal, graph):
    stack = empty stack
    stack.push(start)
    visited = empty set
    parent = empty map
    
    while stack is not empty:
        current = stack.pop()
        
        if current == goal:
            return reconstruct_path(parent, start, goal)
        
        if current not in visited:
            visited.add(current)
            
            for neighbor in graph.neighbors(current):
                if neighbor not in visited:
                    parent[neighbor] = current
                    stack.push(neighbor)
    
    return failure
```

**Python Implementation:**
```python
from typing import Dict, List, Optional

def depth_first_search(start: int, goal: int, graph: Dict[int, List[int]]) -> Optional[List[int]]:
    """
    Performs Depth-First Search (DFS) to find a path in a graph.
    
    Args:
        start: Starting node ID
        goal: Goal node ID
        graph: Dictionary mapping node -> list of neighbor nodes
    
    Returns:
        List of node IDs representing a path, or None if no path exists
    """
    # Stack stores (current_node, path_to_node)
    stack = [(start, [start])]
    visited = set()
    
    while stack:
        current, path = stack.pop()
        
        # Goal reached
        if current == goal:
            return path
        
        if current not in visited:
            visited.add(current)
            
            # Explore neighbors (reverse order for consistent left-to-right traversal)
            for neighbor in reversed(graph.get(current, [])):
                if neighbor not in visited:
                    stack.append((neighbor, path + [neighbor]))
    
    return None  # No path found
```

**Recursive Implementation:**
```python
def dfs_recursive(start: int, goal: int, graph: Dict[int, List[int]], 
                  visited: Optional[set] = None, path: Optional[List[int]] = None) -> Optional[List[int]]:
    """
    Recursive implementation of DFS.
    
    Args:
        start: Current node
        goal: Goal node
        graph: Graph representation
        visited: Set of visited nodes
        path: Current path
    
    Returns:
        Path to goal or None
    """
    if visited is None:
        visited = set()
    if path is None:
        path = []
    
    visited.add(start)
    path = path + [start]
    
    if start == goal:
        return path
    
    for neighbor in graph.get(start, []):
        if neighbor not in visited:
            result = dfs_recursive(neighbor, goal, graph, visited, path)
            if result is not None:
                return result
    
    return None
```

**Time Complexity:** O(V + E) where V is the number of vertices and E is the number of edges

**Space Complexity:** O(V) for storing the visited set and stack (or O(h) for recursion stack where h is the maximum depth)

**Typical Applications:**
- Topological sorting
- Cycle detection in graphs
- Path finding in mazes
- Finding strongly connected components
- Solving puzzles with unique solutions
- Tree/Graph traversal operations

**References:**
- Tarjan, R. (1972). "Depth-first search and linear graph algorithms". SIAM Journal on Computing.
- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2009). *Introduction to Algorithms* (3rd ed.). MIT Press.
- [Python DFS Implementation – Programiz](https://www.programiz.com/dsa/graph-dfs)

---

## Algorithm Comparison Table

| Algorithm | Best For | Time Complexity | Space Complexity | Guarantees Shortest Path |
|-----------|----------|-----------------|------------------|-------------------------|
| A* | Weighted graphs with good heuristic | O((V+E) log V) | O(V) | Yes (if heuristic is admissible) |
| BFS | Unweighted graphs | O(V+E) | O(V) | Yes (by edge count) |
| DFS | Exploring all paths, cycle detection | O(V+E) | O(V) | No |

---

## Integration with MAS-FRO

These algorithms form the foundation of the routing system in MAS-FRO:

1. **A*** - Primary algorithm for risk-aware pathfinding with flood hazard scoring
2. **BFS** - Used for finding nearest evacuation centers in grid-based representations
3. **DFS** - Used for exploring alternative routes and detecting blocked path cycles

All implementations are designed to work with the NetworkX graph structure used in the Dynamic Graph Environment module.

