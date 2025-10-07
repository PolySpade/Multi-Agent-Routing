"""
Risk-Aware Routing Algorithms for MAS-FRO
Implements enhanced pathfinding algorithms that consider flood risk and dynamic hazards.
"""

import heapq
import json
import logging
import time
from typing import Dict, List, Tuple, Optional, Any, Union
import networkx as nx
import numpy as np
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class RiskAwareRoute:
    """Data structure for risk-aware route results"""
    path: List[int]
    total_risk: float
    total_distance: float
    safety_score: float
    computation_time: float
    status: str
    description: Optional[str] = None
    risk_breakdown: Optional[Dict[str, float]] = None
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(asdict(self))


class RiskAwareAStar:
    """
    Risk-Aware A* Algorithm for flood route optimization.
    
    This algorithm extends traditional A* by incorporating:
    - Real-time flood risk hazard scores
    - Dynamic rain rate effects
    - Time-dependent risk evolution
    - Multi-factor risk assessment
    """
    
    def __init__(self):
        self.risk_weight = 0.7  # Weight for risk vs distance (0.7 = 70% risk, 30% distance)
        self.rain_impact_factor = 0.01  # How rain rate affects risk per mm/hr
        self.max_acceptable_risk = 0.95  # Maximum risk threshold before marking impassable
        
    def calculate_route(
        self,
        graph: Union[nx.Graph, Dict],
        start: int,
        goal: int,
        risk_scores: Dict[str, float],
        rain_rate: float = 0.0,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Calculate risk-aware optimal route.
        
        Args:
            graph: NetworkX graph or dict representation of road network
            start: Starting node ID
            goal: Goal node ID
            risk_scores: Dict mapping edge_id (format: "u_v_key") to risk score (0.0-1.0, inf for impassable)
            rain_rate: Current rain rate in mm/hr
            
        Returns:
            Dict with route information in specified format
        """
        start_time = time.time()
        
        try:
            # Input validation
            validation_result = self._validate_inputs(graph, start, goal, risk_scores, rain_rate)
            if validation_result:
                return validation_result
            
            # Convert graph if needed
            if isinstance(graph, dict):
                nx_graph = self._dict_to_networkx(graph)
            else:
                nx_graph = graph
            
            # Execute risk-aware A* algorithm
            path, total_risk = self._risk_aware_astar(
                nx_graph, start, goal, risk_scores, rain_rate
            )
            
            if path is None:
                return {
                    "status": "error",
                    "description": "No viable route found due to flood risk.",
                    "path": [],
                    "total_risk": float('inf')
                }
            
            # Calculate route metrics
            metrics = self._calculate_route_metrics(nx_graph, path, risk_scores, rain_rate)
            
            computation_time = time.time() - start_time
            
            # Success response
            result = {
                "path": path,
                "total_risk": metrics["total_risk"],
                "status": "success",
                "safety_score": 1.0 - min(1.0, metrics["average_risk"]),
                "total_distance": metrics["total_distance"],
                "computation_time": computation_time,
                "risk_breakdown": {
                    "flood_risk": metrics["flood_risk"],
                    "rain_impact": metrics["rain_impact"],
                    "max_segment_risk": metrics["max_risk"]
                }
            }
            
            # Validate output format
            self._validate_output(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Risk-aware routing error: {e}")
            return {
                "status": "error",
                "description": f"Routing calculation failed: {str(e)}",
                "path": [],
                "total_risk": 0.0
            }
    
    def _validate_inputs(
        self,
        graph: Union[nx.Graph, Dict],
        start: int,
        goal: int,
        risk_scores: Dict[str, float],
        rain_rate: float
    ) -> Optional[Dict]:
        """Validate input parameters"""
        
        # Check graph validity
        if graph is None:
            return {
                "status": "error",
                "description": "Invalid input: graph is None",
                "path": [],
                "total_risk": 0.0
            }
        
        # Check if nodes exist in graph
        if isinstance(graph, nx.Graph):
            if start not in graph.nodes():
                return {
                    "status": "error",
                    "description": f"Invalid input: start node {start} not in graph",
                    "path": [],
                    "total_risk": 0.0
                }
            if goal not in graph.nodes():
                return {
                    "status": "error",
                    "description": f"Invalid input: goal node {goal} not in graph",
                    "path": [],
                    "total_risk": 0.0
                }
        
        # Validate risk scores
        if not isinstance(risk_scores, dict):
            return {
                "status": "error",
                "description": "Invalid input: risk_scores must be a dictionary",
                "path": [],
                "total_risk": 0.0
            }
        
        # Validate rain rate
        if rain_rate < 0:
            return {
                "status": "error",
                "description": "Invalid input: rain_rate cannot be negative",
                "path": [],
                "total_risk": 0.0
            }
        
        return None  # All validations passed
    
    def _risk_aware_astar(
        self,
        graph: nx.Graph,
        start: int,
        goal: int,
        risk_scores: Dict[str, float],
        rain_rate: float
    ) -> Tuple[Optional[List[int]], float]:
        """
        Core risk-aware A* implementation.
        
        Returns:
            Tuple of (path, total_risk) or (None, inf) if no path exists
        """
        # Priority queue: (f_score, g_score, risk_accumulation, node, path)
        open_set = []
        initial_h = self._heuristic(graph, start, goal)
        heapq.heappush(open_set, (initial_h, 0, 0, start, [start]))
        
        # Track visited nodes
        visited = set()
        
        # Best scores found so far for each node
        g_scores = {start: 0}
        
        while open_set:
            f_score, g_score, accumulated_risk, current, path = heapq.heappop(open_set)
            
            # Goal reached
            if current == goal:
                return path, accumulated_risk
            
            # Skip if already visited with better score
            if current in visited:
                continue
            visited.add(current)
            
            # Explore neighbors
            for neighbor in graph.neighbors(current):
                if neighbor in visited:
                    continue
                
                # Calculate edge cost with risk consideration
                edge_cost, edge_risk = self._calculate_edge_cost(
                    graph, current, neighbor, risk_scores, rain_rate
                )
                
                # Skip impassable edges
                if edge_cost == float('inf'):
                    continue
                
                new_g_score = g_score + edge_cost
                new_risk = accumulated_risk + edge_risk
                
                # Only proceed if we found a better path to this neighbor
                if neighbor not in g_scores or new_g_score < g_scores[neighbor]:
                    g_scores[neighbor] = new_g_score
                    
                    # Calculate heuristic
                    h_score = self._heuristic(graph, neighbor, goal)
                    
                    # f(n) = g(n) + h(n)
                    new_f_score = new_g_score + h_score
                    
                    heapq.heappush(
                        open_set,
                        (new_f_score, new_g_score, new_risk, neighbor, path + [neighbor])
                    )
        
        return None, float('inf')  # No path found
    
    def _calculate_edge_cost(
        self,
        graph: nx.Graph,
        u: int,
        v: int,
        risk_scores: Dict[str, float],
        rain_rate: float
    ) -> Tuple[float, float]:
        """
        Calculate edge cost incorporating risk and rain effects.
        
        Returns:
            Tuple of (cost, risk_value)
        """
        # Get edge data
        edge_data = graph[u][v]
        if isinstance(edge_data, dict):
            base_distance = edge_data.get('length', 1.0)
        else:
            # MultiDiGraph case
            base_distance = edge_data[0].get('length', 1.0) if edge_data else 1.0
        
        # Get risk score for this edge
        edge_id = f"{u}_{v}_0"
        base_risk = risk_scores.get(edge_id, 0.0)
        
        # Impassable edge
        if base_risk == float('inf'):
            return float('inf'), float('inf')
        
        # Apply rain impact
        rain_adjusted_risk = min(1.0, base_risk + (rain_rate * self.rain_impact_factor))
        
        # Check if risk exceeds maximum acceptable threshold
        if rain_adjusted_risk >= self.max_acceptable_risk:
            return float('inf'), float('inf')
        
        # Combine distance and risk into single cost
        # Higher risk increases the effective cost exponentially
        risk_multiplier = 1.0 + (rain_adjusted_risk ** 2) * 10  # Exponential penalty for risk
        weighted_cost = base_distance * risk_multiplier
        
        # Apply weights for final cost
        final_cost = (
            self.risk_weight * weighted_cost +
            (1 - self.risk_weight) * base_distance
        )
        
        return final_cost, rain_adjusted_risk
    
    def _heuristic(self, graph: nx.Graph, u: int, v: int) -> float:
        """
        Heuristic function for A* (Euclidean distance).
        """
        try:
            u_data = graph.nodes[u]
            v_data = graph.nodes[v]
            
            # Get coordinates
            u_x, u_y = u_data.get('x', 0), u_data.get('y', 0)
            v_x, v_y = v_data.get('x', 0), v_data.get('y', 0)
            
            # Euclidean distance in degrees, converted to approximate meters
            distance = np.sqrt((v_x - u_x)**2 + (v_y - u_y)**2) * 111000
            
            return distance
            
        except Exception:
            # Fallback to 0 (Dijkstra behavior)
            return 0.0
    
    def _calculate_route_metrics(
        self,
        graph: nx.Graph,
        path: List[int],
        risk_scores: Dict[str, float],
        rain_rate: float
    ) -> Dict[str, float]:
        """Calculate comprehensive metrics for the route."""
        total_distance = 0
        total_risk = 0
        flood_risk = 0
        max_risk = 0
        
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            
            # Get edge data
            edge_data = graph[u][v]
            if isinstance(edge_data, dict):
                distance = edge_data.get('length', 1.0)
            else:
                distance = edge_data[0].get('length', 1.0) if edge_data else 1.0
            
            total_distance += distance
            
            # Get risk
            edge_id = f"{u}_{v}_0"
            base_risk = risk_scores.get(edge_id, 0.0)
            
            if base_risk != float('inf'):
                rain_adjusted_risk = min(1.0, base_risk + (rain_rate * self.rain_impact_factor))
                total_risk += rain_adjusted_risk
                flood_risk += base_risk
                max_risk = max(max_risk, rain_adjusted_risk)
        
        num_edges = max(1, len(path) - 1)
        
        return {
            "total_distance": total_distance,
            "total_risk": total_risk,
            "average_risk": total_risk / num_edges,
            "flood_risk": flood_risk / num_edges,
            "rain_impact": (rain_rate * self.rain_impact_factor) * num_edges,
            "max_risk": max_risk
        }
    
    def _dict_to_networkx(self, graph_dict: Dict) -> nx.Graph:
        """Convert dictionary graph representation to NetworkX graph."""
        G = nx.Graph()
        
        # Add nodes
        if 'nodes' in graph_dict:
            for node_id, node_data in graph_dict['nodes'].items():
                G.add_node(node_id, **node_data)
        
        # Add edges
        if 'edges' in graph_dict:
            for edge in graph_dict['edges']:
                G.add_edge(
                    edge['from'],
                    edge['to'],
                    length=edge.get('length', 1.0),
                    **edge.get('data', {})
                )
        
        return G
    
    def _validate_output(self, result: Dict) -> None:
        """Validate output format matches specification."""
        required_fields = ['path', 'total_risk', 'status']
        
        for field in required_fields:
            if field not in result:
                raise ValueError(f"Output missing required field: {field}")
        
        if not isinstance(result['path'], list):
            raise ValueError("Output 'path' must be a list")
        
        if not isinstance(result['total_risk'], (int, float)):
            raise ValueError("Output 'total_risk' must be a number")
        
        if result['status'] not in ['success', 'error']:
            raise ValueError("Output 'status' must be 'success' or 'error'")


class AdaptiveRiskRouter:
    """
    Adaptive routing that adjusts risk tolerance based on conditions.
    """
    
    def __init__(self):
        self.risk_aware_astar = RiskAwareAStar()
        self.algorithm = 'RiskAwareAStar'
        self.available_algorithms = {
            'RiskAwareAStar': self.risk_aware_astar,
            'Dijkstra': None,  # Can be added
            'BFS': None,  # Can be added
            'DFS': None   # Can be added
        }
    
    def set_algorithm(self, algorithm_name: str) -> Dict[str, str]:
        """Set the routing algorithm."""
        if algorithm_name not in self.available_algorithms:
            return {
                "status": "error",
                "message": f"Unknown algorithm: {algorithm_name}",
                "current_algorithm": self.algorithm
            }
        
        self.algorithm = algorithm_name
        return {
            "status": "success",
            "message": f"Algorithm set to {algorithm_name}",
            "current_algorithm": self.algorithm
        }
    
    def get_current_algorithm(self) -> Dict[str, str]:
        """Get the current algorithm in use."""
        return {
            "current_algorithm": self.algorithm,
            "status": "success"
        }
    
    def list_algorithms(self) -> Dict[str, Any]:
        """List available routing algorithms."""
        return {
            "algorithms": list(self.available_algorithms.keys()),
            "current": self.algorithm,
            "status": "success"
        }
    
    def calculate_route(self, **kwargs) -> Dict[str, Any]:
        """Calculate route using current algorithm."""
        if self.algorithm == 'RiskAwareAStar':
            return self.risk_aware_astar.calculate_route(**kwargs)
        else:
            return {
                "status": "error",
                "description": f"Algorithm {self.algorithm} not yet implemented",
                "path": [],
                "total_risk": 0.0
            }


# Example usage and testing
def test_risk_aware_routing():
    """Test the risk-aware routing implementation."""
    
    # Create sample graph
    G = nx.Graph()
    
    # Add nodes with coordinates
    nodes = {
        1: {'x': 121.1000, 'y': 14.6500},
        2: {'x': 121.1010, 'y': 14.6510},
        3: {'x': 121.1020, 'y': 14.6520},
        4: {'x': 121.1030, 'y': 14.6530},
        5: {'x': 121.1040, 'y': 14.6540},
        6: {'x': 121.1050, 'y': 14.6550},
        7: {'x': 121.1060, 'y': 14.6560}
    }
    
    for node_id, coords in nodes.items():
        G.add_node(node_id, **coords)
    
    # Add edges
    edges = [
        (1, 2, 100), (2, 3, 150), (3, 4, 120),
        (1, 5, 200), (5, 6, 180), (6, 7, 150),
        (4, 7, 100), (2, 5, 80), (3, 6, 90)
    ]
    
    for u, v, length in edges:
        G.add_edge(u, v, length=length)
    
    # Create risk scores
    risk_scores = {
        "1_2_0": 0.2,
        "2_1_0": 0.2,
        "2_3_0": 0.8,  # High risk
        "3_2_0": 0.8,
        "3_4_0": 0.3,
        "4_3_0": 0.3,
        "1_5_0": 0.1,
        "5_1_0": 0.1,
        "5_6_0": 0.2,
        "6_5_0": 0.2,
        "6_7_0": 0.1,
        "7_6_0": 0.1,
        "4_7_0": 0.4,
        "7_4_0": 0.4,
        "2_5_0": 0.3,
        "5_2_0": 0.3,
        "3_6_0": 0.9,  # Very high risk
        "6_3_0": 0.9
    }
    
    # Test routing
    router = RiskAwareAStar()
    
    # Test with low rain
    result = router.calculate_route(
        graph=G,
        start=1,
        goal=7,
        risk_scores=risk_scores,
        rain_rate=5.0  # 5mm/hr
    )
    
    print("Low rain scenario:")
    print(json.dumps(result, indent=2))
    
    # Test with heavy rain
    result = router.calculate_route(
        graph=G,
        start=1,
        goal=7,
        risk_scores=risk_scores,
        rain_rate=35.0  # 35mm/hr heavy rain
    )
    
    print("\nHeavy rain scenario:")
    print(json.dumps(result, indent=2))
    
    # Test adaptive router
    adaptive = AdaptiveRiskRouter()
    
    print("\nAvailable algorithms:")
    print(json.dumps(adaptive.list_algorithms(), indent=2))
    
    print("\nCurrent algorithm:")
    print(json.dumps(adaptive.get_current_algorithm(), indent=2))


if __name__ == "__main__":
    test_risk_aware_routing()

