from .base_agent import BaseAgent
from ..data.data_structures import RouteRequest
import networkx as nx
import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)

class RoutingAgent(BaseAgent):
    """Agent responsible for pathfinding computations"""
    
    def __init__(self, agent_id: str, env, input_queue, output_queue, graph_env):
        super().__init__(agent_id, env, input_queue, output_queue)
        self.graph_env = graph_env
    
    def run(self):
        """Process route requests"""
        while self.running:
            try:
                if self.input_queue and not self.input_queue.empty():
                    message = self.input_queue.get()
                    
                    if message.get('type') == 'route_request':
                        route_request = message['data']
                        route = self._calculate_route(route_request)
                        
                        response = {
                            'type': 'route_response',
                            'request_id': route_request.request_id,
                            'route': route,
                            'sender': self.agent_id,
                            'timestamp': time.time()
                        }
                        if self.output_queue:
                            self.output_queue.put(response)
                
                yield self.env.timeout(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"{self.agent_id} error: {e}")
                yield self.env.timeout(30)
    
    def _calculate_route(self, route_request: RouteRequest) -> dict:
        """Calculate optimal route using risk-aware A*"""
        start_time = time.time()
        
        try:
            # Get current network state
            graph = self.graph_env.get_current_state()
            
            # Find nearest nodes to origin and destination
            origin_node = self._find_nearest_node(graph, route_request.origin)
            dest_node = self._find_nearest_evacuation_center(graph)
            
            if origin_node is None or dest_node is None:
                return {'error': 'Could not find valid start or end point', 'success': False}
            
            # Calculate path using risk-aware weights
            try:
                path = nx.astar_path(
                    graph, 
                    origin_node, 
                    dest_node, 
                    weight='risk_aware_weight',
                    heuristic=self._distance_heuristic
                )
                
                # Calculate route metrics
                total_distance = 0
                total_risk = 0
                
                for i in range(len(path) - 1):
                    u, v = path[i], path[i + 1]
                    edge_data = graph[u][v][0]  # Get first edge
                    total_distance += edge_data.get('length', 0)
                    edge_id = f"{u}_{v}_0"
                    total_risk += self.graph_env._risk_scores.get(edge_id, 0)
                
                computation_time = time.time() - start_time
                avg_risk = total_risk / max(1, len(path) - 1)
                safety_score = 1.0 - min(1.0, avg_risk)
                
                return {
                    'path': path,
                    'total_distance': total_distance,
                    'total_risk': total_risk,
                    'average_risk': avg_risk,
                    'safety_score': safety_score,
                    'computation_time': computation_time,
                    'estimated_travel_time': (total_distance / 1000) / 30 * 3600,  # Assuming 30 km/h
                    'success': True
                }
                
            except nx.NetworkXNoPath:
                return {'error': 'No path found', 'success': False}
            
        except Exception as e:
            return {'error': str(e), 'success': False}
    
    def _find_nearest_node(self, graph: nx.MultiDiGraph, location) -> Optional[int]:
        """Find nearest node to given coordinates"""
        try:
            import osmnx as ox
            return ox.distance.nearest_nodes(graph, location[1], location[0])
        except:
            # Fallback for simple graphs
            nodes = list(graph.nodes())
            return nodes[0] if nodes else None
    
    def _find_nearest_evacuation_center(self, graph: nx.MultiDiGraph) -> Optional[int]:
        """Find nearest evacuation center (simplified)"""
        # In real implementation, this would find the actual nearest evacuation center
        nodes = list(graph.nodes())
        return nodes[-1] if nodes else None
    
    def _distance_heuristic(self, u: int, v: int) -> float:
        """Heuristic function for A* (Euclidean distance)"""
        try:
            graph = self.graph_env.graph
            u_data = graph.nodes[u]
            v_data = graph.nodes[v]
            
            lat1, lon1 = u_data.get('y', 0), u_data.get('x', 0)
            lat2, lon2 = v_data.get('y', 0), v_data.get('x', 0)
            
            # Simple Euclidean distance
            return ((lat2 - lat1) ** 2 + (lon2 - lon1) ** 2) ** 0.5 * 111000  # Convert to meters
        except:
            return 0