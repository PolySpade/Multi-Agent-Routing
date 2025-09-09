# Copy the DynamicGraphEnvironment class from the first artifact:
import osmnx as ox
import networkx as nx
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

class DynamicGraphEnvironment:
    """Shared environment representing the road network with dynamic weights"""
    
    def __init__(self, place_name: str = "Marikina, Philippines"):
        self.place_name = place_name
        self.graph = None
        self.evacuation_centers = []
        self._risk_scores = {}  # edge_id -> risk_score
        self.load_network()
    
    def load_network(self):
        """Load road network from OpenStreetMap"""
        try:
            logger.info(f"Loading road network for {self.place_name}")
            self.graph = ox.graph_from_place(self.place_name, network_type='drive')
            logger.info(f"Loaded {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
            
            # Initialize all edges with base risk score (0.0 = safe)
            for u, v, key in self.graph.edges(keys=True):
                edge_id = f"{u}_{v}_{key}"
                self._risk_scores[edge_id] = 0.0
                
        except Exception as e:
            logger.error(f"Failed to load network: {e}")
            # Create a simple test graph for demonstration
            self.graph = nx.MultiDiGraph()
            self.graph.add_edge(1, 2, length=100, key=0)
            self.graph.add_edge(2, 3, length=150, key=0)
    
    def update_edge_risk(self, u: int, v: int, key: int, risk_score: float):
        """Update risk score for a specific edge"""
        edge_id = f"{u}_{v}_{key}"
        self._risk_scores[edge_id] = risk_score
        
        # Update the graph edge weight
        if self.graph.has_edge(u, v, key):
            length = self.graph[u][v][key].get('length', 100)
            if risk_score == float('inf'):
                weight = float('inf')
            else:
                # Composite weight: 80% length, 20% risk
                normalized_length = length / 1000  # normalize by km
                weight = 0.8 * normalized_length + 0.2 * risk_score
            
            self.graph[u][v][key]['risk_aware_weight'] = weight
    
    def get_current_state(self):
        """Get current state of the network"""
        return self.graph.copy()