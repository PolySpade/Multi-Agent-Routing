"""
Routing Algorithm API Interface for MAS-FRO
Provides REST API endpoints for algorithm selection and route calculation.
"""

from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any, Tuple
import json
import logging
import time
import networkx as nx
from datetime import datetime

# Import routing algorithms
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.risk_aware_routing import RiskAwareAStar, AdaptiveRiskRouter
from agents.routing_agent import RoutingAgent
from environment.dynamic_graph import DynamicGraphEnvironment
from data.data_structures import RouteRequest

logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="MAS-FRO Routing API",
    description="Multi-Agent System for Flood Route Optimization - Routing Interface",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response Models
class SetAlgorithmRequest(BaseModel):
    """Request model for setting routing algorithm"""
    algorithm: str = Field(
        ...,
        description="Algorithm name: RiskAwareAStar, Dijkstra, BFS, or DFS"
    )


class RouteCalculationRequest(BaseModel):
    """Request model for route calculation"""
    graph: Optional[Dict[str, Any]] = Field(
        None,
        description="Graph structure (optional if using default graph)"
    )
    start: int = Field(..., description="Starting node ID")
    goal: int = Field(..., description="Goal node ID")
    risk_scores: Dict[str, float] = Field(
        default_factory=dict,
        description="Risk scores for edges (edge_id -> score)"
    )
    rain_rate: float = Field(
        0.0,
        description="Current rain rate in mm/hr"
    )
    algorithm_override: Optional[str] = Field(
        None,
        description="Override current algorithm for this request"
    )


class AlgorithmListResponse(BaseModel):
    """Response model for algorithm list"""
    algorithms: List[str]
    current: str
    status: str
    descriptions: Optional[Dict[str, str]] = None


class AlgorithmStatusResponse(BaseModel):
    """Response model for algorithm status"""
    current_algorithm: str
    status: str
    message: Optional[str] = None


class RouteResponse(BaseModel):
    """Response model for route calculation"""
    path: List[int]
    total_risk: float
    status: str
    safety_score: Optional[float] = None
    total_distance: Optional[float] = None
    computation_time: Optional[float] = None
    risk_breakdown: Optional[Dict[str, float]] = None
    description: Optional[str] = None


# Global router instance
router = AdaptiveRiskRouter()

# Enhanced router with all algorithms
class EnhancedAdaptiveRouter(AdaptiveRiskRouter):
    """Extended router with all algorithm implementations"""
    
    def __init__(self):
        super().__init__()
        self.graph_env = None
        self._initialize_algorithms()
    
    def _initialize_algorithms(self):
        """Initialize all available algorithms"""
        self.available_algorithms = {
            'RiskAwareAStar': self.risk_aware_astar,
            'Dijkstra': self._dijkstra_route,
            'BFS': self._bfs_route,
            'DFS': self._dfs_route
        }
        
        self.algorithm_descriptions = {
            'RiskAwareAStar': 'Risk-aware A* with flood hazard consideration',
            'Dijkstra': 'Shortest path algorithm for weighted graphs',
            'BFS': 'Breadth-first search for unweighted shortest path',
            'DFS': 'Depth-first search for path finding'
        }
    
    def _dijkstra_route(self, graph, start, goal, risk_scores=None, **kwargs):
        """Dijkstra's algorithm implementation"""
        try:
            if isinstance(graph, dict):
                G = self._dict_to_networkx(graph)
            else:
                G = graph
            
            # Apply risk scores as edge weights if provided
            if risk_scores:
                for u, v, data in G.edges(data=True):
                    edge_id = f"{u}_{v}_0"
                    risk = risk_scores.get(edge_id, 0.0)
                    if risk == float('inf'):
                        G[u][v]['weight'] = float('inf')
                    else:
                        base_length = data.get('length', 1.0)
                        G[u][v]['weight'] = base_length * (1 + risk * 5)
            
            # Find shortest path
            try:
                path = nx.dijkstra_path(G, start, goal, weight='weight')
                path_length = nx.dijkstra_path_length(G, start, goal, weight='weight')
                
                # Calculate metrics
                total_risk = 0
                for i in range(len(path) - 1):
                    edge_id = f"{path[i]}_{path[i+1]}_0"
                    total_risk += risk_scores.get(edge_id, 0.0) if risk_scores else 0.0
                
                return {
                    "path": path,
                    "total_risk": total_risk,
                    "total_distance": path_length,
                    "status": "success",
                    "algorithm_used": "Dijkstra"
                }
                
            except nx.NetworkXNoPath:
                return {
                    "status": "error",
                    "description": "No path found using Dijkstra's algorithm",
                    "path": [],
                    "total_risk": 0.0
                }
                
        except Exception as e:
            return {
                "status": "error",
                "description": f"Dijkstra algorithm failed: {str(e)}",
                "path": [],
                "total_risk": 0.0
            }
    
    def _bfs_route(self, graph, start, goal, **kwargs):
        """BFS implementation for unweighted shortest path"""
        try:
            if isinstance(graph, dict):
                G = self._dict_to_networkx(graph)
            else:
                G = graph
            
            # BFS implementation
            from collections import deque
            
            queue = deque([(start, [start])])
            visited = {start}
            
            while queue:
                current, path = queue.popleft()
                
                if current == goal:
                    return {
                        "path": path,
                        "total_risk": 0.0,  # BFS doesn't consider risk
                        "total_distance": len(path) - 1,
                        "status": "success",
                        "algorithm_used": "BFS",
                        "description": "Shortest path by edge count"
                    }
                
                for neighbor in G.neighbors(current):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, path + [neighbor]))
            
            return {
                "status": "error",
                "description": "No path found using BFS",
                "path": [],
                "total_risk": 0.0
            }
            
        except Exception as e:
            return {
                "status": "error",
                "description": f"BFS algorithm failed: {str(e)}",
                "path": [],
                "total_risk": 0.0
            }
    
    def _dfs_route(self, graph, start, goal, **kwargs):
        """DFS implementation for path finding"""
        try:
            if isinstance(graph, dict):
                G = self._dict_to_networkx(graph)
            else:
                G = graph
            
            # DFS implementation
            stack = [(start, [start])]
            visited = set()
            
            while stack:
                current, path = stack.pop()
                
                if current == goal:
                    return {
                        "path": path,
                        "total_risk": 0.0,  # DFS doesn't optimize for risk
                        "total_distance": len(path) - 1,
                        "status": "success",
                        "algorithm_used": "DFS",
                        "description": "Path found (not necessarily optimal)"
                    }
                
                if current not in visited:
                    visited.add(current)
                    
                    for neighbor in reversed(list(G.neighbors(current))):
                        if neighbor not in visited:
                            stack.append((neighbor, path + [neighbor]))
            
            return {
                "status": "error",
                "description": "No path found using DFS",
                "path": [],
                "total_risk": 0.0
            }
            
        except Exception as e:
            return {
                "status": "error",
                "description": f"DFS algorithm failed: {str(e)}",
                "path": [],
                "total_risk": 0.0
            }
    
    def calculate_route(self, **kwargs):
        """Calculate route using selected algorithm"""
        algorithm = kwargs.pop('algorithm_override', None) or self.algorithm
        
        if algorithm == 'RiskAwareAStar':
            return self.risk_aware_astar.calculate_route(**kwargs)
        elif algorithm == 'Dijkstra':
            return self._dijkstra_route(**kwargs)
        elif algorithm == 'BFS':
            return self._bfs_route(**kwargs)
        elif algorithm == 'DFS':
            return self._dfs_route(**kwargs)
        else:
            return {
                "status": "error",
                "description": f"Unknown algorithm: {algorithm}",
                "path": [],
                "total_risk": 0.0
            }
    
    def _dict_to_networkx(self, graph_dict: Dict) -> nx.Graph:
        """Convert dictionary graph to NetworkX"""
        G = nx.Graph()
        
        if 'nodes' in graph_dict:
            for node_id, node_data in graph_dict['nodes'].items():
                G.add_node(int(node_id), **node_data)
        
        if 'edges' in graph_dict:
            for edge in graph_dict['edges']:
                G.add_edge(
                    edge['from'],
                    edge['to'],
                    length=edge.get('length', 1.0),
                    **edge.get('data', {})
                )
        
        return G


# Initialize enhanced router
router = EnhancedAdaptiveRouter()


# API Endpoints

@app.get("/")
async def root():
    """Root endpoint - API information"""
    return {
        "name": "MAS-FRO Routing API",
        "version": "1.0.0",
        "endpoints": {
            "GET /algorithms": "List available algorithms",
            "POST /set_algorithm": "Set routing algorithm",
            "GET /current_algorithm": "Get current algorithm",
            "POST /calculate_route": "Calculate route",
            "GET /health": "Health check"
        }
    }


@app.get("/algorithms", response_model=AlgorithmListResponse)
async def list_algorithms():
    """
    List available routing algorithms.
    
    Returns list of supported algorithms with descriptions.
    """
    result = router.list_algorithms()
    
    # Add descriptions
    result["descriptions"] = router.algorithm_descriptions
    
    return AlgorithmListResponse(**result)


@app.post("/set_algorithm", response_model=AlgorithmStatusResponse)
async def set_algorithm(request: SetAlgorithmRequest):
    """
    Set the routing algorithm to use.
    
    Available algorithms:
    - RiskAwareAStar: Risk-aware pathfinding with flood consideration
    - Dijkstra: Classic shortest path algorithm
    - BFS: Breadth-first search for unweighted graphs
    - DFS: Depth-first search (finds a path, not necessarily optimal)
    """
    result = router.set_algorithm(request.algorithm)
    
    if result["status"] == "error":
        raise HTTPException(status_code=400, detail=result.get("message", "Invalid algorithm"))
    
    return AlgorithmStatusResponse(**result)


@app.get("/current_algorithm", response_model=AlgorithmStatusResponse)
async def get_current_algorithm():
    """
    Get the currently selected routing algorithm.
    """
    result = router.get_current_algorithm()
    return AlgorithmStatusResponse(**result)


@app.post("/calculate_route", response_model=RouteResponse)
async def calculate_route(request: RouteCalculationRequest):
    """
    Calculate optimal route using selected algorithm.
    
    Provide graph structure, start/goal nodes, risk scores, and rain rate.
    The algorithm will return the optimal path based on its strategy.
    """
    try:
        # Use default graph if not provided
        if request.graph is None:
            # Create sample graph for testing
            G = create_sample_graph()
            graph = G
        else:
            graph = request.graph
        
        # Calculate route
        result = router.calculate_route(
            graph=graph,
            start=request.start,
            goal=request.goal,
            risk_scores=request.risk_scores,
            rain_rate=request.rain_rate,
            algorithm_override=request.algorithm_override
        )
        
        if result["status"] == "error":
            raise HTTPException(
                status_code=400,
                detail=result.get("description", "Route calculation failed")
            )
        
        return RouteResponse(**result)
        
    except Exception as e:
        logger.error(f"Route calculation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "current_algorithm": router.algorithm
    }


def create_sample_graph() -> nx.Graph:
    """Create a sample graph for testing"""
    G = nx.Graph()
    
    # Marikina-like coordinates
    nodes = {
        1: {'x': 121.1000, 'y': 14.6500, 'name': 'Start Point'},
        2: {'x': 121.1010, 'y': 14.6510, 'name': 'Junction A'},
        3: {'x': 121.1020, 'y': 14.6520, 'name': 'Junction B'},
        4: {'x': 121.1030, 'y': 14.6530, 'name': 'Junction C'},
        5: {'x': 121.1040, 'y': 14.6540, 'name': 'Junction D'},
        6: {'x': 121.1050, 'y': 14.6550, 'name': 'Junction E'},
        7: {'x': 121.1060, 'y': 14.6560, 'name': 'Evacuation Center'}
    }
    
    for node_id, attrs in nodes.items():
        G.add_node(node_id, **attrs)
    
    # Add edges with lengths
    edges = [
        (1, 2, 100), (2, 3, 150), (3, 4, 120),
        (1, 5, 200), (5, 6, 180), (6, 7, 150),
        (4, 7, 100), (2, 5, 80), (3, 6, 90)
    ]
    
    for u, v, length in edges:
        G.add_edge(u, v, length=length)
    
    return G


# CLI for testing
if __name__ == "__main__":
    import uvicorn
    
    print("Starting MAS-FRO Routing API...")
    print("API will be available at http://localhost:8000")
    print("Documentation at http://localhost:8000/docs")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

