"""
Visualization Agent for MAS-FRO
Manages visualization and communicates with other agents using Agent Communication Protocol.
"""

import json
import logging
import time
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import simpy
from queue import Queue
import networkx as nx

# Import base agent and visualization modules
from .base_agent import BaseAgent
from ..visualization.route_visualizer import RouteVisualizer, SimulationRenderer
from ..data.data_structures import RouteRequest

logger = logging.getLogger(__name__)


class VisualizationAgent(BaseAgent):
    """
    Agent responsible for visualization of routes and risk data.
    
    Communicates bi-directionally with other agents using standardized
    Agent Communication Protocol (ACP) with JSON messages.
    
    Message Protocol:
        Required fields: from, to, type, data, status
        Optional fields: error, timestamp, request_id
    """
    
    def __init__(
        self,
        agent_id: str,
        env: simpy.Environment,
        input_queue: Queue,
        output_queue: Queue,
        graph_env=None
    ):
        """
        Initialize Visualization Agent.
        
        Args:
            agent_id: Unique agent identifier
            env: SimPy environment for discrete event simulation
            input_queue: Queue for receiving messages from other agents
            output_queue: Queue for sending messages to other agents
            graph_env: Dynamic graph environment (optional)
        """
        super().__init__(agent_id, env, input_queue, output_queue)
        
        self.graph_env = graph_env
        self.visualizer = RouteVisualizer()
        self.renderer = SimulationRenderer()
        
        # Storage for visualizations
        self.active_visualizations = {}
        self.simulation_history = []
        
        # Message validation
        self.required_message_fields = {'from', 'to', 'type', 'data', 'status'}
        
        logger.info(f"{agent_id} initialized and ready for visualization tasks")
    
    def run(self):
        """
        Main agent loop processing visualization requests.
        """
        while self.running:
            try:
                # Process incoming messages
                if self.input_queue and not self.input_queue.empty():
                    message = self.input_queue.get()
                    
                    # Validate message format
                    validation_result = self._validate_message(message)
                    if validation_result['valid']:
                        self._process_message(message)
                    else:
                        self._send_error_response(
                            message.get('from', 'unknown'),
                            f"Invalid message format: {validation_result['error']}"
                        )
                
                # Periodic cleanup of old visualizations
                if len(self.active_visualizations) > 100:
                    self._cleanup_old_visualizations()
                
                yield self.env.timeout(5)  # Check every 5 seconds
                
            except Exception as e:
                logger.error(f"{self.agent_id} error in main loop: {e}")
                yield self.env.timeout(10)
    
    def _validate_message(self, message: Dict) -> Dict[str, Any]:
        """
        Validate incoming message format according to ACP.
        
        Args:
            message: Message to validate
            
        Returns:
            Dict with 'valid' boolean and optional 'error' string
        """
        try:
            # Check for required fields
            missing_fields = self.required_message_fields - set(message.keys())
            if missing_fields:
                return {
                    'valid': False,
                    'error': f"Missing required fields: {missing_fields}"
                }
            
            # Validate field types
            if not isinstance(message['from'], str):
                return {'valid': False, 'error': "'from' must be a string"}
            
            if not isinstance(message['to'], str):
                return {'valid': False, 'error': "'to' must be a string"}
            
            if not isinstance(message['type'], str):
                return {'valid': False, 'error': "'type' must be a string"}
            
            if not isinstance(message['data'], (dict, list)):
                return {'valid': False, 'error': "'data' must be dict or list"}
            
            if message['status'] not in ['success', 'error', 'pending', 'request']:
                return {
                    'valid': False,
                    'error': f"Invalid status: {message['status']}"
                }
            
            return {'valid': True}
            
        except Exception as e:
            return {'valid': False, 'error': str(e)}
    
    def _process_message(self, message: Dict):
        """
        Process validated messages based on type.
        
        Args:
            message: Validated ACP message
        """
        msg_type = message['type']
        sender = message['from']
        data = message['data']
        
        logger.info(f"{self.agent_id} processing {msg_type} from {sender}")
        
        try:
            if msg_type == 'routeUpdate':
                self._handle_route_update(message)
            
            elif msg_type == 'visualizeRoute':
                self._handle_visualize_request(message)
            
            elif msg_type == 'riskUpdate':
                self._handle_risk_update(message)
            
            elif msg_type == 'simulationRequest':
                self._handle_simulation_request(message)
            
            elif msg_type == 'exportRequest':
                self._handle_export_request(message)
            
            else:
                self._send_error_response(
                    sender,
                    f"Unknown message type: {msg_type}"
                )
        
        except Exception as e:
            logger.error(f"Error processing {msg_type}: {e}")
            self._send_error_response(sender, str(e))
    
    def _handle_route_update(self, message: Dict):
        """
        Handle route update from Routing Agent.
        
        Expected data format:
            {
                'path': [node_ids],
                'risk': float,
                'request_id': str,
                'graph': optional graph data
            }
        """
        data = message['data']
        sender = message['from']
        request_id = data.get('request_id', f"viz_{int(time.time())}")
        
        try:
            # Extract route information
            path = data.get('path', [])
            total_risk = data.get('risk', 0.0)
            
            # Get graph (from message or environment)
            graph = self._get_graph(data.get('graph'))
            
            # Get risk scores
            risk_scores = data.get('risk_scores', {})
            
            # Generate visualization
            svg_viz = self.visualizer.generate_svg_visualization(
                graph, path, risk_scores
            )
            
            html_viz = self.visualizer.generate_html_overlay(
                graph, path, risk_scores
            )
            
            # Store visualization
            self.active_visualizations[request_id] = {
                'svg': svg_viz,
                'html': html_viz,
                'path': path,
                'risk': total_risk,
                'timestamp': datetime.now().isoformat()
            }
            
            # Send success response
            response = {
                'from': self.agent_id,
                'to': sender,
                'type': 'visualizationReady',
                'data': {
                    'request_id': request_id,
                    'format': 'svg+html',
                    'path_length': len(path),
                    'risk_level': total_risk,
                    'visualization_id': request_id
                },
                'status': 'success'
            }
            
            self._send_message(response)
            
            logger.info(f"Visualization created for request {request_id}")
            
        except Exception as e:
            self._send_error_response(sender, f"Route update failed: {str(e)}")
    
    def _handle_visualize_request(self, message: Dict):
        """
        Handle explicit visualization request.
        
        Expected data format:
            {
                'visualization_id': str,
                'format': 'svg' | 'html' | 'json',
                'include_simulation': bool
            }
        """
        data = message['data']
        sender = message['from']
        viz_id = data.get('visualization_id')
        format_type = data.get('format', 'svg')
        
        try:
            if viz_id not in self.active_visualizations:
                raise ValueError(f"Visualization {viz_id} not found")
            
            viz_data = self.active_visualizations[viz_id]
            
            # Prepare response based on format
            if format_type == 'svg':
                content = viz_data['svg']
            elif format_type == 'html':
                content = viz_data['html']
            elif format_type == 'json':
                content = json.dumps({
                    'path': viz_data['path'],
                    'risk': viz_data['risk'],
                    'timestamp': viz_data['timestamp']
                })
            else:
                raise ValueError(f"Unknown format: {format_type}")
            
            # Send response
            response = {
                'from': self.agent_id,
                'to': sender,
                'type': 'visualizationData',
                'data': {
                    'visualization_id': viz_id,
                    'format': format_type,
                    'content': content,
                    'size_bytes': len(content.encode('utf-8'))
                },
                'status': 'success'
            }
            
            self._send_message(response)
            
        except Exception as e:
            self._send_error_response(sender, str(e))
    
    def _handle_risk_update(self, message: Dict):
        """
        Handle risk update from Hazard Agent.
        
        Expected data format:
            {
                'risk_scores': {edge_id: risk_value},
                'affected_area': optional area specification,
                'timestamp': ISO timestamp
            }
        """
        data = message['data']
        sender = message['from']
        
        try:
            risk_scores = data.get('risk_scores', {})
            
            # Update any active visualizations that might be affected
            for viz_id, viz_data in self.active_visualizations.items():
                # Check if this visualization's path is affected
                path = viz_data.get('path', [])
                if self._is_path_affected(path, risk_scores):
                    # Re-generate visualization with new risk data
                    graph = self._get_graph()
                    
                    updated_svg = self.visualizer.generate_svg_visualization(
                        graph, path, risk_scores
                    )
                    
                    viz_data['svg'] = updated_svg
                    viz_data['last_update'] = datetime.now().isoformat()
            
            # Send acknowledgment
            response = {
                'from': self.agent_id,
                'to': sender,
                'type': 'riskUpdateAcknowledged',
                'data': {
                    'updated_visualizations': len(self.active_visualizations),
                    'timestamp': datetime.now().isoformat()
                },
                'status': 'success'
            }
            
            self._send_message(response)
            
        except Exception as e:
            self._send_error_response(sender, f"Risk update failed: {str(e)}")
    
    def _handle_simulation_request(self, message: Dict):
        """
        Handle simulation visualization request.
        
        Expected data format:
            {
                'path': [node_ids],
                'risk_scores': initial risk scores,
                'rain_profile': [rain_rates],
                'time_steps': number of frames,
                'output_format': 'json' | 'html'
            }
        """
        data = message['data']
        sender = message['from']
        
        try:
            # Extract simulation parameters
            path = data.get('path', [])
            risk_scores = data.get('risk_scores', {})
            rain_profile = data.get('rain_profile', [0] * 10)
            time_steps = data.get('time_steps', 10)
            output_format = data.get('output_format', 'json')
            
            # Get graph
            graph = self._get_graph(data.get('graph'))
            
            # Generate simulation data
            sim_data = self.visualizer.generate_simulation_data(
                graph, path, risk_scores, rain_profile, time_steps
            )
            
            # Format output
            if output_format == 'html':
                output = self.renderer.create_animated_html(sim_data, graph)
            else:  # json
                output = self.visualizer.export_simulation_json(
                    sim_data,
                    metadata={'sender': sender, 'timestamp': datetime.now().isoformat()}
                )
            
            # Store simulation
            sim_id = f"sim_{int(time.time())}"
            self.simulation_history.append({
                'id': sim_id,
                'data': sim_data,
                'timestamp': datetime.now().isoformat()
            })
            
            # Send response
            response = {
                'from': self.agent_id,
                'to': sender,
                'type': 'simulationComplete',
                'data': {
                    'simulation_id': sim_id,
                    'frames': len(sim_data),
                    'format': output_format,
                    'content': output if len(output) < 10000 else 'TOO_LARGE',
                    'download_ready': len(output) >= 10000
                },
                'status': 'success'
            }
            
            self._send_message(response)
            
            logger.info(f"Simulation {sim_id} completed with {len(sim_data)} frames")
            
        except Exception as e:
            self._send_error_response(sender, f"Simulation failed: {str(e)}")
    
    def _handle_export_request(self, message: Dict):
        """
        Handle export request for visualization data.
        
        Expected data format:
            {
                'visualization_id': str or 'all',
                'format': 'json' | 'svg' | 'html'
            }
        """
        data = message['data']
        sender = message['from']
        viz_id = data.get('visualization_id', 'all')
        export_format = data.get('format', 'json')
        
        try:
            if viz_id == 'all':
                # Export all visualizations
                export_data = {
                    'visualizations': self.active_visualizations,
                    'count': len(self.active_visualizations),
                    'export_timestamp': datetime.now().isoformat()
                }
            else:
                # Export specific visualization
                if viz_id not in self.active_visualizations:
                    raise ValueError(f"Visualization {viz_id} not found")
                
                export_data = self.active_visualizations[viz_id]
            
            # Format export
            if export_format == 'json':
                output = json.dumps(export_data, indent=2)
            else:
                output = export_data.get(export_format, '')
            
            # Send response
            response = {
                'from': self.agent_id,
                'to': sender,
                'type': 'exportReady',
                'data': {
                    'export_id': f"export_{int(time.time())}",
                    'format': export_format,
                    'size_bytes': len(output.encode('utf-8')),
                    'content': output if len(output) < 50000 else 'TOO_LARGE',
                    'download_url': f"/download/{viz_id}" if len(output) >= 50000 else None
                },
                'status': 'success'
            }
            
            self._send_message(response)
            
        except Exception as e:
            self._send_error_response(sender, f"Export failed: {str(e)}")
    
    def _send_message(self, message: Dict):
        """
        Send message to output queue with validation.
        
        Args:
            message: Message to send
        """
        # Validate outgoing message
        validation = self._validate_message(message)
        if not validation['valid']:
            logger.error(f"Invalid outgoing message: {validation['error']}")
            return
        
        # Add timestamp if not present
        if 'timestamp' not in message:
            message['timestamp'] = datetime.now().isoformat()
        
        # Send message
        if self.output_queue:
            self.output_queue.put(message)
            logger.debug(f"{self.agent_id} sent {message['type']} to {message['to']}")
    
    def _send_error_response(self, recipient: str, error_msg: str):
        """
        Send standardized error response.
        
        Args:
            recipient: Agent to send error to
            error_msg: Error message
        """
        error_response = {
            'from': self.agent_id,
            'to': recipient,
            'type': 'error',
            'data': {
                'error_message': error_msg,
                'timestamp': datetime.now().isoformat()
            },
            'status': 'error',
            'error': error_msg
        }
        
        self._send_message(error_response)
    
    def _get_graph(self, graph_data: Optional[Any] = None) -> nx.Graph:
        """
        Get graph from various sources.
        
        Args:
            graph_data: Optional graph data
            
        Returns:
            NetworkX graph
        """
        if graph_data:
            if isinstance(graph_data, nx.Graph):
                return graph_data
            elif isinstance(graph_data, dict):
                # Convert dict to graph
                return self._dict_to_graph(graph_data)
        
        if self.graph_env:
            return self.graph_env.get_current_state()
        
        # Create default sample graph
        return self._create_sample_graph()
    
    def _dict_to_graph(self, graph_dict: Dict) -> nx.Graph:
        """Convert dictionary to NetworkX graph."""
        G = nx.Graph()
        
        if 'nodes' in graph_dict:
            for node_id, attrs in graph_dict['nodes'].items():
                G.add_node(int(node_id), **attrs)
        
        if 'edges' in graph_dict:
            for edge in graph_dict['edges']:
                G.add_edge(edge['from'], edge['to'], **edge.get('data', {}))
        
        return G
    
    def _create_sample_graph(self) -> nx.Graph:
        """Create sample graph for testing."""
        G = nx.Graph()
        
        # Sample nodes
        nodes = {
            1: {'x': 121.10, 'y': 14.65, 'name': 'Node 1'},
            2: {'x': 121.11, 'y': 14.66, 'name': 'Node 2'},
            3: {'x': 121.12, 'y': 14.67, 'name': 'Node 3'},
            4: {'x': 121.13, 'y': 14.68, 'name': 'Node 4'},
            5: {'x': 121.14, 'y': 14.69, 'name': 'Node 5'}
        }
        
        for node_id, attrs in nodes.items():
            G.add_node(node_id, **attrs)
        
        # Sample edges
        edges = [
            (1, 2, {'length': 100}),
            (2, 3, {'length': 150}),
            (3, 4, {'length': 120}),
            (4, 5, {'length': 130}),
            (1, 3, {'length': 180}),
            (2, 4, {'length': 170})
        ]
        
        G.add_edges_from([(u, v) for u, v, _ in edges])
        for u, v, attrs in edges:
            G[u][v].update(attrs)
        
        return G
    
    def _is_path_affected(self, path: List[int], risk_scores: Dict[str, float]) -> bool:
        """Check if a path is affected by risk score changes."""
        for i in range(len(path) - 1):
            edge_id = f"{path[i]}_{path[i+1]}_0"
            if edge_id in risk_scores:
                return True
        return False
    
    def _cleanup_old_visualizations(self):
        """Remove old visualizations to free memory."""
        current_time = datetime.now()
        to_remove = []
        
        for viz_id, viz_data in self.active_visualizations.items():
            viz_time = datetime.fromisoformat(viz_data['timestamp'])
            age_hours = (current_time - viz_time).total_seconds() / 3600
            
            if age_hours > 24:  # Remove visualizations older than 24 hours
                to_remove.append(viz_id)
        
        for viz_id in to_remove:
            del self.active_visualizations[viz_id]
        
        if to_remove:
            logger.info(f"Cleaned up {len(to_remove)} old visualizations")


# Example integration test
def test_visualization_agent():
    """Test Visualization Agent with mock messages."""
    
    import simpy
    from queue import Queue
    
    # Create environment and queues
    env = simpy.Environment()
    input_queue = Queue()
    output_queue = Queue()
    
    # Create agent
    viz_agent = VisualizationAgent(
        "VisualizationAgent-1",
        env,
        input_queue,
        output_queue
    )
    
    # Test route update message
    route_update_msg = {
        'from': 'RoutingAgent',
        'to': 'VisualizationAgent',
        'type': 'routeUpdate',
        'data': {
            'path': [1, 2, 3, 4, 5],
            'risk': 0.35,
            'request_id': 'test_route_001',
            'risk_scores': {
                '1_2_0': 0.2,
                '2_3_0': 0.4,
                '3_4_0': 0.5,
                '4_5_0': 0.3
            }
        },
        'status': 'success'
    }
    
    # Process message
    input_queue.put(route_update_msg)
    
    # Run simulation briefly
    env.run(until=10)
    
    # Check output
    if not output_queue.empty():
        response = output_queue.get()
        print("Response from Visualization Agent:")
        print(json.dumps(response, indent=2))
    
    # Test simulation request
    sim_request_msg = {
        'from': 'RoutingAgent',
        'to': 'VisualizationAgent',
        'type': 'simulationRequest',
        'data': {
            'path': [1, 2, 3, 4, 5],
            'risk_scores': {'1_2_0': 0.2, '2_3_0': 0.4},
            'rain_profile': [5, 10, 15, 20, 15, 10, 5],
            'time_steps': 7,
            'output_format': 'json'
        },
        'status': 'request'
    }
    
    input_queue.put(sim_request_msg)
    env.run(until=20)
    
    # Check simulation response
    if not output_queue.empty():
        response = output_queue.get()
        print("\nSimulation Response:")
        print(json.dumps(response, indent=2))
    
    print("\nVisualization Agent test completed successfully!")


if __name__ == "__main__":
    test_visualization_agent()

