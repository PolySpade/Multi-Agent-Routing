from .base_agent import BaseAgent
from ..data.data_structures import RouteRequest
import logging
import time

logger = logging.getLogger(__name__)

class EvacuationManagerAgent(BaseAgent):
    """Agent serving as interface between users and the system"""
    
    def __init__(self, agent_id: str, env, input_queue, output_queue, routing_queue):
        super().__init__(agent_id, env, input_queue, output_queue)
        self.routing_queue = routing_queue
        self.pending_requests = {}
    
    def run(self):
        """Handle user requests and responses"""
        while self.running:
            try:
                # Check for incoming messages
                if self.input_queue and not self.input_queue.empty():
                    message = self.input_queue.get()
                    self._process_message(message)
                
                yield self.env.timeout(5)  # Check every 5 seconds
                
            except Exception as e:
                logger.error(f"{self.agent_id} error: {e}")
                yield self.env.timeout(30)
    
    def _process_message(self, message: dict):
        """Process incoming messages"""
        msg_type = message.get('type')
        
        if msg_type == 'user_request':
            self._handle_route_request(message['data'])
        elif msg_type == 'route_response':
            self._handle_route_response(message)
        elif msg_type == 'user_feedback':
            # Forward feedback to Hazard Agent
            if self.output_queue:
                self.output_queue.put(message)
    
    def _handle_route_request(self, route_request: RouteRequest):
        """Handle incoming route request from user"""
        logger.info(f"{self.agent_id}: Processing route request {route_request.request_id}")
        
        # Store request
        self.pending_requests[route_request.request_id] = route_request
        
        # Forward to Routing Agent
        message = {
            'type': 'route_request',
            'data': route_request,
            'sender': self.agent_id,
            'timestamp': time.time()
        }
        if self.routing_queue:
            self.routing_queue.put(message)
    
    def _handle_route_response(self, message: dict):
        """Handle route response from Routing Agent"""
        request_id = message['request_id']
        route = message['route']
        
        if request_id in self.pending_requests:
            original_request = self.pending_requests[request_id]
            
            # Create user-friendly response
            user_response = {
                'type': 'route_delivered',
                'request_id': request_id,
                'user_id': original_request.user_id,
                'route': route,
                'timestamp': time.time()
            }
            
            # In real implementation, this would be sent to the user's device
            logger.info(f"{self.agent_id}: Route delivered to user {original_request.user_id}")
            if route.get('success'):
                logger.info(f"Safety Score: {route.get('safety_score', 0):.2f}")
                logger.info(f"Estimated Time: {route.get('estimated_travel_time', 0):.0f} seconds")
            
            # Clean up
            del self.pending_requests[request_id]