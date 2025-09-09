import simpy
from multiprocessing import Queue, Manager
import logging
import time
from datetime import datetime
import random

from ..environment.dynamic_graph import DynamicGraphEnvironment
from ..agents.flood_agent import FloodAgent
from ..agents.scout_agent import ScoutAgent
from ..agents.hazard_agent import HazardAgent
from ..agents.routing_agent import RoutingAgent
from ..agents.evacuation_manager_agent import EvacuationManagerAgent
from ..data.data_structures import RouteRequest

logger = logging.getLogger(__name__)

class MASFROController:
    """Main simulation controller for the MAS-FRO system"""
    
    def __init__(self):
        self.env = simpy.Environment()
        self.graph_env = DynamicGraphEnvironment()
        
        # Create message queues
        self.manager = Manager()
        self.flood_to_hazard = Queue()
        self.scout_to_hazard = Queue()
        self.user_to_evacuation = Queue()
        self.evacuation_to_routing = Queue()
        self.routing_to_evacuation = Queue()
        self.feedback_to_hazard = Queue()
        
        # Initialize agents
        self.agents = {}
        self._create_agents()
    
    def _create_agents(self):
        """Create all agents in the system"""
        self.agents['flood'] = FloodAgent(
            'FloodAgent-1', self.env, None, self.flood_to_hazard
        )
        
        self.agents['scout'] = ScoutAgent(
            'ScoutAgent-1', self.env, None, self.scout_to_hazard
        )
        
        # Hazard agent receives from multiple sources
        self.agents['hazard'] = HazardAgent(
            'HazardAgent-1', self.env, self.feedback_to_hazard, None, self.graph_env
        )
        
        self.agents['routing'] = RoutingAgent(
            'RoutingAgent-1', self.env, self.evacuation_to_routing, 
            self.routing_to_evacuation, self.graph_env
        )
        
        self.agents['evacuation'] = EvacuationManagerAgent(
            'EvacuationManager-1', self.env, self.user_to_evacuation,
            self.feedback_to_hazard, self.evacuation_to_routing
        )
    
    def _setup_message_routing(self):
        """Setup message routing between agents"""
        def route_messages():
            while True:
                # Route flood data to hazard agent
                try:
                    while not self.flood_to_hazard.empty():
                        msg = self.flood_to_hazard.get_nowait()
                        self.feedback_to_hazard.put(msg)
                except:
                    pass
                
                # Route scout data to hazard agent  
                try:
                    while not self.scout_to_hazard.empty():
                        msg = self.scout_to_hazard.get_nowait()
                        self.feedback_to_hazard.put(msg)
                except:
                    pass
                
                # Route routing responses back to evacuation manager
                try:
                    while not self.routing_to_evacuation.empty():
                        msg = self.routing_to_evacuation.get_nowait()
                        self.user_to_evacuation.put(msg)
                except:
                    pass
                
                yield self.env.timeout(5)
        
        self.env.process(route_messages())
    
    def simulate_user_request(self):
        """Simulate a user requesting a route"""
        # Create a sample route request
        request = RouteRequest(
            request_id=f"REQ_{int(time.time())}",
            origin=(14.6507 + random.uniform(-0.01, 0.01), 
                   121.1029 + random.uniform(-0.01, 0.01)),
            destination="nearest_evacuation_center",
            timestamp=datetime.now(),
            user_id=f"user_{random.randint(1, 100)}"
        )
        
        # Send request to Evacuation Manager
        message = {
            'type': 'user_request',
            'data': request,
            'sender': 'simulator',
            'timestamp': time.time()
        }
        self.user_to_evacuation.put(message)
        
        logger.info(f"Simulated user request: {request.request_id}")
    
    def run_simulation(self, duration: int = 3600):
        """Run the complete simulation"""
        logger.info("Starting MAS-FRO simulation...")
        
        # Setup message routing
        self._setup_message_routing()
        
        # Schedule user requests
        def user_request_generator():
            while True:
                yield self.env.timeout(300)  # Request every 5 minutes
                self.simulate_user_request()
        
        self.env.process(user_request_generator())
        
        # Run simulation
        logger.info(f"Running simulation for {duration} seconds...")
        self.env.run(until=duration)
        
        logger.info("Simulation completed")
    
    def stop_simulation(self):
        """Stop all agents"""
        for agent in self.agents.values():
            agent.stop()