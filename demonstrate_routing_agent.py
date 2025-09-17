#!/usr/bin/env python3
"""
Standalone demonstration of the RoutingAgent functionality.

This script creates a minimal working example of the RoutingAgent that can be run
independently to demonstrate its core pathfinding capabilities.
"""

import sys
import os
import time
import logging
from datetime import datetime
from multiprocessing import Queue

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Mock SimPy environment for demonstration
class MockSimPyEnv:
    """Mock SimPy environment for standalone testing"""
    def __init__(self):
        self.now = 0

    def timeout(self, delay):
        """Mock timeout - just return a mock event"""
        return MockEvent()

    def process(self, gen):
        """Mock process"""
        pass

class MockEvent:
    """Mock SimPy event"""
    pass

# Mock the simpy module
sys.modules['simpy'] = type(sys)('simpy')
sys.modules['simpy'].Environment = MockSimPyEnv

def setup_demo_logging():
    """Setup logging for the demonstration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def create_demo_request(origin_lat=14.6507, origin_lon=121.1029, destination="Marikina Sports Center"):
    """Create a demo route request"""
    from src.data.data_structures import RouteRequest

    return RouteRequest(
        request_id=f"demo_{int(time.time())}",
        origin=(origin_lat, origin_lon),  # Marikina area coordinates
        destination=destination,
        timestamp=datetime.now(),
        user_id="demo_user"
    )

def demonstrate_routing_agent():
    """Demonstrate the RoutingAgent functionality"""
    logger = setup_demo_logging()
    logger.info("🚀 Starting RoutingAgent Demonstration")

    try:
        # Import required modules
        from src.environment.dynamic_graph import DynamicGraphEnvironment
        from src.agents.routing_agent import RoutingAgent

        logger.info("📦 Initializing components...")

        # Create mock environment and queues
        env = MockSimPyEnv()
        input_queue = Queue()
        output_queue = Queue()

        # Initialize Dynamic Graph Environment
        logger.info("🗺️ Loading road network...")
        graph_env = DynamicGraphEnvironment()

        # Create RoutingAgent
        logger.info("🤖 Creating RoutingAgent...")
        agent = RoutingAgent("DemoRoutingAgent", env, input_queue, output_queue, graph_env)

        # Create demo route request
        logger.info("📍 Creating demo route request...")
        route_request = create_demo_request()

        logger.info(f"🎯 Route Request: {route_request.request_id}")
        logger.info(f"   Origin: {route_request.origin}")
        logger.info(f"   Destination: {route_request.destination}")

        # Calculate route
        logger.info("🧮 Calculating route...")
        start_time = time.time()

        result = agent._calculate_route(route_request)

        computation_time = time.time() - start_time

        # Display results
        if result.get('success', False):
            logger.info("✅ Route calculation successful!")
            logger.info(f"   📏 Total Distance: {result['total_distance']:.1f} meters")
            logger.info(f"   ⚠️  Average Risk: {result['average_risk']:.3f}")
            logger.info(f"   🛡️  Safety Score: {result['safety_score']:.3f}")
            logger.info(f"   ⏱️  Computation Time: {result['computation_time']:.3f} seconds")
            logger.info(f"   🚗 Estimated Travel Time: {result['estimated_travel_time']:.1f} seconds")
            logger.info(f"   🛤️ Path Nodes: {len(result['path'])} nodes")

            # Show first few and last few nodes
            path = result['path']
            if len(path) > 6:
                logger.info(f"   📍 Path: [{path[0]}, {path[1]}, {path[2]} ... {path[-3]}, {path[-2]}, {path[-1]}]")
            else:
                logger.info(f"   📍 Path: {path}")

        else:
            logger.error("❌ Route calculation failed!")
            logger.error(f"   Error: {result.get('error', 'Unknown error')}")

        logger.info(f"   📊 Total Demo Time: {computation_time:.2f} seconds")
        # Demonstrate evacuation center loading
        logger.info("🏢 Evacuation Centers Loaded:")
        if not agent.evacuation_centers.empty:
            for idx, center in agent.evacuation_centers.iterrows():
                logger.info(f"   • {center['name']} (Capacity: {center['capacity']}, Type: {center['type']})")
        else:
            logger.warning("   No evacuation centers loaded")

        # Demonstrate risk scoring
        logger.info("⚠️  Risk Score Demonstration:")
        test_edge_ids = ['1_2_0', '2_3_0', '3_4_0']
        for edge_id in test_edge_ids:
            risk_score = agent._get_risk_score(edge_id)
            if risk_score == float('inf'):
                logger.info(f"   • Edge {edge_id}: IMPASSABLE")
            else:
                logger.info(f"   • Edge {edge_id}: Risk = {risk_score:.3f}")

        logger.info("🎉 Demonstration completed successfully!")

    except ImportError as e:
        logger.error(f"❌ Import Error: {e}")
        logger.error("💡 Make sure all dependencies are installed:")
        logger.error("   pip install simpy geopandas networkx osmnx")
        return False

    except Exception as e:
        logger.error(f"❌ Unexpected Error: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True

def demonstrate_agent_communication():
    """Demonstrate Agent Communication Protocol messaging"""
    logger = setup_demo_logging()
    logger.info("📡 Demonstrating Agent Communication Protocol")

    try:
        from src.data.data_structures import RouteRequest
        from datetime import datetime

        # Create sample ACP messages
        route_request = RouteRequest(
            request_id="acp_demo_123",
            origin=(14.6507, 121.1029),
            destination="Marikina Sports Center",
            timestamp=datetime.now(),
            user_id="demo_user"
        )

        # Sample request message
        request_msg = {
            'type': 'route_request',
            'data': route_request,
            'sender': 'DemoUserAgent',
            'timestamp': time.time()
        }

        # Sample response message
        response_msg = {
            'performative': 'inform',
            'sender': 'DemoRoutingAgent',
            'receiver': 'DemoUserAgent',
            'content': {
                'type': 'route_response',
                'request_id': 'acp_demo_123',
                'route': {'success': True, 'total_distance': 1500.5},
                'timestamp': time.time()
            },
            'language': 'json',
            'ontology': 'routing'
        }

        logger.info("📨 Sample Request Message:")
        logger.info(f"   {request_msg}")

        logger.info("📨 Sample Response Message:")
        logger.info(f"   {response_msg}")

        logger.info("✅ ACP message format demonstration complete")

    except Exception as e:
        logger.error(f"❌ ACP demonstration failed: {e}")
        return False

    return True

if __name__ == "__main__":
    print("=" * 60)
    print("🤖 MAS-FRO RoutingAgent Demonstration")
    print("=" * 60)

    # Run main demonstration
    success1 = demonstrate_routing_agent()

    print("\n" + "=" * 60)

    # Run ACP demonstration
    success2 = demonstrate_agent_communication()

    print("\n" + "=" * 60)
    if success1 and success2:
        print("🎉 All demonstrations completed successfully!")
        print("\n💡 To run the full MAS-FRO system:")
        print("   python src/main.py")
        print("\n📚 For more information, see the README.md file")
    else:
        print("❌ Some demonstrations failed. Check the error messages above.")
    print("=" * 60)