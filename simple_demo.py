#!/usr/bin/env python3
"""
Simple demonstration of RoutingAgent functionality without network dependencies.

This script shows the core RoutingAgent features using mock data to avoid
downloading large OpenStreetMap datasets.
"""

import sys
import os
import time
import logging
from datetime import datetime
from multiprocessing import Queue

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def setup_demo_logging():
    """Setup logging for the demonstration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def demonstrate_core_functionality():
    """Demonstrate core RoutingAgent functionality with mock data"""
    logger = setup_demo_logging()
    logger.info("🚀 Starting RoutingAgent Core Functionality Demo")

    try:
        # Import required modules
        from src.data.data_structures import RouteRequest

        logger.info("📦 Testing data structures...")

        # Create demo route request
        route_request = RouteRequest(
            request_id="demo_mock_123",
            origin=(14.6507, 121.1029),  # Marikina area coordinates
            destination="Marikina Sports Center",
            timestamp=datetime.now(),
            user_id="demo_user"
        )

        logger.info("✅ RouteRequest created successfully")
        logger.info(f"   📍 Origin: {route_request.origin}")
        logger.info(f"   🎯 Destination: {route_request.destination}")
        logger.info(f"   🆔 Request ID: {route_request.request_id}")

        # Demonstrate risk score calculation (mock)
        logger.info("⚠️  Demonstrating risk score calculation...")

        # Mock risk scores for demonstration
        mock_risk_scores = {
            'edge_1_2': 0.1,  # Low risk
            'edge_2_3': 0.7,  # High risk
            'edge_3_4': float('inf'),  # Impassable
            'edge_4_5': 0.3,  # Medium risk
        }

        for edge_id, risk in mock_risk_scores.items():
            if risk == float('inf'):
                logger.info(f"   🚫 {edge_id}: IMPASSABLE (infinite cost)")
            else:
                safety = 1.0 - risk
                logger.info(f"   📊 {edge_id}: Risk = {risk:.1f}, Safety = {safety:.1f}")

        # Demonstrate route metrics calculation
        logger.info("📏 Demonstrating route metrics calculation...")

        mock_route = {
            'path': [1, 2, 3, 4, 5],
            'total_distance': 1250.5,
            'risk_scores': [0.1, 0.7, 0.3, 0.2],
            'computation_time': 0.045
        }

        total_risk = sum(mock_route['risk_scores'])
        avg_risk = total_risk / len(mock_route['risk_scores'])
        max_risk = max(mock_route['risk_scores'])
        safety_score = 1.0 - min(1.0, avg_risk)
        estimated_time = (mock_route['total_distance'] / 1000) / 30 * 3600

        logger.info("   📊 Route Analysis:")
        logger.info(f"   📏 Total Distance: {mock_route['total_distance']:.1f} meters")
        logger.info(f"   ⚠️  Average Risk: {avg_risk:.3f}")
        logger.info(f"   🔴 Max Risk: {max_risk:.3f}")
        logger.info(f"   🛡️  Safety Score: {safety_score:.3f}")
        logger.info(f"   ⏱️  Computation Time: {mock_route['computation_time']:.3f} seconds")
        logger.info(f"   🚗 Estimated Travel Time: {estimated_time:.1f} seconds")
        logger.info(f"   🛤️ Path: {mock_route['path']}")

        # Demonstrate Agent Communication Protocol
        logger.info("📡 Demonstrating Agent Communication Protocol...")

        request_msg = {
            'performative': 'request',
            'sender': 'UserAgent',
            'receiver': 'RoutingAgent',
            'content': {
                'type': 'route_request',
                'data': route_request,
                'timestamp': time.time()
            },
            'language': 'json',
            'ontology': 'routing'
        }

        response_msg = {
            'performative': 'inform',
            'sender': 'RoutingAgent',
            'receiver': 'UserAgent',
            'content': {
                'type': 'route_response',
                'request_id': 'demo_mock_123',
                'route': {
                    'success': True,
                    'total_distance': 1250.5,
                    'safety_score': 0.7,
                    'path': [1, 2, 3, 4, 5]
                },
                'timestamp': time.time()
            },
            'language': 'json',
            'ontology': 'routing'
        }

        logger.info("   📨 Request Message Structure:")
        logger.info(f"      Performative: {request_msg['performative']}")
        logger.info(f"      Content Type: {request_msg['content']['type']}")

        logger.info("   📨 Response Message Structure:")
        logger.info(f"      Performative: {response_msg['performative']}")
        logger.info(f"      Content Type: {response_msg['content']['type']}")
        logger.info(f"      Route Success: {response_msg['content']['route']['success']}")

        logger.info("✅ Core functionality demonstration completed!")

        return True

    except Exception as e:
        logger.error(f"❌ Demonstration failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def show_project_structure():
    """Show the project structure and key files"""
    logger = setup_demo_logging()
    logger.info("📁 MAS-FRO Project Structure:")

    key_files = [
        ("src/agents/routing_agent.py", "⭐ Main RoutingAgent implementation"),
        ("src/environment/dynamic_graph.py", "🗺️ Dynamic road network environment"),
        ("src/data/data_structures.py", "📋 Data models and structures"),
        ("data/evacuation_centers.csv", "🏢 Evacuation center locations"),
        ("src/simulation/mas_controller.py", "🎮 Multi-agent system controller"),
        ("demonstrate_routing_agent.py", "🚀 Full demonstration script"),
        ("requirements.txt", "📦 Python dependencies"),
    ]

    for file_path, description in key_files:
        if os.path.exists(file_path):
            logger.info(f"   ✅ {file_path} - {description}")
        else:
            logger.info(f"   ❌ {file_path} - {description} (missing)")

def show_routing_agent_features():
    """Show the key features of the RoutingAgent"""
    logger = setup_demo_logging()
    logger.info("🎯 RoutingAgent Key Features:")

    features = [
        "Risk-Aware A* Pathfinding Algorithm",
        "Real-Time Dynamic Graph Integration",
        "GeoPandas Spatial Query Support",
        "Agent Communication Protocol (ACP) Compliance",
        "Composite Risk Scoring (Hydrological + Infrastructure)",
        "Research-Based Methodology (Kreibich et al., 2009)",
        "Comprehensive Route Metrics & Safety Scoring",
        "Graceful Error Handling & Logging",
        "Modular Design for Easy Extension",
        "Production-Ready Documentation"
    ]

    for i, feature in enumerate(features, 1):
        logger.info(f"   {i:2d}. {feature}")

if __name__ == "__main__":
    print("=" * 70)
    print("🤖 MAS-FRO RoutingAgent Core Functionality Demo")
    print("=" * 70)

    # Show project structure
    show_project_structure()
    print()

    # Show features
    show_routing_agent_features()
    print()

    # Run core functionality demo
    success = demonstrate_core_functionality()

    print("\n" + "=" * 70)
    if success:
        print("🎉 Core functionality demonstration completed successfully!")
        print("\n💡 To see the full demonstration with real map data:")
        print("   python demonstrate_routing_agent.py")
        print("\n📚 For more information, see the README.md file")
        print("\n🔧 To run the complete MAS-FRO system:")
        print("   python src/main.py")
    else:
        print("❌ Some demonstrations failed. Check the error messages above.")
    print("=" * 70)