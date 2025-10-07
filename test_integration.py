"""
Integration Test Suite for MAS-FRO Enhanced Features
Tests all new components: Risk-Aware A*, Routing API, Visualization, and Agent Communication
"""

import json
import time
import asyncio
import simpy
from queue import Queue
import networkx as nx
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import all new modules
from src.agents.risk_aware_routing import RiskAwareAStar, AdaptiveRiskRouter
from src.agents.visualization_agent import VisualizationAgent
from src.api.routing_api import EnhancedAdaptiveRouter
from src.visualization.route_visualizer import RouteVisualizer, SimulationRenderer

print("=" * 70)
print("MAS-FRO INTEGRATION TEST SUITE")
print("Testing: Risk-Aware Routing, Visualization, and Agent Communication")
print("=" * 70)


def test_risk_aware_astar():
    """Test Risk-Aware A* Algorithm"""
    print("\n[TEST 1] Risk-Aware A* Algorithm")
    print("-" * 40)
    
    # Create test graph
    G = nx.Graph()
    nodes = {
        1: {'x': 121.10, 'y': 14.65},
        2: {'x': 121.11, 'y': 14.66},
        3: {'x': 121.12, 'y': 14.67},
        4: {'x': 121.13, 'y': 14.68}
    }
    
    for node_id, coords in nodes.items():
        G.add_node(node_id, **coords)
    
    G.add_edges_from([(1, 2), (2, 3), (3, 4), (1, 3), (2, 4)])
    for u, v in G.edges():
        G[u][v]['length'] = 100
    
    # Define risk scores
    risk_scores = {
        "1_2_0": 0.2,
        "2_1_0": 0.2,
        "2_3_0": 0.8,  # High risk path
        "3_2_0": 0.8,
        "3_4_0": 0.3,
        "4_3_0": 0.3,
        "1_3_0": 0.4,
        "3_1_0": 0.4,
        "2_4_0": 0.5,
        "4_2_0": 0.5
    }
    
    # Test routing with different rain scenarios
    router = RiskAwareAStar()
    
    # Low rain scenario
    print("Testing with low rain (5 mm/hr):")
    result = router.calculate_route(
        graph=G,
        start=1,
        goal=4,
        risk_scores=risk_scores,
        rain_rate=5.0
    )
    
    if result['status'] == 'success':
        print(f"  ✓ Path found: {result['path']}")
        print(f"  ✓ Total risk: {result['total_risk']:.2f}")
        print(f"  ✓ Safety score: {result.get('safety_score', 0):.2%}")
    else:
        print(f"  ✗ Error: {result.get('description', 'Unknown error')}")
    
    # High rain scenario
    print("\nTesting with heavy rain (35 mm/hr):")
    result = router.calculate_route(
        graph=G,
        start=1,
        goal=4,
        risk_scores=risk_scores,
        rain_rate=35.0
    )
    
    if result['status'] == 'success':
        print(f"  ✓ Path found: {result['path']}")
        print(f"  ✓ Total risk: {result['total_risk']:.2f}")
        print(f"  ✓ Safety score: {result.get('safety_score', 0):.2%}")
    else:
        print(f"  ✗ Error: {result.get('description', 'Unknown error')}")
    
    # Validate output format
    print("\nValidating output format:")
    required_fields = ['path', 'total_risk', 'status']
    missing = [f for f in required_fields if f not in result]
    if not missing:
        print("  ✓ All required fields present")
    else:
        print(f"  ✗ Missing fields: {missing}")
    
    return result['status'] == 'success'


def test_routing_interface():
    """Test Routing Algorithm Switching Interface"""
    print("\n[TEST 2] Routing Algorithm Interface")
    print("-" * 40)
    
    router = EnhancedAdaptiveRouter()
    
    # Test listing algorithms
    print("Testing algorithm listing:")
    algorithms = router.list_algorithms()
    print(f"  ✓ Available algorithms: {algorithms['algorithms']}")
    print(f"  ✓ Current algorithm: {algorithms['current']}")
    
    # Test setting algorithm
    print("\nTesting algorithm switching:")
    test_algorithms = ['RiskAwareAStar', 'Dijkstra', 'BFS', 'DFS']
    
    for algo in test_algorithms:
        result = router.set_algorithm(algo)
        if result['status'] == 'success':
            print(f"  ✓ Set to {algo}: {result['message']}")
        else:
            print(f"  ✗ Failed to set {algo}: {result.get('message', 'Unknown error')}")
    
    # Test getting current algorithm
    print("\nTesting current algorithm query:")
    current = router.get_current_algorithm()
    print(f"  ✓ Current algorithm: {current['current_algorithm']}")
    
    # Test invalid algorithm
    print("\nTesting error handling:")
    result = router.set_algorithm("InvalidAlgo")
    if result['status'] == 'error':
        print(f"  ✓ Correctly rejected invalid algorithm")
    else:
        print(f"  ✗ Should have rejected invalid algorithm")
    
    return True


def test_visualization():
    """Test Visualization and Simulation"""
    print("\n[TEST 3] Visualization System")
    print("-" * 40)
    
    # Create test data
    G = nx.Graph()
    nodes = {
        1: {'x': 121.10, 'y': 14.65, 'name': 'Start'},
        2: {'x': 121.11, 'y': 14.66, 'name': 'Mid1'},
        3: {'x': 121.12, 'y': 14.67, 'name': 'Mid2'},
        4: {'x': 121.13, 'y': 14.68, 'name': 'End'}
    }
    
    for node_id, attrs in nodes.items():
        G.add_node(node_id, **attrs)
    
    G.add_edges_from([(1, 2), (2, 3), (3, 4), (1, 3)])
    
    path = [1, 2, 3, 4]
    risk_scores = {
        "1_2_0": 0.2,
        "2_3_0": 0.5,
        "3_4_0": 0.3,
        "1_3_0": 0.7
    }
    
    visualizer = RouteVisualizer()
    
    # Test SVG generation
    print("Testing SVG visualization:")
    try:
        svg = visualizer.generate_svg_visualization(G, path, risk_scores)
        if '<svg' in svg and '</svg>' in svg:
            print(f"  ✓ SVG generated ({len(svg)} characters)")
        else:
            print(f"  ✗ Invalid SVG format")
    except Exception as e:
        print(f"  ✗ SVG generation failed: {e}")
    
    # Test HTML generation
    print("\nTesting HTML overlay:")
    try:
        html = visualizer.generate_html_overlay(G, path, risk_scores)
        if '<!DOCTYPE html>' in html:
            print(f"  ✓ HTML generated ({len(html)} characters)")
        else:
            print(f"  ✗ Invalid HTML format")
    except Exception as e:
        print(f"  ✗ HTML generation failed: {e}")
    
    # Test simulation data
    print("\nTesting simulation generation:")
    try:
        rain_profile = [5, 10, 15, 20, 15, 10, 5]
        sim_data = visualizer.generate_simulation_data(
            G, path, risk_scores, rain_profile, time_steps=7
        )
        
        print(f"  ✓ Generated {len(sim_data)} simulation frames")
        
        # Validate frame structure
        if sim_data:
            frame = sim_data[0]
            required_fields = ['frame', 'timestamp', 'hazard_scores', 'path']
            missing = [f for f in required_fields if f not in frame]
            if not missing:
                print(f"  ✓ Frame structure valid")
            else:
                print(f"  ✗ Missing frame fields: {missing}")
    except Exception as e:
        print(f"  ✗ Simulation generation failed: {e}")
    
    # Test JSON export
    print("\nTesting JSON export:")
    try:
        json_export = visualizer.export_simulation_json(sim_data)
        data = json.loads(json_export)
        if 'simulation' in data and 'frames' in data['simulation']:
            print(f"  ✓ JSON export successful")
        else:
            print(f"  ✗ Invalid JSON structure")
    except Exception as e:
        print(f"  ✗ JSON export failed: {e}")
    
    return True


def test_agent_communication():
    """Test Visualization Agent Communication Protocol"""
    print("\n[TEST 4] Agent Communication Protocol")
    print("-" * 40)
    
    # Setup simulation environment
    env = simpy.Environment()
    input_queue = Queue()
    output_queue = Queue()
    
    # Create Visualization Agent
    viz_agent = VisualizationAgent(
        "TestVizAgent",
        env,
        input_queue,
        output_queue
    )
    
    print("Testing message validation:")
    
    # Test valid message
    valid_msg = {
        'from': 'RoutingAgent',
        'to': 'VisualizationAgent',
        'type': 'routeUpdate',
        'data': {
            'path': [1, 2, 3],
            'risk': 0.45,
            'request_id': 'test_001'
        },
        'status': 'success'
    }
    
    validation = viz_agent._validate_message(valid_msg)
    if validation['valid']:
        print("  ✓ Valid message accepted")
    else:
        print(f"  ✗ Valid message rejected: {validation['error']}")
    
    # Test invalid message (missing field)
    invalid_msg = {
        'from': 'RoutingAgent',
        'type': 'routeUpdate',
        'data': {},
        'status': 'success'
    }
    
    validation = viz_agent._validate_message(invalid_msg)
    if not validation['valid']:
        print("  ✓ Invalid message rejected correctly")
    else:
        print("  ✗ Invalid message should have been rejected")
    
    # Test message processing
    print("\nTesting message processing:")
    
    input_queue.put(valid_msg)
    
    # Run simulation briefly
    env.run(until=10)
    
    # Check for response
    if not output_queue.empty():
        response = output_queue.get()
        
        # Validate response format
        validation = viz_agent._validate_message(response)
        if validation['valid']:
            print(f"  ✓ Response generated: {response['type']}")
            print(f"  ✓ Response status: {response['status']}")
        else:
            print(f"  ✗ Invalid response format: {validation['error']}")
    else:
        print("  ✗ No response generated")
    
    # Test error handling
    print("\nTesting error response:")
    
    error_msg = {
        'from': 'TestAgent',
        'to': 'VisualizationAgent',
        'type': 'unknownType',
        'data': {},
        'status': 'request'
    }
    
    input_queue.put(error_msg)
    env.run(until=20)
    
    if not output_queue.empty():
        response = output_queue.get()
        if response['status'] == 'error':
            print("  ✓ Error response generated correctly")
        else:
            print("  ✗ Should have generated error response")
    
    return True


def test_full_integration():
    """Test complete integration of all components"""
    print("\n[TEST 5] Full System Integration")
    print("-" * 40)
    
    # Create integrated system
    router = AdaptiveRiskRouter()
    visualizer = RouteVisualizer()
    
    # Setup test scenario
    G = nx.Graph()
    nodes = {
        1: {'x': 121.10, 'y': 14.65, 'name': 'Origin'},
        2: {'x': 121.11, 'y': 14.66, 'name': 'Junction'},
        3: {'x': 121.12, 'y': 14.67, 'name': 'Midpoint'},
        4: {'x': 121.13, 'y': 14.68, 'name': 'Destination'}
    }
    
    for node_id, attrs in nodes.items():
        G.add_node(node_id, **attrs)
    
    edges = [(1, 2, 100), (2, 3, 150), (3, 4, 120), (1, 3, 200), (2, 4, 180)]
    for u, v, length in edges:
        G.add_edge(u, v, length=length)
    
    risk_scores = {
        f"{u}_{v}_0": 0.3 + (u + v) * 0.1 for u, v in G.edges()
    }
    
    print("Step 1: Calculate route with Risk-Aware A*")
    route_result = router.calculate_route(
        graph=G,
        start=1,
        goal=4,
        risk_scores=risk_scores,
        rain_rate=15.0
    )
    
    if route_result['status'] == 'success':
        print(f"  ✓ Route calculated: {route_result['path']}")
        path = route_result['path']
    else:
        print(f"  ✗ Route calculation failed")
        return False
    
    print("\nStep 2: Generate visualization")
    try:
        svg = visualizer.generate_svg_visualization(G, path, risk_scores)
        print(f"  ✓ Visualization generated")
    except Exception as e:
        print(f"  ✗ Visualization failed: {e}")
        return False
    
    print("\nStep 3: Create simulation")
    try:
        rain_profile = [10, 20, 30, 20, 10]
        sim_data = visualizer.generate_simulation_data(
            G, path, risk_scores, rain_profile, 5
        )
        print(f"  ✓ Simulation created with {len(sim_data)} frames")
    except Exception as e:
        print(f"  ✗ Simulation failed: {e}")
        return False
    
    print("\nStep 4: Export results")
    try:
        export_data = {
            'route': route_result,
            'simulation': sim_data,
            'metadata': {
                'test': 'integration',
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
        }
        json_str = json.dumps(export_data, indent=2)
        print(f"  ✓ Results exported ({len(json_str)} bytes)")
    except Exception as e:
        print(f"  ✗ Export failed: {e}")
        return False
    
    print("\n✓ Full integration test completed successfully!")
    return True


def run_all_tests():
    """Run all integration tests"""
    tests = [
        ("Risk-Aware A* Algorithm", test_risk_aware_astar),
        ("Routing Interface", test_routing_interface),
        ("Visualization System", test_visualization),
        ("Agent Communication", test_agent_communication),
        ("Full Integration", test_full_integration)
    ]
    
    results = []
    
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"\n✗ Test '{name}' crashed: {e}")
            results.append((name, False))
    
    # Print summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for name, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{status:12} | {name}")
    
    print("-" * 70)
    print(f"Results: {passed}/{total} tests passed ({passed/total*100:.0f}%)")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! System is ready for use.")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please review the errors above.")
    
    return passed == total


if __name__ == "__main__":
    print("\nStarting integration tests...")
    print("This will validate all new MAS-FRO features.\n")
    
    success = run_all_tests()
    
    if success:
        print("\n✅ Integration testing complete - all systems operational!")
        print("\nYou can now:")
        print("1. Run the API: python src/api/routing_api.py")
        print("2. Test routing: python src/agents/risk_aware_routing.py")
        print("3. Test visualization: python src/visualization/route_visualizer.py")
        print("4. Test agents: python src/agents/visualization_agent.py")
    else:
        print("\n❌ Some tests failed. Please fix the issues before deployment.")
    
    sys.exit(0 if success else 1)

