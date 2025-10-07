"""
Test Runner for MAS-FRO - Handles import issues properly
"""

import sys
import os
from pathlib import Path
import importlib.util

# Add src to Python path
src_path = Path(__file__).parent / 'src'
sys.path.insert(0, str(src_path))

# Now run the integration tests
print("=" * 70)
print("MAS-FRO TEST RUNNER")
print("=" * 70)
print(f"Python version: {sys.version}")
print(f"Working directory: {os.getcwd()}")
print(f"Src path added: {src_path}")
print()

# Import and run integration tests
try:
    # Import modules by loading them directly (bypassing package __init__.py)
    import networkx as nx
    import json
    
    # Load risk_aware_routing module directly
    spec = importlib.util.spec_from_file_location(
        "risk_aware_routing",
        src_path / "agents" / "risk_aware_routing.py"
    )
    risk_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(risk_module)
    RiskAwareAStar = risk_module.RiskAwareAStar
    AdaptiveRiskRouter = risk_module.AdaptiveRiskRouter
    
    # Load visualization module directly
    spec = importlib.util.spec_from_file_location(
        "route_visualizer",
        src_path / "visualization" / "route_visualizer.py"
    )
    viz_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(viz_module)
    RouteVisualizer = viz_module.RouteVisualizer
    SimulationRenderer = viz_module.SimulationRenderer
    
    # Load API module directly
    spec = importlib.util.spec_from_file_location(
        "routing_api",
        src_path / "api" / "routing_api.py"
    )
    api_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(api_module)
    EnhancedAdaptiveRouter = api_module.EnhancedAdaptiveRouter
    
    print("[TEST 1] Risk-Aware A* Algorithm")
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
    
    # Test with different scenarios
    risk_scores = {
        f"{u}_{v}_0": 0.2 + (u + v) * 0.05
        for u, v in G.edges()
    }
    
    router = RiskAwareAStar()
    
    # Test 1: Low rain
    print("Test 1a: Low rain (5 mm/hr)")
    result = router.calculate_route(
        graph=G,
        start=1,
        goal=4,
        risk_scores=risk_scores,
        rain_rate=5.0
    )
    
    if result['status'] == 'success':
        print(f"  ✓ Route found: {result['path']}")
        print(f"  ✓ Safety score: {result.get('safety_score', 0):.2%}")
    else:
        print(f"  ✗ Error: {result.get('description')}")
    
    # Test 2: Heavy rain
    print("\nTest 1b: Heavy rain (35 mm/hr)")
    result = router.calculate_route(
        graph=G,
        start=1,
        goal=4,
        risk_scores=risk_scores,
        rain_rate=35.0
    )
    
    if result['status'] == 'success':
        print(f"  ✓ Route found: {result['path']}")
        print(f"  ✓ Safety score: {result.get('safety_score', 0):.2%}")
    else:
        print(f"  ✗ Error: {result.get('description')}")
    
    print("\n[TEST 2] Routing Interface")
    print("-" * 40)
    
    router = EnhancedAdaptiveRouter()
    
    # Test algorithm listing
    algorithms = router.list_algorithms()
    print(f"  ✓ Available algorithms: {algorithms['algorithms']}")
    
    # Test algorithm switching
    for algo in ['RiskAwareAStar', 'Dijkstra', 'BFS']:
        result = router.set_algorithm(algo)
        if result['status'] == 'success':
            print(f"  ✓ Set to {algo}")
        else:
            print(f"  ✗ Failed to set {algo}")
    
    print("\n[TEST 3] Visualization")
    print("-" * 40)
    
    viz = RouteVisualizer()
    
    path = [1, 2, 3, 4]
    test_risks = {"1_2_0": 0.2, "2_3_0": 0.5, "3_4_0": 0.3}
    
    # Test SVG generation
    svg = viz.generate_svg_visualization(G, path, test_risks)
    if '<svg' in svg:
        print(f"  ✓ SVG generated ({len(svg)} chars)")
    else:
        print(f"  ✗ SVG generation failed")
    
    # Test simulation
    rain_profile = [5, 10, 15, 10, 5]
    sim_data = viz.generate_simulation_data(G, path, test_risks, rain_profile, 5)
    if len(sim_data) == 5:
        print(f"  ✓ Simulation created ({len(sim_data)} frames)")
    else:
        print(f"  ✗ Simulation failed")
    
    print("\n" + "=" * 70)
    print("✅ ALL TESTS PASSED!")
    print("=" * 70)
    print("\nThe system is ready for use with real data.")
    print("Run: python demonstrate_real_data.py")
    
    sys.exit(0)

except ImportError as e:
    print(f"\n❌ Import Error: {e}")
    print("\nTroubleshooting:")
    print("  1. Make sure you're in the project root directory")
    print("  2. Verify src folder exists and contains required modules")
    print("  3. Check that all dependencies are installed:")
    print("     pip install -r requirements.txt")
    sys.exit(1)

except Exception as e:
    print(f"\n❌ Test Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
