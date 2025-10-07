"""
Complete Real Data Integration Demonstration for MAS-FRO
Shows end-to-end workflow using authentic elevation and network data.

This script demonstrates:
1. Loading real data from data/ folder
2. Calculating flood risks using actual elevations
3. Routing with risk-aware algorithms
4. Generating visualizations
5. Exporting results

Usage:
    python demonstrate_real_data.py

Requirements:
    - Python >= 3.9
    - All dependencies installed (pip install -r requirements.txt)
    - Data files in data/ folder
"""

import sys
import os
import json
import logging
from pathlib import Path
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / 'src'))

# Import real data modules
from data.real_data_loader import RealDataLoader, FloodRiskCalculator
from agents.risk_aware_routing import RiskAwareAStar
from visualization.route_visualizer import RouteVisualizer, SimulationRenderer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_section(title: str):
    """Print formatted section header"""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def print_subsection(title: str):
    """Print formatted subsection"""
    print(f"\n{title}")
    print("-" * 40)


def demonstrate_complete_workflow():
    """
    Demonstrate complete MAS-FRO workflow with real data.
    """
    print_section("MAS-FRO: Real Data Integration Demonstration")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Python Version: {sys.version.split()[0]}")
    
    try:
        # ============================================================
        # STEP 1: Load Real Data
        # ============================================================
        print_section("STEP 1: Load Real Data from data/ Folder")
        
        loader = RealDataLoader(data_folder="data")
        
        # Load evacuation centers
        print_subsection("Loading Evacuation Centers")
        try:
            evac_centers = loader.load_evacuation_centers(use_csv=True)
            print(f"✓ Loaded {len(evac_centers)} evacuation centers")
            print(f"\nSample Centers:")
            sample_centers = evac_centers[['name', 'latitude', 'longitude', 'capacity', 'type']].head(3)
            print(sample_centers.to_string(index=False))
        except Exception as e:
            print(f"✗ Error loading evacuation centers: {e}")
            return False
        
        # Load road network
        print_subsection("Loading Road Network")
        try:
            # Try GPKG first, fallback to GeoJSON
            try:
                road_network = loader.load_road_network(use_gpkg=True)
                data_source = "GPKG"
            except FileNotFoundError:
                print("  GPKG not found, trying GeoJSON...")
                road_network = loader.load_road_network(use_gpkg=False)
                data_source = "GeoJSON"
            
            print(f"✓ Loaded {len(road_network)} road segments from {data_source}")
            print(f"  Columns: {list(road_network.columns)[:8]}...")
        except Exception as e:
            print(f"✗ Error loading road network: {e}")
            return False
        
        # Convert to NetworkX graph
        print_subsection("Creating NetworkX Graph")
        try:
            G = loader.create_networkx_graph(road_network)
            print(f"✓ Graph created:")
            print(f"  Nodes: {G.number_of_nodes()}")
            print(f"  Edges: {G.number_of_edges()}")
        except Exception as e:
            print(f"✗ Error creating graph: {e}")
            return False
        
        # ============================================================
        # STEP 2: Export and Calculate Flood Risks
        # ============================================================
        print_section("STEP 2: Calculate Flood Risks from Real Elevations")
        
        # Export to CSV
        print_subsection("Exporting Network to CSV")
        try:
            nodes_path, edges_path = loader.export_to_csv(G)
            print(f"✓ Nodes: {nodes_path}")
            print(f"✓ Edges: {edges_path}")
        except Exception as e:
            print(f"✗ Error exporting: {e}")
            return False
        
        # Calculate risks for different scenarios
        print_subsection("Calculating Flood Risks (Multiple Scenarios)")
        
        calculator = FloodRiskCalculator(flood_threshold=12.0)
        
        scenarios = [
            (0.0, "No flooding"),
            (2.0, "Minor flooding (2m water)"),
            (5.0, "Major flooding (5m water)")
        ]
        
        risk_scores_dict = {}
        
        for water_level, description in scenarios:
            print(f"\nScenario: {description}")
            output_file = f"adjacency_matrix_water_{int(water_level)}m.csv"
            
            try:
                updated_df = calculator.update_adjacency_matrix(
                    adjacency_file=str(edges_path),
                    nodes_file="nodes.csv",
                    output_file=output_file,
                    water_level=water_level
                )
                
                # Store risk scores for 2m scenario (typical flood)
                if water_level == 2.0:
                    for _, row in updated_df.iterrows():
                        edge_id = f"{int(row['from_node'])}_{int(row['to_node'])}_{int(row['edge_key'])}"
                        risk_scores_dict[edge_id] = float(row['flood_risk (after)'])
                
                print(f"  ✓ Processed {len(updated_df)} edges")
                print(f"  ✓ Output: data/{output_file}")
                
                # Show sample
                sample = updated_df[['from_node', 'to_node', 'distance', 
                                    'avg_elevation', 'flood_risk (after)']].head(3)
                print(f"\n  Sample Edges:")
                print(sample.to_string(index=False))
                
            except Exception as e:
                print(f"  ✗ Error: {e}")
        
        # ============================================================
        # STEP 3: Calculate Route with Real Data
        # ============================================================
        print_section("STEP 3: Calculate Route Using Real Network")
        
        print_subsection("Risk-Aware Pathfinding")
        
        # Find start and goal nodes from real network
        if G.number_of_nodes() > 0:
            nodes = list(G.nodes())
            start_node = nodes[0]
            goal_node = nodes[min(len(nodes)-1, 100)] if len(nodes) > 100 else nodes[-1]
            
            print(f"Start Node: {start_node}")
            print(f"Goal Node: {goal_node}")
            print(f"Using {len(risk_scores_dict)} real risk scores")
            
            # Calculate route
            router = RiskAwareAStar()
            
            for rain_rate in [0, 15, 30]:
                print(f"\nRain Rate: {rain_rate} mm/hr")
                result = router.calculate_route(
                    graph=G,
                    start=start_node,
                    goal=goal_node,
                    risk_scores=risk_scores_dict,
                    rain_rate=rain_rate
                )
                
                if result['status'] == 'success':
                    print(f"  ✓ Route found:")
                    print(f"    Path length: {len(result['path'])} nodes")
                    print(f"    Safety score: {result.get('safety_score', 0):.2%}")
                    print(f"    Total distance: {result.get('total_distance', 0):.1f}m")
                    print(f"    Computation time: {result.get('computation_time', 0):.4f}s")
                else:
                    print(f"  ✗ {result.get('description', 'Route calculation failed')}")
        
        # ============================================================
        # STEP 4: Generate Visualization
        # ============================================================
        print_section("STEP 4: Generate Visualization with Real Network")
        
        print_subsection("Creating Visual Outputs")
        
        if result['status'] == 'success':
            path = result['path']
            
            viz = RouteVisualizer()
            
            # Generate SVG
            print("\nGenerating SVG visualization...")
            svg = viz.generate_svg_visualization(G, path, risk_scores_dict)
            print(f"  ✓ SVG created ({len(svg)} characters)")
            
            # Generate HTML with UTF-8 encoding
            print("\nGenerating HTML visualization...")
            html = viz.generate_html_overlay(G, path, risk_scores_dict)
            with open("test_visualization.html", "w", encoding='utf-8') as f:
                f.write(html)
            print(f"  ✓ HTML saved: test_visualization.html")
            
            # Generate simulation
            print("\nGenerating time-stepped simulation...")
            rain_profile = [5, 10, 15, 20, 25, 20, 15, 10, 5, 0]
            sim_data = viz.generate_simulation_data(
                G, path, risk_scores_dict, rain_profile
            )
            print(f"  ✓ Created {len(sim_data)} simulation frames")
            
            # Export JSON with UTF-8
            json_export = viz.export_simulation_json(
                sim_data,
                metadata={
                    "data_source": "real_marikina_network",
                    "water_level": 2.0,
                    "flood_threshold": 12.0,
                    "timestamp": datetime.now().isoformat()
                }
            )
            with open("simulation_data.json", "w", encoding='utf-8') as f:
                f.write(json_export)
            print(f"  ✓ JSON exported: simulation_data.json")
            
            # Create animated visualization
            renderer = SimulationRenderer()
            animated = renderer.create_animated_html(sim_data, G)
            with open("animated_simulation.html", "w", encoding='utf-8') as f:
                f.write(animated)
            print(f"  ✓ Animation saved: animated_simulation.html")
        
        # ============================================================
        # STEP 5: Summary and Next Steps
        # ============================================================
        print_section("Summary and Next Steps")
        
        print("\n✅ Successfully processed real data!")
        print("\nGenerated Files:")
        print("  📄 data/nodes.csv - Node elevations from real network")
        print("  📄 data/adjacency_matrix.csv - Road connections")
        print("  📄 data/adjacency_matrix_water_*m.csv - Risk scores at different water levels")
        print("  🌐 test_visualization.html - Route visualization (open in browser)")
        print("  🎬 animated_simulation.html - Animated simulation")
        print("  📊 simulation_data.json - Raw simulation data")
        
        print("\nNext Steps:")
        print("  1. Open test_visualization.html in browser to see route")
        print("  2. Review adjacency_matrix_updated.csv for risk calculations")
        print("  3. Start API server: python -m src.api.routing_api")
        print("  4. Run integration tests: python test_integration.py")
        
        print("\n" + "=" * 70)
        print("✅ Real Data Integration: COMPLETE")
        print("=" * 70)
        
        return True
    
    except Exception as e:
        print(f"\n✗ Fatal Error: {e}")
        import traceback
        traceback.print_exc()
        
        print("\n" + "=" * 70)
        print("❌ Real Data Integration: FAILED")
        print("=" * 70)
        print("\nTroubleshooting:")
        print("  1. Verify data files exist: ls data/")
        print("  2. Check Python version: python --version (need >= 3.9)")
        print("  3. Verify dependencies: pip list | grep geopandas")
        print("  4. See REAL_DATA_USAGE_GUIDE.md for detailed help")
        
        return False


if __name__ == "__main__":
    success = demonstrate_complete_workflow()
    sys.exit(0 if success else 1)
