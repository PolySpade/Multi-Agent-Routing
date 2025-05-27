from gis.utils import load_gis_data
from routing.engine import create_road_network, find_path
from agents.flood_agent import FloodAgent
import networkx as nx

def run_simulation():
    """
    Main function to run the routing simulations.
    """
    # 1. Load Data
    print("--- Loading GIS Data ---")
    gdf_roads = load_gis_data('data/your_marikina_roads.shp') # CHANGE FILENAME
    gdf_flood_zones = load_gis_data('data/your_marikina_flood_zones.shp') # CHANGE FILENAME
    
    if gdf_roads is None or gdf_flood_zones is None:
        print("Could not load necessary data. Exiting.")
        return

    # Define start and end points for a sample evacuation route
    # You'll need to get actual coordinates from your map
    start_point = (121.0, 14.6) # Replace with actual coordinates
    end_point = (121.1, 14.7)   # Replace with actual coordinates

    # 2. Scenario 1: Normal Routing
    print("\n--- Running Scenario 1: Normal Routing ---")
    road_network_normal = create_road_network(gdf_roads)
    # Note: You may need a function to find the closest network nodes to your start/end points
    normal_path = find_path(road_network_normal, start_point, end_point, algorithm='astar')
    if normal_path:
        print(f"Normal Path Found: {len(normal_path)} nodes.")
        # Add logic here to calculate distance, time, etc.

    # 3. Scenario 2: Disaster-Aware Routing
    print("\n--- Running Scenario 2: Disaster-Aware Routing ---")
    # Initialize the agent
    flood_agent = FloodAgent(gdf_flood_zones)
    
    # Get impassable roads from the agent
    affected_roads = flood_agent.get_affected_roads(gdf_roads)
    
    # Create a new graph for the disaster scenario
    road_network_disaster = create_road_network(gdf_roads)
    
    # Modify the graph based on agent's input: make affected roads impassable
    for _, road in affected_roads.iterrows():
        start_node = road.geometry.coords[0]
        end_node = road.geometry.coords[-1]
        if road_network_disaster.has_edge(start_node, end_node):
            road_network_disaster.remove_edge(start_node, end_node)
            
    disaster_path = find_path(road_network_disaster, start_point, end_point, algorithm='astar')
    if disaster_path:
        print(f"Disaster-Aware Path Found: {len(disaster_path)} nodes.")
        # Add logic here to calculate distance, time, and safety score

if __name__ == "__main__":
    run_simulation()