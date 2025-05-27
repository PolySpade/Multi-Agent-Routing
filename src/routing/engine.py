import networkx as nx
import geopandas as gpd

def create_road_network(gdf_roads):
    """
    Creates a NetworkX graph from a GeoDataFrame of roads.
    
    Args:
        gdf_roads (gpd.GeoDataFrame): GeoDataFrame containing road linestrings.
        
    Returns:
        nx.Graph: A graph representing the road network.
    """
    G = nx.Graph()
    for _, row in gdf_roads.iterrows():
        # Add edges for each road segment with weight as its length
        G.add_edge(row.geometry.coords[0], row.geometry.coords[-1], weight=row.geometry.length)
    print("Road network graph created.")
    return G

def find_path(graph, start_node, end_node, algorithm='astar'):
    """
    Finds the shortest path in the graph using a specified algorithm.
    
    Args:
        graph (nx.Graph): The network graph.
        start_node (tuple): The (lon, lat) of the starting point.
        end_node (tuple): The (lon, lat) of the ending point.
        algorithm (str): 'astar' or 'dijkstra'.
        
    Returns:
        list: A list of nodes representing the path, or None if no path is found.
    """
    try:
        if algorithm == 'astar':
            path = nx.astar_path(graph, start_node, end_node, heuristic=None, weight='weight')
            return path
        elif algorithm == 'dijkstra':
            path = nx.dijkstra_path(graph, start_node, end_node, weight='weight')
            return path
    except nx.NetworkXNoPath:
        print("No path could be found between the specified nodes.")
        return None