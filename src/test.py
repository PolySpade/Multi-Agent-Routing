import geopandas as gpd
import networkx as nx
import osmnx as ox
import matplotlib

# Explicitly set the backend for Matplotlib to avoid IDE-specific issues.
# 'TkAgg' is a widely compatible backend.
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
# from shapely.geometry import Point
import contextily as ctx
import random
from geopy.distance import great_circle


def download_real_network(place_name="Marikina, Philippines"):
    """
    Downloads a real street network from OpenStreetMap using OSMnx.
    This version returns a DIRECTED graph to respect one-way streets.

    Args:
        place_name (str): The name of the place to download the network for.

    Returns:
        tuple: A tuple containing the directed NetworkX graph (MultiDiGraph),
               and GeoDataFrames for nodes and edges.
    """
    print(f"Downloading street network for {place_name}...")
    # Download the street network. This is a MultiDiGraph, which is directed
    # and can have multiple edges between the same two nodes.
    graph = ox.graph_from_place(place_name, network_type='drive')

    # Convert the graph to GeoDataFrames for plotting and analysis
    nodes, edges = ox.graph_to_gdfs(graph, nodes=True, edges=True)

    return graph, nodes, edges

def add_risk_and_weights(G, gdf_edges):
    """
    Adds simulated risk to each edge and calculates a composite weight.
    If risk_factor is 1, the road is considered impassable (weight = infinity).
    """
    print("2. Simulating risk and creating a risk-aware weight...")

    # Simulate risk, ensuring some roads are impassable (risk=1)
    risk_factors = []
    for _ in range(len(gdf_edges)):
        # Let's say 5% of roads are completely flooded/impassable
        if random.random() < 0.05:
            risk_factors.append(1.0)
        else:
            # Keep other risks below 1 so they don't trigger the 'impassable' condition
            risk_factors.append(random.uniform(0.0, 0.9))
    gdf_edges['risk_factor'] = risk_factors

    # Normalize length for weighting
    max_length = gdf_edges['length'].max()
    gdf_edges['norm_length'] = gdf_edges['length'] / max_length

    # Calculate weights conditionally
    weights = []
    for index, row in gdf_edges.iterrows():
        if row['risk_factor'] == 1.0:
            # If risk is 1, the road is impassable
            weights.append(float('inf'))
        else:
            # Otherwise, calculate the composite weight
            weight = 0.8 * row['norm_length'] + 0.2 * row['risk_factor']
            weights.append(weight)

    gdf_edges['risk_aware_weight'] = weights

    # Add this new weight back to the graph itself
    for u, v, key, data in G.edges(keys=True, data=True):
        edge_data = gdf_edges.query(f"u == {u} and v == {v} and key == {key}").iloc[0]
        data['risk_aware_weight'] = edge_data['risk_aware_weight']

    return G, gdf_edges

# def load_and_join_risk_data(gdf_edges, risk_filepath):
#     """
#     Loads risk data from a GIS file and spatially joins it to the street network.
#     """
#     print("-> Loading custom risk data from GIS file...")
#     # Load the risk zones (e.g., flood map shapefile)
#     risk_zones = gpd.read_file(risk_filepath)

#     # IMPORTANT: Ensure the Coordinate Reference System (CRS) matches the street network
#     print(f"   - Projecting risk zones to match street network CRS ({gdf_edges.crs})...")
#     risk_zones = risk_zones.to_crs(gdf_edges.crs)

#     # Perform the spatial join
#     # This adds columns from risk_zones to each edge that intersects a risk zone.
#     # 'how=left' ensures we keep all original streets.
#     print("   - Performing spatial join between streets and risk zones...")
#     gdf_edges_with_risk = gpd.sjoin(gdf_edges, risk_zones, how='left', predicate='intersects')

#     return gdf_edges_with_risk


def distance_heuristic(u, v, G):
    """
    A heuristic function for A* algorithm based on great-circle distance.
    This version correctly accesses node coordinates from the graph attributes.
    """
    # Get the node data from the graph
    u_node_data = G.nodes[u]
    v_node_data = G.nodes[v]

    # Use the 'y' (latitude) and 'x' (longitude) attributes directly
    return great_circle((u_node_data['y'], u_node_data['x']), (v_node_data['y'], v_node_data['x'])).meters


def reconstruct_path_gdf(G, path_nodes, gdf_edges, weight_label):
    """
    Reconstructs a GeoDataFrame for a path from a list of nodes.
    This is necessary for MultiDiGraphs where nx.shortest_path doesn't return edge keys,
    so we must determine which parallel edge was chosen based on the minimum weight.
    """
    path_edge_list = []
    for u, v in zip(path_nodes[:-1], path_nodes[1:]):
        # Get all parallel edges between the two nodes
        edge_data_dict = G.get_edge_data(u, v)

        if edge_data_dict is None:
            continue

        min_weight = float('inf')
        min_key = None

        # Find the edge with the minimum weight that the pathfinding algorithm would have chosen
        for key, data in edge_data_dict.items():
            if data.get(weight_label, float('inf')) < min_weight:
                min_weight = data[weight_label]
                min_key = key

        # Now, retrieve this specific edge from the main GeoDataFrame
        if min_key is not None:
            edge_gdf = gdf_edges.query(f"u == {u} and v == {v} and key == {min_key}")
            if not edge_gdf.empty:
                path_edge_list.append(edge_gdf.iloc[0])

    return gpd.GeoDataFrame(path_edge_list, crs=gdf_edges.crs)


def main():
    """
    Main function to run the demonstration.
    """
    # --- Download Real Network Data ---
    print("1. Downloading a real street network with OSMnx...")
    G, gdf_nodes, gdf_edges = download_real_network()
    G, gdf_edges = add_risk_and_weights(G, gdf_edges)

    print(f"   - Graph for Pasay created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

    # --- Select Random Points for Simulation ---
    origin_node, dest_node = random.sample(list(G.nodes), 2)
    origin_point = gdf_nodes.loc[origin_node].geometry
    dest_point = gdf_nodes.loc[dest_node].geometry

    # --- Perform Network Analysis ---
    print("3. Calculating shortest path (Dijkstra)...")
    try:
        shortest_path_nodes = nx.shortest_path(G, source=origin_node, target=dest_node, weight='risk_aware_weight')
        # Reconstruct the GeoDataFrame from the list of nodes
        gdf_shortest_path = reconstruct_path_gdf(G, shortest_path_nodes, gdf_edges, 'length')
    except nx.NetworkXNoPath:
        print("   - No path found using standard shortest path.")
        gdf_shortest_path = None
    #add computational time
    print("4. Calculating risk-aware path (A*)...")
    try:
        risk_aware_path_nodes = nx.astar_path(G,
                                              source=origin_node,
                                              target=dest_node,
                                              weight='risk_aware_weight',
                                              heuristic=lambda u, v: distance_heuristic(u, v, G))
        # Reconstruct the GeoDataFrame from the list of nodes
        gdf_risk_aware_path = reconstruct_path_gdf(G, risk_aware_path_nodes, gdf_edges, 'risk_aware_weight')
    except nx.NetworkXNoPath:
        print("   - No path found using risk-aware A* path.")
        gdf_risk_aware_path = None

    # --- Visualization ---
    print("5. Visualizing the network and paths...")

    # Project layers for plotting
    gdf_edges_proj = gdf_edges.to_crs(epsg=3857)
    if gdf_shortest_path is not None and not gdf_shortest_path.empty:
        gdf_shortest_path_proj = gdf_shortest_path.to_crs(epsg=3857)
    else:
        gdf_shortest_path_proj = None

    if gdf_risk_aware_path is not None and not gdf_risk_aware_path.empty:
        gdf_risk_aware_path_proj = gdf_risk_aware_path.to_crs(epsg=3857)
    else:
        gdf_risk_aware_path_proj = None

    fig, ax = plt.subplots(figsize=(15, 15))

    # Plot the full network
    gdf_edges_proj.plot(ax=ax, color='gray', linestyle='-', linewidth=0.5, label='Full Street Network', alpha=0.6)

    # Plot the shortest path (Dijkstra)
    if gdf_shortest_path_proj is not None:
        gdf_shortest_path_proj.plot(ax=ax, color='dodgerblue', linewidth=2.5, label='Shortest Path (Distance)',
                                    zorder=2)

    # Plot the risk-aware path (A*)
    if gdf_risk_aware_path_proj is not None:
        gdf_risk_aware_path_proj.plot(ax=ax, color='springgreen', linewidth=2.5, linestyle='--',
                                      label='Risk-Aware Path (A*)', zorder=3)

    # Highlight the origin and destination points
    origin_dest_points = gpd.GeoDataFrame(geometry=[origin_point, dest_point], crs=gdf_edges.crs).to_crs(epsg=3857)
    origin_dest_points.plot(ax=ax, color=['lime', 'magenta'], markersize=150, zorder=4, edgecolor='black')

    ax.set_title('Shortest vs. Risk-Aware Routing (Marikina, Philippines)', fontsize=18)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.legend()

    # Add the basemap using contextily
    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)

    plt.show()


if __name__ == '__main__':
    main()
