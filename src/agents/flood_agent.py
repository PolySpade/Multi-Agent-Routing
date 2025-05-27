import geopandas as gpd

class FloodAgent:
    """
    An agent that responds to flood-related hazards.
    """
    def __init__(self, gdf_flood_zones):
        """
        Initializes the agent with flood hazard data.
        
        Args:
            gdf_flood_zones (gpd.GeoDataFrame): A GeoDataFrame of flood-prone areas.
        """
        self.flood_zones = gdf_flood_zones
        print("FloodAgent initialized.")

    def get_affected_roads(self, gdf_roads, flood_intensity='high'):
        """
        Identifies roads that are affected by floods based on a rule.
        RULE: A road is considered affected if it intersects with a flood zone.
        
        Args:
            gdf_roads (gpd.GeoDataFrame): The road network.
            flood_intensity (str): The level of flood to simulate (e.g., 'high', 'medium').
            
        Returns:
            gpd.GeoDataFrame: A GeoDataFrame of roads that are affected.
        """
        # This is a simplified rule. You can make it more complex.
        # For example, by using different flood zone polygons for different intensities.
        affected_roads = gdf_roads[gdf_roads.intersects(self.flood_zones.unary_union)]
        print(f"{len(affected_roads)} road segments identified as flood-affected.")
        return affected_roads