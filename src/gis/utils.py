import geopandas as gpd

def load_gis_data(file_path):
    """
    Loads GIS data (e.g., shapefile) into a GeoDataFrame.
    
    Args:
        file_path (str): The path to the GIS file.
        
    Returns:
        gpd.GeoDataFrame: The loaded geospatial data.
    """
    try:
        gdf = gpd.read_file(file_path)
        print(f"Successfully loaded {file_path}")
        # Optional: Reproject to a projected CRS for accurate distance calculations
        # gdf = gdf.to_crs("EPSG:32651") # WGS 84 / UTM zone 51N for the Philippines
        return gdf
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None