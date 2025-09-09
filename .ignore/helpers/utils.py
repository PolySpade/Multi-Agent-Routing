import geopandas as gpd
def load_gis_data(file_path):
    """
    Loads GIS data (e.g., shapefile) into a GeoDataFrame.
    """
    try:
        gdf = gpd.read_file(file_path)
        print(f"Successfully loaded data from {file_path}")
        print(f"Coordinate Reference System (CRS): {gdf.crs}")
        return gdf
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None