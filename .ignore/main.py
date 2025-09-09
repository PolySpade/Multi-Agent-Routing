from helpers.utils import load_gis_data
from pathlib import Path 
import matplotlib.pyplot as plt

main_dir = Path(__file__).parent 

if __name__ == '__main__':





    # shapefile_path = main_dir / 'data' / 'marikina_boundary_shapefiles' / 'marikina_boundary_shapefile.shp'
    # # print(f"Constructed path: {shapefile_path}")
    # # print(f"Does the file exist? {shapefile_path.exists()}")

    # my_geodataframe = load_gis_data(shapefile_path)
    

    # # You can now work with the loaded data
    # if my_geodataframe is not None:
    #     print("--- First 5 Rows (shows column names) ---")
    #     print(my_geodataframe.head())
    #     print("Generating Plot")
    #     fig, ax = plt.subplots(1,1, figsize=(10,10))
    #     my_geodataframe.plot(ax=ax, color='lightblue', edgecolor='black')
    #     ax.set_title("Map of Marikina Boundary", fontsize=16)
    #     ax.set_xlabel("Longitude")
    #     ax.set_ylabel("Latitude")
    #     plt.tight_layout()
    #     plt.show()

    