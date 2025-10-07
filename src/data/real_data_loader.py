"""
Real Data Loader for MAS-FRO
Loads authentic elevation and network data from the data folder.
All outputs are based on actual processed inputs, not mock data.
"""

import os
import logging
import pandas as pd
import geopandas as gpd
import networkx as nx
from typing import Dict, List, Tuple, Optional, Any
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class RealDataLoader:
    """
    Loads real data from the data folder for MAS-FRO system.
    
    Data Files Expected:
    - road_networks_marikina.gpkg or road_networks.geojson: Road network with elevation data
    - evacuation_centers_marikina.gpkg or evacuation_centers.csv: Evacuation center locations
    
    Required Fields:
    - Road Network: geometry, osmid (optional: elevation, highway, name)
    - Evacuation Centers: name, latitude, longitude, capacity, type
    """
    
    def __init__(self, data_folder: str = "data"):
        """
        Initialize data loader.
        
        Args:
            data_folder: Path to folder containing data files
        """
        self.data_folder = Path(data_folder)
        
        if not self.data_folder.exists():
            raise FileNotFoundError(
                f"Data folder not found: {self.data_folder}\n"
                f"Please ensure the 'data' directory exists with required files."
            )
        
        logger.info(f"Initialized RealDataLoader with folder: {self.data_folder}")
    
    def load_road_network(self, use_gpkg: bool = True) -> gpd.GeoDataFrame:
        """
        Load real road network data from GPKG or GeoJSON file.
        
        Args:
            use_gpkg: If True, load from GPKG file; otherwise use GeoJSON
        
        Returns:
            GeoDataFrame containing road network with geometry and attributes
        
        Required Fields in Output:
            - geometry: LineString geometries for roads
            - osmid: OpenStreetMap ID (optional but recommended)
            - elevation_start: Elevation at start of road segment (meters)
            - elevation_end: Elevation at end of road segment (meters)
            - highway: Road type classification
            - length: Road segment length in meters
        
        Raises:
            FileNotFoundError: If data file doesn't exist
            ValueError: If data is malformed or missing required fields
        """
        try:
            if use_gpkg:
                file_path = self.data_folder / "road_networks_marikina.gpkg"
                if not file_path.exists():
                    raise FileNotFoundError(
                        f"GPKG file not found: {file_path}\n"
                        f"Available files: {list(self.data_folder.glob('*.gpkg'))}"
                    )
                
                logger.info(f"Loading road network from GPKG: {file_path}")
                gdf = gpd.read_file(file_path)
            
            else:
                file_path = self.data_folder / "road_networks.geojson"
                if not file_path.exists():
                    raise FileNotFoundError(
                        f"GeoJSON file not found: {file_path}\n"
                        f"Available files: {list(self.data_folder.glob('*.geojson'))}"
                    )
                
                logger.info(f"Loading road network from GeoJSON: {file_path}")
                gdf = gpd.read_file(file_path, encoding='utf-8')
            
            # Validate required fields
            if 'geometry' not in gdf.columns:
                raise ValueError("Road network data missing 'geometry' column")
            
            # Add elevation data if not present (extract from z-coordinates or use default)
            if 'elevation' not in gdf.columns:
                logger.warning("Elevation data not found, extracting from geometry or using defaults")
                gdf = self._add_elevation_data(gdf)
            
            # Calculate length if not present
            if 'length' not in gdf.columns:
                logger.info("Calculating road segment lengths")
                gdf['length'] = gdf.geometry.length * 111000  # Convert degrees to meters
            
            logger.info(f"Loaded {len(gdf)} road segments with {len(gdf.columns)} attributes")
            
            return gdf
        
        except Exception as e:
            logger.error(f"Error loading road network: {e}")
            raise
    
    def load_evacuation_centers(self, use_csv: bool = True) -> gpd.GeoDataFrame:
        """
        Load real evacuation center data.
        
        Args:
            use_csv: If True, load from CSV; otherwise use GPKG
        
        Returns:
            GeoDataFrame containing evacuation centers
        
        Required CSV Columns:
            - name: Evacuation center name
            - latitude: Latitude coordinate (WGS84)
            - longitude: Longitude coordinate (WGS84)
            - capacity: Maximum evacuee capacity
            - type: Facility type (e.g., school, sports_facility)
        
        Raises:
            FileNotFoundError: If data file doesn't exist
            ValueError: If required columns are missing
        """
        try:
            if use_csv:
                file_path = self.data_folder / "evacuation_centers.csv"
                if not file_path.exists():
                    raise FileNotFoundError(
                        f"CSV file not found: {file_path}\n"
                        f"Please ensure evacuation_centers.csv exists in data folder"
                    )
                
                logger.info(f"Loading evacuation centers from CSV: {file_path}")
                df = pd.read_csv(file_path, encoding='utf-8')
                
                # Validate required columns
                required_cols = ['name', 'latitude', 'longitude', 'capacity', 'type']
                missing_cols = [col for col in required_cols if col not in df.columns]
                if missing_cols:
                    raise ValueError(
                        f"CSV missing required columns: {missing_cols}\n"
                        f"Required: {required_cols}\n"
                        f"Found: {list(df.columns)}"
                    )
                
                # Convert to GeoDataFrame
                gdf = gpd.GeoDataFrame(
                    df,
                    geometry=gpd.points_from_xy(df.longitude, df.latitude),
                    crs='EPSG:4326'
                )
            
            else:
                file_path = self.data_folder / "evacuation_centers_marikina.gpkg"
                if not file_path.exists():
                    raise FileNotFoundError(f"GPKG file not found: {file_path}")
                
                logger.info(f"Loading evacuation centers from GPKG: {file_path}")
                gdf = gpd.read_file(file_path)
            
            logger.info(f"Loaded {len(gdf)} evacuation centers")
            
            return gdf
        
        except Exception as e:
            logger.error(f"Error loading evacuation centers: {e}")
            raise
    
    def _add_elevation_data(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Add elevation data to road network.
        
        Strategy:
        1. Try to extract from geometry z-coordinates
        2. Use topographic data if available
        3. Assign default based on Marikina elevation range (10-50m)
        
        Args:
            gdf: Road network GeoDataFrame
        
        Returns:
            GeoDataFrame with elevation columns added
        """
        from shapely.geometry import LineString, MultiLineString
        
        # Helper function to safely get coordinates from any geometry type
        def get_coords(geom):
            """Extract coordinates from LineString or MultiLineString"""
            try:
                if isinstance(geom, LineString):
                    return list(geom.coords)
                elif isinstance(geom, MultiLineString):
                    # For MultiLineString, use the first LineString
                    return list(geom.geoms[0].coords) if len(geom.geoms) > 0 else []
                else:
                    # For other geometry types, try to get coords
                    return list(geom.coords) if hasattr(geom, 'coords') else []
            except:
                return []
        
        # Check if geometry has z-coordinates
        first_geom = gdf.geometry.iloc[0] if len(gdf) > 0 else None
        has_z = False
        
        if first_geom is not None:
            try:
                if isinstance(first_geom, LineString):
                    has_z = first_geom.has_z
                elif isinstance(first_geom, MultiLineString):
                    has_z = first_geom.geoms[0].has_z if len(first_geom.geoms) > 0 else False
            except:
                has_z = False
        
        if has_z:
            logger.info("Extracting elevation from z-coordinates")
            
            def get_start_elevation(geom):
                coords = get_coords(geom)
                return coords[0][2] if len(coords) > 0 and len(coords[0]) > 2 else 20.0
            
            def get_end_elevation(geom):
                coords = get_coords(geom)
                return coords[-1][2] if len(coords) > 0 and len(coords[-1]) > 2 else 20.0
            
            gdf['elevation_start'] = gdf.geometry.apply(get_start_elevation)
            gdf['elevation_end'] = gdf.geometry.apply(get_end_elevation)
        else:
            # Assign default elevation based on Marikina topography
            # Marikina elevation ranges from ~10m (river areas) to ~50m (higher ground)
            logger.warning(
                "No z-coordinates found. Using elevation estimates based on location.\n"
                "River areas: 10-15m, Low-lying: 15-25m, Medium: 25-35m, High: 35-50m"
            )
            
            # Estimate elevation based on latitude (lower lat = lower elevation near river)
            def estimate_elevation(geom):
                try:
                    coords = get_coords(geom)
                    if not coords:
                        return 20.0
                    
                    # Calculate average latitude
                    avg_lat = sum(c[1] for c in coords) / len(coords)
                    
                    # Marikina: lat ~14.61-14.68, lower = near river
                    # Map to 10-40m elevation range
                    normalized = (avg_lat - 14.61) / (14.68 - 14.61)
                    elevation = 10 + normalized * 30
                    return max(10.0, min(40.0, elevation))
                except Exception as e:
                    logger.debug(f"Error estimating elevation: {e}")
                    return 20.0
            
            gdf['elevation_start'] = gdf.geometry.apply(estimate_elevation)
            gdf['elevation_end'] = gdf['elevation_start']
        
        return gdf
    
    def create_networkx_graph(self, gdf: gpd.GeoDataFrame) -> nx.MultiDiGraph:
        """
        Convert GeoDataFrame to NetworkX graph.
        
        Args:
            gdf: Road network GeoDataFrame
        
        Returns:
            NetworkX MultiDiGraph with nodes and edges
        """
        from shapely.geometry import LineString, MultiLineString
        
        # Helper function to safely extract coordinates
        def get_coords_safe(geom):
            """Safely extract coordinates from any geometry type"""
            try:
                if isinstance(geom, LineString):
                    return list(geom.coords)
                elif isinstance(geom, MultiLineString):
                    # Use first LineString for MultiLineString
                    return list(geom.geoms[0].coords) if len(geom.geoms) > 0 else []
                else:
                    # Try direct access for other types
                    return list(geom.coords) if hasattr(geom, 'coords') else []
            except:
                return []
        
        try:
            import osmnx as ox
            
            # If the GeoDataFrame has OSM format, try to use osmnx
            if 'osmid' in gdf.columns and 'u' in gdf.columns and 'v' in gdf.columns:
                logger.info("Creating NetworkX graph from OSM-format data using osmnx")
                try:
                    # OSMnx expects specific format
                    G = ox.graph_from_gdfs(gdf, gdf.geometry.apply(lambda x: x.bounds))
                    logger.info(f"Created graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
                    return G
                except Exception as e:
                    logger.warning(f"OSMnx conversion failed ({e}), using manual conversion")
            
            # Manual conversion (more robust for various formats)
            logger.info("Creating NetworkX graph manually from GeoDataFrame")
            G = nx.MultiDiGraph()
            
            # Create nodes from line endpoints
            node_id = 0
            node_map = {}
            edges_created = 0
            
            for idx, row in gdf.iterrows():
                try:
                    geom = row.geometry
                    
                    # Skip invalid geometries
                    if geom is None or geom.is_empty:
                        continue
                    
                    # Get coordinates safely
                    coords = get_coords_safe(geom)
                    
                    if len(coords) < 2:
                        logger.debug(f"Skipping edge {idx}: insufficient coordinates")
                        continue
                    
                    # Start node (round to avoid floating point issues)
                    start_coord = (round(coords[0][0], 7), round(coords[0][1], 7))
                    if start_coord not in node_map:
                        node_map[start_coord] = node_id
                        G.add_node(
                            node_id,
                            x=start_coord[0],
                            y=start_coord[1],
                            elevation=row.get('elevation_start', 20.0)
                        )
                        node_id += 1
                    
                    # End node (round to avoid floating point issues)
                    end_coord = (round(coords[-1][0], 7), round(coords[-1][1], 7))
                    if end_coord not in node_map:
                        node_map[end_coord] = node_id
                        G.add_node(
                            node_id,
                            x=end_coord[0],
                            y=end_coord[1],
                            elevation=row.get('elevation_end', 20.0)
                        )
                        node_id += 1
                    
                    # Add edge
                    u = node_map[start_coord]
                    v = node_map[end_coord]
                    
                    # Calculate length if needed
                    length = row.get('length', None)
                    if length is None or pd.isna(length):
                        # Estimate from geometry
                        length = geom.length * 111000  # degrees to meters approximation
                    
                    G.add_edge(
                        u, v,
                        length=float(length),
                        highway=row.get('highway', 'unclassified'),
                        name=row.get('name', ''),
                        osmid=row.get('osmid', idx)
                    )
                    edges_created += 1
                    
                except Exception as e:
                    logger.debug(f"Error processing edge {idx}: {e}")
                    continue
            
            logger.info(f"Created graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
            
            if G.number_of_nodes() == 0:
                raise ValueError(
                    "No valid nodes created from road network data.\n"
                    "Check that the data contains valid LineString geometries."
                )
            
            return G
        
        except Exception as e:
            logger.error(f"Error creating NetworkX graph: {e}")
            raise
    
    def export_to_csv(
        self,
        graph: nx.MultiDiGraph,
        nodes_file: str = "nodes.csv",
        edges_file: str = "adjacency_matrix.csv"
    ):
        """
        Export NetworkX graph to CSV files for inspection and updates.
        
        Output Files:
        1. nodes.csv - Node data with elevations
        2. adjacency_matrix.csv - Edge data with distances and flood risk
        
        Args:
            graph: NetworkX graph to export
            nodes_file: Output filename for nodes
            edges_file: Output filename for edges
        """
        try:
            # Export nodes
            nodes_data = []
            for node, data in graph.nodes(data=True):
                nodes_data.append({
                    'node_id': node,
                    'latitude': data.get('y', 0),
                    'longitude': data.get('x', 0),
                    'elevation': data.get('elevation', 20.0)
                })
            
            nodes_df = pd.DataFrame(nodes_data)
            nodes_path = self.data_folder / nodes_file
            nodes_df.to_csv(nodes_path, index=False, encoding='utf-8')
            logger.info(f"Exported {len(nodes_df)} nodes to {nodes_path}")
            
            # Export edges
            edges_data = []
            for u, v, key, data in graph.edges(keys=True, data=True):
                edges_data.append({
                    'from_node': u,
                    'to_node': v,
                    'edge_key': key,
                    'distance': data.get('length', 0),
                    'highway_type': data.get('highway', 'unclassified'),
                    'flood_risk': 0.0  # Will be calculated
                })
            
            edges_df = pd.DataFrame(edges_data)
            edges_path = self.data_folder / edges_file
            edges_df.to_csv(edges_path, index=False, encoding='utf-8')
            logger.info(f"Exported {len(edges_df)} edges to {edges_path}")
            
            return nodes_path, edges_path
        
        except Exception as e:
            logger.error(f"Error exporting to CSV: {e}")
            raise


class FloodRiskCalculator:
    """
    Calculate flood risk scores for road segments using real elevation data.
    
    Flood Risk Formula:
        flood_risk = max(0, (flood_threshold - avg_elevation) / flood_threshold)
    
    Where:
        - flood_threshold: Water level at which flooding becomes critical (default: 12m)
        - avg_elevation: Average elevation of road segment
        - flood_risk: Normalized risk score (0.0 = no risk, 1.0 = maximum risk)
    """
    
    def __init__(self, flood_threshold: float = 12.0):
        """
        Initialize flood risk calculator.
        
        Args:
            flood_threshold: Critical flood water level in meters (default: 12m)
        """
        self.flood_threshold = flood_threshold
        logger.info(f"Initialized FloodRiskCalculator with threshold: {flood_threshold}m")
    
    def calculate_risk_from_elevation(
        self,
        elevation_start: float,
        elevation_end: float,
        water_level: float = 0.0
    ) -> float:
        """
        Calculate flood risk for a road segment based on elevation.
        
        Formula:
            avg_elevation = (elevation_start + elevation_end) / 2
            effective_threshold = flood_threshold - water_level
            flood_risk = max(0, (effective_threshold - avg_elevation) / effective_threshold)
        
        Args:
            elevation_start: Elevation at start of segment (meters)
            elevation_end: Elevation at end of segment (meters)
            water_level: Current water level in area (meters, default: 0)
        
        Returns:
            Flood risk score (0.0 to 1.0, or inf if submerged)
        
        Example:
            >>> calc = FloodRiskCalculator(flood_threshold=12.0)
            >>> risk = calc.calculate_risk_from_elevation(8.0, 10.0, water_level=2.0)
            >>> # avg_elevation = 9.0
            >>> # effective_threshold = 12.0 - 2.0 = 10.0
            >>> # flood_risk = (10.0 - 9.0) / 10.0 = 0.1
            >>> print(risk)  # 0.1
        """
        try:
            # Calculate average elevation
            avg_elevation = (elevation_start + elevation_end) / 2
            
            # Adjust threshold for current water level
            effective_threshold = self.flood_threshold - water_level
            
            # If already submerged, mark as impassable
            if avg_elevation <= water_level:
                logger.debug(f"Road submerged: elevation={avg_elevation:.1f}m, water={water_level:.1f}m")
                return float('inf')
            
            # Calculate normalized risk
            if avg_elevation >= effective_threshold:
                # Above flood threshold - very low risk
                return 0.0
            
            # Linear risk increase as elevation decreases
            flood_risk = max(0.0, (effective_threshold - avg_elevation) / effective_threshold)
            
            return min(1.0, flood_risk)  # Clamp to [0, 1]
        
        except Exception as e:
            logger.error(f"Error calculating flood risk: {e}")
            return 0.5  # Default moderate risk on error
    
    def update_adjacency_matrix(
        self,
        adjacency_file: str,
        nodes_file: str,
        output_file: str = "adjacency_matrix_updated.csv",
        water_level: float = 0.0
    ) -> pd.DataFrame:
        """
        Update adjacency matrix with recalculated flood risk scores.
        
        Process:
        1. Load nodes.csv to get elevations
        2. Load adjacency_matrix.csv
        3. For each edge, retrieve node elevations
        4. Calculate flood risk using formula
        5. Update flood_risk column
        6. Save updated matrix
        
        Args:
            adjacency_file: Path to adjacency matrix CSV
            nodes_file: Path to nodes CSV with elevations
            output_file: Output filename for updated matrix
            water_level: Current water level (meters)
        
        Returns:
            Updated adjacency matrix DataFrame
        
        Output CSV Columns:
            - from_node: Source node ID
            - to_node: Target node ID
            - edge_key: Edge identifier
            - distance: Road segment length (meters)
            - highway_type: Road classification
            - flood_risk (before): Original risk score
            - avg_elevation: Average elevation of segment
            - flood_risk (after): Recalculated risk score
        """
        try:
            # Load data files
            nodes_path = Path(adjacency_file).parent / nodes_file
            adjacency_path = Path(adjacency_file)
            
            if not nodes_path.exists():
                raise FileNotFoundError(f"Nodes file not found: {nodes_path}")
            if not adjacency_path.exists():
                raise FileNotFoundError(f"Adjacency matrix not found: {adjacency_path}")
            
            logger.info(f"Loading nodes from: {nodes_path}")
            nodes_df = pd.read_csv(nodes_path, encoding='utf-8')
            
            logger.info(f"Loading adjacency matrix from: {adjacency_path}")
            edges_df = pd.read_csv(adjacency_path, encoding='utf-8')
            
            # Create elevation lookup
            elevation_map = dict(zip(nodes_df['node_id'], nodes_df['elevation']))
            
            # Backup original flood risk
            edges_df['flood_risk_before'] = edges_df.get('flood_risk', 0.0)
            
            # Calculate new flood risks
            updated_risks = []
            avg_elevations = []
            
            for idx, row in edges_df.iterrows():
                from_node = row['from_node']
                to_node = row['to_node']
                
                # Get elevations
                elev_start = elevation_map.get(from_node, 20.0)
                elev_end = elevation_map.get(to_node, 20.0)
                
                # Calculate risk
                risk = self.calculate_risk_from_elevation(
                    elev_start, elev_end, water_level
                )
                
                avg_elevation = (elev_start + elev_end) / 2
                
                updated_risks.append(risk)
                avg_elevations.append(avg_elevation)
            
            # Update dataframe
            edges_df['avg_elevation'] = avg_elevations
            edges_df['flood_risk'] = updated_risks
            
            # Rename for clarity
            edges_df.rename(columns={'flood_risk_before': 'flood_risk (before)'}, inplace=True)
            edges_df.rename(columns={'flood_risk': 'flood_risk (after)'}, inplace=True)
            
            # Save updated matrix
            output_path = Path(adjacency_file).parent / output_file
            edges_df.to_csv(output_path, index=False, encoding='utf-8')
            
            logger.info(f"Updated adjacency matrix saved to: {output_path}")
            logger.info(f"Processed {len(edges_df)} edges with water level: {water_level}m")
            
            # Print summary statistics
            logger.info(f"Risk Statistics:")
            logger.info(f"  Mean flood risk: {edges_df['flood_risk (after)'].mean():.3f}")
            logger.info(f"  Max flood risk: {edges_df['flood_risk (after)'].max():.3f}")
            logger.info(f"  Min elevation: {edges_df['avg_elevation'].min():.1f}m")
            logger.info(f"  Max elevation: {edges_df['avg_elevation'].max():.1f}m")
            
            return edges_df
        
        except Exception as e:
            logger.error(f"Error updating adjacency matrix: {e}")
            raise


# Example usage and testing
def demonstrate_real_data_usage():
    """
    Demonstrate loading and processing real data files.
    """
    print("=" * 70)
    print("MAS-FRO Real Data Integration Demonstration")
    print("=" * 70)
    
    # Initialize data loader
    print("\n[Step 1] Loading Real Data Files")
    print("-" * 40)
    
    try:
        loader = RealDataLoader(data_folder="data")
        
        # Load evacuation centers
        print("\nLoading evacuation centers from CSV...")
        evac_centers = loader.load_evacuation_centers(use_csv=True)
        print(f"✓ Loaded {len(evac_centers)} evacuation centers")
        print(f"  Columns: {list(evac_centers.columns)}")
        print(f"\n  Sample Data:")
        print(evac_centers[['name', 'latitude', 'longitude', 'capacity', 'type']].head(3).to_string(index=False))
        
        # Load road network
        print("\nLoading road network from GPKG...")
        try:
            road_network = loader.load_road_network(use_gpkg=True)
            print(f"✓ Loaded {len(road_network)} road segments")
            print(f"  Columns: {list(road_network.columns)[:10]}...")  # First 10 columns
        except FileNotFoundError:
            print("  GPKG not found, trying GeoJSON...")
            road_network = loader.load_road_network(use_gpkg=False)
            print(f"✓ Loaded {len(road_network)} road segments from GeoJSON")
        
        # Convert to NetworkX
        print("\n[Step 2] Converting to NetworkX Graph")
        print("-" * 40)
        G = loader.create_networkx_graph(road_network)
        print(f"✓ Created graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        
        # Export to CSV
        print("\n[Step 3] Exporting to CSV for Updates")
        print("-" * 40)
        nodes_path, edges_path = loader.export_to_csv(G)
        print(f"✓ Nodes exported to: {nodes_path}")
        print(f"✓ Edges exported to: {edges_path}")
        
        # Calculate flood risks
        print("\n[Step 4] Calculating Flood Risks")
        print("-" * 40)
        
        calculator = FloodRiskCalculator(flood_threshold=12.0)
        
        # Update with different water levels
        for water_level in [0.0, 2.0, 5.0]:
            print(f"\nScenario: Water Level = {water_level}m")
            output_file = f"adjacency_matrix_water_{int(water_level)}m.csv"
            
            updated_df = calculator.update_adjacency_matrix(
                adjacency_file=str(edges_path),
                nodes_file="nodes.csv",
                output_file=output_file,
                water_level=water_level
            )
            
            # Show sample updates
            sample = updated_df[['from_node', 'to_node', 'distance', 
                                'flood_risk (before)', 'avg_elevation', 
                                'flood_risk (after)']].head(5)
            print(f"\n  Sample Updates:")
            print(sample.to_string(index=False))
        
        print("\n" + "=" * 70)
        print("✓ Real Data Integration Complete!")
        print("=" * 70)
        
        return loader, G, evac_centers
    
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    demonstrate_real_data_usage()
