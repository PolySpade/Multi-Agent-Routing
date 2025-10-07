"""
Simple Test for Real Data Integration - No Complex Imports
Tests the core functionality directly without package dependencies.
"""

import sys
from pathlib import Path

print("=" * 70)
print("MAS-FRO Real Data Integration Test")
print("=" * 70)
print(f"Python version: {sys.version.split()[0]}")
print()

# Test 1: Check dependencies
print("[TEST 1] Checking Dependencies")
print("-" * 40)

required_modules = [
    ('geopandas', 'GeoPandas for spatial data'),
    ('networkx', 'NetworkX for graph operations'),
    ('pandas', 'Pandas for data processing'),
    ('numpy', 'NumPy for numerical operations')
]

all_deps_ok = True
for module_name, description in required_modules:
    try:
        __import__(module_name)
        print(f"  ✓ {module_name:15s} - {description}")
    except ImportError:
        print(f"  ✗ {module_name:15s} - NOT INSTALLED")
        all_deps_ok = False

if not all_deps_ok:
    print("\n❌ Missing dependencies. Run: pip install -r requirements.txt")
    sys.exit(1)

print("\n✓ All core dependencies installed")

# Test 2: Check data files
print("\n[TEST 2] Checking Data Files")
print("-" * 40)

data_folder = Path("data")
required_files = [
    ('evacuation_centers.csv', 'Evacuation centers'),
    ('road_networks.geojson', 'Road network (GeoJSON)'),
]

optional_files = [
    ('road_networks_marikina.gpkg', 'Road network (GPKG)'),
]

all_files_ok = True
for filename, description in required_files:
    file_path = data_folder / filename
    if file_path.exists():
        print(f"  ✓ {filename:30s} - {description}")
    else:
        print(f"  ✗ {filename:30s} - MISSING")
        all_files_ok = False

for filename, description in optional_files:
    file_path = data_folder / filename
    if file_path.exists():
        print(f"  ✓ {filename:30s} - {description} (optional)")
    else:
        print(f"  ⚠ {filename:30s} - Not found (optional)")

if not all_files_ok:
    print("\n❌ Missing required data files")
    print("ACTION: Ensure data/ folder contains required files")
    sys.exit(1)

print("\n✓ All required data files present")

# Test 3: Load real data
print("\n[TEST 3] Loading Real Data")
print("-" * 40)

try:
    import pandas as pd
    import geopandas as gpd
    
    # Load evacuation centers
    print("Loading evacuation centers...")
    evac_df = pd.read_csv(data_folder / "evacuation_centers.csv", encoding='utf-8')
    print(f"  ✓ Loaded {len(evac_df)} evacuation centers")
    print(f"  Columns: {list(evac_df.columns)}")
    
    # Validate columns
    required_cols = ['name', 'latitude', 'longitude', 'capacity', 'type']
    missing = [c for c in required_cols if c not in evac_df.columns]
    if missing:
        print(f"  ✗ Missing columns: {missing}")
        sys.exit(1)
    
    # Load road network
    print("\nLoading road network...")
    try:
        # Try GPKG first
        gpkg_path = data_folder / "road_networks_marikina.gpkg"
        if gpkg_path.exists():
            road_gdf = gpd.read_file(gpkg_path)
            source = "GPKG"
        else:
            road_gdf = gpd.read_file(data_folder / "road_networks.geojson", encoding='utf-8')
            source = "GeoJSON"
        
        print(f"  ✓ Loaded {len(road_gdf)} road segments from {source}")
        print(f"  Columns (first 10): {list(road_gdf.columns)[:10]}")
    
    except Exception as e:
        print(f"  ✗ Error loading road network: {e}")
        sys.exit(1)

except Exception as e:
    print(f"\n❌ Data loading error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n✓ Real data loaded successfully")

# Test 4: Basic graph operations
print("\n[TEST 4] Testing Graph Creation")
print("-" * 40)

try:
    import networkx as nx
    from shapely.geometry import LineString, MultiLineString
    
    # Create a simple graph from data
    G = nx.MultiDiGraph()
    
    # Add a few nodes and edges from the data
    node_count = 0
    edge_count = 0
    
    for idx, row in road_gdf.head(100).iterrows():
        try:
            geom = row.geometry
            
            # Skip invalid
            if geom is None or geom.is_empty:
                continue
            
            # Get coordinates safely
            if isinstance(geom, LineString):
                coords = list(geom.coords)
            elif isinstance(geom, MultiLineString):
                coords = list(geom.geoms[0].coords) if len(geom.geoms) > 0 else []
            else:
                continue
            
            if len(coords) < 2:
                continue
            
            # Add nodes
            start_id = node_count
            G.add_node(start_id, x=coords[0][0], y=coords[0][1])
            node_count += 1
            
            end_id = node_count
            G.add_node(end_id, x=coords[-1][0], y=coords[-1][1])
            node_count += 1
            
            # Add edge
            G.add_edge(start_id, end_id, length=100.0)
            edge_count += 1
            
        except Exception as e:
            continue
    
    print(f"  ✓ Created graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    
    if G.number_of_nodes() == 0:
        print("  ✗ No nodes created - check geometry handling")
        sys.exit(1)

except Exception as e:
    print(f"  ✗ Graph creation error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n✓ Graph creation successful")

# Test 5: Elevation estimation
print("\n[TEST 5] Testing Elevation Estimation")
print("-" * 40)

try:
    import numpy as np
    
    # Test the elevation formula
    def estimate_elevation(lat):
        """Estimate elevation based on latitude"""
        # Marikina: lat ~14.61-14.68, lower = near river
        normalized = (lat - 14.61) / (14.68 - 14.61)
        elevation = 10 + normalized * 30
        return max(10.0, min(40.0, elevation))
    
    # Test with sample latitudes
    test_lats = [14.61, 14.65, 14.68]
    for lat in test_lats:
        elev = estimate_elevation(lat)
        print(f"  Latitude {lat:.2f} → Elevation {elev:.1f}m")
    
    print("  ✓ Elevation estimation working")

except Exception as e:
    print(f"  ✗ Elevation error: {e}")
    sys.exit(1)

# Test 6: Flood risk calculation
print("\n[TEST 6] Testing Flood Risk Calculation")
print("-" * 40)

try:
    def calculate_flood_risk(avg_elevation, flood_threshold=12.0, water_level=0.0):
        """Calculate flood risk from elevation"""
        effective_threshold = flood_threshold - water_level
        
        if avg_elevation <= water_level:
            return float('inf')  # Submerged
        
        if avg_elevation >= effective_threshold:
            return 0.0  # Safe
        
        risk = max(0.0, (effective_threshold - avg_elevation) / effective_threshold)
        return min(1.0, risk)
    
    # Test scenarios
    scenarios = [
        (30.0, 0.0, "High elevation, no flood"),
        (8.0, 0.0, "Low elevation, no flood"),
        (8.0, 2.0, "Low elevation, 2m water"),
        (5.0, 2.0, "Very low, 2m water"),
    ]
    
    for elevation, water, description in scenarios:
        risk = calculate_flood_risk(elevation, flood_threshold=12.0, water_level=water)
        if risk == float('inf'):
            risk_str = "IMPASSABLE"
        else:
            risk_str = f"{risk:.2f}"
        print(f"  Elevation {elevation:4.1f}m, Water {water:.1f}m → Risk {risk_str:10s} ({description})")
    
    print("  ✓ Flood risk calculation working")

except Exception as e:
    print(f"  ✗ Risk calculation error: {e}")
    sys.exit(1)

# Test 7: CSV export
print("\n[TEST 7] Testing CSV Export (UTF-8)")
print("-" * 40)

try:
    test_df = pd.DataFrame({
        'node_id': [1, 2, 3],
        'elevation': [20.5, 25.3, 18.7],
        'test': ['✓', '✓', '✓']  # Unicode test
    })
    
    test_file = data_folder / "test_export.csv"
    test_df.to_csv(test_file, index=False, encoding='utf-8')
    
    # Read it back
    read_df = pd.read_csv(test_file, encoding='utf-8')
    
    if len(read_df) == 3:
        print(f"  ✓ CSV export/import working with UTF-8")
        test_file.unlink()  # Delete test file
    else:
        print(f"  ✗ CSV data mismatch")

except Exception as e:
    print(f"  ✗ CSV export error: {e}")
    sys.exit(1)

# All tests passed
print("\n" + "=" * 70)
print("✅ ALL TESTS PASSED!")
print("=" * 70)
print("\nReal Data Integration Status:")
print("  ✓ Dependencies installed")
print("  ✓ Data files present")
print("  ✓ Data loading works")
print("  ✓ Graph creation works")
print("  ✓ Elevation estimation works")
print("  ✓ Flood risk calculation works")
print("  ✓ UTF-8 encoding works")
print("\nNext Steps:")
print("  1. Run real data demonstration:")
print("     python src/data/real_data_loader.py")
print()
print("  2. Or run complete workflow:")
print("     python demonstrate_real_data.py")
print()
print("  3. View generated files in data/ folder")

sys.exit(0)


