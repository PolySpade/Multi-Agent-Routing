# Quick Start: MAS-FRO with Real Data
**5-Minute Guide to Using Authentic Marikina Network Data**

---

## ⚡ Super Quick Start (3 Commands)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Load and process real data
python demonstrate_real_data.py

# 3. View results
# Open test_visualization.html in your browser
```

---

## 📋 Installation (One-Time Setup)

```bash
# Ensure Python 3.9+
python --version

# Install dependencies
pip install -r requirements.txt
pip install fastapi uvicorn  # For API features

# Verify installation
python -c "import geopandas, networkx, fastapi; print('✓ Ready!')"
```

---

## 🗺️ Real Data Files

**Already in your `data/` folder:**
- ✅ `evacuation_centers.csv` - 6 real evacuation centers in Marikina
- ✅ `road_networks.geojson` - Actual Marikina road network
- ✅ `road_networks_marikina.gpkg` - GeoPackage format (if available)

**Auto-generated after running scripts:**
- `nodes.csv` - Node elevations (extracted from network)
- `adjacency_matrix.csv` - Road connections with distances
- `adjacency_matrix_water_*m.csv` - Flood risk scores at different water levels

---

## 🚀 Usage Examples

### Example 1: Load Real Data and Calculate Risks

```python
from src.data.real_data_loader import RealDataLoader, FloodRiskCalculator

# Load real Marikina data
loader = RealDataLoader("data")
network = loader.load_road_network()  # 2500+ real road segments
centers = loader.load_evacuation_centers()  # 6 real centers

# Create graph
G = loader.create_networkx_graph(network)
print(f"Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# Export to CSV
loader.export_to_csv(G)

# Calculate flood risks (formula: flood_risk = (threshold - elevation) / threshold)
calc = FloodRiskCalculator(flood_threshold=12.0)
updated = calc.update_adjacency_matrix(
    adjacency_file="data/adjacency_matrix.csv",
    nodes_file="data/nodes.csv",
    water_level=2.0  # 2m flood water
)

print(f"✓ Updated {len(updated)} road segments with real flood risks")
```

### Example 2: Route Calculation with Real Network

```python
from src.agents.risk_aware_routing import RiskAwareAStar

# Use real graph and risk scores from Example 1
router = RiskAwareAStar()

result = router.calculate_route(
    graph=G,
    start=1,      # Real node from Marikina network
    goal=500,     # Real node from Marikina network
    risk_scores=risk_scores_dict,
    rain_rate=15.0  # Current rainfall
)

# Result contains:
# - path: Actual node IDs from Marikina road network
# - total_risk: Calculated from real elevation data
# - safety_score: Based on actual flood risk assessment
```

### Example 3: Start API Server

```bash
# Start server with real data backend
python -m src.api.routing_api

# API runs at: http://localhost:8000
# Docs at: http://localhost:8000/docs
```

**Test API:**
```bash
curl http://localhost:8000/algorithms
```

---

## 📊 Flood Risk Calculation

### Formula

```
avg_elevation = (elevation_start + elevation_end) / 2
effective_threshold = flood_threshold - water_level
flood_risk = max(0, (effective_threshold - avg_elevation) / effective_threshold)
```

### Examples (flood_threshold=12m)

| Water Level | Elevation | Calculation | Risk | Status |
|-------------|-----------|-------------|------|--------|
| 0m | 30m | (12-30)/12 | 0.00 | ✓ Safe |
| 0m | 8m | (12-8)/12 | 0.33 | ⚠ Moderate |
| 2m | 8m | (10-8)/10 | 0.20 | ⚠ Low |
| 2m | 5m | (10-5)/10 | 0.50 | ⚠ High |
| 5m | 5m | (7-5)/7 | 0.29 | ⚠ Moderate |
| 5m | 3m | (7-3)/7 | 0.57 | ⚠ High |
| 5m | ≤5m | N/A | ∞ | ❌ Impassable |

---

## 🔧 Troubleshooting (30 Seconds)

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError: fastapi` | `pip install fastapi uvicorn` |
| `ImportError: relative import` | Run with `python -m src.api.routing_api` |
| `UnicodeEncodeError` | Already fixed - all files use `encoding='utf-8'` |
| `FileNotFoundError: road_networks` | Check `ls data/` - files must exist |
| API won't start | Check port 8000 not in use: `netstat -an \| find "8000"` |

---

## 📁 Output Files

After running `demonstrate_real_data.py`:

```
test_visualization.html       # ← Open this in browser!
animated_simulation.html      # ← Interactive time-stepped simulation
simulation_data.json          # Raw data export
data/nodes.csv                # Node elevations
data/adjacency_matrix.csv     # Road connections
data/adjacency_matrix_water_0m.csv
data/adjacency_matrix_water_2m.csv  # ← Use this for typical flooding
data/adjacency_matrix_water_5m.csv
```

---

## ✅ Validation Checklist

**Before Using:**
- [ ] Python 3.9+ installed
- [ ] Dependencies installed (`pip list | grep geopandas`)
- [ ] Data files exist (`ls data/*.csv data/*.geojson`)

**After Processing:**
- [ ] `nodes.csv` created with elevations
- [ ] `adjacency_matrix.csv` contains edges
- [ ] Flood risks calculated (check `*_water_*.csv` files)
- [ ] Visualizations generated (`.html` files)
- [ ] No errors in console output

---

## 🎯 Key Differences: Sample vs Real Data

| Aspect | Sample/Mock Data | Real Data (NEW) |
|--------|------------------|-----------------|
| **Network Source** | Hardcoded coordinates | `road_networks_marikina.gpkg` |
| **Nodes** | 5-10 sample nodes | 1800+ real nodes |
| **Edges** | 10-15 sample edges | 2500+ real edges |
| **Elevations** | Random/fixed (20m) | Actual topography (10-50m) |
| **Evacuation Centers** | 3 sample centers | 6 real Marikina centers |
| **Flood Risk** | Random generation | Calculated from real elevations |
| **Coordinates** | Approximate | Actual WGS84 coordinates |

---

## 📖 Full Documentation

For detailed information, see:
- **REAL_DATA_USAGE_GUIDE.md** - Complete end-to-end guide
- **FEATURE_IMPLEMENTATION_SUMMARY.md** - All new features
- **project-specs.md** - Technical specifications

---

**Last Updated:** October 2025  
**Status:** ✅ Production-ready with real Marikina data  
**Python Version:** >= 3.9
