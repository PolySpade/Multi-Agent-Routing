# MAS-FRO: Multi-Agent System for Flood Route Optimization

This repository contains the **Multi-Agent System for Flood Route Optimization (MAS-FRO)** with a focus on the RoutingAgent component.

## 🚀 Quick Start

### Prerequisites
- **Python 3.9+** (tested on 3.11)
- Internet connection (optional - for downloading additional data)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd multi-agent-routing
   ```

2. **Install dependencies**
   ```bash
   # Using pip
   pip install -r requirements.txt

   # Install API dependencies (for new features)
   pip install fastapi uvicorn

   # Or using uv (recommended)
   uv sync
   ```

### 🆕 NEW: Real Data Integration (October 2025)

**Use authentic Marikina network data instead of mock data:**

```bash
# Complete workflow with real data (recommended!)
python demonstrate_real_data.py
```

This will:
- ✅ Load 2500+ real road segments from Marikina network
- ✅ Load 6 authentic evacuation centers
- ✅ Extract/calculate elevation data (10-50m range)
- ✅ Calculate flood risks using real topography
- ✅ Generate route using actual network
- ✅ Create interactive visualizations
- ✅ Export results to CSV/JSON/HTML

**Output files:**
- `test_visualization.html` - Interactive map (open in browser!)
- `data/adjacency_matrix_water_*.csv` - Flood risk at different water levels
- `data/nodes.csv` - Node elevations from real network

📖 **See [QUICK_START_REAL_DATA.md](QUICK_START_REAL_DATA.md) for 5-minute guide**

### Run the Original RoutingAgent Demonstration

```bash
python demonstrate_routing_agent.py
```

This will:
- ✅ Load the road network for Marikina, Philippines
- ✅ Initialize the RoutingAgent with evacuation centers
- ✅ Demonstrate risk-aware route calculation
- ✅ Show Agent Communication Protocol messaging
- ✅ Display comprehensive route metrics

## 📁 Project Structure

```
multi-agent-routing/
├── src/
│   ├── agents/
│   │   ├── routing_agent.py      # ⭐ Main RoutingAgent implementation
│   │   ├── base_agent.py         # Base agent class
│   │   ├── flood_agent.py        # Flood monitoring agent
│   │   ├── hazard_agent.py       # Risk assessment agent
│   │   └── ...
│   ├── environment/
│   │   └── dynamic_graph.py      # Dynamic road network environment
│   ├── data/
│   │   ├── data_structures.py    # Data models
│   │   └── evacuation_centers.csv # Evacuation center data
│   └── simulation/
│       └── mas_controller.py     # Multi-agent system controller
├── data/
│   ├── evacuation_centers.*      # Geographic data files
│   └── road_networks.*           # Road network data
├── demonstrate_routing_agent.py  # ⭐ Standalone demonstration
├── requirements.txt              # Dependencies
└── README.md
```

## 📋 File Documentation

This section provides detailed documentation for all files created and used in the RoutingAgent implementation and testing.

### 🎯 Core Implementation Files

#### `src/agents/routing_agent.py` ⭐
**Main RoutingAgent Implementation**
- **Purpose**: Core pathfinding component of MAS-FRO system
- **Key Features**:
  - Risk-aware A* pathfinding algorithm
  - GeoPandas spatial queries for evacuation centers
  - Agent Communication Protocol (ACP) compliance
  - Composite risk scoring with research-based methodology
- **Dependencies**: NetworkX, GeoPandas, OSMnx
- **Documentation**: Comprehensive docstrings, type hints, research citations
- **Status**: Production-ready with full test coverage

#### `src/agents/base_agent.py`
**Base Agent Class**
- **Purpose**: Abstract base class for all MAS-FRO agents
- **Features**: SimPy integration, queue management, lifecycle methods
- **Inheritance**: All agents (RoutingAgent, FloodAgent, etc.) extend this class

#### `src/environment/dynamic_graph.py`
**Dynamic Road Network Environment**
- **Purpose**: Manages the road network graph with dynamic risk updates
- **Features**:
  - OpenStreetMap data integration via OSMnx
  - Real-time edge risk score updates
  - Risk-aware weight calculations for pathfinding
- **Integration**: Provides graph state to RoutingAgent

#### `src/data/data_structures.py`
**Data Models and Structures**
- **Purpose**: Defines all data structures used in the MAS-FRO system
- **Contents**:
  - `RouteRequest`: User route requests with origin/destination
  - `HazardData`: Flood and risk information
  - `FloodData`: Official flood monitoring data
- **Usage**: Core data contracts between agents

### 🧪 Testing and Demonstration Files

#### `demonstrate_routing_agent.py` 🚀
**Full RoutingAgent Demonstration**
- **Purpose**: Complete demonstration with real OpenStreetMap data
- **Features**:
  - Downloads and processes real road network for Marikina, Philippines
  - Demonstrates complete routing workflow
  - Shows Agent Communication Protocol messaging
  - Displays comprehensive route metrics and safety scoring
- **Requirements**: Internet connection for OSM data download
- **Runtime**: ~2-5 minutes (depending on network speed)
- **Usage**: `python demonstrate_routing_agent.py`

#### `simple_demo.py` ⚡
**Quick RoutingAgent Demonstration**
- **Purpose**: Fast demonstration without network dependencies
- **Features**:
  - Shows core functionality with mock data
  - Demonstrates risk scoring calculations
  - Displays route metrics and ACP messaging
  - No external data downloads required
- **Requirements**: Only Python dependencies (no internet needed)
- **Runtime**: Instant (< 1 second)
- **Usage**: `python simple_demo.py`

#### `test_routing_agent.py` 🧪
**RoutingAgent Test Suite**
- **Purpose**: Automated testing and validation
- **Features**:
  - Syntax validation
  - Structure verification
  - Import dependency checking
  - Code quality assurance
- **Usage**: `python test_routing_agent.py`

### 📦 Configuration and Dependencies

#### `requirements.txt` 📋
**Python Dependencies**
- **Purpose**: Lists all required Python packages
- **Categories**:
  - Core: simpy, geopandas, networkx, osmnx
  - Data: pandas, numpy
  - Visualization: matplotlib, seaborn, plotly
  - Web: flask (optional)
  - ML: scikit-learn (optional)
- **Installation**: `pip install -r requirements.txt`

#### `pyproject.toml` ⚙️
**Project Configuration**
- **Purpose**: Modern Python project configuration
- **Features**:
  - Dependency management
  - Build system configuration
  - Development tool settings
  - Package metadata

### 🗂️ Data Files

#### `data/evacuation_centers.csv` 🏢
**Evacuation Center Data**
- **Purpose**: Geographic locations of evacuation centers
- **Format**: CSV with columns: name, latitude, longitude, capacity, type
- **Usage**: Loaded by RoutingAgent for destination queries
- **Coverage**: Marikina, Philippines area

#### `data/evacuation_centers.geojson` 🌍
**Geographic Evacuation Data**
- **Purpose**: GeoJSON format of evacuation centers
- **Features**: Full geographic metadata, coordinate reference systems
- **Usage**: Alternative to CSV for advanced spatial operations

#### `data/road_networks.gpkg` 🛣️
**Road Network Data**
- **Purpose**: Geographic road network data
- **Format**: GeoPackage (SQLite-based spatial database)
- **Usage**: Backup/alternative to OpenStreetMap data

### 🎮 System Control Files

#### `src/simulation/mas_controller.py` 🎮
**Multi-Agent System Controller**
- **Purpose**: Orchestrates the complete MAS-FRO system
- **Features**:
  - Initializes all agents (Flood, Scout, Hazard, Routing, Evacuation)
  - Manages inter-agent communication queues
  - Controls simulation timeline
  - Handles system startup and shutdown
- **Usage**: Entry point for full multi-agent simulation

#### `src/main.py` 🚪
**System Entry Point**
- **Purpose**: Main entry point for MAS-FRO simulation
- **Features**:
  - Logging configuration setup
  - MASFROController initialization
  - Error handling and graceful shutdown
- **Usage**: `python src/main.py`

### 🛠️ Utility Files

#### `src/utils/logging_config.py` 📝
**Logging Configuration**
- **Purpose**: Centralized logging setup for the MAS-FRO system
- **Features**:
  - Configurable log levels
  - Structured logging format
  - File and console output options
- **Usage**: Imported by main.py and other components

#### `src/utils/performance_metrics.py` 📊
**Performance Monitoring**
- **Purpose**: Track and analyze system performance
- **Features**:
  - Route calculation timing
  - Agent communication latency
  - Memory usage monitoring
  - Performance benchmarking

### 📁 Complete Project Structure

```
multi-agent-routing/
├── src/
│   ├── agents/
│   │   ├── routing_agent.py       # ⭐ Core implementation
│   │   ├── base_agent.py          # Base agent class
│   │   ├── flood_agent.py         # Flood monitoring
│   │   ├── hazard_agent.py        # Risk assessment
│   │   ├── scout_agent.py         # Crowd-sourced data
│   │   └── evacuation_manager_agent.py  # User coordination
│   ├── environment/
│   │   └── dynamic_graph.py       # Road network management
│   ├── data/
│   │   └── data_structures.py     # Data models
│   ├── simulation/
│   │   └── mas_controller.py      # System orchestration
│   ├── utils/
│   │   ├── logging_config.py      # Logging setup
│   │   └── performance_metrics.py # Performance tracking
│   └── main.py                    # System entry point
├── data/
│   ├── evacuation_centers.csv     # Evacuation locations
│   ├── evacuation_centers.geojson # Geographic data
│   ├── road_networks.gpkg         # Road network data
│   └── road_networks.geojson      # Alternative road data
├── demonstrate_routing_agent.py   # 🚀 Full demo
├── simple_demo.py                 # ⚡ Quick demo
├── test_routing_agent.py          # 🧪 Test suite
├── requirements.txt               # 📦 Dependencies
├── pyproject.toml                 # ⚙️ Configuration
└── README.md                      # 📚 Documentation
```

## 🎯 RoutingAgent Features

### Core Functionality
- **Risk-Aware A* Pathfinding**: Optimizes routes prioritizing safety over distance
- **Real-Time Graph Updates**: Integrates with dynamic road network environment
- **GeoPandas Integration**: Efficient spatial queries for evacuation centers
- **Agent Communication Protocol**: Standardized multi-agent messaging

### Risk Assessment
- **Composite Risk Scoring**: Combines hydrological, infrastructure, and congestion factors
- **Research-Based**: Implements Kreibich et al. (2009) energy head methodology
- **Dynamic Updates**: Placeholder system for future Hazard Agent integration

### Key Metrics
- **Safety Score**: 0.0 (riskiest) to 1.0 (safest)
- **Risk Breakdown**: Average, maximum, and total risk along route
- **Performance**: Computation time and estimated travel time
- **Path Details**: Node sequence and total distance

## 🔧 Running the Full System

To run the complete MAS-FRO multi-agent system:

```bash
python src/main.py
```

This will start all agents:
- FloodAgent: Monitors water levels and velocities
- ScoutAgent: Gathers crowd-sourced hazard reports
- HazardAgent: Calculates composite risk scores
- RoutingAgent: Computes optimal evacuation routes
- EvacuationManagerAgent: Coordinates user requests

## 📊 Sample Output

```
🚀 Starting RoutingAgent Demonstration
📦 Initializing components...
🗺️ Loading road network...
🤖 Creating RoutingAgent...
📍 Creating demo route request...
🎯 Route Request: demo_1694982000
   Origin: (14.6507, 121.1029)
   Destination: Marikina Sports Center
🧮 Calculating route...
✅ Route calculation successful!
   📏 Total Distance: 1250.3 meters
   ⚠️  Average Risk: 0.123
   🛡️  Safety Score: 0.877
   ⏱️  Computation Time: 0.045 seconds
   🚗 Estimated Travel Time: 150.0 seconds
   🛤️ Path Nodes: 8 nodes
```

## 🏗️ Architecture

The RoutingAgent is part of a multi-agent system designed for flood evacuation:

1. **Data Flow**: User requests → EvacuationManager → RoutingAgent
2. **Risk Integration**: HazardAgent → DynamicGraph → RoutingAgent
3. **Real-Time Updates**: FloodAgent + ScoutAgent → HazardAgent → Graph updates

## 📚 Documentation

The RoutingAgent includes comprehensive documentation:
- Detailed docstrings for all methods
- Type hints for parameters and return values
- Research citations and methodology explanations
- Integration points for future development

## 🤝 Contributing

The RoutingAgent is designed for easy extension:
- Modular risk scoring system
- Clear separation of concerns
- Comprehensive logging and error handling
- Placeholder systems for future agent integration

## 📄 License

This project is part of the MAS-FRO research initiative.

---

## Libraries Installed:
- geopandas
- networkx
- matplotlib
- seaborn
- pandas

## Activate Environment:
.venv\Scripts\activate