# MAS-FRO: Multi-Agent System for Flood Route Optimization

This repository contains the **Multi-Agent System for Flood Route Optimization (MAS-FRO)** with a focus on the RoutingAgent component.

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Internet connection (for downloading OpenStreetMap data)

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

   # Or using uv (recommended)
   uv sync
   ```

### Run the RoutingAgent Demonstration

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