"""
Route Visualization and Simulation Module for MAS-FRO
Generates visual representations of routes and flood risk evolution.
"""

import json
import logging
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from datetime import datetime, timedelta
import networkx as nx
import base64
from io import StringIO

logger = logging.getLogger(__name__)


class RouteVisualizer:
    """
    Generates visualization data for routes and flood risk.
    Outputs SVG, HTML overlays, and simulation data.
    """
    
    def __init__(self):
        self.default_colors = {
            'safe': '#00FF00',      # Green
            'low_risk': '#FFFF00',  # Yellow
            'medium_risk': '#FFA500', # Orange
            'high_risk': '#FF0000',  # Red
            'impassable': '#8B0000', # Dark red
            'route': '#800080',      # Purple for ideal path
            'node': '#4169E1',       # Royal blue for nodes
            'evacuation': '#FFD700'  # Gold for evacuation centers
        }
        
    def generate_svg_visualization(
        self,
        graph: nx.Graph,
        path: List[int],
        risk_scores: Dict[str, float],
        width: int = 800,
        height: int = 600
    ) -> str:
        """
        Generate SVG visualization of the route and risk levels.
        
        Args:
            graph: NetworkX graph
            path: List of node IDs forming the route
            risk_scores: Risk scores for edges
            width: SVG canvas width
            height: SVG canvas height
            
        Returns:
            SVG markup as string
        """
        try:
            # Get graph bounds
            bounds = self._get_graph_bounds(graph)
            
            # Start SVG
            svg = StringIO()
            svg.write(f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">\n')
            svg.write('  <defs>\n')
            svg.write('    <marker id="arrowhead" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto">\n')
            svg.write('      <polygon points="0 0, 10 3, 0 6" fill="#800080" />\n')
            svg.write('    </marker>\n')
            svg.write('  </defs>\n')
            
            # Background
            svg.write(f'  <rect width="{width}" height="{height}" fill="#F0F0F0"/>\n')
            
            # Draw edges with risk coloring
            svg.write('  <g id="edges">\n')
            for u, v, data in graph.edges(data=True):
                edge_id = f"{u}_{v}_0"
                risk = risk_scores.get(edge_id, 0.0)
                color = self._get_risk_color(risk)
                opacity = 0.3 if (u, v) not in zip(path[:-1], path[1:]) else 1.0
                
                u_pos = self._transform_coordinates(
                    graph.nodes[u]['x'], graph.nodes[u]['y'],
                    bounds, width, height
                )
                v_pos = self._transform_coordinates(
                    graph.nodes[v]['x'], graph.nodes[v]['y'],
                    bounds, width, height
                )
                
                svg.write(f'    <line x1="{u_pos[0]}" y1="{u_pos[1]}" ')
                svg.write(f'x2="{v_pos[0]}" y2="{v_pos[1]}" ')
                svg.write(f'stroke="{color}" stroke-width="2" opacity="{opacity}"/>\n')
            svg.write('  </g>\n')
            
            # Draw the optimal path in purple
            if path and len(path) > 1:
                svg.write('  <g id="route">\n')
                path_points = []
                for node in path:
                    pos = self._transform_coordinates(
                        graph.nodes[node]['x'], graph.nodes[node]['y'],
                        bounds, width, height
                    )
                    path_points.append(f"{pos[0]},{pos[1]}")
                
                svg.write(f'    <polyline points="{" ".join(path_points)}" ')
                svg.write(f'fill="none" stroke="{self.default_colors["route"]}" ')
                svg.write('stroke-width="4" marker-end="url(#arrowhead)"/>\n')
                svg.write('  </g>\n')
            
            # Draw nodes
            svg.write('  <g id="nodes">\n')
            for node in graph.nodes():
                pos = self._transform_coordinates(
                    graph.nodes[node]['x'], graph.nodes[node]['y'],
                    bounds, width, height
                )
                
                # Different styling for path nodes
                if node in path:
                    if node == path[0]:
                        color = '#00FF00'  # Green for start
                        radius = 8
                    elif node == path[-1]:
                        color = self.default_colors['evacuation']  # Gold for destination
                        radius = 8
                    else:
                        color = self.default_colors['route']
                        radius = 6
                else:
                    color = self.default_colors['node']
                    radius = 4
                
                svg.write(f'    <circle cx="{pos[0]}" cy="{pos[1]}" r="{radius}" ')
                svg.write(f'fill="{color}" stroke="black" stroke-width="1"/>\n')
                
                # Add labels for start and end
                if path and node == path[0]:
                    svg.write(f'    <text x="{pos[0]}" y="{pos[1]-10}" ')
                    svg.write('text-anchor="middle" font-size="12" font-weight="bold">START</text>\n')
                elif path and node == path[-1]:
                    svg.write(f'    <text x="{pos[0]}" y="{pos[1]-10}" ')
                    svg.write('text-anchor="middle" font-size="12" font-weight="bold">EVACUATION</text>\n')
            svg.write('  </g>\n')
            
            # Add legend
            svg.write('  <g id="legend" transform="translate(10, 20)">\n')
            svg.write('    <rect x="0" y="0" width="150" height="120" fill="white" stroke="black" opacity="0.9"/>\n')
            svg.write('    <text x="75" y="15" text-anchor="middle" font-weight="bold">Risk Levels</text>\n')
            
            legend_items = [
                ('Safe', self.default_colors['safe']),
                ('Low Risk', self.default_colors['low_risk']),
                ('Medium Risk', self.default_colors['medium_risk']),
                ('High Risk', self.default_colors['high_risk']),
                ('Optimal Route', self.default_colors['route'])
            ]
            
            for i, (label, color) in enumerate(legend_items):
                y_pos = 35 + i * 20
                svg.write(f'    <rect x="10" y="{y_pos-5}" width="15" height="10" fill="{color}"/>\n')
                svg.write(f'    <text x="30" y="{y_pos+3}" font-size="12">{label}</text>\n')
            svg.write('  </g>\n')
            
            svg.write('</svg>')
            
            return svg.getvalue()
            
        except Exception as e:
            logger.error(f"SVG generation error: {e}")
            return f'<svg width="{width}" height="{height}"><text x="10" y="30">Visualization Error: {str(e)}</text></svg>'
    
    def generate_html_overlay(
        self,
        graph: nx.Graph,
        path: List[int],
        risk_scores: Dict[str, float]
    ) -> str:
        """
        Generate HTML with embedded SVG and interactive features.
        """
        svg_content = self.generate_svg_visualization(graph, path, risk_scores)
        
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MAS-FRO Route Visualization</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background: #f5f5f5;
        }}
        .container {{
            max-width: 900px;
            margin: 0 auto;
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            text-align: center;
        }}
        .visualization {{
            display: flex;
            justify-content: center;
            margin: 20px 0;
        }}
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }}
        .stat-card {{
            background: #f9f9f9;
            padding: 15px;
            border-radius: 5px;
            border-left: 4px solid #800080;
        }}
        .stat-label {{
            font-size: 12px;
            color: #666;
            text-transform: uppercase;
        }}
        .stat-value {{
            font-size: 24px;
            font-weight: bold;
            color: #333;
            margin-top: 5px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>MAS-FRO Route Visualization</h1>
        <div class="visualization">
            {svg_content}
        </div>
        <div class="stats">
            <div class="stat-card">
                <div class="stat-label">Route Length</div>
                <div class="stat-value">{len(path) - 1} segments</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Risk Level</div>
                <div class="stat-value">{self._calculate_average_risk(path, risk_scores):.2%}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Status</div>
                <div class="stat-value">Optimal</div>
            </div>
        </div>
    </div>
</body>
</html>"""
        
        return html
    
    def generate_simulation_data(
        self,
        graph: nx.Graph,
        path: List[int],
        risk_scores: Dict[str, float],
        rain_rate_profile: List[float],
        time_steps: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Generate time-stepped simulation data showing risk evolution.
        
        Args:
            graph: Network graph
            path: Optimal path
            risk_scores: Initial risk scores
            rain_rate_profile: Rain rate at each time step
            time_steps: Number of simulation frames
            
        Returns:
            List of simulation frames with hazard evolution
        """
        simulation_frames = []
        base_time = datetime.now()
        
        for frame in range(time_steps):
            # Current rain rate
            rain_rate = rain_rate_profile[frame % len(rain_rate_profile)]
            
            # Update risk scores based on rain
            current_risks = {}
            for edge_id, base_risk in risk_scores.items():
                if base_risk != float('inf'):
                    # Risk increases with rain
                    adjusted_risk = min(1.0, base_risk + (rain_rate * 0.01))
                    current_risks[edge_id] = adjusted_risk
                else:
                    current_risks[edge_id] = base_risk
            
            # Create frame data
            frame_data = {
                "frame": frame + 1,
                "timestamp": (base_time + timedelta(minutes=frame * 10)).isoformat(),
                "rain_rate": rain_rate,
                "hazard_scores": current_risks,
                "path": path,
                "average_risk": self._calculate_average_risk(path, current_risks),
                "max_risk": max([current_risks.get(f"{path[i]}_{path[i+1]}_0", 0.0) 
                               for i in range(len(path)-1)] or [0])
            }
            
            simulation_frames.append(frame_data)
        
        return simulation_frames
    
    def export_simulation_json(
        self,
        simulation_data: List[Dict[str, Any]],
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Export simulation data as JSON for external rendering.
        
        Args:
            simulation_data: List of simulation frames
            metadata: Additional metadata to include
            
        Returns:
            JSON string with complete simulation data
        """
        export_data = {
            "version": "1.0",
            "generated_at": datetime.now().isoformat(),
            "simulation": {
                "frames": simulation_data,
                "total_frames": len(simulation_data),
                "metadata": metadata or {}
            }
        }
        
        try:
            json_str = json.dumps(export_data, indent=2)
            
            # Validate JSON structure
            json.loads(json_str)  # Parse to validate
            
            return json_str
            
        except Exception as e:
            logger.error(f"JSON export error: {e}")
            error_response = {
                "status": "error",
                "message": f"Failed to export simulation data: {str(e)}"
            }
            return json.dumps(error_response)
    
    def _get_graph_bounds(self, graph: nx.Graph) -> Dict[str, float]:
        """Get the bounding box of the graph."""
        x_coords = [data.get('x', 0) for _, data in graph.nodes(data=True)]
        y_coords = [data.get('y', 0) for _, data in graph.nodes(data=True)]
        
        return {
            'min_x': min(x_coords),
            'max_x': max(x_coords),
            'min_y': min(y_coords),
            'max_y': max(y_coords)
        }
    
    def _transform_coordinates(
        self,
        x: float,
        y: float,
        bounds: Dict[str, float],
        width: int,
        height: int
    ) -> Tuple[float, float]:
        """Transform graph coordinates to SVG coordinates."""
        # Add padding
        padding = 50
        usable_width = width - 2 * padding
        usable_height = height - 2 * padding
        
        # Normalize to [0, 1]
        x_range = bounds['max_x'] - bounds['min_x'] or 1
        y_range = bounds['max_y'] - bounds['min_y'] or 1
        
        norm_x = (x - bounds['min_x']) / x_range
        norm_y = (y - bounds['min_y']) / y_range
        
        # Transform to SVG coordinates (y is inverted)
        svg_x = padding + norm_x * usable_width
        svg_y = padding + (1 - norm_y) * usable_height
        
        return (svg_x, svg_y)
    
    def _get_risk_color(self, risk: float) -> str:
        """Get color for risk level."""
        if risk == float('inf'):
            return self.default_colors['impassable']
        elif risk < 0.25:
            return self.default_colors['safe']
        elif risk < 0.5:
            return self.default_colors['low_risk']
        elif risk < 0.75:
            return self.default_colors['medium_risk']
        else:
            return self.default_colors['high_risk']
    
    def _calculate_average_risk(self, path: List[int], risk_scores: Dict[str, float]) -> float:
        """Calculate average risk along path."""
        if len(path) < 2:
            return 0.0
        
        total_risk = 0
        edge_count = 0
        
        for i in range(len(path) - 1):
            edge_id = f"{path[i]}_{path[i+1]}_0"
            risk = risk_scores.get(edge_id, 0.0)
            if risk != float('inf'):
                total_risk += risk
                edge_count += 1
        
        return total_risk / max(1, edge_count)


class SimulationRenderer:
    """
    Handles rendering of simulation data in various formats.
    """
    
    def __init__(self):
        self.visualizer = RouteVisualizer()
    
    def render_frame(
        self,
        frame_data: Dict[str, Any],
        graph: nx.Graph,
        format: str = 'svg'
    ) -> str:
        """
        Render a single simulation frame.
        
        Args:
            frame_data: Frame data from simulation
            graph: Network graph
            format: Output format ('svg', 'html', 'json')
            
        Returns:
            Rendered frame in requested format
        """
        if format == 'svg':
            return self.visualizer.generate_svg_visualization(
                graph,
                frame_data['path'],
                frame_data['hazard_scores']
            )
        elif format == 'html':
            return self.visualizer.generate_html_overlay(
                graph,
                frame_data['path'],
                frame_data['hazard_scores']
            )
        elif format == 'json':
            return json.dumps(frame_data, indent=2)
        else:
            raise ValueError(f"Unknown format: {format}")
    
    def create_animated_html(
        self,
        simulation_data: List[Dict[str, Any]],
        graph: nx.Graph
    ) -> str:
        """
        Create animated HTML visualization of simulation.
        """
        # Generate SVG for each frame
        frames = []
        for frame_data in simulation_data:
            svg = self.visualizer.generate_svg_visualization(
                graph,
                frame_data['path'],
                frame_data['hazard_scores']
            )
            frames.append(base64.b64encode(svg.encode()).decode())
        
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>MAS-FRO Simulation</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background: #f5f5f5;
        }}
        .container {{
            max-width: 900px;
            margin: 0 auto;
            background: white;
            padding: 20px;
            border-radius: 8px;
        }}
        #visualization {{
            width: 100%;
            height: 600px;
            border: 1px solid #ccc;
        }}
        .controls {{
            margin-top: 20px;
            text-align: center;
        }}
        button {{
            margin: 0 5px;
            padding: 10px 20px;
            font-size: 16px;
            cursor: pointer;
        }}
        .frame-info {{
            margin-top: 10px;
            font-size: 14px;
            color: #666;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>MAS-FRO Route Simulation</h1>
        <div id="visualization"></div>
        <div class="controls">
            <button onclick="previousFrame()">← Previous</button>
            <button onclick="togglePlay()" id="playBtn">▶ Play</button>
            <button onclick="nextFrame()">Next →</button>
            <button onclick="reset()">↻ Reset</button>
        </div>
        <div class="frame-info">
            Frame: <span id="currentFrame">1</span> / {len(frames)}
            | Rain Rate: <span id="rainRate">0</span> mm/hr
            | Avg Risk: <span id="avgRisk">0</span>%
        </div>
    </div>
    
    <script>
        const frames = {json.dumps(frames)};
        const frameData = {json.dumps(simulation_data)};
        let currentFrameIndex = 0;
        let isPlaying = false;
        let playInterval = null;
        
        function showFrame(index) {{
            if (index < 0 || index >= frames.length) return;
            
            currentFrameIndex = index;
            const svgData = frames[index];
            const vizDiv = document.getElementById('visualization');
            vizDiv.innerHTML = atob(svgData);
            
            // Update info
            document.getElementById('currentFrame').textContent = index + 1;
            document.getElementById('rainRate').textContent = frameData[index].rain_rate.toFixed(1);
            document.getElementById('avgRisk').textContent = (frameData[index].average_risk * 100).toFixed(1);
        }}
        
        function nextFrame() {{
            showFrame((currentFrameIndex + 1) % frames.length);
        }}
        
        function previousFrame() {{
            showFrame((currentFrameIndex - 1 + frames.length) % frames.length);
        }}
        
        function togglePlay() {{
            isPlaying = !isPlaying;
            const btn = document.getElementById('playBtn');
            
            if (isPlaying) {{
                btn.textContent = '⏸ Pause';
                playInterval = setInterval(nextFrame, 1000);
            }} else {{
                btn.textContent = '▶ Play';
                clearInterval(playInterval);
            }}
        }}
        
        function reset() {{
            if (isPlaying) togglePlay();
            showFrame(0);
        }}
        
        // Initialize
        showFrame(0);
    </script>
</body>
</html>"""
        
        return html


# Testing function
def test_visualization():
    """Test visualization generation."""
    
    # Create sample graph
    G = nx.Graph()
    nodes = {
        1: {'x': 121.10, 'y': 14.65},
        2: {'x': 121.11, 'y': 14.66},
        3: {'x': 121.12, 'y': 14.67},
        4: {'x': 121.13, 'y': 14.68},
        5: {'x': 121.14, 'y': 14.69}
    }
    
    for node_id, attrs in nodes.items():
        G.add_node(node_id, **attrs)
    
    edges = [(1, 2), (2, 3), (3, 4), (4, 5), (1, 3), (2, 4)]
    G.add_edges_from(edges)
    
    # Sample path and risks
    path = [1, 2, 3, 4, 5]
    risk_scores = {
        "1_2_0": 0.2,
        "2_3_0": 0.5,
        "3_4_0": 0.8,
        "4_5_0": 0.3,
        "1_3_0": 0.4,
        "2_4_0": 0.6
    }
    
    # Test visualization
    viz = RouteVisualizer()
    
    # Generate SVG
    svg = viz.generate_svg_visualization(G, path, risk_scores)
    print(f"SVG generated: {len(svg)} characters")
    
    # Generate HTML
    html = viz.generate_html_overlay(G, path, risk_scores)
    with open("test_visualization.html", "w", encoding='utf-8') as f:
        f.write(html)
    print("HTML saved to test_visualization.html")
    
    # Generate simulation data
    rain_profile = [5, 10, 15, 20, 25, 20, 15, 10, 5, 0]
    sim_data = viz.generate_simulation_data(G, path, risk_scores, rain_profile)
    print(f"Simulation data: {len(sim_data)} frames")
    
    # Export JSON
    json_export = viz.export_simulation_json(sim_data, {"test": "metadata"})
    with open("simulation_data.json", "w", encoding='utf-8') as f:
        f.write(json_export)
    print("Simulation data exported to simulation_data.json")
    
    # Create animated HTML
    renderer = SimulationRenderer()
    animated = renderer.create_animated_html(sim_data, G)
    with open("animated_simulation.html", "w", encoding='utf-8') as f:
        f.write(animated)
    print("Animated HTML saved to animated_simulation.html")


if __name__ == "__main__":
    test_visualization()

