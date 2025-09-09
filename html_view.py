# Web Interface and Visualization for MAS-FRO
# This provides a simple web dashboard for monitoring the system

from flask import Flask, render_template, request, jsonify, send_from_directory
import folium
from folium import plugins
import json
import os
import sys
from datetime import datetime
import threading
import time
import queue
import pandas as pd
import plotly
import plotly.graph_objs as go
import plotly.express as px
from werkzeug.serving import make_server
import networkx as nx
import numpy as np
import random
import logging

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import MAS-FRO components
from src.data.data_structures import RouteRequest
from src.agents.advanced_hazard_agent import IntegratedMASFROSystem
from src.utils.logging_config import setup_logging

# Setup logging
setup_logging(logging.INFO)
logger = logging.getLogger(__name__)

class MASFROWebInterface:
    """Web interface for monitoring and controlling MAS-FRO system"""
    
    def __init__(self, mas_fro_system):
        self.app = Flask(__name__)
        self.mas_fro_system = mas_fro_system
        self.setup_routes()
        
        # Data for web interface
        self.current_routes = []
        self.system_status = {
            'agents_active': 0,
            'routes_computed': 0,
            'avg_computation_time': 0,
            'success_rate': 0
        }
        
        # Create templates directory if it doesn't exist
        os.makedirs('templates', exist_ok=True)
        os.makedirs('static', exist_ok=True)
        
        self._create_html_templates()
    
    def setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route('/')
        def dashboard():
            return render_template('dashboard.html')
        
        @self.app.route('/api/status')
        def get_status():
            """Get current system status"""
            return jsonify(self.system_status)
        
        @self.app.route('/api/routes')
        def get_routes():
            """Get current active routes"""
            return jsonify(self.current_routes)
        
        @self.app.route('/api/map')
        def get_map():
            """Generate current flood risk map"""
            return jsonify(self._generate_risk_map_data())
        
        @self.app.route('/api/performance')
        def get_performance():
            """Get performance metrics"""
            if hasattr(self.mas_fro_system.agents.get('routing'), 'performance_monitor'):
                stats = self.mas_fro_system.agents['routing'].get_performance_statistics()
                return jsonify(stats)
            return jsonify({})
        
        @self.app.route('/api/request_route', methods=['POST'])
        def request_route():
            """Handle route request from web interface"""
            data = request.json
            
            route_request = RouteRequest(
                request_id=f"WEB_{int(time.time())}",
                origin=(data['origin_lat'], data['origin_lon']),
                destination=data.get('destination', 'nearest_evacuation_center'),
                timestamp=datetime.now(),
                user_id=data.get('user_id', 'web_user')
            )
            
            # Submit request to system
            message = {
                'type': 'user_request',
                'data': route_request,
                'sender': 'web_interface',
                'timestamp': time.time()
            }
            
            self.mas_fro_system.user_to_evacuation.put(message)
            
            return jsonify({'status': 'submitted', 'request_id': route_request.request_id})
        
        @self.app.route('/visualization')
        def visualization():
            return render_template('visualization.html')
        
        @self.app.route('/api/visualization_data')
        def get_visualization_data():
            """Get data for interactive visualizations"""
            return jsonify(self._generate_visualization_data())
    
    def _create_html_templates(self):
        """Create HTML templates for the web interface"""
        
        # Dashboard template
        dashboard_html = """
<!DOCTYPE html>
<html>
<head>
    <title>MAS-FRO Dashboard</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.7.1/dist/leaflet.css"/>
    <script src="https://unpkg.com/leaflet@1.7.1/dist/leaflet.js"></script>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
</head>
<body>
    <nav class="navbar navbar-dark bg-dark">
        <div class="container">
            <span class="navbar-brand">MAS-FRO: Multi-Agent Flood Route Optimization</span>
            <span class="navbar-text" id="status">System Status: <span class="badge bg-success">Active</span></span>
        </div>
    </nav>
    
    <div class="container-fluid mt-3">
        <div class="row">
            <!-- System Status Panel -->
            <div class="col-md-3">
                <div class="card">
                    <div class="card-header">
                        <h5>System Status</h5>
                    </div>
                    <div class="card-body">
                        <div class="mb-2">
                            <strong>Active Agents:</strong> <span id="agents-count">0</span>
                        </div>
                        <div class="mb-2">
                            <strong>Routes Computed:</strong> <span id="routes-count">0</span>
                        </div>
                        <div class="mb-2">
                            <strong>Success Rate:</strong> <span id="success-rate">0%</span>
                        </div>
                        <div class="mb-2">
                            <strong>Avg Response Time:</strong> <span id="response-time">0ms</span>
                        </div>
                    </div>
                </div>
                
                <!-- Route Request Panel -->
                <div class="card mt-3">
                    <div class="card-header">
                        <h5>Request Route</h5>
                    </div>
                    <div class="card-body">
                        <form id="route-form">
                            <div class="mb-2">
                                <label class="form-label">Origin Latitude</label>
                                <input type="number" class="form-control" id="origin-lat" 
                                       value="14.6507" step="0.0001" required>
                            </div>
                            <div class="mb-2">
                                <label class="form-label">Origin Longitude</label>
                                <input type="number" class="form-control" id="origin-lon" 
                                       value="121.1029" step="0.0001" required>
                            </div>
                            <div class="mb-2">
                                <label class="form-label">User ID</label>
                                <input type="text" class="form-control" id="user-id" 
                                       value="web_user_1" required>
                            </div>
                            <button type="submit" class="btn btn-primary">Request Route</button>
                        </form>
                        <div id="route-result" class="mt-2"></div>
                    </div>
                </div>
            </div>
            
            <!-- Map Panel -->
            <div class="col-md-6">
                <div class="card">
                    <div class="card-header">
                        <h5>Real-time Flood Risk Map</h5>
                    </div>
                    <div class="card-body p-0">
                        <div id="map" style="height: 600px;"></div>
                    </div>
                </div>
            </div>
            
            <!-- Performance Panel -->
            <div class="col-md-3">
                <div class="card">
                    <div class="card-header">
                        <h5>Performance Metrics</h5>
                    </div>
                    <div class="card-body">
                        <div id="performance-chart" style="height: 300px;"></div>
                    </div>
                </div>
                
                <div class="card mt-3">
                    <div class="card-header">
                        <h5>Recent Activity</h5>
                    </div>
                    <div class="card-body">
                        <div id="activity-log" style="height: 250px; overflow-y: auto;">
                            <!-- Activity items will be added here -->
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        // Initialize map
        var map = L.map('map').setView([14.6507, 121.1029], 13);
        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png').addTo(map);
        
        // Initialize performance chart
        var performanceData = {
            x: [],
            y: [],
            type: 'scatter',
            mode: 'lines+markers',
            name: 'Response Time'
        };
        
        Plotly.newPlot('performance-chart', [performanceData], {
            title: 'Response Time Trend',
            xaxis: { title: 'Time' },
            yaxis: { title: 'Response Time (ms)' }
        });
        
        // Update functions
        function updateStatus() {
            fetch('/api/status')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('agents-count').textContent = data.agents_active || 5;
                    document.getElementById('routes-count').textContent = data.routes_computed || 0;
                    document.getElementById('success-rate').textContent = 
                        ((data.success_rate || 0) * 100).toFixed(1) + '%';
                    document.getElementById('response-time').textContent = 
                        ((data.avg_computation_time || 0) * 1000).toFixed(0) + 'ms';
                });
        }
        
        function updateMap() {
            fetch('/api/map')
                .then(response => response.json())
                .then(data => {
                    // Clear existing layers
                    map.eachLayer(function(layer) {
                        if (layer instanceof L.Circle || layer instanceof L.Polyline) {
                            map.removeLayer(layer);
                        }
                    });
                    
                    // Add risk zones
                    if (data.risk_zones) {
                        data.risk_zones.forEach(zone => {
                            var color = zone.risk > 0.7 ? 'red' : zone.risk > 0.4 ? 'orange' : 'yellow';
                            L.circle([zone.lat, zone.lon], {
                                color: color,
                                fillColor: color,
                                fillOpacity: 0.3,
                                radius: zone.radius || 200
                            }).addTo(map).bindPopup(`Risk Level: ${(zone.risk * 100).toFixed(0)}%`);
                        });
                    }
                    
                    // Add active routes
                    if (data.active_routes) {
                        data.active_routes.forEach(route => {
                            L.polyline(route.path, {color: 'blue', weight: 3}).addTo(map)
                                .bindPopup(`Route ID: ${route.id}<br>Safety Score: ${route.safety_score.toFixed(2)}`);
                        });
                    }
                });
        }
        
        function updatePerformance() {
            fetch('/api/performance')
                .then(response => response.json())
                .then(data => {
                    if (data.avg_computation_time) {
                        performanceData.x.push(new Date());
                        performanceData.y.push(data.avg_computation_time * 1000);
                        
                        // Keep only last 20 points
                        if (performanceData.x.length > 20) {
                            performanceData.x.shift();
                            performanceData.y.shift();
                        }
                        
                        Plotly.redraw('performance-chart');
                    }
                });
        }
        
        function addActivityLog(message) {
            var log = document.getElementById('activity-log');
            var item = document.createElement('div');
            item.className = 'alert alert-info alert-sm mb-1';
            item.innerHTML = `<small>${new Date().toLocaleTimeString()}</small><br>${message}`;
            log.insertBefore(item, log.firstChild);
            
            // Keep only last 10 items
            while (log.children.length > 10) {
                log.removeChild(log.lastChild);
            }
        }
        
        // Route request form
        document.getElementById('route-form').addEventListener('submit', function(e) {
            e.preventDefault();
            
            var requestData = {
                origin_lat: parseFloat(document.getElementById('origin-lat').value),
                origin_lon: parseFloat(document.getElementById('origin-lon').value),
                user_id: document.getElementById('user-id').value
            };
            
            fetch('/api/request_route', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(requestData)
            })
            .then(response => response.json())
            .then(data => {
                document.getElementById('route-result').innerHTML = 
                    `<div class="alert alert-success">Route requested: ${data.request_id}</div>`;
                addActivityLog(`Route requested for user ${requestData.user_id}`);
            })
            .catch(error => {
                document.getElementById('route-result').innerHTML = 
                    `<div class="alert alert-danger">Error: ${error}</div>`;
            });
        });
        
        // Auto-update every 5 seconds
        setInterval(() => {
            updateStatus();
            updateMap();
            updatePerformance();
        }, 5000);
        
        // Initial load
        updateStatus();
        updateMap();
        updatePerformance();
        addActivityLog('Dashboard initialized');
    </script>
</body>
</html>
        """
        
        # Visualization template
        visualization_html = """
<!DOCTYPE html>
<html>
<head>
    <title>MAS-FRO Visualizations</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
</head>
<body>
    <nav class="navbar navbar-dark bg-dark">
        <div class="container">
            <span class="navbar-brand">MAS-FRO Visualizations</span>
            <a href="/" class="btn btn-outline-light">Back to Dashboard</a>
        </div>
    </nav>
    
    <div class="container-fluid mt-3">
        <div class="row">
            <div class="col-md-6">
                <div class="card">
                    <div class="card-header">
                        <h5>Route Computation Performance</h5>
                    </div>
                    <div class="card-body">
                        <div id="performance-histogram"></div>
                    </div>
                </div>
            </div>
            
            <div class="col-md-6">
                <div class="card">
                    <div class="card-header">
                        <h5>Safety Score Distribution</h5>
                    </div>
                    <div class="card-body">
                        <div id="safety-distribution"></div>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="row mt-3">
            <div class="col-md-12">
                <div class="card">
                    <div class="card-header">
                        <h5>System Performance Over Time</h5>
                    </div>
                    <div class="card-body">
                        <div id="time-series"></div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        function loadVisualizationData() {
            fetch('/api/visualization_data')
                .then(response => response.json())
                .then(data => {
                    createPerformanceHistogram(data.computation_times);
                    createSafetyDistribution(data.safety_scores);
                    createTimeSeries(data.time_series);
                });
        }
        
        function createPerformanceHistogram(data) {
            var trace = {
                x: data,
                type: 'histogram',
                nbinsx: 20,
                marker: {color: 'blue', opacity: 0.7}
            };
            
            Plotly.newPlot('performance-histogram', [trace], {
                title: 'Route Computation Time Distribution',
                xaxis: {title: 'Computation Time (seconds)'},
                yaxis: {title: 'Frequency'}
            });
        }
        
        function createSafetyDistribution(data) {
            var trace = {
                x: data,
                type: 'histogram',
                nbinsx: 20,
                marker: {color: 'green', opacity: 0.7}
            };
            
            Plotly.newPlot('safety-distribution', [trace], {
                title: 'Route Safety Score Distribution',
                xaxis: {title: 'Safety Score (0-1)'},
                yaxis: {title: 'Frequency'}
            });
        }
        
        function createTimeSeries(data) {
            var traces = [];
            
            if (data.success_rate) {
                traces.push({
                    x: data.timestamps,
                    y: data.success_rate,
                    name: 'Success Rate',
                    yaxis: 'y'
                });
            }
            
            if (data.avg_computation_time) {
                traces.push({
                    x: data.timestamps,
                    y: data.avg_computation_time,
                    name: 'Avg Computation Time',
                    yaxis: 'y2'
                });
            }
            
            var layout = {
                title: 'System Performance Metrics Over Time',
                xaxis: {title: 'Time'},
                yaxis: {
                    title: 'Success Rate',
                    side: 'left'
                },
                yaxis2: {
                    title: 'Computation Time (s)',
                    side: 'right',
                    overlaying: 'y'
                }
            };
            
            Plotly.newPlot('time-series', traces, layout);
        }
        
        // Load data on page load
        loadVisualizationData();
        
        // Refresh every 30 seconds
        setInterval(loadVisualizationData, 30000);
    </script>
</body>
</html>
        """
        
        # Write templates to files
        with open('templates/dashboard.html', 'w') as f:
            f.write(dashboard_html)
        
        with open('templates/visualization.html', 'w') as f:
            f.write(visualization_html)
    
    def _generate_risk_map_data(self):
        """Generate data for the risk map visualization"""
        risk_zones = []
        active_routes = []
        
        # Generate sample risk zones based on current graph state
        if hasattr(self.mas_fro_system, 'graph_env'):
            graph_env = self.mas_fro_system.graph_env
            
            # Sample some edges and their risk levels
            edges_sample = list(graph_env.graph.edges(keys=True))[:20]  # Sample 20 edges
            
            for u, v, key in edges_sample:
                try:
                    edge_id = f"{u}_{v}_{key}"
                    risk_level = graph_env._risk_scores.get(edge_id, 0.0)
                    
                    if risk_level > 0.1:  # Only show areas with some risk
                        # Get node coordinates
                        u_data = graph_env.graph.nodes[u]
                        v_data = graph_env.graph.nodes[v]
                        
                        lat = (u_data.get('y', 0) + v_data.get('y', 0)) / 2
                        lon = (u_data.get('x', 0) + v_data.get('x', 0)) / 2
                        
                        if lat != 0 and lon != 0:  # Valid coordinates
                            risk_zones.append({
                                'lat': lat,
                                'lon': lon,
                                'risk': min(risk_level, 1.0),
                                'radius': max(100, min(500, risk_level * 500))
                            })
                except:
                    continue
        
        # Add some sample evacuation centers
        evacuation_centers = [
            {'lat': 14.6507, 'lon': 121.1029, 'name': 'Marikina Sports Center'},
            {'lat': 14.6350, 'lon': 121.1120, 'name': 'Marikina Elementary School'},
            {'lat': 14.6180, 'lon': 121.1200, 'name': 'Barangay Hall'}
        ]
        
        return {
            'risk_zones': risk_zones,
            'active_routes': active_routes,
            'evacuation_centers': evacuation_centers
        }
    
    def _generate_visualization_data(self):
        """Generate data for performance visualizations"""
        # Get performance data from routing agent
        computation_times = []
        safety_scores = []
        timestamps = []
        success_rates = []
        avg_computation_times = []
        
        if hasattr(self.mas_fro_system.agents.get('routing'), 'performance_monitor'):
            monitor = self.mas_fro_system.agents['routing'].performance_monitor
            
            for metric in monitor.metrics:
                computation_times.append(metric.computation_time)
                safety_scores.append(metric.route_safety_score)
                timestamps.append(datetime.fromtimestamp(metric.timestamp).isoformat())
            
            # Calculate rolling statistics
            window_size = 10
            for i in range(len(monitor.metrics)):
                start_idx = max(0, i - window_size)
                window_metrics = monitor.metrics[start_idx:i+1]
                
                successful = [m for m in window_metrics if m.success]
                success_rate = len(successful) / len(window_metrics) if window_metrics else 0
                avg_time = sum(m.computation_time for m in successful) / len(successful) if successful else 0
                
                success_rates.append(success_rate)
                avg_computation_times.append(avg_time)
        
        return {
            'computation_times': computation_times,
            'safety_scores': safety_scores,
            'time_series': {
                'timestamps': timestamps,
                'success_rate': success_rates,
                'avg_computation_time': avg_computation_times
            }
        }
    
    def start_server(self, host='127.0.0.1', port=5000, debug=False):
        """Start the web server"""
        print(f"Starting MAS-FRO Web Interface at http://{host}:{port}")
        self.server = make_server(host, port, self.app)
        self.server_thread = threading.Thread(target=self.server.serve_forever)
        self.server_thread.daemon = True
        self.server_thread.start()
        
        return f"http://{host}:{port}"
    
    def stop_server(self):
        """Stop the web server"""
        if hasattr(self, 'server'):
            self.server.shutdown()
            self.server_thread.join()

# Integrated system with web interface
class WebEnabledMASFRO(IntegratedMASFROSystem):
    """MAS-FRO system with integrated web interface"""
    
    def __init__(self):
        super().__init__()
        self.web_interface = MASFROWebInterface(self)
        self.web_url = None
    
    def run_with_web_interface(self, duration=3600, web_host='127.0.0.1', web_port=5000):
        """Run simulation with web interface"""
        
        # Start web server
        self.web_url = self.web_interface.start_server(web_host, web_port)
        print(f"Web interface available at: {self.web_url}")
        
        # Start simulation in background
        simulation_thread = threading.Thread(
            target=self.run_enhanced_simulation, 
            args=(duration,)
        )
        simulation_thread.daemon = True
        simulation_thread.start()
        
        # Update web interface data periodically
        def update_web_data():
            while simulation_thread.is_alive():
                try:
                    # Update system status
                    if hasattr(self.agents.get('routing'), 'performance_monitor'):
                        stats = self.agents['routing'].get_performance_statistics()
                        self.web_interface.system_status.update({
                            'agents_active': len(self.agents),
                            'routes_computed': stats.get('total_requests', 0),
                            'avg_computation_time': stats.get('avg_computation_time', 0),
                            'success_rate': stats.get('success_rate', 0)
                        })
                    
                    time.sleep(5)  # Update every 5 seconds
                    
                except Exception as e:
                    logger.error(f"Error updating web data: {e}")
                    time.sleep(10)
        
        web_update_thread = threading.Thread(target=update_web_data)
        web_update_thread.daemon = True
        web_update_thread.start()
        
        try:
            print("Simulation running with web interface...")
            print("Press Ctrl+C to stop")
            simulation_thread.join()
            
        except KeyboardInterrupt:
            print("\nShutting down...")
        finally:
            self.web_interface.stop_server()
    
    def run_simulation_only(self, duration=3600):
        """Run simulation without web interface"""
        self.run_enhanced_simulation(duration)

# Example evaluation and comparison functions
class MASFROEvaluator:
    """Evaluation system for comparing MAS-FRO with baseline algorithms"""
    
    def __init__(self, mas_fro_system):
        self.mas_fro_system = mas_fro_system
        self.baseline_results = []
        self.mas_fro_results = []
    
    def run_comparative_evaluation(self, num_scenarios=50):
        """Run comparative evaluation between MAS-FRO and baseline"""
        print("Starting comparative evaluation...")
        
        # Generate test scenarios
        test_scenarios = self._generate_test_scenarios(num_scenarios)
        
        for i, scenario in enumerate(test_scenarios):
            print(f"Running scenario {i+1}/{num_scenarios}")
            
            # Test baseline algorithm (Dijkstra on static graph)
            baseline_result = self._test_baseline_algorithm(scenario)
            self.baseline_results.append(baseline_result)
            
            # Test MAS-FRO system
            mas_fro_result = self._test_mas_fro_algorithm(scenario)
            self.mas_fro_results.append(mas_fro_result)
            
            time.sleep(0.1)  # Small delay between tests
        
        # Generate comparison report
        self._generate_comparison_report()
    
    def _generate_test_scenarios(self, num_scenarios):
        """Generate test scenarios with varying flood conditions"""
        scenarios = []
        
        # Base coordinates for Marikina
        base_lat, base_lon = 14.6507, 121.1029
        
        for i in range(num_scenarios):
            # Random origin within Marikina area
            origin = (
                base_lat + random.uniform(-0.02, 0.02),
                base_lon + random.uniform(-0.02, 0.02)
            )
            
            # Simulate different flood intensities
            flood_intensity = random.uniform(0.0, 1.0)
            
            # Create scenario
            scenario = {
                'id': f'scenario_{i}',
                'origin': origin,
                'destination': 'nearest_evacuation_center',
                'flood_intensity': flood_intensity,
                'timestamp': time.time()
            }
            scenarios.append(scenario)
        
        return scenarios
    
    def _test_baseline_algorithm(self, scenario):
        """Test baseline shortest-path algorithm"""
        start_time = time.time()
        
        try:
            # Get static graph (without risk weights)
            graph = self.mas_fro_system.graph_env.graph.copy()
            
            # Remove risk-aware weights, use only distance
            for u, v, key in graph.edges(keys=True):
                edge_data = graph[u][v][key]
                graph[u][v][key]['weight'] = edge_data.get('length', 100)
            
            # Find nearest nodes
            origin_node = self._find_nearest_node_simple(graph, scenario['origin'])
            dest_node = self._find_nearest_evacuation_center_simple(graph)
            
            if origin_node and dest_node:
                # Calculate shortest path using distance only
                path = nx.shortest_path(graph, origin_node, dest_node, weight='weight')
                
                # Calculate metrics
                total_distance = sum(
                    graph[path[i]][path[i+1]][0].get('length', 0) 
                    for i in range(len(path)-1)
                )
                
                # Calculate actual risk encountered (for comparison)
                total_risk = 0
                for i in range(len(path)-1):
                    edge_id = f"{path[i]}_{path[i+1]}_0"
                    risk = self.mas_fro_system.graph_env._risk_scores.get(edge_id, 0)
                    total_risk += risk
                
                avg_risk = total_risk / max(1, len(path)-1)
                safety_score = 1.0 - min(1.0, avg_risk)
                
                computation_time = time.time() - start_time
                
                return {
                    'algorithm': 'baseline_dijkstra',
                    'scenario_id': scenario['id'],
                    'success': True,
                    'path_length': len(path),
                    'total_distance': total_distance,
                    'total_risk': total_risk,
                    'average_risk': avg_risk,
                    'safety_score': safety_score,
                    'computation_time': computation_time,
                    'flood_intensity': scenario['flood_intensity']
                }
            
        except Exception as e:
            computation_time = time.time() - start_time
            return {
                'algorithm': 'baseline_dijkstra',
                'scenario_id': scenario['id'],
                'success': False,
                'error': str(e),
                'computation_time': computation_time,
                'flood_intensity': scenario['flood_intensity']
            }
    
    def _test_mas_fro_algorithm(self, scenario):
        """Test MAS-FRO risk-aware algorithm"""
        start_time = time.time()
        
        try:
            # Create route request
            route_request = RouteRequest(
                request_id=f"EVAL_{scenario['id']}",
                origin=scenario['origin'],
                destination=scenario['destination'],
                timestamp=datetime.now(),
                user_id='evaluator'
            )
            
            # Get routing agent and calculate route
            routing_agent = self.mas_fro_system.agents.get('routing')
            if routing_agent:
                result = routing_agent._calculate_optimal_route(route_request)
                
                if result.get('success'):
                    computation_time = time.time() - start_time
                    
                    return {
                        'algorithm': 'mas_fro_astar',
                        'scenario_id': scenario['id'],
                        'success': True,
                        'path_length': len(result.get('path', [])),
                        'total_distance': result.get('total_distance', 0),
                        'total_risk': result.get('total_risk', 0),
                        'average_risk': result.get('average_risk', 0),
                        'safety_score': result.get('safety_score', 0),
                        'computation_time': computation_time,
                        'flood_intensity': scenario['flood_intensity'],
                        'high_risk_segments': result.get('high_risk_segments', 0)
                    }
            
            computation_time = time.time() - start_time
            return {
                'algorithm': 'mas_fro_astar',
                'scenario_id': scenario['id'],
                'success': False,
                'error': 'No routing agent available',
                'computation_time': computation_time,
                'flood_intensity': scenario['flood_intensity']
            }
            
        except Exception as e:
            computation_time = time.time() - start_time
            return {
                'algorithm': 'mas_fro_astar',
                'scenario_id': scenario['id'],
                'success': False,
                'error': str(e),
                'computation_time': computation_time,
                'flood_intensity': scenario['flood_intensity']
            }
    
    def _find_nearest_node_simple(self, graph, location):
        """Simple nearest node finder"""
        nodes = list(graph.nodes())
        return nodes[0] if nodes else None
    
    def _find_nearest_evacuation_center_simple(self, graph):
        """Simple evacuation center finder"""
        nodes = list(graph.nodes())
        return nodes[-1] if nodes else None
    
    def _generate_comparison_report(self):
        """Generate detailed comparison report"""
        print("\n" + "="*60)
        print("MAS-FRO EVALUATION REPORT")
        print("="*60)
        
        # Filter successful results
        baseline_success = [r for r in self.baseline_results if r.get('success')]
        mas_fro_success = [r for r in self.mas_fro_results if r.get('success')]
        
        print(f"\nSUCCESS RATES:")
        print(f"Baseline: {len(baseline_success)}/{len(self.baseline_results)} ({len(baseline_success)/len(self.baseline_results)*100:.1f}%)")
        print(f"MAS-FRO:  {len(mas_fro_success)}/{len(self.mas_fro_results)} ({len(mas_fro_success)/len(self.mas_fro_results)*100:.1f}%)")
        
        if baseline_success and mas_fro_success:
            print(f"\nSAFETY SCORES:")
            baseline_safety = np.mean([r['safety_score'] for r in baseline_success])
            mas_fro_safety = np.mean([r['safety_score'] for r in mas_fro_success])
            print(f"Baseline: {baseline_safety:.3f}")
            print(f"MAS-FRO:  {mas_fro_safety:.3f}")
            print(f"Improvement: {((mas_fro_safety - baseline_safety) / baseline_safety * 100):+.1f}%")
            
            print(f"\nCOMPUTATION TIME:")
            baseline_time = np.mean([r['computation_time'] for r in baseline_success])
            mas_fro_time = np.mean([r['computation_time'] for r in mas_fro_success])
            print(f"Baseline: {baseline_time:.3f}s")
            print(f"MAS-FRO:  {mas_fro_time:.3f}s")
            
            print(f"\nDISTANCE COMPARISON:")
            baseline_dist = np.mean([r['total_distance'] for r in baseline_success])
            mas_fro_dist = np.mean([r['total_distance'] for r in mas_fro_success])
            print(f"Baseline: {baseline_dist:.0f}m")
            print(f"MAS-FRO:  {mas_fro_dist:.0f}m")
            print(f"Distance trade-off: {((mas_fro_dist - baseline_dist) / baseline_dist * 100):+.1f}%")
        
        # Save detailed results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"results/evaluation_results_{timestamp}.json"
        
        os.makedirs('results', exist_ok=True)
        with open(results_file, 'w') as f:
            json.dump({
                'baseline_results': self.baseline_results,
                'mas_fro_results': self.mas_fro_results,
                'evaluation_timestamp': timestamp
            }, f, indent=2)
        
        print(f"\nDetailed results saved to: {results_file}")

def main_with_web():
    """Main function with web interface"""
    try:
        system = WebEnabledMASFRO()
        
        # Option to run with or without web interface
        import sys
        if '--no-web' in sys.argv:
            system.run_simulation_only(duration=3600)
        else:
            system.run_with_web_interface(duration=3600, web_port=5000)
            
    except KeyboardInterrupt:
        print("Simulation interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

def main_evaluation():
    """Run evaluation comparison"""
    try:
        system = IntegratedMASFROSystem()
        evaluator = MASFROEvaluator(system)
        
        # Run quick evaluation
        evaluator.run_comparative_evaluation(num_scenarios=20)
        
    except Exception as e:
        print(f"Evaluation error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import sys
    
    if '--eval' in sys.argv:
        main_evaluation()
    else:
        main_with_web()