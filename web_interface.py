#!/usr/bin/env python3
"""
Web interface for MAS-FRO system
Run this file to start the web dashboard
"""

from flask import Flask, render_template, request, jsonify
import threading
import time
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.simulation.mas_controller import MASFROController
from src.utils.logging_config import setup_logging
import logging

app = Flask(__name__)
mas_controller = None
simulation_thread = None
system_stats = {
    'agents_active': 0,
    'routes_computed': 0,
    'success_rate': 0.0,
    'avg_computation_time': 0.0
}

@app.route('/')
def dashboard():
    """Main dashboard page"""
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>MAS-FRO Dashboard</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
            .container { max-width: 1200px; margin: 0 auto; }
            .card { background: white; padding: 20px; margin: 20px 0; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            .header { background: #2c3e50; color: white; padding: 20px; border-radius: 8px; text-align: center; }
            .stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; }
            .stat-card { background: #3498db; color: white; padding: 20px; border-radius: 8px; text-align: center; }
            .stat-value { font-size: 2em; font-weight: bold; }
            .stat-label { font-size: 0.9em; opacity: 0.9; }
            .status-indicator { display: inline-block; width: 12px; height: 12px; border-radius: 50%; margin-right: 8px; }
            .status-active { background-color: #27ae60; }
            .status-inactive { background-color: #e74c3c; }
            .log-container { max-height: 300px; overflow-y: auto; background: #2c3e50; color: #ecf0f1; padding: 15px; border-radius: 4px; font-family: monospace; font-size: 12px; }
            .button { background: #3498db; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; margin: 5px; }
            .button:hover { background: #2980b9; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🚨 MAS-FRO: Multi-Agent Flood Route Optimization</h1>
                <p>Real-time Monitoring Dashboard</p>
            </div>
            
            <div class="card">
                <h2><span class="status-indicator status-active"></span>System Status</h2>
                <div class="stats" id="stats-container">
                    <div class="stat-card">
                        <div class="stat-value" id="agents-count">5</div>
                        <div class="stat-label">Active Agents</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value" id="routes-count">0</div>
                        <div class="stat-label">Routes Computed</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value" id="success-rate">0%</div>
                        <div class="stat-label">Success Rate</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value" id="avg-time">0ms</div>
                        <div class="stat-label">Avg Response Time</div>
                    </div>
                </div>
            </div>
            
            <div class="card">
                <h3>System Control</h3>
                <button class="button" onclick="requestRoute()">🚗 Request Test Route</button>
                <button class="button" onclick="triggerFloodEvent()">🌊 Trigger Flood Event</button>
                <button class="button" onclick="clearLogs()">🗑️ Clear Logs</button>
            </div>
            
            <div class="card">
                <h3>📊 System Activity Log</h3>
                <div class="log-container" id="activity-log">
                    <div>System initialized...</div>
                </div>
            </div>
        </div>

        <script>
            function updateStats() {
                fetch('/api/stats')
                    .then(response => response.json())
                    .then(data => {
                        document.getElementById('agents-count').textContent = data.agents_active || 5;
                        document.getElementById('routes-count').textContent = data.routes_computed || 0;
                        document.getElementById('success-rate').textContent = 
                            (data.success_rate * 100).toFixed(1) + '%';
                        document.getElementById('avg-time').textContent = 
                            (data.avg_computation_time * 1000).toFixed(0) + 'ms';
                    })
                    .catch(error => console.error('Error updating stats:', error));
            }
            
            function addLog(message) {
                const log = document.getElementById('activity-log');
                const timestamp = new Date().toLocaleTimeString();
                const logEntry = document.createElement('div');
                logEntry.innerHTML = `[${timestamp}] ${message}`;
                log.appendChild(logEntry);
                log.scrollTop = log.scrollHeight;
            }
            
            function requestRoute() {
                fetch('/api/request_route', { method: 'POST' })
                    .then(response => response.json())
                    .then(data => {
                        addLog(`📍 Route requested: ${data.request_id || 'Unknown'}`);
                    })
                    .catch(error => {
                        addLog(`❌ Error requesting route: ${error}`);
                    });
            }
            
            function triggerFloodEvent() {
                addLog('🌊 Flood event triggered - updating risk assessments...');
                // In a full implementation, this would trigger a scenario
            }
            
            function clearLogs() {
                document.getElementById('activity-log').innerHTML = '<div>Logs cleared...</div>';
            }
            
            // Update stats every 5 seconds
            setInterval(updateStats, 5000);
            updateStats(); // Initial load
            
            // Add some initial log entries
            setTimeout(() => addLog('🤖 All agents initialized and active'), 1000);
            setTimeout(() => addLog('🗺️ Road network loaded: Marikina, Philippines'), 1500);
            setTimeout(() => addLog('📡 Data collection agents started'), 2000);
        </script>
    </body>
    </html>
    '''

@app.route('/api/stats')
def get_stats():
    """Get current system statistics"""
    return jsonify(system_stats)

@app.route('/api/request_route', methods=['POST'])
def request_route():
    """Handle test route request"""
    global mas_controller
    if mas_controller:
        mas_controller.simulate_user_request()
        return jsonify({'status': 'success', 'request_id': f'WEB_{int(time.time())}'})
    return jsonify({'status': 'error', 'message': 'System not running'})

def run_simulation():
    """Run the MAS-FRO simulation in background"""
    global mas_controller, system_stats
    
    setup_logging(logging.INFO)
    mas_controller = MASFROController()
    
    # Update system stats periodically
    def update_stats():
        while True:
            try:
                system_stats.update({
                    'agents_active': len(mas_controller.agents),
                    'routes_computed': system_stats.get('routes_computed', 0) + 1,
                    'success_rate': 0.95,  # Mock data
                    'avg_computation_time': 0.045  # Mock data
                })
                time.sleep(10)
            except:
                break
    
    stats_thread = threading.Thread(target=update_stats, daemon=True)
    stats_thread.start()
    
    # Run simulation
    mas_controller.run_simulation(duration=7200)  # 2 hours

def main():
    """Main function for web interface"""
    global simulation_thread
    
    print("🚀 Starting MAS-FRO Web Interface...")
    print("📊 Dashboard will be available at: http://localhost:5000")
    
    # Start simulation in background
    simulation_thread = threading.Thread(target=run_simulation, daemon=True)
    simulation_thread.start()
    
    # Start web server
    app.run(host='0.0.0.0', port=5000, debug=False)

if __name__ == "__main__":
    main()