# Advanced Agent Implementations for MAS-FRO
# This file contains enhanced agent implementations with ML integration

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import re
import json
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import logging
from geopy.distance import geodesic

logger = logging.getLogger(__name__)

class AdvancedHazardAgent(BaseAgent):
    """Enhanced Hazard Agent with ML-based risk prediction and NLP processing"""
    
    def __init__(self, agent_id: str, env: simpy.Environment, 
                 input_queue: Queue, output_queue: Queue, 
                 graph_env: DynamicGraphEnvironment):
        super().__init__(agent_id, env, input_queue, output_queue)
        self.graph_env = graph_env
        
        # Data buffers
        self.flood_data_buffer = []
        self.crowdsourced_buffer = []
        self.historical_data = []
        
        # ML Models
        self.flood_predictor = None
        self.nlp_model = None
        self.nlp_tokenizer = None
        self.scaler = StandardScaler()
        
        # Risk decay parameters
        self.risk_decay_rate = 0.1  # Risk decreases by 10% every hour
        self.last_update_time = {}
        
        # Initialize ML models
        self._initialize_ml_models()
    
    def _initialize_ml_models(self):
        """Initialize machine learning models"""
        try:
            # Initialize flood prediction model (Random Forest)
            self.flood_predictor = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            
            # Initialize NLP model for crowdsourced data processing
            model_name = "distilbert-base-uncased"
            self.nlp_tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # For demonstration, we'll use a pre-trained sentiment model
            # In practice, you'd fine-tune on disaster-related text
            self.nlp_model = AutoModelForSequenceClassification.from_pretrained(
                "cardiffnlp/twitter-roberta-base-sentiment-latest"
            )
            
            logger.info(f"{self.agent_id}: ML models initialized successfully")
            
        except Exception as e:
            logger.warning(f"{self.agent_id}: Could not initialize ML models: {e}")
            # Continue without ML models for basic functionality
    
    def run(self):
        """Enhanced run method with ML integration"""
        while self.running:
            try:
                # Process incoming messages
                self._process_incoming_messages()
                
                # Update risk assessments with ML predictions
                if self.flood_data_buffer or self.crowdsourced_buffer:
                    self._update_risk_assessments_ml()
                
                # Apply risk decay over time
                self._apply_risk_decay()
                
                # Train models with accumulated data
                if len(self.historical_data) > 100:  # Minimum data for training
                    self._retrain_models()
                
                yield self.env.timeout(30)  # Update every 30 seconds
                
            except Exception as e:
                logger.error(f"{self.agent_id} error: {e}")
                yield self.env.timeout(60)
    
    def _process_incoming_messages(self):
        """Process all incoming messages"""
        while not self.input_queue.empty():
            try:
                message = self.input_queue.get_nowait()
                msg_type = message.get('type')
                
                if msg_type == 'flood_data':
                    self.flood_data_buffer.extend(message['data'])
                elif msg_type == 'crowdsourced_data':
                    processed_reports = self._process_crowdsourced_nlp(message['data'])
                    self.crowdsourced_buffer.extend(processed_reports)
                elif msg_type == 'user_feedback':
                    self._apply_user_feedback(message['data'])
                    
            except Exception as e:
                logger.error(f"Error processing message: {e}")
    
    def _process_crowdsourced_nlp(self, reports: List[HazardData]) -> List[HazardData]:
        """Process crowdsourced text data using NLP"""
        processed_reports = []
        
        for report in reports:
            try:
                if hasattr(report, 'text_content'):
                    # Extract flood severity from text using NLP
                    severity = self._extract_flood_severity(report.text_content)
                    confidence = self._calculate_text_confidence(report.text_content)
                    
                    # Update report with NLP-derived values
                    report.risk_level = severity
                    report.confidence = confidence
                
                processed_reports.append(report)
                
            except Exception as e:
                logger.warning(f"Error processing crowdsourced report: {e}")
                processed_reports.append(report)  # Keep original if processing fails
        
        return processed_reports
    
    def _extract_flood_severity(self, text: str) -> float:
        """Extract flood severity from text using NLP and rules"""
        if not text or not self.nlp_model:
            return 0.5  # Default moderate risk
        
        text = text.lower()
        
        # Rule-based severity extraction
        severity_keywords = {
            'impassable': 1.0,
            'blocked': 1.0,
            'flooded': 0.8,
            'knee-deep': 0.7,
            'ankle-deep': 0.4,
            'wet': 0.2,
            'clear': 0.0
        }
        
        # Check for explicit severity keywords
        for keyword, severity in severity_keywords.items():
            if keyword in text:
                return severity
        
        # Use NLP model for more nuanced analysis
        try:
            inputs = self.nlp_tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
            
            with torch.no_grad():
                outputs = self.nlp_model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                
                # Convert sentiment to risk (negative sentiment = higher risk)
                # This is a simplified mapping - in practice, you'd train on disaster data
                risk_score = 1.0 - predictions[0][2].item()  # Invert positive sentiment
                return max(0.0, min(1.0, risk_score))
        
        except Exception as e:
            logger.warning(f"NLP processing failed: {e}")
            return 0.5
    
    def _calculate_text_confidence(self, text: str) -> float:
        """Calculate confidence score based on text characteristics"""
        if not text:
            return 0.1
        
        confidence = 0.5  # Base confidence
        
        # Increase confidence for specific details
        if re.search(r'\d+\s*(cm|meter|feet|inch)', text.lower()):
            confidence += 0.2  # Specific measurements
        
        if any(word in text.lower() for word in ['photo', 'image', 'picture', 'video']):
            confidence += 0.3  # Visual evidence
        
        if len(text.split()) > 10:
            confidence += 0.1  # Detailed description
        
        # Decrease confidence for vague terms
        if any(word in text.lower() for word in ['maybe', 'think', 'seems', 'probably']):
            confidence -= 0.2
        
        return max(0.1, min(1.0, confidence))
    
    def _update_risk_assessments_ml(self):
        """Update risk assessments using ML predictions"""
        current_time = datetime.now()
        
        # Process flood data with ML predictions
        if self.flood_data_buffer and self.flood_predictor:
            for flood_data in self.flood_data_buffer:
                # Prepare features for ML model
                features = self._prepare_flood_features(flood_data, current_time)
                
                # Predict future risk if model is trained
                if hasattr(self.flood_predictor, 'feature_importances_'):
                    try:
                        features_scaled = self.scaler.transform([features])
                        predicted_risk = self.flood_predictor.predict(features_scaled)[0]
                        predicted_risk = max(0.0, min(1.0, predicted_risk))
                    except:
                        predicted_risk = self._calculate_flood_risk(flood_data)
                else:
                    predicted_risk = self._calculate_flood_risk(flood_data)
                
                # Apply prediction to nearby edges
                affected_edges = self._find_nearby_edges(flood_data.location, radius=500)
                for u, v, key in affected_edges:
                    self.graph_env.update_edge_risk(u, v, key, predicted_risk)
                    self.last_update_time[f"{u}_{v}_{key}"] = current_time
                
                # Store for model training
                self.historical_data.append({
                    'features': features,
                    'actual_risk': predicted_risk,
                    'timestamp': current_time
                })
        
        # Process crowdsourced data
        for report in self.crowdsourced_buffer:
            weighted_risk = report.risk_level * report.confidence
            affected_edges = self._find_nearby_edges(report.location, radius=200)
            
            for u, v, key in affected_edges:
                current_risk = self.graph_env._risk_scores.get(f"{u}_{v}_{key}", 0.0)
                new_risk = max(current_risk, weighted_risk)
                self.graph_env.update_edge_risk(u, v, key, new_risk)
                self.last_update_time[f"{u}_{v}_{key}"] = current_time
        
        # Clear buffers
        self.flood_data_buffer.clear()
        self.crowdsourced_buffer.clear()
        
        logger.info(f"{self.agent_id}: Updated risk assessments using ML predictions")
    
    def _prepare_flood_features(self, flood_data: FloodData, current_time: datetime) -> List[float]:
        """Prepare features for ML flood prediction model"""
        # Time-based features
        hour_of_day = current_time.hour
        day_of_week = current_time.weekday()
        month = current_time.month
        
        # Cyclical encoding for time features
        hour_sin = np.sin(2 * np.pi * hour_of_day / 24)
        hour_cos = np.cos(2 * np.pi * hour_of_day / 24)
        month_sin = np.sin(2 * np.pi * month / 12)
        month_cos = np.cos(2 * np.pi * month / 12)
        
        # Flood data features
        water_level = flood_data.water_level
        rainfall_intensity = flood_data.rainfall_intensity
        
        # Location features (simplified)
        lat, lon = flood_data.location
        
        # Historical context (if available)
        recent_rainfall = self._get_recent_rainfall(flood_data.location, hours=6)
        
        return [
            water_level, rainfall_intensity, lat, lon,
            hour_sin, hour_cos, month_sin, month_cos,
            day_of_week, recent_rainfall
        ]
    
    def _get_recent_rainfall(self, location: Tuple[float, float], hours: int = 6) -> float:
        """Get recent rainfall data for location (simplified implementation)"""
        # In practice, this would query historical weather data
        # For simulation, return random value based on current data
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        recent_data = [
            data for data in self.historical_data 
            if data['timestamp'] > cutoff_time
        ]
        
        if recent_data:
            return np.mean([data['features'][1] for data in recent_data])  # rainfall_intensity
        
        return 0.0
    
    def _apply_risk_decay(self):
        """Apply time-based risk decay"""
        current_time = datetime.now()
        
        for edge_id, last_update in list(self.last_update_time.items()):
            time_diff = (current_time - last_update).total_seconds() / 3600  # hours
            
            if time_diff > 0:
                current_risk = self.graph_env._risk_scores.get(edge_id, 0.0)
                
                if current_risk > 0 and current_risk != float('inf'):
                    # Exponential decay
                    decayed_risk = current_risk * np.exp(-self.risk_decay_rate * time_diff)
                    
                    # Update if significant change
                    if abs(current_risk - decayed_risk) > 0.01:
                        u, v, key = edge_id.split('_')
                        self.graph_env.update_edge_risk(int(u), int(v), int(key), decayed_risk)
                        self.last_update_time[edge_id] = current_time
    
    def _retrain_models(self):
        """Retrain ML models with accumulated data"""
        try:
            if len(self.historical_data) < 100:
                return
            
            # Prepare training data
            features = [data['features'] for data in self.historical_data[-1000:]]  # Use last 1000 samples
            targets = [data['actual_risk'] for data in self.historical_data[-1000:]]
            
            X = np.array(features)
            y = np.array(targets)
            
            # Fit scaler and model
            X_scaled = self.scaler.fit_transform(X)
            self.flood_predictor.fit(X_scaled, y)
            
            logger.info(f"{self.agent_id}: Retrained ML models with {len(features)} samples")
            
        except Exception as e:
            logger.error(f"Error retraining models: {e}")
    
    def _find_nearby_edges(self, location: Tuple[float, float], radius: int = 500) -> List[Tuple]:
        """Find edges within radius of location using spatial indexing"""
        nearby_edges = []
        
        try:
            # Get all edges with their coordinates
            for u, v, key in self.graph_env.graph.edges(keys=True):
                try:
                    # Get node coordinates
                    u_data = self.graph_env.graph.nodes[u]
                    v_data = self.graph_env.graph.nodes[v]
                    
                    u_coord = (u_data.get('y', 0), u_data.get('x', 0))
                    v_coord = (v_data.get('y', 0), v_data.get('x', 0))
                    
                    # Calculate distance to edge midpoint
                    midpoint = (
                        (u_coord[0] + v_coord[0]) / 2,
                        (u_coord[1] + v_coord[1]) / 2
                    )
                    
                    distance = geodesic(location, midpoint).meters
                    
                    if distance <= radius:
                        nearby_edges.append((u, v, key))
                
                except (KeyError, ValueError):
                    continue
                    
        except Exception as e:
            logger.warning(f"Error finding nearby edges: {e}")
            # Fallback: return random sample
            edges = list(self.graph_env.graph.edges(keys=True))
            nearby_edges = edges[:min(3, len(edges))]
        
        return nearby_edges

class EnhancedRoutingAgent(BaseAgent):
    """Enhanced routing agent with multiple algorithms and performance optimization"""
    
    def __init__(self, agent_id: str, env: simpy.Environment, 
                 input_queue: Queue, output_queue: Queue, 
                 graph_env: DynamicGraphEnvironment):
        super().__init__(agent_id, env, input_queue, output_queue)
        self.graph_env = graph_env
        self.performance_monitor = PerformanceMonitor()
        self.algorithm_config = {
            'default': 'astar',
            'fallback': 'dijkstra',
            'max_computation_time': 5.0,
            'weight_factors': {
                'distance': 0.8,
                'risk': 0.2
            }
        }
    
    def run(self):
        """Enhanced routing with performance monitoring"""
        while self.running:
            try:
                if not self.input_queue.empty():
                    message = self.input_queue.get()
                    
                    if message.get('type') == 'route_request':
                        route_request = message['data']
                        
                        # Calculate route with performance monitoring
                        start_time = time.time()
                        route_result = self._calculate_optimal_route(route_request)
                        computation_time = time.time() - start_time
                        
                        # Record performance metrics
                        metrics = RouteMetrics(
                            request_id=route_request.request_id,
                            computation_time=computation_time,
                            route_distance=route_result.get('total_distance', 0),
                            route_safety_score=route_result.get('total_risk', 0),
                            success=route_result.get('success', False),
                            timestamp=time.time()
                        )
                        self.performance_monitor.record_route_computation(metrics)
                        
                        # Send response
                        response = {
                            'type': 'route_response',
                            'request_id': route_request.request_id,
                            'route': route_result,
                            'sender': self.agent_id,
                            'timestamp': time.time(),
                            'computation_time': computation_time
                        }
                        self.output_queue.put(response)
                
                yield self.env.timeout(5)
                
            except Exception as e:
                logger.error(f"{self.agent_id} error: {e}")
                yield self.env.timeout(30)
    
    def _calculate_optimal_route(self, route_request: RouteRequest) -> dict:
        """Calculate optimal route using multiple algorithms with fallback"""
        try:
            # Try primary algorithm first
            result = self._try_algorithm(route_request, self.algorithm_config['default'])
            
            if result.get('success'):
                return result
            
            # Fallback to secondary algorithm
            logger.warning(f"Primary algorithm failed, trying fallback for {route_request.request_id}")
            result = self._try_algorithm(route_request, self.algorithm_config['fallback'])
            
            return result
            
        except Exception as e:
            logger.error(f"All routing algorithms failed: {e}")
            return {'error': str(e), 'success': False}
    
    def _try_algorithm(self, route_request: RouteRequest, algorithm: str) -> dict:
        """Try specific routing algorithm"""
        start_time = time.time()
        
        try:
            graph = self.graph_env.get_current_state()
            
            # Find start and end nodes
            origin_node = self._find_nearest_node(graph, route_request.origin)
            dest_node = self._find_nearest_evacuation_center(graph, route_request.destination)
            
            if origin_node is None or dest_node is None:
                return {'error': 'Invalid start or end point', 'success': False}
            
            # Calculate path based on algorithm
            if algorithm == 'astar':
                path = nx.astar_path(
                    graph, origin_node, dest_node,
                    weight='risk_aware_weight',
                    heuristic=self._distance_heuristic
                )
            elif algorithm == 'dijkstra':
                path = nx.shortest_path(
                    graph, origin_node, dest_node,
                    weight='risk_aware_weight'
                )
            else:
                raise ValueError(f"Unknown algorithm: {algorithm}")
            
            # Calculate metrics
            metrics = self._calculate_route_metrics(graph, path)
            
            return {
                'path': path,
                'algorithm_used': algorithm,
                'success': True,
                **metrics
            }
            
        except nx.NetworkXNoPath:
            return {'error': 'No path found', 'success': False}
        except Exception as e:
            return {'error': str(e), 'success': False}
    
    def _calculate_route_metrics(self, graph: nx.MultiDiGraph, path: List[int]) -> dict:
        """Calculate comprehensive route metrics"""
        total_distance = 0
        total_risk = 0
        max_risk_segment = 0
        risk_segments = 0
        
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            
            # Get edge data (take first edge if multiple)
            edge_data = graph[u][v][0]
            segment_distance = edge_data.get('length', 0)
            
            # Get risk score
            edge_id = f"{u}_{v}_0"
            segment_risk = self.graph_env._risk_scores.get(edge_id, 0)
            
            total_distance += segment_distance
            total_risk += segment_risk
            
            if segment_risk > 0.5:  # High risk threshold
                risk_segments += 1
                max_risk_segment = max(max_risk_segment, segment_risk)
        
        # Calculate safety score (inverse of risk)
        avg_risk = total_risk / max(1, len(path) - 1)
        safety_score = 1.0 - min(1.0, avg_risk)
        
        # Estimate travel time (simplified)
        avg_speed = 30  # km/h in urban areas
        travel_time = (total_distance / 1000) / avg_speed * 3600  # seconds
        
        return {
            'total_distance': total_distance,
            'total_risk': total_risk,
            'average_risk': avg_risk,
            'safety_score': safety_score,
            'max_risk_segment': max_risk_segment,
            'high_risk_segments': risk_segments,
            'estimated_travel_time': travel_time,
            'num_segments': len(path) - 1
        }
    
    def get_performance_statistics(self) -> dict:
        """Get routing performance statistics"""
        return self.performance_monitor.get_statistics()

# Additional utility classes for performance monitoring
from dataclasses import dataclass

@dataclass
class RouteMetrics:
    request_id: str
    computation_time: float
    route_distance: float
    route_safety_score: float
    success: bool
    timestamp: float

class PerformanceMonitor:
    """Performance monitoring class for the MAS-FRO system"""
    
    def __init__(self):
        self.metrics: List[RouteMetrics] = []
        self.system_start_time = time.time()
    
    def record_route_computation(self, metrics: RouteMetrics):
        """Record route computation metrics"""
        self.metrics.append(metrics)
    
    def get_statistics(self) -> Dict:
        """Get comprehensive performance statistics"""
        if not self.metrics:
            return {
                'total_requests': 0,
                'successful_routes': 0,
                'success_rate': 0.0,
                'avg_computation_time': 0.0,
                'avg_route_distance': 0.0,
                'avg_safety_score': 0.0
            }
        
        successful_routes = [m for m in self.metrics if m.success]
        failed_routes = [m for m in self.metrics if not m.success]
        
        # Calculate percentiles for computation time
        computation_times = [m.computation_time for m in successful_routes]
        
        stats = {
            'total_requests': len(self.metrics),
            'successful_routes': len(successful_routes),
            'failed_routes': len(failed_routes),
            'success_rate': len(successful_routes) / len(self.metrics),
            'system_uptime': time.time() - self.system_start_time
        }
        
        if successful_routes:
            stats.update({
                'avg_computation_time': np.mean(computation_times),
                'median_computation_time': np.median(computation_times),
                'p95_computation_time': np.percentile(computation_times, 95),
                'max_computation_time': max(computation_times),
                'avg_route_distance': np.mean([m.route_distance for m in successful_routes]),
                'avg_safety_score': np.mean([m.route_safety_score for m in successful_routes])
            })
        
        return stats
    
    def save_metrics(self, filepath: str):
        """Save metrics to file"""
        import json
        from dataclasses import asdict
        
        with open(filepath, 'w') as f:
            json.dump([asdict(m) for m in self.metrics], f, indent=2)

class SmartEvacuationManagerAgent(BaseAgent):
    """Enhanced Evacuation Manager with user profiling and adaptive routing"""
    
    def __init__(self, agent_id: str, env: simpy.Environment, 
                 input_queue: Queue, output_queue: Queue,
                 routing_queue: Queue, hazard_queue: Queue):
        super().__init__(agent_id, env, input_queue, output_queue)
        self.routing_queue = routing_queue
        self.hazard_queue = hazard_queue
        self.pending_requests = {}
        self.user_profiles = {}
        self.route_history = {}
        
    def run(self):
        """Enhanced user management with profiling"""
        while self.running:
            try:
                # Process incoming messages
                while not self.input_queue.empty():
                    message = self.input_queue.get()
                    self._process_message(message)
                
                # Check for routing responses
                while not self.routing_queue.empty():
                    try:
                        response = self.routing_queue.get_nowait()
                        if response.get('type') == 'route_response':
                            self._handle_route_response(response)
                    except:
                        break
                
                yield self.env.timeout(5)
                
            except Exception as e:
                logger.error(f"{self.agent_id} error: {e}")
                yield self.env.timeout(30)
    
    def _process_message(self, message: dict):
        """Process different types of user messages"""
        msg_type = message.get('type')
        
        if msg_type == 'user_request':
            self._handle_route_request(message['data'])
        elif msg_type == 'user_feedback':
            self._handle_user_feedback(message['data'])
        elif msg_type == 'user_profile_update':
            self._update_user_profile(message['data'])
        elif msg_type == 'emergency_broadcast':
            self._handle_emergency_broadcast(message['data'])
    
    def _handle_route_request(self, route_request: RouteRequest):
        """Handle route request with user profiling"""
        logger.info(f"{self.agent_id}: Processing route request {route_request.request_id}")
        
        # Update user profile
        self._update_user_activity(route_request.user_id)
        
        # Check if user needs special routing considerations
        user_profile = self.user_profiles.get(route_request.user_id, {})
        enhanced_request = self._enhance_request_with_profile(route_request, user_profile)
        
        # Store request
        self.pending_requests[route_request.request_id] = {
            'original_request': route_request,
            'enhanced_request': enhanced_request,
            'user_profile': user_profile,
            'request_time': time.time()
        }
        
        # Forward to Routing Agent
        message = {
            'type': 'route_request',
            'data': enhanced_request,
            'sender': self.agent_id,
            'timestamp': time.time()
        }
        
        # Send to routing queue instead of output queue
        if hasattr(self, 'routing_agent_queue'):
            self.routing_agent_queue.put(message)
        else:
            self.output_queue.put(message)
    
    def _enhance_request_with_profile(self, request: RouteRequest, profile: dict) -> RouteRequest:
        """Enhance route request based on user profile"""
        enhanced_request = request
        
        # Modify request based on user characteristics
        if profile.get('mobility_impaired', False):
            # Prefer accessible routes
            enhanced_request.preferences = getattr(request, 'preferences', {})
            enhanced_request.preferences['avoid_stairs'] = True
            enhanced_request.preferences['prefer_wide_roads'] = True
        
        if profile.get('has_vehicle', True):
            enhanced_request.transport_mode = 'vehicle'
        else:
            enhanced_request.transport_mode = 'pedestrian'
        
        # Risk tolerance based on past behavior
        risk_tolerance = profile.get('risk_tolerance', 0.5)
        enhanced_request.risk_weight = 1.0 - risk_tolerance
        
        return enhanced_request
    
    def _handle_route_response(self, response: dict):
        """Handle route response with user-specific formatting"""
        request_id = response['request_id']
        route_data = response['route']
        
        if request_id in self.pending_requests:
            request_info = self.pending_requests[request_id]
            original_request = request_info['original_request']
            user_profile = request_info['user_profile']
            
            # Format response based on user preferences
            formatted_route = self._format_route_for_user(route_data, user_profile)
            
            # Store in route history for learning
            self._store_route_history(original_request.user_id, {
                'request': original_request,
                'route': route_data,
                'timestamp': time.time()
            })
            
            # Create user-friendly response
            user_response = {
                'type': 'route_delivered',
                'request_id': request_id,
                'user_id': original_request.user_id,
                'route': formatted_route,
                'estimated_time': route_data.get('estimated_travel_time', 0),
                'safety_score': route_data.get('safety_score', 0),
                'instructions': self._generate_turn_by_turn_instructions(route_data),
                'timestamp': time.time()
            }
            
            # In real implementation, send to user's device
            logger.info(f"{self.agent_id}: Route delivered to user {original_request.user_id}")
            logger.info(f"Safety Score: {route_data.get('safety_score', 0):.2f}")
            logger.info(f"Estimated Time: {route_data.get('estimated_travel_time', 0):.0f} seconds")
            
            # Clean up
            del self.pending_requests[request_id]
    
    def _format_route_for_user(self, route_data: dict, user_profile: dict) -> dict:
        """Format route data based on user preferences"""
        formatted = route_data.copy()
        
        # Add user-specific warnings
        warnings = []
        
        if route_data.get('high_risk_segments', 0) > 0:
            warnings.append("Route includes areas with elevated flood risk")
        
        if user_profile.get('first_time_user', False):
            warnings.append("Drive slowly and follow evacuation signs")
        
        if route_data.get('max_risk_segment', 0) > 0.8:
            warnings.append("CAUTION: Route includes potentially dangerous areas")
        
        formatted['warnings'] = warnings
        formatted['user_specific'] = True
        
        return formatted
    
    def _generate_turn_by_turn_instructions(self, route_data: dict) -> List[str]:
        """Generate simple turn-by-turn instructions"""
        # Simplified implementation - in practice, would use detailed road network data
        path = route_data.get('path', [])
        
        if len(path) < 2:
            return ["No route instructions available"]
        
        instructions = [
            "Start from your current location",
            f"Follow the recommended path through {len(path)-1} segments",
            "Monitor road conditions and follow evacuation signs",
            "Proceed to the designated evacuation center"
        ]
        
        # Add risk-specific warnings
        if route_data.get('high_risk_segments', 0) > 0:
            instructions.insert(-1, "CAUTION: Drive carefully through flood-prone areas")
        
        return instructions
    
    def _handle_user_feedback(self, feedback: dict):
        """Process user feedback and forward to Hazard Agent"""
        user_id = feedback.get('user_id')
        condition = feedback.get('condition')
        location = feedback.get('location')
        
        # Update user profile based on feedback
        if user_id:
            self._update_user_profile_from_feedback(user_id, feedback)
        
        # Forward high-confidence feedback to Hazard Agent
        hazard_message = {
            'type': 'user_feedback',
            'data': {
                'location': location,
                'condition': condition,
                'confidence': 1.0,  # User feedback has high confidence
                'timestamp': time.time(),
                'user_id': user_id
            },
            'sender': self.agent_id,
            'timestamp': time.time()
        }
        
        self.hazard_queue.put(hazard_message)
        
        logger.info(f"{self.agent_id}: Processed feedback from user {user_id}: {condition}")
    
    def _update_user_activity(self, user_id: str):
        """Update user activity and profile"""
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = {
                'first_seen': time.time(),
                'request_count': 0,
                'last_request': time.time(),
                'risk_tolerance': 0.5,  # Default moderate risk tolerance
                'has_vehicle': True,
                'mobility_impaired': False,
                'first_time_user': True
            }
        
        profile = self.user_profiles[user_id]
        profile['request_count'] += 1
        profile['last_request'] = time.time()
        
        # Update first-time user status
        if profile['request_count'] > 1:
            profile['first_time_user'] = False
    
    def _update_user_profile_from_feedback(self, user_id: str, feedback: dict):
        """Update user risk tolerance based on feedback patterns"""
        if user_id not in self.user_profiles:
            return
        
        profile = self.user_profiles[user_id]
        condition = feedback.get('condition', '')
        
        # Adjust risk tolerance based on feedback
        if condition in ['clear', 'safe']:
            # User took suggested route and found it safe - maintain trust
            pass
        elif condition in ['blocked', 'dangerous']:
            # Route was problematic - user might prefer safer routes
            profile['risk_tolerance'] = max(0.0, profile['risk_tolerance'] - 0.1)
        
        # Store feedback history
        if 'feedback_history' not in profile:
            profile['feedback_history'] = []
        
        profile['feedback_history'].append({
            'condition': condition,
            'timestamp': time.time()
        })
        
        # Keep only recent feedback (last 10)
        profile['feedback_history'] = profile['feedback_history'][-10:]
    
    def _store_route_history(self, user_id: str, route_info: dict):
        """Store route history for analysis"""
        if user_id not in self.route_history:
            self.route_history[user_id] = []
        
        self.route_history[user_id].append(route_info)
        
        # Keep only recent routes (last 20)
        self.route_history[user_id] = self.route_history[user_id][-20:]
    
    def _handle_emergency_broadcast(self, broadcast: dict):
        """Handle emergency broadcast messages"""
        message = broadcast.get('message', '')
        severity = broadcast.get('severity', 'medium')
        affected_areas = broadcast.get('affected_areas', [])
        
        logger.warning(f"{self.agent_id}: Emergency broadcast ({severity}): {message}")
        
        # In real implementation, would push notifications to affected users
        # For now, log the broadcast
        if affected_areas:
            logger.warning(f"Affected areas: {affected_areas}")

# Example usage and integration
class IntegratedMASFROSystem:
    """Integrated MAS-FRO system with all enhanced agents"""
    
    def __init__(self):
        self.env = simpy.Environment()
        self.graph_env = DynamicGraphEnvironment()
        
        # Enhanced message queues
        self.manager = Manager()
        self.flood_to_hazard = Queue()
        self.scout_to_hazard = Queue()
        self.user_to_evacuation = Queue()
        self.evacuation_to_routing = Queue()
        self.routing_to_evacuation = Queue()
        self.feedback_to_hazard = Queue()
        
        # Performance monitoring
        self.performance_monitor = PerformanceMonitor()
        
        # Initialize enhanced agents
        self.agents = {}
        self._create_enhanced_agents()
    
    def _create_enhanced_agents(self):
        """Create all enhanced agents"""
        # Enhanced Hazard Agent with ML
        self.agents['hazard'] = AdvancedHazardAgent(
            'AdvancedHazardAgent-1', self.env, 
            self.feedback_to_hazard, None, self.graph_env
        )
        
        # Enhanced Routing Agent
        self.agents['routing'] = EnhancedRoutingAgent(
            'EnhancedRoutingAgent-1', self.env,
            self.evacuation_to_routing, self.routing_to_evacuation,
            self.graph_env
        )
        
        # Smart Evacuation Manager
        self.agents['evacuation'] = SmartEvacuationManagerAgent(
            'SmartEvacuationManager-1', self.env,
            self.user_to_evacuation, self.feedback_to_hazard,
            self.evacuation_to_routing, self.feedback_to_hazard
        )
        
        # Basic data collection agents (can be enhanced similarly)
        self.agents['flood'] = FloodAgent(
            'FloodAgent-1', self.env, None, self.flood_to_hazard
        )
        
        self.agents['scout'] = ScoutAgent(
            'ScoutAgent-1', self.env, None, self.scout_to_hazard
        )
    
    def run_enhanced_simulation(self, duration: int = 3600):
        """Run enhanced simulation with performance monitoring"""
        logger.info("Starting Enhanced MAS-FRO simulation...")
        
        # Message routing process
        def route_messages():
            while True:
                # Route flood data to hazard agent
                while not self.flood_to_hazard.empty():
                    msg = self.flood_to_hazard.get()
                    self.feedback_to_hazard.put(msg)
                
                # Route scout data to hazard agent  
                while not self.scout_to_hazard.empty():
                    msg = self.scout_to_hazard.get()
                    self.feedback_to_hazard.put(msg)
                
                yield self.env.timeout(10)
        
        self.env.process(route_messages())
        
        # Enhanced user request generator
        def enhanced_user_generator():
            user_counter = 0
            while True:
                yield self.env.timeout(random.randint(180, 420))  # 3-7 minutes
                
                user_counter += 1
                request = RouteRequest(
                    request_id=f"REQ_{int(time.time())}_{user_counter}",
                    origin=(
                        14.6507 + random.uniform(-0.01, 0.01),
                        121.1029 + random.uniform(-0.01, 0.01)
                    ),
                    destination="nearest_evacuation_center",
                    timestamp=datetime.now(),
                    user_id=f"user_{user_counter % 10}"  # Simulate 10 different users
                )
                
                message = {
                    'type': 'user_request',
                    'data': request,
                    'sender': 'simulator',
                    'timestamp': time.time()
                }
                self.user_to_evacuation.put(message)
                
                logger.info(f"Generated user request: {request.request_id}")
        
        self.env.process(enhanced_user_generator())
        
        # Performance monitoring process
        def performance_logger():
            while True:
                yield self.env.timeout(600)  # Log every 10 minutes
                
                # Get routing performance
                if 'routing' in self.agents:
                    routing_agent = self.agents['routing']
                    if hasattr(routing_agent, 'performance_monitor'):
                        stats = routing_agent.get_performance_statistics()
                        logger.info(f"Performance Stats: {stats}")
        
        self.env.process(performance_logger())
        
        # Run simulation
        logger.info(f"Running enhanced simulation for {duration} seconds...")
        self.env.run(until=duration)
        
        # Final performance report
        self._generate_performance_report()
    
    def _generate_performance_report(self):
        """Generate comprehensive performance report"""
        logger.info("=== ENHANCED MAS-FRO PERFORMANCE REPORT ===")
        
        if 'routing' in self.agents and hasattr(self.agents['routing'], 'performance_monitor'):
            stats = self.agents['routing'].get_performance_statistics()
            
            logger.info(f"Total Requests: {stats.get('total_requests', 0)}")
            logger.info(f"Success Rate: {stats.get('success_rate', 0):.2%}")
            logger.info(f"Avg Computation Time: {stats.get('avg_computation_time', 0):.3f}s")
            logger.info(f"P95 Computation Time: {stats.get('p95_computation_time', 0):.3f}s")
            logger.info(f"Avg Safety Score: {stats.get('avg_safety_score', 0):.3f}")
            
            # Save detailed metrics
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            metrics_file = f"results/metrics/enhanced_simulation_{timestamp}.json"
            self.agents['routing'].performance_monitor.save_metrics(metrics_file)
            logger.info(f"Detailed metrics saved to: {metrics_file}")

def main_enhanced():
    """Main function for enhanced MAS-FRO system"""
    try:
        system = IntegratedMASFROSystem()
        system.run_enhanced_simulation(duration=3600)
        
    except KeyboardInterrupt:
        logger.info("Enhanced simulation interrupted by user")
    except Exception as e:
        logger.error(f"Enhanced simulation error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main_enhanced()