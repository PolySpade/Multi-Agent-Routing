# Copy the HazardAgent class from the first artifact:
from .base_agent import BaseAgent
import logging
import random

logger = logging.getLogger(__name__)

class HazardAgent(BaseAgent):
    """Central agent for data fusion and risk assessment"""
    
    def __init__(self, agent_id: str, env, input_queue, output_queue, graph_env):
        super().__init__(agent_id, env, input_queue, output_queue)
        self.graph_env = graph_env
        self.flood_data_buffer = []
        self.crowdsourced_buffer = []
    
    def run(self):
        """Process incoming data and update risk assessments"""
        while self.running:
            try:
                # Check for incoming messages
                if self.input_queue and not self.input_queue.empty():
                    message = self.input_queue.get()
                    self._process_message(message)
                
                # Update risk assessments
                if self.flood_data_buffer or self.crowdsourced_buffer:
                    self._update_risk_assessments()
                
                yield self.env.timeout(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"{self.agent_id} error: {e}")
                yield self.env.timeout(60)
    
    def _process_message(self, message: dict):
        """Process incoming data messages"""
        msg_type = message.get('type')
        
        if msg_type == 'flood_data':
            self.flood_data_buffer.extend(message['data'])
        elif msg_type == 'crowdsourced_data':
            self.crowdsourced_buffer.extend(message['data'])
        elif msg_type == 'user_feedback':
            # Process user feedback with high confidence
            feedback = message['data']
            self._apply_user_feedback(feedback)
    
    def _update_risk_assessments(self):
        """Update risk scores for road segments"""
        logger.info(f"{self.agent_id}: Updating risk assessments")
        
        # Process flood data
        for flood_data in self.flood_data_buffer:
            risk_score = self._calculate_flood_risk(flood_data)
            affected_edges = self._find_nearby_edges(flood_data.location, radius=500)
            
            for u, v, key in affected_edges:
                self.graph_env.update_edge_risk(u, v, key, risk_score)
        
        # Process crowdsourced data
        for report in self.crowdsourced_buffer:
            # Weight by confidence
            weighted_risk = report.risk_level * report.confidence
            affected_edges = self._find_nearby_edges(report.location, radius=200)
            
            for u, v, key in affected_edges:
                current_risk = self.graph_env._risk_scores.get(f"{u}_{v}_{key}", 0.0)
                # Take maximum of current and new risk
                new_risk = max(current_risk, weighted_risk)
                self.graph_env.update_edge_risk(u, v, key, new_risk)
        
        # Clear buffers
        self.flood_data_buffer.clear()
        self.crowdsourced_buffer.clear()
    
    def _calculate_flood_risk(self, flood_data) -> float:
        """Calculate risk score based on flood data"""
        # Simple risk calculation based on water level and rainfall
        water_risk = min(flood_data.water_level / 10.0, 1.0)  # Normalize to 0-1
        rainfall_risk = min(flood_data.rainfall_intensity / 30.0, 1.0)  # Normalize to 0-1
        
        combined_risk = 0.7 * water_risk + 0.3 * rainfall_risk
        
        # Mark as impassable if extremely high
        if flood_data.water_level > 12.0 or flood_data.rainfall_intensity > 40.0:
            return float('inf')
        
        return combined_risk
    
    def _find_nearby_edges(self, location, radius: int = 500):
        """Find edges near a given location"""
        # Simplified: return random sample of edges for demonstration
        # In real implementation, use spatial indexing
        edges = list(self.graph_env.graph.edges(keys=True))
        return random.sample(edges, min(3, len(edges)))
    
    def _apply_user_feedback(self, feedback: dict):
        """Apply high-confidence user feedback"""
        # User feedback has high confidence and immediate impact
        location = feedback.get('location')
        condition = feedback.get('condition')  # 'clear', 'blocked', 'flooded'
        
        if condition == 'blocked' or condition == 'flooded':
            risk_score = float('inf')
        else:
            risk_score = 0.0
        
        affected_edges = self._find_nearby_edges(location, radius=100)
        for u, v, key in affected_edges:
            self.graph_env.update_edge_risk(u, v, key, risk_score)