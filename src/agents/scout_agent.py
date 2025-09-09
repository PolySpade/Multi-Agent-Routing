# Copy the ScoutAgent class from the first artifact:
from .base_agent import BaseAgent
from ..data.data_structures import HazardData
import logging
import random
from datetime import datetime

logger = logging.getLogger(__name__)

class ScoutAgent(BaseAgent):
    """Agent for collecting crowdsourced data"""
    
    def run(self):
        """Simulate collecting crowdsourced reports"""
        while self.running:
            try:
                # Simulate crowdsourced reports
                reports = self._simulate_crowdsourced_data()
                
                # Send data to Hazard Agent
                message = {
                    'type': 'crowdsourced_data',
                    'data': reports,
                    'sender': self.agent_id,
                    'timestamp': self.env.now
                }
                if self.output_queue:
                    self.output_queue.put(message)
                
                logger.info(f"{self.agent_id}: Collected {len(reports)} crowdsourced reports")
                
                # Wait 2 minutes
                yield self.env.timeout(120)
                
            except Exception as e:
                logger.error(f"{self.agent_id} error: {e}")
                yield self.env.timeout(60)
    
    def _simulate_crowdsourced_data(self):
        """Simulate crowdsourced hazard reports"""
        # Simulate reports from different locations in Marikina
        locations = [
            (14.6507, 121.1029), (14.6350, 121.1120), 
            (14.6180, 121.1200), (14.6420, 121.1050)
        ]
        
        reports = []
        num_reports = random.randint(0, 5)  # 0-5 reports per cycle
        
        for _ in range(num_reports):
            location = random.choice(locations)
            
            # Simulate different types of reports
            report_types = [
                (0.3, "Minor flooding"),
                (0.7, "Road partially blocked"),
                (1.0, "Road completely flooded")
            ]
            
            risk_level, description = random.choice(report_types)
            confidence = random.uniform(0.6, 0.9)  # Crowdsourced data has lower confidence
            
            report = HazardData(
                location=location,
                risk_level=risk_level,
                data_type='crowdsourced',
                timestamp=datetime.now(),
                confidence=confidence
            )
            reports.append(report)
        
        return reports