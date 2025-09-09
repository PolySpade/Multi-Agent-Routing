# Copy the FloodAgent class from the first artifact:
from .base_agent import BaseAgent
from ..data.data_structures import FloodData
import logging
import random
from datetime import datetime

logger = logging.getLogger(__name__)

class FloodAgent(BaseAgent):
    """Agent responsible for collecting official flood data"""
    
    def run(self):
        """Simulate collecting flood data every 5 minutes"""
        while self.running:
            try:
                # Simulate data collection from PAGASA sensors
                flood_data = self._simulate_flood_data()
                
                # Send data to Hazard Agent
                message = {
                    'type': 'flood_data',
                    'data': flood_data,
                    'sender': self.agent_id,
                    'timestamp': self.env.now
                }
                if self.output_queue:
                    self.output_queue.put(message)
                
                logger.info(f"{self.agent_id}: Collected flood data from {len(flood_data)} stations")
                
                # Wait 5 minutes (300 seconds in simulation)
                yield self.env.timeout(300)
                
            except Exception as e:
                logger.error(f"{self.agent_id} error: {e}")
                yield self.env.timeout(60)  # Wait 1 minute before retry
    
    def _simulate_flood_data(self):
        """Simulate flood data from various monitoring stations"""
        stations = [
            {"id": "MAR001", "location": (14.6507, 121.1029)},  # Marikina River
            {"id": "MAR002", "location": (14.6350, 121.1120)},
            {"id": "MAR003", "location": (14.6180, 121.1200)},
        ]
        
        flood_data = []
        for station in stations:
            # Simulate varying flood conditions
            base_level = random.uniform(2.0, 15.0)  # meters
            rainfall = random.uniform(0, 50)  # mm/hr
            
            data = FloodData(
                station_id=station["id"],
                water_level=base_level,
                rainfall_intensity=rainfall,
                location=station["location"],
                timestamp=datetime.now()
            )
            flood_data.append(data)
        
        return flood_data