import random
from typing import List, Tuple
from datetime import datetime, timedelta
from ..data.data_structures import FloodData, HazardData

class FloodScenario:
    """Flood scenario configuration"""
    
    def __init__(self, scenario_name: str, intensity: float = 0.5, 
                 duration: int = 1800, affected_areas: List[Tuple[float, float]] = None):
        self.scenario_name = scenario_name
        self.intensity = intensity  # 0.0 to 1.0
        self.duration = duration  # seconds
        self.affected_areas = affected_areas or []
        self.start_time = None
    
    def generate_flood_data(self, current_time: datetime) -> List[FloodData]:
        """Generate flood data based on scenario parameters"""
        if not self.start_time:
            self.start_time = current_time
        
        elapsed = (current_time - self.start_time).total_seconds()
        
        # Scenario has ended
        if elapsed > self.duration:
            return []
        
        # Calculate intensity curve (peaks in middle of scenario)
        progress = elapsed / self.duration
        intensity_curve = 4 * progress * (1 - progress)  # Parabolic curve
        current_intensity = self.intensity * intensity_curve
        
        flood_data = []
        
        # Generate data for affected areas
        for i, location in enumerate(self.affected_areas):
            # Add some randomness
            water_level = current_intensity * 15 + random.uniform(-2, 2)
            rainfall = current_intensity * 50 + random.uniform(-5, 5)
            
            data = FloodData(
                station_id=f"SCENARIO_{self.scenario_name}_{i}",
                water_level=max(0, water_level),
                rainfall_intensity=max(0, rainfall),
                location=location,
                timestamp=current_time
            )
            flood_data.append(data)
        
        return flood_data
    
    def generate_crowdsourced_reports(self, current_time: datetime) -> List[HazardData]:
        """Generate crowdsourced reports based on scenario"""
        if not self.start_time:
            return []
        
        elapsed = (current_time - self.start_time).total_seconds()
        if elapsed > self.duration:
            return []
        
        reports = []
        
        # Generate reports with probability based on intensity
        for location in self.affected_areas:
            if random.random() < self.intensity * 0.3:  # 30% chance at full intensity
                risk_level = min(1.0, self.intensity + random.uniform(-0.2, 0.2))
                confidence = random.uniform(0.6, 0.9)
                
                report = HazardData(
                    location=location,
                    risk_level=risk_level,
                    data_type='crowdsourced',
                    timestamp=current_time,
                    confidence=confidence
                )
                reports.append(report)
        
        return reports

class ScenarioManager:
    """Manages different flood scenarios"""
    
    def __init__(self):
        self.scenarios = self._create_default_scenarios()
        self.current_scenario = None
    
    def _create_default_scenarios(self) -> dict:
        """Create default flood scenarios for Marikina"""
        scenarios = {
            'light_rain': FloodScenario(
                'light_rain',
                intensity=0.3,
                duration=1800,  # 30 minutes
                affected_areas=[(14.6507, 121.1029)]
            ),
            'moderate_flood': FloodScenario(
                'moderate_flood',
                intensity=0.6,
                duration=3600,  # 1 hour
                affected_areas=[
                    (14.6507, 121.1029),
                    (14.6350, 121.1120)
                ]
            ),
            'severe_typhoon': FloodScenario(
                'severe_typhoon',
                intensity=0.9,
                duration=7200,  # 2 hours
                affected_areas=[
                    (14.6507, 121.1029),
                    (14.6350, 121.1120),
                    (14.6180, 121.1200),
                    (14.6420, 121.1050)
                ]
            )
        }
        return scenarios
    
    def get_scenario(self, scenario_name: str) -> FloodScenario:
        """Get scenario by name"""
        return self.scenarios.get(scenario_name)
    
    def set_active_scenario(self, scenario_name: str):
        """Set the active scenario"""
        if scenario_name in self.scenarios:
            self.current_scenario = self.scenarios[scenario_name]
        else:
            raise ValueError(f"Unknown scenario: {scenario_name}")
    
    def get_current_data(self, current_time: datetime) -> dict:
        """Get current scenario data"""
        if not self.current_scenario:
            return {'flood_data': [], 'crowdsourced_data': []}
        
        return {
            'flood_data': self.current_scenario.generate_flood_data(current_time),
            'crowdsourced_data': self.current_scenario.generate_crowdsourced_reports(current_time)
        }