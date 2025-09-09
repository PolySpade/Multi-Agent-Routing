"""
Simulation data generators and helpers
"""

import random
import pandas as pd
from datetime import datetime, timedelta
from typing import List
from .data_structures import FloodData, HazardData

class SimulationDataGenerator:
    """Generate realistic simulation data for testing"""
    
    def __init__(self):
        self.marikina_locations = [
            (14.6507, 121.1029, "Marikina River Bridge"),
            (14.6350, 121.1120, "Malanday Area"),
            (14.6180, 121.1200, "Tumana Bridge"),
            (14.6420, 121.1050, "Sta Elena"),
            (14.6580, 121.1100, "Riverbanks"),
            (14.6250, 121.1150, "Fortune"),
            (14.6300, 121.1080, "Concepcion"),
            (14.6450, 121.1020, "Nangka")
        ]
    
    def generate_flood_stations(self) -> List[dict]:
        """Generate realistic flood monitoring stations"""
        stations = []
        for i, (lat, lon, name) in enumerate(self.marikina_locations[:5]):
            station = {
                'id': f'MAR{i+1:03d}',
                'name': f'{name} Station',
                'location': (lat, lon),
                'elevation': random.uniform(10, 50),  # meters above sea level
                'basin_area': random.uniform(5, 20)   # sq km drainage area
            }
            stations.append(station)
        return stations
    
    def generate_historical_flood_data(self, days: int = 30) -> pd.DataFrame:
        """Generate historical flood data for analysis"""
        data = []
        base_date = datetime.now() - timedelta(days=days)
        
        stations = self.generate_flood_stations()
        
        for day in range(days):
            current_date = base_date + timedelta(days=day)
            
            # Simulate weather patterns
            if random.random() < 0.3:  # 30% chance of rain
                rainfall_intensity = random.uniform(5, 40)
                is_rainy_day = True
            else:
                rainfall_intensity = random.uniform(0, 2)
                is_rainy_day = False
            
            for station in stations:
                # Base water level varies by station
                base_level = random.uniform(2, 6)
                
                # Water level increases with rainfall
                if is_rainy_day:
                    water_level = base_level + (rainfall_intensity * random.uniform(0.2, 0.8))
                else:
                    water_level = base_level + random.uniform(-1, 1)
                
                data.append({
                    'date': current_date,
                    'station_id': station['id'],
                    'station_name': station['name'],
                    'latitude': station['location'][0],
                    'longitude': station['location'][1],
                    'water_level': max(0, water_level),
                    'rainfall_intensity': max(0, rainfall_intensity),
                    'is_flood_event': water_level > 8.0
                })
        
        return pd.DataFrame(data)
    
    def generate_crowdsourced_reports(self, num_reports: int = 50) -> List[HazardData]:
        """Generate realistic crowdsourced reports"""
        reports = []
        
        report_templates = [
            ("flooding", 0.8, "knee-deep water on main road"),
            ("blocked", 1.0, "road completely impassable"),
            ("minor_flood", 0.4, "water on road but passable"),
            ("debris", 0.6, "fallen tree blocking lane"),
            ("clear", 0.0, "road clear and safe"),
            ("traffic", 0.3, "heavy traffic due to detours")
        ]
        
        for i in range(num_reports):
            location = random.choice(self.marikina_locations)
            report_type, base_risk, description = random.choice(report_templates)
            
            # Add some randomness to risk level
            risk_level = max(0.0, min(1.0, base_risk + random.uniform(-0.2, 0.2)))
            
            # Confidence varies by report type
            if report_type in ["clear", "blocked"]:
                confidence = random.uniform(0.8, 1.0)  # High confidence for clear statements
            else:
                confidence = random.uniform(0.6, 0.9)  # Lower confidence for subjective reports
            
            report = HazardData(
                location=(location[0], location[1]),
                risk_level=risk_level,
                data_type='crowdsourced',
                timestamp=datetime.now() - timedelta(minutes=random.randint(0, 120)),
                confidence=confidence
            )
            reports.append(report)
        
        return reports

def load_evacuation_centers() -> pd.DataFrame:
    """Load evacuation centers from CSV file"""
    try:
        return pd.read_csv('data/evacuation_centers.csv')
    except FileNotFoundError:
        # Return default data if file doesn't exist
        default_centers = [
            {'name': 'Marikina Sports Center', 'latitude': 14.6507, 'longitude': 121.1029, 'capacity': 5000, 'type': 'sports_facility'},
            {'name': 'Marikina Elementary School', 'latitude': 14.6350, 'longitude': 121.1120, 'capacity': 2000, 'type': 'school'},
            {'name': 'Barangay Hall', 'latitude': 14.6180, 'longitude': 121.1200, 'capacity': 500, 'type': 'community_center'}
        ]
        return pd.DataFrame(default_centers)