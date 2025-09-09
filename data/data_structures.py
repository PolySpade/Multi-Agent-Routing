# Copy from the first artifact - the dataclass definitions:
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from datetime import datetime

@dataclass
class RouteRequest:
    """Data structure for route requests"""
    request_id: str
    origin: Tuple[float, float]  # (lat, lon)
    destination: str  # evacuation center name or coordinates
    timestamp: datetime
    user_id: str

@dataclass
class HazardData:
    """Data structure for hazard information"""
    location: Tuple[float, float]
    risk_level: float  # 0.0 (safe) to 1.0 (high risk), inf for impassable
    data_type: str  # 'official' or 'crowdsourced'
    timestamp: datetime
    confidence: float = 1.0

@dataclass
class FloodData:
    """Data structure for official flood data"""
    station_id: str
    water_level: float
    rainfall_intensity: float
    location: Tuple[float, float]
    timestamp: datetime