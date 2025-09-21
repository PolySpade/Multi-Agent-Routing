import requests
import os
import logging
from datetime import datetime
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

class WeatherClient:
    """
    A client for fetching weather data from the OpenWeatherMap One Call API 3.0.
    """
    BASE_URL = "https://api.openweathermap.org/data/3.0/onecall"

    def __init__(self):
        self.api_key = os.environ.get("OPENWEATHER_API_KEY")
        if not self.api_key:
            logger.error("OPENWEATHER_API_KEY is not set. The WeatherClient will not be able to fetch data.")
            raise ValueError("API key for OpenWeather is not configured.")

    def get_weather(self, lat: float, lon: float) -> Optional[Dict[str, Any]]:
        """
        Fetches the current weather data for a specific latitude and longitude.
        """
        params = {
            "lat": lat,
            "lon": lon,
            "appid": self.api_key,
            "units": "metric",
            "exclude": "minutely,hourly,daily,alerts"
        }

        try:
            response = requests.get(self.BASE_URL, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            return data.get('current')

        except requests.exceptions.RequestException as e:
            logger.error(f"API request to OpenWeather failed: {e}")
            return None
        except KeyError as e:
            logger.error(f"Failed to parse OpenWeather API response. Unexpected structure: {e}")
            return None