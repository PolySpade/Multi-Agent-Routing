from .base_agent import BaseAgent
from ..data.data_structures import FloodData
from ..utils.weather_client import WeatherClient  # <-- IMPORT our new client
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class FloodAgent(BaseAgent):
    """Agent responsible for collecting real-time flood data using a weather client."""

    def __init__(self, agent_id, env, input_queue, output_queue):
        # We now accept input_queue and pass it to the parent class
        super().__init__(agent_id, env, input_queue, output_queue)
        try:
            self.weather_client = WeatherClient()
        except ValueError as e:
            logger.error(f"Could not initialize FloodAgent: {e}")
            self.weather_client = None # Agent will be disabled

    def run(self):
        """Fetches real-time weather data every 5 minutes."""
        if not self.weather_client:
            logger.warning(f"{self.agent_id} is disabled because the WeatherClient could not be initialized.")
            return # Stop the run method if client is not available

        while self.running:
            try:
                # The agent's logic is now much cleaner
                flood_data = self._fetch_and_process_data()

                if flood_data:
                    message = {
                        'type': 'flood_data',
                        'data': flood_data,
                        'sender': self.agent_id,
                        'timestamp': self.env.now
                    }
                    if self.output_queue:
                        self.output_queue.put(message)
                    logger.info(f"{self.agent_id}: Collected and sent real-time weather data.")
                else:
                    logger.info(f"{self.agent_id}: No new weather data fetched.")

                yield self.env.timeout(300)

            except Exception as e:
                logger.error(f"{self.agent_id} experienced an error: {e}")
                yield self.env.timeout(60)

    def _fetch_and_process_data(self):
        """
        Uses the WeatherClient to get data and maps it to the FloodData structure.
        """
        # Define the location to monitor
        lat, lon = 14.6507, 121.1029
        
        current_weather = self.weather_client.get_weather(lat, lon)
        
        if not current_weather:
            return []

        # Safely extract rainfall, defaulting to 0.0 if not present
        rainfall = current_weather.get('rain', {}).get('1h', 0.0)
        
        # Map to our internal data structure
        flood_instance = FloodData(
            station_id="MARIKINA_CENTER",
            water_level=-1.0,  # Placeholder, as this doesn't come from the weather API
            rainfall_intensity=float(rainfall),
            location=(lat, lon),
            timestamp=datetime.fromtimestamp(current_weather.get('dt', datetime.now().timestamp()))
        )
        
        return [flood_instance]