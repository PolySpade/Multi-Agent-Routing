import unittest
import sys
import os
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.simulation.mas_controller import MASFROController
from src.simulation.scenarios import FloodScenario, ScenarioManager
from src.data.data_structures import RouteRequest

class TestMASFROController(unittest.TestCase):
    def setUp(self):
        self.controller = MASFROController()
    
    def test_controller_creation(self):
        """Test that controller creates all necessary components"""
        self.assertIsNotNone(self.controller.env)
        self.assertIsNotNone(self.controller.graph_env)
        self.assertIsNotNone(self.controller.agents)
        
        # Check that all agents were created
        expected_agents = ['flood', 'scout', 'hazard', 'routing', 'evacuation']
        for agent_name in expected_agents:
            self.assertIn(agent_name, self.controller.agents)
    
    def test_user_request_simulation(self):
        """Test user request simulation"""
        initial_queue_size = self.controller.user_to_evacuation.qsize()
        self.controller.simulate_user_request()
        
        # Queue should have one more item
        self.assertEqual(self.controller.user_to_evacuation.qsize(), initial_queue_size + 1)
    
    def test_short_simulation_run(self):
        """Test running a short simulation"""
        # Run simulation for 60 simulation seconds
        try:
            self.controller.run_simulation(duration=60)
            # If we get here, simulation ran without errors
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"Simulation failed with error: {e}")

class TestFloodScenarios(unittest.TestCase):
    def test_scenario_creation(self):
        """Test flood scenario creation"""
        scenario = FloodScenario(
            'test_scenario',
            intensity=0.7,
            duration=1800,
            affected_areas=[(14.65, 121.10)]
        )
        
        self.assertEqual(scenario.scenario_name, 'test_scenario')
        self.assertEqual(scenario.intensity, 0.7)
        self.assertEqual(scenario.duration, 1800)
    
    def test_scenario_data_generation(self):
        """Test scenario data generation"""
        scenario = FloodScenario(
            'test_scenario',
            intensity=0.8,
            duration=3600,
            affected_areas=[(14.65, 121.10), (14.66, 121.11)]
        )
        
        current_time = datetime.now()
        
        # Generate flood data
        flood_data = scenario.generate_flood_data(current_time)
        self.assertIsInstance(flood_data, list)
        
        # Should generate data for each affected area
        self.assertLessEqual(len(flood_data), len(scenario.affected_areas))
        
        # Generate crowdsourced data
        reports = scenario.generate_crowdsourced_reports(current_time)
        self.assertIsInstance(reports, list)

class TestScenarioManager(unittest.TestCase):
    def setUp(self):
        self.manager = ScenarioManager()
    
    def test_default_scenarios(self):
        """Test that default scenarios are created"""
        scenarios = self.manager.scenarios
        
        expected_scenarios = ['light_rain', 'moderate_flood', 'severe_typhoon']
        for scenario_name in expected_scenarios:
            self.assertIn(scenario_name, scenarios)
    
    def test_scenario_selection(self):
        """Test scenario selection"""
        self.manager.set_active_scenario('moderate_flood')
        self.assertIsNotNone(self.manager.current_scenario)
        self.assertEqual(self.manager.current_scenario.scenario_name, 'moderate_flood')
    
    def test_invalid_scenario(self):
        """Test handling of invalid scenario names"""
        with self.assertRaises(ValueError):
            self.manager.set_active_scenario('nonexistent_scenario')

if __name__ == '__main__':
    unittest.main()