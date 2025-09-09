import unittest
import simpy
from multiprocessing import Queue
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.agents.flood_agent import FloodAgent
from src.agents.scout_agent import ScoutAgent
from src.agents.hazard_agent import HazardAgent
from src.environment.dynamic_graph import DynamicGraphEnvironment
from src.data.data_structures import RouteRequest
from datetime import datetime

class TestFloodAgent(unittest.TestCase):
    def setUp(self):
        self.env = simpy.Environment()
        self.output_queue = Queue()
    
    def test_flood_agent_creation(self):
        """Test FloodAgent can be created"""
        agent = FloodAgent('test_flood', self.env, None, self.output_queue)
        self.assertEqual(agent.agent_id, 'test_flood')
        self.assertIsNotNone(agent.process)
    
    def test_flood_data_simulation(self):
        """Test flood data simulation"""
        agent = FloodAgent('test_flood', self.env, None, self.output_queue)
        data = agent._simulate_flood_data()
        
        self.assertIsInstance(data, list)
        self.assertGreater(len(data), 0)
        
        # Check data structure
        for item in data:
            self.assertIsNotNone(item.station_id)
            self.assertIsNotNone(item.location)
            self.assertGreaterEqual(item.water_level, 0)
            self.assertGreaterEqual(item.rainfall_intensity, 0)

class TestScoutAgent(unittest.TestCase):
    def setUp(self):
        self.env = simpy.Environment()
        self.output_queue = Queue()
    
    def test_scout_agent_creation(self):
        """Test ScoutAgent can be created"""
        agent = ScoutAgent('test_scout', self.env, None, self.output_queue)
        self.assertEqual(agent.agent_id, 'test_scout')
    
    def test_crowdsourced_data_simulation(self):
        """Test crowdsourced data simulation"""
        agent = ScoutAgent('test_scout', self.env, None, self.output_queue)
        reports = agent._simulate_crowdsourced_data()
        
        self.assertIsInstance(reports, list)
        
        # Check report structure if any reports generated
        for report in reports:
            self.assertIsNotNone(report.location)
            self.assertGreaterEqual(report.risk_level, 0)
            self.assertLessEqual(report.risk_level, 1)
            self.assertEqual(report.data_type, 'crowdsourced')

class TestHazardAgent(unittest.TestCase):
    def setUp(self):
        self.env = simpy.Environment()
        self.input_queue = Queue()
        self.graph_env = DynamicGraphEnvironment()
    
    def test_hazard_agent_creation(self):
        """Test HazardAgent can be created"""
        agent = HazardAgent('test_hazard', self.env, self.input_queue, None, self.graph_env)
        self.assertEqual(agent.agent_id, 'test_hazard')
        self.assertIsNotNone(agent.graph_env)
    
    def test_risk_calculation(self):
        """Test risk calculation methods"""
        agent = HazardAgent('test_hazard', self.env, self.input_queue, None, self.graph_env)
        
        # Create mock flood data
        from src.data.data_structures import FloodData
        
        test_data = FloodData(
            station_id='TEST01',
            water_level=5.0,
            rainfall_intensity=20.0,
            location=(14.65, 121.10),
            timestamp=datetime.now()
        )
        
        risk = agent._calculate_flood_risk(test_data)
        self.assertIsInstance(risk, float)
        self.assertGreaterEqual(risk, 0)

class TestIntegration(unittest.TestCase):
    def test_message_flow(self):
        """Test basic message flow between agents"""
        env = simpy.Environment()
        
        # Create queues
        flood_to_hazard = Queue()
        graph_env = DynamicGraphEnvironment()
        
        # Create agents
        flood_agent = FloodAgent('flood_test', env, None, flood_to_hazard)
        hazard_agent = HazardAgent('hazard_test', env, flood_to_hazard, None, graph_env)
        
        # Run short simulation
        env.run(until=10)
        
        # Check that agents were created successfully
        self.assertIsNotNone(flood_agent)
        self.assertIsNotNone(hazard_agent)

if __name__ == '__main__':
    unittest.main()