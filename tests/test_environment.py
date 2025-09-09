import unittest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.environment.dynamic_graph import DynamicGraphEnvironment

class TestDynamicGraphEnvironment(unittest.TestCase):
    def setUp(self):
        # Use a simpler test location to avoid network issues
        self.env = DynamicGraphEnvironment("Test City")
    
    def test_environment_creation(self):
        """Test that environment can be created"""
        self.assertIsNotNone(self.env.graph)
        self.assertIsInstance(self.env._risk_scores, dict)
    
    def test_risk_score_update(self):
        """Test updating risk scores"""
        # Add test edges to graph
        self.env.graph.add_edge(1, 2, length=100, key=0)
        
        # Update risk score
        self.env.update_edge_risk(1, 2, 0, 0.5)
        
        # Check that risk score was updated
        edge_id = "1_2_0"
        self.assertEqual(self.env._risk_scores[edge_id], 0.5)
        
        # Check that graph weight was updated
        weight = self.env.graph[1][2][0].get('risk_aware_weight')
        self.assertIsNotNone(weight)
    
    def test_impassable_roads(self):
        """Test handling of impassable roads"""
        self.env.graph.add_edge(3, 4, length=200, key=0)
        
        # Mark road as impassable
        self.env.update_edge_risk(3, 4, 0, float('inf'))
        
        # Check that weight is infinite
        weight = self.env.graph[3][4][0].get('risk_aware_weight')
        self.assertEqual(weight, float('inf'))
    
    def test_get_current_state(self):
        """Test getting current network state"""
        current_state = self.env.get_current_state()
        self.assertIsNotNone(current_state)
        
        # Should be a copy, not the same object
        self.assertIsNot(current_state, self.env.graph)

if __name__ == '__main__':
    unittest.main()