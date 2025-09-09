#!/usr/bin/env python3
"""
Quick test script to verify MAS-FRO installation and basic functionality
"""

import sys
import os
import traceback

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test that all critical modules can be imported"""
    print("🔍 Testing imports...")
    
    try:
        from src.agents.base_agent import BaseAgent
        print("  ✅ BaseAgent imported successfully")
        
        from src.environment.dynamic_graph import DynamicGraphEnvironment
        print("  ✅ DynamicGraphEnvironment imported successfully")
        
        from src.simulation.mas_controller import MASFROController
        print("  ✅ MASFROController imported successfully")
        
        from src.data.data_structures import RouteRequest, FloodData, HazardData
        print("  ✅ Data structures imported successfully")
        
        from src.utils.logging_config import setup_logging
        print("  ✅ Logging utilities imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"  ❌ Import error: {e}")
        traceback.print_exc()
        return False

def test_basic_functionality():
    """Test basic system functionality"""
    print("\n🧪 Testing basic functionality...")
    
    try:
        # Test data structures
        from src.data.data_structures import RouteRequest
        from datetime import datetime
        
        request = RouteRequest(
            request_id="test_001",
            origin=(14.6507, 121.1029),
            destination="test_destination",
            timestamp=datetime.now(),
            user_id="test_user"
        )
        print("  ✅ RouteRequest creation successful")
        
        # Test graph environment (with fallback)
        from src.environment.dynamic_graph import DynamicGraphEnvironment
        
        # This might fail if no internet, but that's OK
        try:
            graph_env = DynamicGraphEnvironment("Marikina, Philippines")
            print("  ✅ DynamicGraphEnvironment with real network successful")
        except:
            print("  ⚠️  Real network failed, but that's expected without internet")
            graph_env = DynamicGraphEnvironment("Test City")  # This will create simple test graph
            print("  ✅ DynamicGraphEnvironment with test network successful")
        
        # Test risk scoring
        graph_env.update_edge_risk(1, 2, 0, 0.5)
        print("  ✅ Risk scoring successful")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Functionality test error: {e}")
        traceback.print_exc()
        return False

def test_simulation_setup():
    """Test that simulation can be set up"""
    print("\n🎮 Testing simulation setup...")
    
    try:
        from src.simulation.mas_controller import MASFROController
        
        controller = MASFROController()
        print("  ✅ MASFROController creation successful")
        
        # Check that all agents were created
        expected_agents = ['flood', 'scout', 'hazard', 'routing', 'evacuation']
        for agent_name in expected_agents:
            if agent_name in controller.agents:
                print(f"  ✅ {agent_name.capitalize()}Agent created successfully")
            else:
                print(f"  ❌ {agent_name.capitalize()}Agent missing")
                return False
        
        return True
        
    except Exception as e:
        print(f"  ❌ Simulation setup error: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🚀 MAS-FRO System Verification")
    print("="*50)
    
    # Test imports
    imports_ok = test_imports()
    
    # Test basic functionality
    func_ok = test_basic_functionality()
    
    # Test simulation setup
    sim_ok = test_simulation_setup()
    
    print("\n" + "="*50)
    print("📋 Test Summary:")
    print(f"  Imports: {'✅ PASS' if imports_ok else '❌ FAIL'}")
    print(f"  Basic Functionality: {'✅ PASS' if func_ok else '❌ FAIL'}")
    print(f"  Simulation Setup: {'✅ PASS' if sim_ok else '❌ FAIL'}")
    
    if imports_ok and func_ok and sim_ok:
        print("\n🎉 All tests passed! Your MAS-FRO system is ready!")
        print("\nNext steps:")
        print("  1. Install any missing dependencies: uv sync")
        print("  2. Run basic simulation: python main.py")
        print("  3. Start web interface: python web_interface.py")
        print("  4. Run tests: python -m pytest tests/")
        return True
    else:
        print("\n⚠️  Some tests failed. Check the error messages above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
