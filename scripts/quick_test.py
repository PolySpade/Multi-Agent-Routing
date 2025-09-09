#!/usr/bin/env python3
"""
Quick test script to verify installation
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_imports():
    """Test that all modules can be imported"""
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
        
        return True
        
    except ImportError as e:
        print(f"  ❌ Import error: {e}")
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
        
        # Test graph environment
        from src.environment.dynamic_graph import DynamicGraphEnvironment
        
        # Use simple test to avoid network issues
        graph_env = DynamicGraphEnvironment("Test City")
        print("  ✅ DynamicGraphEnvironment creation successful")
        
        # Test agent creation
        import simpy
        from multiprocessing import Queue
        from src.agents.flood_agent import FloodAgent
        
        env = simpy.Environment()
        queue = Queue()
        agent = FloodAgent('test_flood', env, None, queue)
        print("  ✅ FloodAgent creation successful")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Functionality test error: {e}")
        return False

def test_dependencies():
    """Test that required dependencies are available"""
    print("\n📦 Testing dependencies...")
    
    dependencies = [
        ('simpy', 'SimPy simulation framework'),
        ('networkx', 'NetworkX graph library'),
        ('pandas', 'Pandas data analysis'),
        ('numpy', 'NumPy numerical computing'),
    ]
    
    all_good = True
    
    for module, description in dependencies:
        try:
            __import__(module)
            print(f"  ✅ {module} - {description}")
        except ImportError:
            print(f"  ❌ {module} - {description} (MISSING)")
            all_good = False
    
    # Test optional dependencies
    optional_deps = [
        ('osmnx', 'OSMnx for road networks'),
        ('geopandas', 'GeoPandas for geospatial data'),
        ('sklearn', 'Scikit-learn for ML'),
        ('flask', 'Flask for web interface')
    ]
    
    print("\n🔧 Optional dependencies:")
    for module, description in optional_deps:
        try:
            __import__(module)
            print(f"  ✅ {module} - {description}")
        except ImportError:
            print(f"  ⚠️  {module} - {description} (optional, but recommended)")
    
    return all_good

def main():
    """Run all tests"""
    print("🚀 MAS-FRO Quick Test")
    print("="*50)
    
    # Test imports
    imports_ok = test_imports()
    
    # Test dependencies
    deps_ok = test_dependencies()
    
    # Test basic functionality
    func_ok = test_basic_functionality()
    
    print("\n" + "="*50)
    print("📋 Test Summary:")
    print(f"  Imports: {'✅ PASS' if imports_ok else '❌ FAIL'}")
    print(f"  Dependencies: {'✅ PASS' if deps_ok else '❌ FAIL'}")
    print(f"  Functionality: {'✅ PASS' if func_ok else '❌ FAIL'}")
    
    if imports_ok and func_ok:
        print("\n🎉 All tests passed! System is ready to use.")
        print("\nNext steps:")
        print("  1. Run basic simulation: python main.py")
        print("  2. Start web interface: python web_interface.py")
        print("  3. Run evaluation: python scripts/run_evaluation.py")
    else:
        print("\n⚠️  Some tests failed. Please check the installation.")
        if not deps_ok:
            print("  - Install missing dependencies: pip install -r requirements.txt")
        if not imports_ok or not func_ok:
            print("  - Check Python path and file structure")

if __name__ == "__main__":
    main()