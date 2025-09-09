#!/usr/bin/env python3
"""
Test script for html_view.py
This script tests the web interface components
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test that all imports work"""
    print("🧪 Testing imports...")
    
    try:
        from html_view import MASFROWebInterface, WebEnabledMASFRO, MASFROEvaluator
        print("✅ All main classes imported successfully")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_web_interface_creation():
    """Test creating web interface"""
    print("\n🧪 Testing web interface creation...")
    
    try:
        from html_view import WebEnabledMASFRO
        
        print("Creating WebEnabledMASFRO system...")
        system = WebEnabledMASFRO()
        print("✅ WebEnabledMASFRO created successfully")
        
        print("Web interface available at:", system.web_interface)
        return True
        
    except Exception as e:
        print(f"❌ Web interface creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_evaluation_system():
    """Test evaluation system"""
    print("\n🧪 Testing evaluation system...")
    
    try:
        from html_view import IntegratedMASFROSystem, MASFROEvaluator
        
        print("Creating system...")
        system = IntegratedMASFROSystem()
        print("✅ System created successfully")
        
        print("Creating evaluator...")
        evaluator = MASFROEvaluator(system)
        print("✅ Evaluator created successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Evaluation system test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🚀 Testing HTML View Components")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_web_interface_creation,
        test_evaluation_system
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! HTML view is ready to use.")
        print("\nYou can now:")
        print("1. Run with web interface: uv run python html_view.py")
        print("2. Run evaluation mode: uv run python html_view.py --eval")
        print("3. Run without web interface: uv run python html_view.py --no-web")
    else:
        print("⚠️  Some tests failed. Check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
