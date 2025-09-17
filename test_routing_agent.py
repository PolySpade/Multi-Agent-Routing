#!/usr/bin/env python3
"""
Simple syntax and structure test for RoutingAgent
"""
import sys
import os
import ast

def test_syntax():
    """Test if routing_agent.py has valid Python syntax"""
    try:
        with open('src/agents/routing_agent.py', 'r') as f:
            source = f.read()

        # Parse the AST to check syntax
        ast.parse(source)
        print("✓ routing_agent.py syntax is valid")
        return True
    except SyntaxError as e:
        print(f"✗ Syntax error in routing_agent.py: {e}")
        return False
    except Exception as e:
        print(f"✗ Error reading file: {e}")
        return False

def test_structure():
    """Test if the expected methods and classes are present"""
    try:
        with open('src/agents/routing_agent.py', 'r') as f:
            source = f.read()

        tree = ast.parse(source)

        classes = []
        methods = []

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                classes.append(node.name)
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        methods.append(f"{node.name}.{item.name}")

        expected_classes = ['RoutingAgent']
        expected_methods = [
            'RoutingAgent.__init__',
            'RoutingAgent.run',
            'RoutingAgent._calculate_route',
            'RoutingAgent._get_risk_score',
            'RoutingAgent._find_nearest_node',
            'RoutingAgent._find_nearest_evacuation_center',
            'RoutingAgent._distance_heuristic'
        ]

        print("✓ Found classes:", classes)
        print("✓ Found methods:", [m for m in methods if 'RoutingAgent' in m])

        for cls in expected_classes:
            if cls not in classes:
                print(f"✗ Missing class: {cls}")
                return False

        for method in expected_methods:
            if method not in methods:
                print(f"✗ Missing method: {method}")
                return False

        print("✓ All expected classes and methods found")
        return True

    except Exception as e:
        print(f"✗ Error analyzing structure: {e}")
        return False

def test_imports():
    """Test if expected imports are present"""
    try:
        with open('src/agents/routing_agent.py', 'r') as f:
            source = f.read()

        expected_imports = [
            'BaseAgent',
            'RouteRequest',
            'networkx',
            'geopandas',
            'logging',
            'time',
            'Optional',
            'Dict',
            'Any',
            'math'
        ]

        found_imports = []
        tree = ast.parse(source)

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    found_imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    found_imports.append(alias.name)

        print("✓ Found imports:", found_imports)

        for imp in expected_imports:
            if imp not in found_imports:
                print(f"✗ Missing import: {imp}")
                return False

        print("✓ All expected imports found")
        return True

    except Exception as e:
        print(f"✗ Error analyzing imports: {e}")
        return False

if __name__ == "__main__":
    print("Testing RoutingAgent implementation...")
    print()

    tests = [
        ("Syntax validation", test_syntax),
        ("Structure validation", test_structure),
        ("Import validation", test_imports)
    ]

    results = []
    for name, test_func in tests:
        print(f"Running {name}...")
        result = test_func()
        results.append(result)
        print()

    if all(results):
        print("🎉 All tests passed! RoutingAgent implementation looks good.")
        sys.exit(0)
    else:
        print("❌ Some tests failed. Please review the implementation.")
        sys.exit(1)