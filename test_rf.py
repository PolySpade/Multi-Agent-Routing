#!/usr/bin/env python3
"""
Test RandomForestRegressor in isolation to identify the estimators_ issue
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

def test_random_forest():
    print("Testing RandomForestRegressor...")
    
    # Create model
    rf = RandomForestRegressor(n_estimators=10, max_depth=5, random_state=42)
    scaler = StandardScaler()
    
    # Generate sample data
    X = np.random.rand(20, 5)
    y = np.random.rand(20)
    
    print(f"Before fitting - has estimators_: {hasattr(rf, 'estimators_')}")
    
    # Fit scaler
    X_scaled = scaler.fit_transform(X)
    
    # Fit model
    print("Fitting model...")
    rf.fit(X_scaled, y)
    
    print(f"After fitting - has estimators_: {hasattr(rf, 'estimators_')}")
    
    if hasattr(rf, 'estimators_'):
        print(f"Number of estimators: {len(rf.estimators_)}")
    
    # Test prediction
    test_X = np.random.rand(1, 5)
    test_X_scaled = scaler.transform(test_X)
    pred = rf.predict(test_X_scaled)
    print(f"Test prediction: {pred[0]:.3f}")
    
    print("Random Forest test completed successfully!")

if __name__ == "__main__":
    test_random_forest()
