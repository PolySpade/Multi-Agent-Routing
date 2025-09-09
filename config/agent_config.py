AGENT_CONFIG = {
    'flood_agent': {
        'data_collection_interval': 300,  # 5 minutes
        'max_retries': 3,
        'stations': [
            {"id": "MAR001", "location": (14.6507, 121.1029)},
            {"id": "MAR002", "location": (14.6350, 121.1120)},
            {"id": "MAR003", "location": (14.6180, 121.1200)},
        ]
    },
    'scout_agent': {
        'data_collection_interval': 120,  # 2 minutes
        'confidence_threshold': 0.6,
        'max_reports_per_cycle': 10
    },
    'hazard_agent': {
        'risk_update_interval': 30,  # 30 seconds
        'risk_decay_rate': 0.1,  # Risk decreases over time
        'confidence_weights': {
            'official': 1.0,
            'crowdsourced': 0.7,
            'user_feedback': 1.0
        }
    },
    'routing_agent': {
        'algorithm': 'astar',  # 'dijkstra' or 'astar'
        'weight_composition': {
            'distance': 0.8,
            'risk': 0.2
        },
        'max_computation_time': 5.0  # seconds
    }
}