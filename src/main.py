import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.simulation.mas_controller import MASFROController
from src.utils.logging_config import setup_logging
import logging

def main():
    """Main entry point for MAS-FRO simulation"""
    # Setup logging
    setup_logging(logging.INFO)
    logger = logging.getLogger(__name__)
    
    try:
        # Create and run simulation
        controller = MASFROController()
        
        # Run for 1 hour simulation time
        controller.run_simulation(duration=3600)
        
    except KeyboardInterrupt:
        logger.info("Simulation interrupted by user")
    except Exception as e:
        logger.error(f"Simulation error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        logger.info("Cleaning up...")

if __name__ == "__main__":
    main()