#!/usr/bin/env python3
"""
Simple launcher script for MAS-FRO HTML View
This makes it easier to start the web interface with different options
"""

import sys
import os

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def main():
    """Main launcher function"""
    
    print("🚀 MAS-FRO Web Interface Launcher")
    print("=" * 40)
    
    # Check command line arguments
    if len(sys.argv) > 1:
        if '--help' in sys.argv or '-h' in sys.argv:
            print("Available options:")
            print("  --eval          Run evaluation mode")
            print("  --no-web        Run simulation without web interface")
            print("  --port=5000     Set web server port (default: 5000)")
            print("  --duration=3600 Set simulation duration in seconds (default: 3600)")
            return
    
    # Parse arguments
    port = 5000
    duration = 3600
    no_web = False
    eval_mode = False
    
    for arg in sys.argv[1:]:
        if arg.startswith('--port='):
            port = int(arg.split('=')[1])
        elif arg.startswith('--duration='):
            duration = int(arg.split('=')[1])
        elif arg == '--no-web':
            no_web = True
        elif arg == '--eval':
            eval_mode = True
    
    try:
        if eval_mode:
            print("🧪 Starting evaluation mode...")
            from html_view import main_evaluation
            main_evaluation()
        else:
            from html_view import WebEnabledMASFRO
            
            print("🌐 Creating web-enabled system...")
            system = WebEnabledMASFRO()
            
            if no_web:
                print(f"🏃 Running simulation without web interface for {duration} seconds...")
                system.run_simulation_only(duration=duration)
            else:
                print(f"🌐 Starting web interface on port {port}...")
                print(f"🏃 Running simulation for {duration} seconds...")
                print(f"📊 Dashboard will be available at: http://localhost:{port}")
                system.run_with_web_interface(
                    duration=duration, 
                    web_port=port
                )
    
    except KeyboardInterrupt:
        print("\n⏹️  Shutting down...")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
