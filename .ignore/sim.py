import simpy

env = simpy.Environment()

class FloodAgent:
    def __init__(self, env):
        self.env = env

    def run(self):
        while True:
            # Simulate fetching new flood data every 5 minutes
            yield self.env.timeout(300) 
            print(f"Time {self.env.now}: FloodAgent fetching new data.")
            # Add logic to generate and pass data to the Hazard Agent
    


flood_agent = FloodAgent(env)

# Start the agent processes
env.process(flood_agent.run())

env.run(until=10000)
