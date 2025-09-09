from .base_agent import BaseAgent

class FloodAgent(BaseAgent):
    """
    The FloodAgent is responsible for sourcing official environmental data.
    """
    def run(self):

        print(f"{self.agent_id} starting at time {self.env.now}")
        while True:
            # Simulate fetching new flood data every 5 minutes (300 seconds)
            yield self.env.timeout(300)
            print(f"Time {self.env.now}: {self.agent_id} fetching new data.")
            # In a full implementation, you would add the data to a shared
            # resource (e.g., a simpy.Store) here.