import simpy

class BaseAgent:
    """
    A base class for all agents in the MAS-FRO simulation.

    This class provides a common structure, ensuring that every agent
    is initialized with a reference to the simulation environment and
    has a run method to define its core logic.
    """
    def __init__(self, env: simpy.Environment, agent_id: str):
        """
        Initializes the BaseAgent.

        Args:
            env: The SimPy simulation environment.
            agent_id: A unique identifier for the agent (e.g., 'FloodAgent-1').
        """
        self.env = env
        self.agent_id = agent_id
        self.process = self.env.process(self.run())

    def run(self):
        """
        The main logic loop for the agent.

        This method must be overridden by all subclasses. It should be a
        generator function that yields SimPy events.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement the run() method."
        )
        # The 'yield' statement is included to ensure the method is a generator,
        # even though it's unreachable.
        yield self.env.timeout(0)