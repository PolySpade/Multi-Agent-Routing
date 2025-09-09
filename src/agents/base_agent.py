# Copy the BaseAgent class from the first artifact:
import simpy
from abc import ABC, abstractmethod

class BaseAgent(ABC):
    """Base class for all agents in the MAS-FRO system"""
    
    def __init__(self, agent_id: str, env: simpy.Environment, 
                 input_queue, output_queue):
        self.agent_id = agent_id
        self.env = env
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.running = True
        self.process = env.process(self.run())
    
    @abstractmethod
    def run(self):
        """Main agent logic - must be implemented by subclasses"""
        pass
    
    def stop(self):
        """Stop the agent"""
        self.running = False