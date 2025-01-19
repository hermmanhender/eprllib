"""
Agent Spec
===========

This module implement the base for an agent specification to safe configuration of the object.
"""

from eprllib.ActionFunctions.ActionFunctions import ActionSpec
from eprllib.ObservationFunctions.ObservationFunctions import ObservationSpec
from eprllib.RewardFunctions.RewardFunctions import RewardSpec

class AgentSpec:
    """
    AgentSpec is the base class for an agent specification to safe configuration of the object.
    """
    def __init__(
        self,
        observation: ObservationSpec = NotImplemented,
        action: ActionSpec = NotImplemented,
        reward: RewardSpec = NotImplemented,
        **kwargs):
        
        self.observation = observation
        self.action = action
        self.reward = reward
        
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)
