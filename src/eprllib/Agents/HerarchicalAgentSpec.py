"""
Herarchical Agent Spec
=======================

This module implement the base for an agent specification to safe configuration of the object.
"""

from gymnasium.spaces import Discrete, MultiDiscrete
from eprllib.Agents.AgentSpec import RewardSpec, ObservationSpec


class HerarchicalActionSpec:
    """
    HerarchicalActionSpec is the base class for an action specification to safe configuration of the object.
    Note that the top level agent not control actuators, but define a strategy that can be used to select a 
    policy, an agent or augmentate the observation space with a goal variable.
    """
    def __init__(
        self,
        action_space: Discrete|MultiDiscrete = NotImplemented
        ):
        
        if action_space == NotImplemented:
            raise NotImplementedError("action_space must be defined.")
        if type(action_space) not in [Discrete, MultiDiscrete]:
            raise ValueError(f"The action space must be a Discrete or MultiDiscrete space, but {type(action_space)} was used.")
        
        self.action_space = action_space
        self.action_fn = self.herarchical_action_fn
        self.action_fn_config = {}
    
    def herarchical_action_fn(self, action_fn_config={}):
        return self.action_space
    
    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

class HerarchicalAgentSpec:
    """
    AgentSpec is the base class for an agent specification to safe configuration of the object.
    """
    def __init__(
        self,
        observation: ObservationSpec = NotImplemented,
        action: HerarchicalActionSpec = NotImplemented,
        reward: RewardSpec = NotImplemented,
        **kwargs):
        
        if observation == NotImplemented:
            raise NotImplementedError("observation must be deffined.")
        if action == NotImplemented:
            raise NotImplementedError("action must be deffined.")
        if reward == NotImplemented:
            raise NotImplementedError("reward must be deffined.")
        
        self.observation = observation
        self.action = action
        self.reward = reward
        
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)
