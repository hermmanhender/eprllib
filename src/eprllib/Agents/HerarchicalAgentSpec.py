"""
Herarchical Agent Spec
=======================

This module implement the base for an agent specification to safe configuration of the object.
"""
from eprllib.Agents.AgentSpec import RewardSpec, ObservationSpec
from eprllib.ActionFunctions.ActionFunctions import HerarchicalActionFunction
from typing import Any
from typing import Dict


class HerarchicalActionSpec:
    """
    HerarchicalActionSpec is the base class for an action specification to safe configuration of the object.
    Note that the top level agent not control actuators, but define a strategy that can be used to select a 
    policy, an agent or augmentate the observation space with a goal variable.
    """
    def __init__(
        self,
        action_fn: HerarchicalActionFunction = NotImplemented,
        action_fn_config: Dict[str, Any] = {},
        ):
        self.action_fn = action_fn
        self.action_fn_config = action_fn_config
    
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
