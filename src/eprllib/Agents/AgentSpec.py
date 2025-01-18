"""
Agent Spec
===========

This module implement the base for an agent specification to safe configuration of the object.
"""

from typing import Optional, List, Dict, Tuple, Any
from eprllib.ActionFunctions.ActionFunctions import ActionSpec
from eprllib.ObservationFunctions.ObservationFunctions import ObservationSpec
from eprllib.RewardFunctions.RewardFunctions import RewardSpec

class AgentSpec:
    """
    AgentSpec is the base class for an agent specification to safe configuration of the object.
    """
    def __init__(self, **kwargs):
        observation: ObservationSpec = NotImplemented
        action: ActionSpec = NotImplemented
        reward: RewardSpec = NotImplemented
        
        for key, value in kwargs.items():
            setattr(self, key, value)
            



        
