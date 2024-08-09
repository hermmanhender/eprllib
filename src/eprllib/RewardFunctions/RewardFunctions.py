"""
Reward Funtion
==============

This module contains the base reward function.

It has been preferred to use the `infos` dictionary and not the observation, since the latter is 
a numpy array and cannot be called by key values, which is prone to errors when developing the program 
and indexing a arrangement may change.
"""

from typing import Dict, Any
from eprllib.Env.MultiAgent.EnergyPlusEnvironment import EnergyPlusEnv_v0

class RewardFunction:
    def __init__(
        self,
        EnvObject: EnergyPlusEnv_v0
    ):
        self.EnvObject = EnvObject
    
    def calculate_reward(
        self,
        infos: Dict[str,Dict[str,Any]]
        ) -> Dict[str,float]:
        return NotImplementedError("This method must be implemented in the subclass.")
