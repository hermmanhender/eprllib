"""
Reward Funtion
==============

This module contains the base reward function.

It has been preferred to use the `infos` dictionary and not the observation, since the latter is 
a numpy array and cannot be called by key values, which is prone to errors when developing the program 
and indexing a arrangement may change.
"""

from typing import Dict, Any

class RewardSpec:
    """
    RewardSpec is the base class for an reward specification to safe configuration of the object.
    """
    def __init__(self):
        reward_fn: RewardFunction = None
        reward_fn_config: Dict[str, Any] = {}

class RewardFunction:
    """
    This class is the base class for the reward function.
    """
    def __init__(
        self,
        reward_fn_config: Dict[str,Any] = {}
    ):
        self.reward_fn_config = reward_fn_config
    
    def get_reward(
        self,
        infos: Dict[str,Any],
        terminated: bool,
        truncated: bool,
        ) -> Dict[str,float]:
        """
        This method must be implemented in the subclass.

        Args:
            infos (Dict[str,Dict[str,Any]]): The infos dictionary containing the necessary information for calculating the reward.

        Returns:
            Dict[str,float]: The calculated reward as a dictionary with the keys 'agent'.
        """
        return NotImplementedError("This method must be implemented in the subclass.")
