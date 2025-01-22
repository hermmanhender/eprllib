"""
Reward Funtion
==============

This module contains the base reward function.

It has been preferred to use the `infos` dictionary and not the observation, since the latter is 
a numpy array and cannot be called by key values, which is prone to errors when developing the program 
and indexing a arrangement may change.
"""

from typing import Dict, Any

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

class RewardSpec:
    """
    RewardSpec is the base class for an reward specification to safe configuration of the object.
    """
    def __init__(
        self,
        reward_fn: RewardFunction = NotImplemented,
        reward_fn_config: Dict[str, Any] = {},
        ):
        """
        _Description_
        
        Args:
            reward_fn (RewardFunction): The reward funtion take the arguments EnvObject (the GymEnv class) and the infos 
            dictionary. As a return, gives a float number as reward. See eprllib.RewardFunctions for examples.
            
        """
        if reward_fn == NotImplemented:
            raise NotImplementedError("reward_fn must be defined.")
        
        self.reward_fn = reward_fn
        self.reward_fn_config = reward_fn_config
        
    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)
        