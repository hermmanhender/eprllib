"""
Reward Function
================

This module contains the base class for defining reward functions.

It is preferred to use the `infos` dictionary and not the observation, since the latter is 
a numpy array and cannot be called by key values, which is prone to errors when developing the program 
and indexing an array may change.

The terminated and truncated flags are arguments in the reward function ``get_reward`` method to allow
implementations with dispersed reward. This flags allow return the final reward when the episode ends.
"""

from typing import Dict, Any

class BaseReward:
    """
    This class is the base class for defining reward functions.
    """
    def __init__(
        self,
        reward_fn_config: Dict[str, Any] = {}
    ):
        """
        Initializes the base reward function with the given configuration.

        Args:
            reward_fn_config (Dict[str, Any]): Configuration dictionary for the reward function.
        """
        self.reward_fn_config = reward_fn_config
    
    def get_reward(
        self,
        infos: Dict[str, Any],
        terminated: bool,
        truncated: bool,
    ) -> float:
        """
        This method must be implemented in the subclass to calculate the reward.

        Args:
            infos (Dict[str, Any]): The infos dictionary containing the necessary information for calculating the reward.
            terminated (bool): Indicates if the episode has terminated.
            truncated (bool): Indicates if the episode has been truncated.

        Returns:
            float: The calculated reward.
        """
        raise NotImplementedError("This method must be implemented in the subclass.")
