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
from typing import Dict, Any # type: ignore
from numpy.typing import NDArray
from numpy import float32
from eprllib import logger

class BaseReward:
    """
    This class is the base class for defining reward functions.
    """
    def __init__(
        self,
        reward_fn_config: Dict[str, Any] = {}
    ) -> None:
        """
        Initializes the base reward function with the given configuration.

        Args:
            reward_fn_config (Dict[str, Any]): Configuration dictionary for the reward function.
        """
        self.reward_fn_config = reward_fn_config
    
    def set_initial_parameters(
        self,
        agent_name: str,
        obs_indexed: Dict[str, int]
    ) -> None:
        """
        This method can be overridden in subclasses to set initial parameters based on the provided infos.

        Args:
            infos (Dict[str, Any]): The infos dictionary containing necessary information for initialization.
        """
        pass
    
    def get_reward(
        self,
        obs: NDArray[float32],
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
        msg = "This method must be implemented in the subclass."
        logger.error(msg)
        raise NotImplementedError(msg)
