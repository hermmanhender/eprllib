"""
Specification for agent reward functions
=========================================
This module defines the `RewardSpec` class, which is used to specify the configuration of reward functions for agents in reinforcement learning environments.
It ensures that the reward function is properly defined and adheres to the expected interface.
"""
import logging
from typing import Dict, Any
from eprllib.Agents.Rewards.BaseReward import BaseReward

logger = logging.getLogger("ray.rllib")

class RewardSpec:
    """
    RewardSpec is the base class for an reward specification to safe configuration of the object.
    """
    def __init__(
        self,
        reward_fn: BaseReward = NotImplemented,
        reward_fn_config: Dict[str, Any] = {},
        ):
        """
        Construction method.
        
        Args:
            reward_fn (BaseReward): The reward funtion take the arguments EnvObject (the GymEnv class) and the infos 
            dictionary. As a return, gives a float number as reward. See eprllib.Agents.Rewards for examples.
            reward_fn_config (Dict[str, Any]): The configuration of the reward function.
            
        Raises:
            NotImplementedError: If the reward_fn is NotImplemented.
        """
        self.reward_fn = reward_fn
        self.reward_fn_config = reward_fn_config
    
    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        valid_keys = self.__dict__.keys()
        if key not in valid_keys:
            msg = f"Invalid key: {key}."
            logger.error(msg)
            raise KeyError(msg)
        setattr(self, key, value)
    
    def build(self) -> Dict:
        """
        This method is used to build the RewardSpec object.
        """
        if self.reward_fn == NotImplemented:
            msg = "No reward function provided."
            logger.error(msg)
            raise NotImplementedError(msg)
        
        # Check if the reward_fn is a subclass of BaseReward, and raise an error if not.
        if not issubclass(self.reward_fn, BaseReward):
            msg = f"The reward function must be based on BaseReward class but {type(self.reward_fn)} was given."
            logger.error(msg)
            raise TypeError(msg)
        
        if not isinstance(self.reward_fn_config, dict):
            msg = f"The configuration for the reward function must be a dictionary but {type(self.reward_fn_config)} was given."
            logger.error(msg)
            raise TypeError(msg)
        
        return vars(self)
    