"""
Specification for agent reward functions
=========================================
This module defines the `RewardSpec` class, which is used to specify the configuration of reward functions for agents in reinforcement learning environments.
It ensures that the reward function is properly defined and adheres to the expected interface.
"""
from typing import Dict, Any, Optional, Type # type: ignore
from eprllib.Agents.Rewards.BaseReward import BaseReward
from eprllib import logger

class RewardSpec:
    """
    RewardSpec is the base class for an reward specification to safe configuration of the object.
    """
    def __init__(
        self,
        reward_fn: Optional[Type[BaseReward]] = None,
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
    
    def __getitem__(self, key:str):
        return getattr(self, key)

    def __setitem__(self, key:str, value:Any):
        valid_keys = self.__dict__.keys()
        if key not in valid_keys:
            msg = f"Invalid key: {key}."
            logger.error(msg)
            raise KeyError(msg)
        setattr(self, key, value)
    
    def build(self) -> Dict[str, Any]:
        """
        This method is used to build the RewardSpec object.
        """
        if self.reward_fn == None:
            msg = "No reward function provided."
            logger.error(msg)
            raise NotImplementedError(msg)
        
        return vars(self)
    