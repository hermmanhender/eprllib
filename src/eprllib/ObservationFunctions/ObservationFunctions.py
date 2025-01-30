"""
Observaton Functions
=====================

Work in progress...
"""
from typing import Any, Dict
import numpy as np
import gymnasium as gym
from eprllib.MultiagentFunctions.MultiagentFunctions import MultiagentFunction

class ObservationFunction:
    def __init__(
        self,
        obs_fn_config: Dict[str,Any]
        ):
        self.obs_fn_config = obs_fn_config
    
    def get_agent_obs_dim(
        self,
        env_config: Dict[str,Any],
        multiagent_fn: MultiagentFunction,
        agent: str = None
        ) -> Dict[str, gym.Space]:
        """
        This method construct the observation space. Now is implemented in the multiagent function. Not modify this method.
        """
        return multiagent_fn.get_agent_obs_dim(env_config, agent)
        
    def set_agent_obs(
        self,
        env_config: Dict[str,Any],
        agent_states: Dict[str,Any] = NotImplemented,
        ) -> Dict[str,Any]:
        
        return np.array(list(agent_states.values()), dtype='float32')
    