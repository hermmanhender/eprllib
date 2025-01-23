"""
Observaton Functions
=====================

Work in progress...
"""
from typing import Any, Dict
import gymnasium as gym

class ObservationFunction:
    def __init__(
        self,
        obs_fn_config: Dict[str,Any]
        ):
        self.obs_fn_config = obs_fn_config
    
    def get_agent_obs_dim(
        self,
        env_config: Dict[str,Any],
        ) -> Dict[str, gym.Space]:
        return NotImplementedError("You must implement this method.")
        
    def set_agent_obs(
        self,
        env_config: Dict[str,Any],
        agent_states: Dict[str, Dict[str,Any]] = NotImplemented,
        ) -> Dict[str,Any]:
        
        return NotImplementedError("You must implement this method.")
    