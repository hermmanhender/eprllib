"""

"""
from typing import Any, Dict, Tuple, Set, List
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
        agents: Set,
        _thermal_zone_ids: Set,
        ) -> Tuple[gym.Space, Dict[str,List[str]]]:
        return NotImplementedError("You must implement this method.")
        
    def set_agent_obs_and_infos(
        self,
        env_config: Dict[str,Any],
        agents: Set,
        _thermal_zone_ids: Set,
        actuator_states: Dict[str,Any] = NotImplemented,
        actuator_infos: Dict[str,Any] = NotImplemented,
        site_state: Dict[str,Any] = NotImplemented,
        site_infos: Dict[str,Any] = NotImplemented,
        thermal_zone_states: Dict[str, Dict[str,Any]] = NotImplemented,
        thermal_zone_infos: Dict[str, Dict[str,Any]] = NotImplemented,
        agent_states: Dict[str, Dict[str,Any]] = NotImplemented,
        agent_infos: Dict[str, Dict[str,Any]] = NotImplemented,
        ) -> Tuple[Dict[str,Any],Dict[str,Dict[str,Any]]]:
        
        return NotImplementedError("You must implement this method.")
    