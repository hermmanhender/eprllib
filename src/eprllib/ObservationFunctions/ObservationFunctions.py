"""

"""
from typing import Any, Dict, Tuple, Set

class ObservationFunction:
    def __init__(
        self,
        config: Dict[str,Any]
        ):
        self.config = config
    
    def get_agent_obs_dim(
        self,
        env_config: Dict[str,Any],
        _agent_ids: Set,
        _thermal_zone_ids: Set,
        ) -> int:
        return NotImplementedError("You must implement this method.")
        
    def set_agent_obs(
        self,
        env_config: Dict[str,Any],
        _agent_ids: Set,
        _thermal_zone_ids: Set,
        actuator_states: Dict[str,Any] = NotImplemented,
        actuator_infos: Dict[str,Any] = NotImplemented,
        site_state: Dict[str,Any] = NotImplemented,
        site_infos: Dict[str,Any] = NotImplemented,
        thermal_zone_states: Dict[str, Dict[str,Any]] = NotImplemented,
        thermal_zone_infos: Dict[str, Dict[str,Any]] = NotImplemented,
        agent_states: Dict[str, Dict[str,Any]] = NotImplemented,
        agent_infos: Dict[str, Dict[str,Any]] = NotImplemented,
        ) -> Tuple[Dict[str,Any],Dict[Dict[str,Any]]]:
        
        return NotImplementedError("You must implement this method.")
    