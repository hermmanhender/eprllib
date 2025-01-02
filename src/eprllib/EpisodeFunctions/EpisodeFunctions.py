"""
Episode Functions Base Class
============================

This module contains the base class for the episode functions. This class is used to define the episode configuration
for the EnergyPlus environment.
"""
from typing import Dict, Any, List

class EpisodeFunction:
    """
    This class contains the methods to configure the episode in EnergyPlus with RLlib.
    """
    def __init__(
        self,
        episode_fn_config: Dict[str,Any] = {}
        ):
        self.episode_fn_config = episode_fn_config

    def get_episode_config(self, env_config: Dict[str,Any]) -> Dict[str,Any]:
        """
        This method returns the episode configuration for the EnergyPlus environment.

        Returns:
            Dict: The episode configuration.
        """
        return env_config
    
    def get_episode_agents(self, env_config: Dict[str,Any], possible_agents: List[str]) -> Dict[str,Any]:
        """
        This method returns the agents for the episode configuration in the EnergyPlus environment.

        Returns:
            List[str]: The agent that are acting for the episode configuration. Default: possible_agent list.
        """
        return possible_agents
    
    def get_timestep_agents(self, env_config: Dict[str,Any], possible_agents: List[str]) -> Dict[str,Any]:
        """
        This method returns the agents for the episode configuration in the EnergyPlus environment.

        Returns:
            List[str]: The agent that are acting for the episode configuration. Default: possible_agent list.
        """
        return possible_agents
    