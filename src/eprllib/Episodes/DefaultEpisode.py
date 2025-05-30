"""
Default Episode
================

This module contains the default implementation of the episode functions for the EnergyPlus environment.
"""

from typing import Dict, Any, List
from eprllib.Episodes.BaseEpisode import BaseEpisode
from eprllib.Utils.annotations import override
from eprllib import logger

class DefaultEpisode(BaseEpisode):
    """
    This class provides the default implementation of the episode functions for the EnergyPlus environment.
    It inherits from the BaseEpisode class.
    """
    def __init__(
        self,
        episode_fn_config: Dict[str, Any] = {}
    ):
        """
        Initializes the DefaultEpisode object.

        Args:
            episode_fn_config (Dict[str, Any]): Configuration dictionary for the episode function.
        """
        super().__init__(episode_fn_config)

    @override(BaseEpisode)
    def get_episode_config(self, env_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Returns the episode configuration for the EnergyPlus environment.

        Args:
            env_config (Dict[str, Any]): The environment configuration.

        Returns:
            Dict[str, Any]: The episode configuration.
        """
        return super().get_episode_config(env_config)
    
    @override(BaseEpisode)
    def get_episode_agents(self, env_config: Dict[str, Any], possible_agents: List[str]) -> Dict[str, Any]:
        """
        Returns the agents for the episode configuration in the EnergyPlus environment.

        Args:
            env_config (Dict[str, Any]): The environment configuration.
            possible_agents (List[str]): List of possible agents.

        Returns:
            Dict[str, Any]: The agents that are acting for the episode configuration. Default: possible_agents list.
        """
        return super().get_episode_agents(env_config, possible_agents)
    
    @override(BaseEpisode)
    def get_timestep_agents(self, env_config: Dict[str, Any], possible_agents: List[str]) -> Dict[str, Any]:
        """
        Returns the agents for the timestep configuration in the EnergyPlus environment.

        Args:
            env_config (Dict[str, Any]): The environment configuration.
            possible_agents (List[str]): List of possible agents.

        Returns:
            Dict[str, Any]: The agents that are acting for the timestep configuration. Default: possible_agents list.
        """
        return super().get_timestep_agents(env_config, possible_agents)