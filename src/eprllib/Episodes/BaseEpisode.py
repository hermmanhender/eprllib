"""
Episode Functions Base Class
=============================

This module contains the base class for the episode functions. This class is used to define the episode configuration
for the EnergyPlus environment.
"""

from typing import Dict, Any, List

class BaseEpisode:
    """
    This class contains the methods to configure the episode in EnergyPlus with RLlib.
    """
    def __init__(
        self,
        episode_fn_config: Dict[str, Any] = {}
    ):
        """
        Initializes the BaseEpisode object.

        Args:
            episode_fn_config (Dict[str, Any]): Configuration dictionary for the episode function.
        """
        # Set the episode_fn_config attribute
        self.episode_fn_config = episode_fn_config

    def get_episode_config(self, env_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Returns the episode configuration for the EnergyPlus environment.

        Args:
            env_config (Dict[str, Any]): The environment configuration.

        Returns:
            Dict[str, Any]: The episode configuration.
        """
        return env_config
    
    def get_episode_agents(self, env_config: Dict[str, Any], possible_agents: List[str]) -> List[str]:
        """
        Returns the agents for the episode configuration in the EnergyPlus environment.

        Args:
            env_config (Dict[str, Any]): The environment configuration.
            possible_agents (List[str]): List of possible agents.

        Returns:
            Dict[str, Any]: The agents that are acting for the episode configuration. Default: possible_agents list.
        """
        return possible_agents
    
    def get_timestep_agents(self, env_config: Dict[str, Any], possible_agents: List[str]) -> List[str]:
        """
        Returns the agents for the timestep configuration in the EnergyPlus environment.

        Args:
            env_config (Dict[str, Any]): The environment configuration.
            possible_agents (List[str]): List of possible agents.

        Returns:
            Dict[str, Any]: The agents that are acting for the timestep configuration. Default: possible_agents list.
        """
        return possible_agents