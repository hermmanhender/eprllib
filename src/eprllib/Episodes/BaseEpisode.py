"""
Episode Functions Base Class
=============================

This module contains the base class for the ``Episode`` functions. This class is used to define the ``Episode`` configuration
for the EnergyPlus environment.

The methods provided here are used during inizialization and execution of the environment.
You can overwrite the following methods for sophisticated executions:

    - ``setup(self)``
    - ``get_episode_config(self, env_config: Dict[str, Any])``
    - ``get_episode_agents(self, env_config: Dict[str, Any], possible_agents: List[str])``
    - ``get_timestep_agents(self, env_config: Dict[str, Any], possible_agents: List[str])``

The ``DefaultEpisode`` configuration will be used by default.
"""

from typing import Dict, Any, List

from eprllib import logger

class BaseEpisode:
    """
    This class contains the methods to configure the episode in EnergyPlus with RLlib.
    """
    episode_fn_config: Dict[str, Any] = {}
    
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
        
        logger.info(f"BaseEpisode: The BaseEpisode was correctly inicializated with {self.episode_fn_config} config.")
        
        # Make sure, `setup()` is only called once, no matter what.
        if hasattr(self, "_is_setup") and self._is_setup:
            raise RuntimeError(
                "``BaseActionMapper.setup()`` called twice within your ActionMapper implementation "
                f"{self}! Make sure you are using the proper inheritance order "
                " and that you are NOT overriding the constructor, but "
                "only the ``setup()`` method of your subclass."
            )
        try:
            self.setup()
        except AttributeError as e:
            raise e

        self._is_setup:bool = True
    
    # ===========================
    # === OVERRIDABLE METHODS ===
    # ===========================
    
    def setup(self) -> None:
        """
        Sets up the episode function.
        
        This is called automatically during the __init__ method of this class.
        """
        pass
    
    
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