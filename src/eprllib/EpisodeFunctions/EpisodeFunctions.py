"""
Episode Functions Base Class
============================

This module contains the base class for the episode functions. This class is used to define the episode configuration
for the EnergyPlus environment.
"""
from typing import Dict, Any

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