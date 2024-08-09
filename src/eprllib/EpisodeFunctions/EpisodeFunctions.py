"""
Episode Functions Base Class
============================

This module contains the base class for the episode functions. This class is used to define the episode configuration
for the EnergyPlus environment.
"""
from typing import Dict

class EpisodeFunction:
    """
    This class contains the methods to configure the episode in EnergyPlus with RLlib.
    """
    def __init__(
        self,
        env_config: Dict,
        **kargs
        ):
        self.env_config = env_config

    def get_episode_config(self) -> Dict:
        """
        This method returns the episode configuration for the EnergyPlus environment.

        Returns:
            Dict: The episode configuration.
        """
        return self.env_config