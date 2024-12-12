from typing import Dict, Any
from eprllib.EpisodeFunctions.EpisodeFunctions import EpisodeFunction
from eprllib.Tools.Utils import random_weather_config

class RandomWeather(EpisodeFunction):
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
        
        return random_weather_config(env_config, self.episode_fn_config['epw_files_folder_path'])
    