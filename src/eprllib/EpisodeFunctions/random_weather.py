"""
Ramdom weather for episode
===========================


"""

from typing import Dict, Any
from eprllib.EpisodeFunctions.EpisodeFunctions import EpisodeFunction
from eprllib.Utils.random_weather import get_random_weather
from eprllib.Utils.annotations import override

class random_weather_episode(EpisodeFunction):
    """
    This class contains the methods to configure the episode in EnergyPlus with RLlib.
    """
    def __init__(
        self,
        episode_fn_config: Dict[str,Any] = {}
        ):
        super().__init__(episode_fn_config)
        
        # check that 'epw_files_folder_path' exist in the episode_fn_config
        if 'epw_files_folder_path' not in self.episode_fn_config:
            raise ValueError("The 'epw_files_folder_path' must be defined in the episode_fn_config.")

    @override(EpisodeFunction)
    def get_episode_config(self, env_config: Dict[str,Any]) -> Dict[str,Any]:
        """
        This method update the 'epw_path' property of the env_config Dict and returns it
        with the change appied.

        Returns:
            Dict: The episode configuration.
        """
        env_config['epw_path'] = get_random_weather(self.episode_fn_config['epw_files_folder_path'])
        return env_config
    