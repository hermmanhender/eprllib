"""
Default Agents Connector
=========================

This module defines the default connector class that allows the combination of agents' observations 
to provide a flexible configuration of the communication between agents. Built-in hierarchical 
(only two levels), fully-shared, centralized, and independent configurations are provided.
"""
from typing import Dict, Any, Tuple
from gymnasium.spaces import Box
from eprllib.AgentsConnectors.BaseConnector import BaseConnector
from eprllib.Utils.annotations import override

class Coraci2021Connector(BaseConnector):
    def __init__(
        self,
        connector_fn_config: Dict[str,Any] = {}
    ):
        """
        Base class for multiagent functions.
        
        :param connector_fn_config: configuration of the multiagent function
        :type connector_fn_config: Dict[str,Any], optional
        """
        super().__init__(connector_fn_config)
    
    @override(BaseConnector)
    def get_agent_obs_dim(
        self,
        env_config: Dict[str,Any],
        agent: str
        ) -> Box:
        """
        Get the agent observation dimension.

        :param env_config: environment configuration
        :type env_config: Dict[str,Any]
        :return: agent observation spaces
        :rtype: Dict[str, gym.Space]
        """
        obs_space_len: int = 18
        self.obs_indexed[agent] = {
            f"{agent}: hour": 0,
            f"{agent}: minutes": 1,
            f"{agent}: dt: -3ts": 2,
            f"{agent}: dt: -2ts": 3,
            f"{agent}: dt: -1ts": 4,
            f"{agent}: dt: 0": 5,
            f"{agent}: Site Outdoor Air Drybulb Temperature: Environment": 6,
            f"{agent}: outdoor_dry_bulb: 0": 7,
            f"{agent}: outdoor_dry_bulb: 1": 8,
            f"{agent}: outdoor_dry_bulb: 2": 9,
            f"{agent}: Cooling:DistrictCooling": 10,
            f"{agent}: Heating:DistrictHeatingWater": 11,
            f"{agent}: Zone People Occupant Count: Thermal Zone": 12,
            f"{agent}: User Occupation Forecast +1h": 13,
            f"{agent}: User Occupation Forecast +2h": 14,
            f"{agent}: User Occupation Forecast +3h": 15,
            f"{agent}: Site Outdoor Air Relative Humidity: Environment": 16,
            f"{agent}: Zone Air Relative Humidity: Thermal Zone": 17,
        }
        return Box(float("-inf"), float("inf"), (obs_space_len, ))
    
    @override(BaseConnector)
    def get_agent_obs_indexed(
        self,
        env_config: Dict[str, Any],
        agent: str
    ) -> Dict[str, int]:
        """
        Get a dictionary of the agent observation parameters and their respective index in the observation array.

        :param env_config: Environment configuration.
        :type env_config: Dict[str, Any]
        :param agent: Agent identifier, optional.
        :type agent: str, optional
        :return: Agent observation spaces.
        :rtype: gym.spaces.Space
        """
        if self.obs_indexed == {}:
            self.get_agent_obs_dim(env_config, agent)
        return self.obs_indexed[agent]
    
    @override(BaseConnector)    
    def set_top_level_obs(
        self,
        env_config: Dict[str, Any],
        agent_states: Dict[str,Dict[str,Any]],
        dict_agents_obs: Dict[str,Any],
        infos: Dict[str, Dict[str, Any]],
        is_last_timestep: bool = False
    ) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]], bool]:
        """
        Set the multiagent observation.

        :param env_config: environment configuration
        :type env_config: Dict[str,Any]
        :param agent_states: agent states
        :type agent_states: Dict[str,Any]
        :param dict_agents_obs: dictionary of agents observations
        :type dict_agents_obs: Dict[str,Any]
        :return: multiagent observation
        :rtype: Dict[str,Any]
        """
        is_lowest_level = True
        return dict_agents_obs, infos, is_lowest_level
    
    @override(BaseConnector)
    def set_low_level_obs(
        self,
        env_config: Dict[str, Any],
        agent_states: Dict[str,Dict[str,Any]],
        dict_agents_obs: Dict[str,Any],
        infos: Dict[str, Dict[str, Any]],
        goals: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]], bool]:
        """
        Set the multiagent observation.

        :param env_config: environment configuration
        :type env_config: Dict[str,Any]
        :param agent_states: agent states
        :type agent_states: Dict[str,Any]
        :param dict_agents_obs: dictionary of agents observations
        :type dict_agents_obs: Dict[str,Any]
        :return: multiagent observation
        :rtype: Dict[str,Any]
        """
        is_lowest_level = True
        return dict_agents_obs, infos, is_lowest_level
