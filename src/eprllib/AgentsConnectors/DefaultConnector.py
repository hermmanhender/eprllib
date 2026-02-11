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
from eprllib.Utils.connector_utils import (
    set_variables_in_obs,
    set_internal_variables_in_obs,
    set_meters_in_obs,
    set_simulation_parameters_in_obs,
    set_zone_simulation_parameters_in_obs,
    set_prediction_variables_in_obs,
    set_other_obs_in_obs,
    set_actuators_in_obs,
    set_user_occupation_forecast_in_obs
    )
from eprllib import logger

class DefaultConnector(BaseConnector):
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
        obs_space_len: int = 0
        self.obs_indexed[agent] = {}
        
        self.obs_indexed[agent], obs_space_len = set_variables_in_obs(env_config, agent, self.obs_indexed[agent], obs_space_len)
        self.obs_indexed[agent], obs_space_len = set_internal_variables_in_obs(env_config, agent, self.obs_indexed[agent], obs_space_len)
        self.obs_indexed[agent], obs_space_len = set_meters_in_obs(env_config, agent, self.obs_indexed[agent], obs_space_len)
        self.obs_indexed[agent], obs_space_len = set_simulation_parameters_in_obs(env_config, agent, self.obs_indexed[agent], obs_space_len)
        self.obs_indexed[agent], obs_space_len = set_zone_simulation_parameters_in_obs(env_config, agent, self.obs_indexed[agent], obs_space_len)
        self.obs_indexed[agent], obs_space_len = set_prediction_variables_in_obs(env_config, agent, self.obs_indexed[agent], obs_space_len)
        self.obs_indexed[agent], obs_space_len = set_other_obs_in_obs(env_config, agent, self.obs_indexed[agent], obs_space_len)
        self.obs_indexed[agent], obs_space_len = set_actuators_in_obs(env_config, agent, self.obs_indexed[agent], obs_space_len)
        self.obs_indexed[agent], obs_space_len = set_user_occupation_forecast_in_obs(env_config, agent, self.obs_indexed[agent], obs_space_len)
                
        assert obs_space_len > 0, "The observation space length must be greater than 0."
        assert len(self.obs_indexed[agent]) == obs_space_len, f"The observation space length must be equal to the number of indexed observations. Obs indexed:{len(self.obs_indexed[agent])} != Obs space len:{obs_space_len}."
        # obs_space_len += 1
        logger.debug(f"DefaultConnector: Observation space length for agent {agent}: {obs_space_len}")
        
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
