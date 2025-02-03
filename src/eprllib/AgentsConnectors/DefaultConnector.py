"""
Default Agents Connector
=========================

This module defines the default connector class that allows the combination of agents' observations 
to provide a flexible configuration of the communication between agents. Built-in hierarchical 
(only two levels), fully-shared, centralized, and independent configurations are provided.
"""

import gymnasium as gym
from typing import Dict, Any, Tuple
from gymnasium.spaces import Space
from eprllib.AgentsConnectors.BaseConnector import BaseConnector
from eprllib.Utils.annotations import override

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
        agent:str = None
        ) -> Space:
        """
        Get the agent observation dimension.

        :param env_config: environment configuration
        :type env_config: Dict[str,Any]
        :return: agent observation spaces
        :rtype: Dict[str, gym.Space]
        """
        obs_space_len = 0
        if env_config["agents_config"][agent]["observation"]['variables'] is not None:
            obs_space_len += len(env_config["agents_config"][agent]["observation"]['variables'])
            
        if env_config["agents_config"][agent]["observation"]['internal_variables'] is not None:
            obs_space_len += len(env_config["agents_config"][agent]["observation"]['internal_variables'])
            
        if env_config["agents_config"][agent]["observation"]['meters'] is not None:
            obs_space_len += len(env_config["agents_config"][agent]["observation"]['meters'])
            
        if env_config["agents_config"][agent]["observation"]['simulation_parameters'] is not None:
            sp_len = 0
            for value in env_config["agents_config"][agent]["observation"]['simulation_parameters'].values():
                if value:
                    sp_len += 1
            obs_space_len += sp_len
            
        if env_config["agents_config"][agent]["observation"]['zone_simulation_parameters'] is not None:
            sp_len = 0
            for value in env_config["agents_config"][agent]["observation"]['zone_simulation_parameters'].values():
                if value:
                    sp_len += 1
            obs_space_len += sp_len
            
        if env_config["agents_config"][agent]["observation"]['use_one_day_weather_prediction']:
            count_variables = 0
            for key in env_config["agents_config"][agent]["observation"]['prediction_variables'].keys():
                if env_config["agents_config"][agent]["observation"]['prediction_variables'][key]:
                    count_variables += 1
            obs_space_len += env_config["agents_config"][agent]["observation"]['prediction_hours']*count_variables
        
        if env_config["agents_config"][agent]["observation"]['other_obs'] is not None:
            obs_space_len += len(env_config["agents_config"][agent]["observation"]['other_obs'])

        if env_config["agents_config"][agent]["observation"]['use_actuator_state']:
            obs_space_len += len(env_config['agents_config'][agent]["action"]["actuators"])
    
        return gym.spaces.Box(float("-inf"), float("inf"), (obs_space_len, ))
    
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
