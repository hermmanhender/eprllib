"""
Agents Connector Base Method
=============================

This module defines the base class for connector functions that allow the combination of 
agents' observations to provide a flexible configuration of the communication between agents. 
Built-in hierarchical (only two levels), fully-shared, centralized, and independent configurations 
are provided.
"""
from typing import Dict, Any, Tuple # type: ignore
from gymnasium import spaces
from eprllib import logger

class BaseConnector:
    def __init__(
        self,
        connector_fn_config: Dict[str, Any] = {}
    ):
        """
        Base class for connector functions.
        
        :param connector_fn_config: Configuration of the connector function.
        :type connector_fn_config: Dict[str, Any], optional
        """
        self.connector_fn_config = connector_fn_config
        self.obs_indexed: Dict[str,Dict[str, int]] = {}
    
    def __name__(self):
        """
        Returns the name of the connector function.

        :return: Name of the connector function.
        :rtype: str
        """
        return self.__class__.__name__
    
    def get_agent_obs_dim(
        self,
        env_config: Dict[str, Any],
        agent: str
    ) -> spaces.Space[Any]:
        """
        Get the agent observation dimension.

        :param env_config: Environment configuration.
        :type env_config: Dict[str, Any]
        :param agent: Agent identifier, optional.
        :type agent: str, optional
        :return: Agent observation spaces.
        :rtype: gym.spaces.Space
        """
        msg = "This method must be implemented in the child class."
        logger.error(msg)
        raise NotImplementedError(msg)
    
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
        msg = "This method must be implemented in the child class."
        logger.error(msg)
        raise NotImplementedError(msg)
    
    def get_all_agents_obs_spaces_dict(
        self,
        env_config: Dict[str, Any],
    ) -> spaces.Dict:
        """
        Get all the agents observations spaces putting togheter in a Dict space dimension.

        :param env_config: Environment configuration.
        :type env_config: Dict[str, Any]
        :return: Agents observation spaces.
        :rtype: gym.spaces.Dict
        """
        possible_agents = [key for key in env_config["agents_config"].keys()]
        observation_space_dict: Dict[str, Any] = {}
        for agent in possible_agents:
            observation_space_dict[agent] = self.get_agent_obs_dim(env_config, agent)
        return spaces.Dict(observation_space_dict)
        
    def set_top_level_obs(
        self,
        env_config: Dict[str, Any],
        agent_states: Dict[str, Dict[str, Any]],
        dict_agents_obs: Dict[str, Any],
        infos: Dict[str, Dict[str, Any]],
        is_last_timestep: bool = False
    ) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]], bool]:
        """
        Set the multi-agent observation.

        :param env_config: Environment configuration.
        :type env_config: Dict[str, Any]
        :param agent_states: Agent states.
        :type agent_states: Dict[str, Dict[str, Any]]
        :param dict_agents_obs: Dictionary of agents' observations.
        :type dict_agents_obs: Dict[str, Any]
        :param infos: Additional information.
        :type infos: Dict[str, Dict[str, Any]]
        :param is_last_timestep: Flag indicating if it is the last timestep, defaults to False.
        :type is_last_timestep: bool, optional
        :return: Multi-agent observation, updated infos, and a flag indicating if it is the lowest level.
        :rtype: Tuple[Dict[str, Any], Dict[str, Dict[str, Any]], bool]
        """
        is_lowest_level = True
        return dict_agents_obs, infos, is_lowest_level
    
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
