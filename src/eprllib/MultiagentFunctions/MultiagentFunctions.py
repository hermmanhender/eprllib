"""
Multiagent Functions Base Methods
==================================

This method allows the combination of agents observations to provide a flexible configuration
of the comunication between agents. Build-in herarchical (only two levels), fully-shared,
centralized and independent configurations are provided.
"""
from typing import Dict, Any, Tuple
from gymnasium.spaces import Space

class MultiagentFunction:
    def __init__(
        self,
        multiagent_fn_config: Dict[str,Any] = {}
    ):
        """
        Base class for multiagent functions.
        
        :param multiagent_fn_config: configuration of the multiagent function
        :type multiagent_fn_config: Dict[str,Any], optional
        """
        self.multiagent_fn_config = multiagent_fn_config
    
    def get_agent_obs_dim(
        self,
        env_config: Dict[str,Any]
        ) -> Dict[str, Space]:
        """
        Get the agent observation dimension.

        :param env_config: environment configuration
        :type env_config: Dict[str,Any]
        :return: agent observation spaces
        :rtype: Dict[str, gym.Space]
        """
        return NotImplementedError("This method must be implemented in the child class.")
        
    def set_top_level_obs(
        self,
        env_config: Dict[str, Any],
        agent_states: Dict[str,Dict[str,Any]],
        dict_agents_obs: Dict[str,Any],
        infos: Dict[str, Dict[str, Any]]
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