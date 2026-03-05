"""
Connector Base Method
=============================

This module defines the base class for connector functions that allow the combination of 
agents' observations to provide a flexible configuration of the communication between agents. 
Built-in hierarchical (only two levels), fully-shared, centralized, and independent configurations 
are provided.
"""
from typing import Dict, Any, Tuple, List
from gymnasium import spaces

from eprllib import logger
from eprllib.Utils.annotations import OverrideToImplementCustomLogic

class BaseConnector:
    """
    Base class for connector functions.
    """
    connector_fn_config: Dict[str, Any]
    obs_indexed: Dict[str,Dict[str, int]]
    _is_setup: bool
    
    def __init__(
        self,
        connector_fn_config: Dict[str, Any] = {}
    ):
        """
        Base class for connector functions.
        
        Args:
            connector_fn_config (Dict[str, Any]): Configuration of the connector function.
        
        Raises:
            RuntimeError: If the ``setup()`` method is called more than once.
            AttributeError: If the ``setup()`` method is not implemented in the child class.
        """
        self.connector_fn_config = connector_fn_config
        self.obs_indexed = {}
        
        logger.info(f"BaseConnector: The BaseConnector was correctly inicializated with {self.connector_fn_config} config.")
        
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
        
    
    def __name__(self):
        """
        Returns the name of the connector function.
        
        Returns:
            str: Name of the connector function.
        """
        return self.__class__.__name__
    
    def get_all_agents_obs_spaces_dict(
        self,
        possible_agents: List[str]
    ) -> spaces.Dict:
        """
        Get all the agents observations spaces putting togheter in a Dict space dimension.
        
        Args:
            possible_agents (List[str]): List of possible agents.
        
        Returns:
            gym.spaces.Dict: Agents observation spaces.
        """
        observation_space_dict: Dict[str, Any] = {}
        for agent in possible_agents:
            observation_space_dict[agent] = self.get_agent_obs_dim(agent)
        return spaces.Dict(observation_space_dict)
    
    
    # ===========================
    # === OVERRIDABLE METHODS ===
    # ===========================
    
    @OverrideToImplementCustomLogic
    def setup(self):
        """
        This method can be overridden in subclasses to perform setup tasks.
        """
        pass
    
    @OverrideToImplementCustomLogic
    def get_agent_obs_dim(
        self,
        agent: str
    ) -> spaces.Space[Any]:
        """
        Get the agent observation dimension.
        
        Args:
            agent (str): Agent identifier.
        
        Returns:
            gym.spaces.Space: Agent observation dimension.
            
        Raises:
            NotImplementedError: If the method is not implemented in the child class.
        """
        msg = "BaseConnector: This method must be implemented in the child class."
        logger.error(msg)
        raise NotImplementedError(msg)
    
    @OverrideToImplementCustomLogic
    def get_agent_obs_indexed(
        self,
        agent: str
    ) -> Dict[str, int]:
        """
        Get a dictionary of the agent observation parameters and their respective index in the observation array.
        
        Args:
            agent (str): Agent identifier.
            
        Returns:
            Dict[str, int]: Dictionary of the agent observation parameters and their respective index in the observation array.
        
        Raises:
            NotImplementedError: If the method is not implemented in the child class.

        """
        msg = "BaseConnector: This method must be implemented in the child class."
        logger.error(msg)
        raise NotImplementedError(msg)
    
    
    def set_top_level_obs(
        self,
        agent_states: Dict[str, Dict[str, Any]],
        dict_agents_obs: Dict[str, Any],
        infos: Dict[str, Dict[str, Any]],
        is_last_timestep: bool = False
    ) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]], bool]:
        """
        Set the multi-agent observation.
        
        Args:
            agent_states (Dict[str, Dict[str, Any]]): Agent states.
            dict_agents_obs (Dict[str, Any]): Dictionary of agents' observations.
            infos (Dict[str, Dict[str, Any]]): Additional information.
            is_last_timestep (bool, optional): Flag indicating if it is the last timestep. Defaults to False.
        
        Returns:
            Tuple[Dict[str, Any], Dict[str, Dict[str, Any]], bool]: Multi-agent observation, updated infos, and a flag indicating if it is the lowest level.
        
        """
        is_lowest_level = True
        return dict_agents_obs, infos, is_lowest_level
    
    
    def set_low_level_obs(
        self,
        agent_states: Dict[str,Dict[str,Any]],
        dict_agents_obs: Dict[str,Any],
        infos: Dict[str, Dict[str, Any]],
        goals: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]], bool]:
        """
        Set the multiagent observation.
        
        Args:
            agent_states (Dict[str, Dict[str, Any]]): Agent states.
            dict_agents_obs (Dict[str, Any]): Dictionary of agents' observations.
            infos (Dict[str, Dict[str, Any]]): Additional information.
            goals (Dict[str, Any]): Goals.
        
        Returns:
            Tuple[Dict[str, Any], Dict[str, Dict[str, Any]], bool]: Multi-agent observation, updated infos, and a flag indicating if it is the lowest level.
            
        """
        is_lowest_level = True
        return dict_agents_obs, infos, is_lowest_level
