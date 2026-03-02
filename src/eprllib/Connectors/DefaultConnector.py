"""
Default Agents Connector
=========================

This module defines the default connector class that allows the combination of agents' observations 
to provide a flexible configuration of the communication between agents. Built-in hierarchical 
(only two levels), fully-shared, centralized, and independent configurations are provided.
"""
from typing import Dict, Any
from gymnasium.spaces import Box

from eprllib.Connectors.BaseConnector import BaseConnector
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
    
    @override(BaseConnector)
    def setup(self) -> None:
        """
        This method can be overridden in subclasses to perform setup tasks.
        """
        self.env_config: Dict[str, Any] = {}
    
    @override(BaseConnector)
    def get_agent_obs_dim(
        self,
        agent: str
        ) -> Box:
        """
        Get the agent observation dimension.
        
        Args:
            agent (str): Agent identifier.
        
        Returns:
            gym.spaces.Space: Agent observation dimension.
            
        Raises:
            NotImplementedError: If the method is not implemented in the child class.
        """
        obs_space_len: int = 0
        self.obs_indexed[agent] = {}
        
        self.obs_indexed[agent], obs_space_len = set_variables_in_obs(self.env_config, agent, self.obs_indexed[agent], obs_space_len)
        self.obs_indexed[agent], obs_space_len = set_internal_variables_in_obs(self.env_config, agent, self.obs_indexed[agent], obs_space_len)
        self.obs_indexed[agent], obs_space_len = set_meters_in_obs(self.env_config, agent, self.obs_indexed[agent], obs_space_len)
        self.obs_indexed[agent], obs_space_len = set_simulation_parameters_in_obs(self.env_config, agent, self.obs_indexed[agent], obs_space_len)
        self.obs_indexed[agent], obs_space_len = set_zone_simulation_parameters_in_obs(self.env_config, agent, self.obs_indexed[agent], obs_space_len)
        self.obs_indexed[agent], obs_space_len = set_prediction_variables_in_obs(self.env_config, agent, self.obs_indexed[agent], obs_space_len)
        self.obs_indexed[agent], obs_space_len = set_other_obs_in_obs(self.env_config, agent, self.obs_indexed[agent], obs_space_len)
        self.obs_indexed[agent], obs_space_len = set_actuators_in_obs(self.env_config, agent, self.obs_indexed[agent], obs_space_len)
        self.obs_indexed[agent], obs_space_len = set_user_occupation_forecast_in_obs(self.env_config, agent, self.obs_indexed[agent], obs_space_len)
                
        assert obs_space_len > 0, "The observation space length must be greater than 0."
        assert len(self.obs_indexed[agent]) == obs_space_len, f"The observation space length must be equal to the number of indexed observations. Obs indexed:{len(self.obs_indexed[agent])} != Obs space len:{obs_space_len}."
        # obs_space_len += 1
        logger.debug(f"DefaultConnector: Observation space length for agent {agent}: {obs_space_len}")
        
        return Box(float("-inf"), float("inf"), (obs_space_len, ))
    
    @override(BaseConnector)
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
        if self.obs_indexed == {}:
            self.get_agent_obs_dim(agent)
        return self.obs_indexed[agent]
    