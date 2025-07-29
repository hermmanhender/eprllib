"""
Centralized Agents Connector
=============================

A central agent takes the observations of all the agents involved in the environment and concatenates them
to create a single observation. After transforming the multiple observations into one, this is used in the 
central policy to select multiple discrete actions, one for each agent.

To avoid parameter repetitions in the central agent observation, only implement an observation parameter
in a single agent. For example, if two agents are present in the same thermal zone and both of them have
access to the thermal zone mean air temperature, only declare this parameter in one of them.
"""
from gymnasium.spaces import Box
import numpy as np
from typing import Any, Dict, Tuple # type: ignore
from eprllib.AgentsConnectors.BaseConnector import BaseConnector
from eprllib.Utils.annotations import override
from eprllib.Utils.connector_utils import (
    set_variables_in_obs,
    set_internal_variables_in_obs,
    set_meters_in_obs,
    set_zone_simulation_parameters_in_obs,
    set_prediction_variables_in_obs,
    set_other_obs_in_obs,
    set_actuators_in_obs
    )
from eprllib import logger

class CentralizedConnector(BaseConnector):
    def __init__(
        self,
        connector_fn_config: Dict[str,Any]
        ):
        """
        This class implements a centralized policy for the observation function.

        Args:
            connector_fn_config (Dict[str, Any]): The configuration dictionary for the observation function.
            This must contain the key 'number_of_agents_total', which represents the maximum
            quantity to which the policy is prepared. It is related to the unitary vector.
        """
        super().__init__(connector_fn_config)
    
    @override(BaseConnector)
    def get_agent_obs_dim(
        self,
        env_config: Dict[str, Any],
        agent: str
    ) -> Box:
        """
        Construct the observation space of the environment.

        Args:
            env_config (Dict[str, Any]): The environment configuration dictionary.
            agent (str, optional): The agent identifier.

        Returns:
            gym.Space: The observation space of the environment.
        """
        obs_space_len: int = 0
        
        self.obs_indexed, obs_space_len = set_variables_in_obs(env_config, agent, self.obs_indexed)
        self.obs_indexed, obs_space_len = set_internal_variables_in_obs(env_config, agent, self.obs_indexed)
        self.obs_indexed, obs_space_len = set_meters_in_obs(env_config, agent, self.obs_indexed)
        self.obs_indexed, obs_space_len = set_zone_simulation_parameters_in_obs(env_config, agent, self.obs_indexed)
        self.obs_indexed, obs_space_len = set_prediction_variables_in_obs(env_config, agent, self.obs_indexed)
        self.obs_indexed, obs_space_len = set_other_obs_in_obs(env_config, agent, self.obs_indexed)
        self.obs_indexed, obs_space_len = set_actuators_in_obs(env_config, agent, self.obs_indexed)
        
        assert obs_space_len > 0, "The observation space length must be greater than 0."
        assert len(self.obs_indexed) == obs_space_len, "The observation space length must be equal to the number of indexed observations."
        
        logger.debug(f"Observation space length for agent {agent}: {obs_space_len}")
        
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
        return self.obs_indexed
    
    @override(BaseConnector)
    def set_top_level_obs(
        self,
        env_config: Dict[str, Any],
        agent_states: Dict[str,Dict[str,Any]],
        dict_agents_obs: Dict[str,Any],
        infos: Dict[str, Dict[str, Any]],
        is_last_timestep: bool = False
    ) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]], bool]:
        
        # agents in this timestep
        agent_list = [key for key in dict_agents_obs.keys()]
        # Add agent indicator for the observation for each agent
        agents_obs = {"central_agent": np.array([], dtype='float32')}
        
        for agent in agent_list:
            agents_obs["central_agent"] = np.concatenate(
                (
                    agents_obs["central_agent"],
                    dict_agents_obs[agent]
                ),
                dtype='float32'
            )
            
        return agents_obs, infos, True
