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
import gymnasium as gym
import numpy as np
from typing import Any, Dict, Tuple
from eprllib.AgentsConnectors.BaseConnector import BaseConnector
from eprllib.Utils.annotations import override
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
        agent: str = None
    ) -> gym.Space:
        """
        Construct the observation space of the environment.

        Args:
            env_config (Dict[str, Any]): The environment configuration dictionary.
            agent (str, optional): The agent identifier.

        Returns:
            gym.Space: The observation space of the environment.
        """
        obs_space_len: int = 0
        
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
            obs_space_len += len(env_config["agents_config"][agent]["action"]['actuators'])
        
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
