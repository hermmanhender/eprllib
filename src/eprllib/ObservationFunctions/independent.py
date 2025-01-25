"""
Independent Agents Policy
==========================

This is the default observation function. Here each agent has his oown observation space and is returned
without modifications considering only the agent_states provided in the EnergyPlusRunner class.
"""
from typing import Any, Dict
from eprllib.ObservationFunctions.ObservationFunctions import ObservationFunction

import numpy as np
import gymnasium as gym

class independent(ObservationFunction):
    def __init__(
        self,
        obs_fn_config: Dict[str,Any] = {}
        ):
        """This class implements an independent observation space for each agent.

        Args:
            obs_fn_config (Dict[str,Any]): An emptly dict.
            
        """
        super().__init__(obs_fn_config)
    
    def get_agent_obs_dim(
        self,
        env_config: Dict[str,Any]
        ) -> gym.Space:
        """
        This method construct the observation space of the environment.

        Args:
            env_config (Dict): The environment configuration dictionary.

        Returns:
            space.Box: The observation space of the environment.
        """
        agent_list = [key for key in env_config["agents_config"].keys()]
        obs_space_len: Dict[str,int] = {agent:0 for agent in agent_list}
        
        for agent in agent_list:
            # Variables
            if env_config["agents_config"][agent]["observation"]['variables'] is not None:
                obs_space_len[agent] += len(env_config["agents_config"][agent]["observation"]['variables'])
            
            # Internal variables
            if env_config["agents_config"][agent]["observation"]['internal_variables'] is not None:
                obs_space_len[agent] += len(env_config["agents_config"][agent]["observation"]['internal_variables'])
            
            # Meters
            if env_config["agents_config"][agent]["observation"]['meters'] is not None:
                obs_space_len[agent] += len(env_config["agents_config"][agent]["observation"]['meters'])
            
            # Simulation parameters
            sp_len = 0
            for value in env_config["agents_config"][agent]["observation"]['simulation_parameters'].values():
                if value:
                    sp_len += 1
            obs_space_len[agent] += sp_len
            
            # Zone simulation parameters
            sp_len = 0
            for value in env_config["agents_config"][agent]["observation"]['zone_simulation_parameters'].values():
                if value:
                    sp_len += 1
            obs_space_len[agent] += sp_len
            
            # One day weather prediction
            if env_config["agents_config"][agent]["observation"]['use_one_day_weather_prediction']:
                count_variables = 0
                for key in env_config["agents_config"][agent]["observation"]['prediction_variables'].keys():
                    if env_config["agents_config"][agent]["observation"]['prediction_variables'][key]:
                        count_variables += 1
                obs_space_len[agent] += env_config["agents_config"][agent]["observation"]['prediction_hours']*count_variables
            
            # Actuators state
            if env_config["agents_config"][agent]["observation"]['use_actuator_state']:
                obs_space_len[agent] += len(env_config['agents_config'][agent]["action"]["actuators"])
            
            # Other obs
            if env_config["agents_config"][agent]["observation"]['other_obs'] is not None:
                obs_space_len[agent] += len(env_config["agents_config"][agent]["observation"]['other_obs'])
        
        return gym.spaces.Dict(
            {
                agent: gym.spaces.Box(float("-inf"), float("inf"), (obs_space_len[agent], )) 
                for agent 
                in agent_list
            }
        )
        
    def set_agent_obs(
        self,
        env_config: Dict[str,Any],
        agent_states: Dict[str, Dict[str,Any]] = NotImplemented,
        ) -> Dict[str,Any]:
        
        # Agents in this timestep
        agent_list = [key for key in agent_states.keys()]
        agents_obs = {agent: [] for agent in agent_list}
        
        # Transform agent_state dict to np.array for each agent.
        for agent in agent_list:
            agents_obs[agent] = np.array(list(agent_states[agent].values()), dtype='float32')
            
        return agents_obs
