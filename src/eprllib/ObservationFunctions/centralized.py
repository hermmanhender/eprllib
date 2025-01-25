"""
Centralized Agent Policy
=========================

A central agent take the observation of all the agents involve in the environment and concatenate them
to create a single observation. After transform the multiple observations in one, this is used in the 
central policy to select the multiple discrete action, one for each agent.

To avoid parameter repetitions in the central agent observation, only implement an observation parameter
in one single agent. For example, if two agents are present in the same thermal zone and both of them has
access to the thermal zone mean air temperature, only declares this parameter in one of them.
"""
import gymnasium as gym
import numpy as np
from typing import Any, Dict
from eprllib.ObservationFunctions.ObservationFunctions import ObservationFunction

class centralized(ObservationFunction):
    def __init__(
        self,
        obs_fn_config: Dict[str,Any]
        ):
        """This class implements a centralized policy for the observation function.

        Args:
            obs_fn_config (Dict[str,Any]): The configuration dictionary for the observation function.
            This must to contain the key 'number_of_agents_total', that represent the maximal
            quantity to wich the policy is prepared. It is related with the unitary vector.
            
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
        obs_space_len: int = 0
        
        for agent in agent_list:
            
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
        
        return gym.spaces.Dict({"central_agent": gym.spaces.Box(float("-inf"), float("inf"), (obs_space_len, ))})
        
    def set_agent_obs(
        self,
        env_config: Dict[str,Any],
        agent_states: Dict[str, Dict[str,Any]] = NotImplemented,
        ) -> Dict[str,Any]:
        
        # agents in this timestep
        agent_list = [key for key in agent_states.keys()]
        # Add agent indicator for the observation for each agent
        agents_obs = {"central_agent": np.array([], dtype='float32')}
        
        for agent in agent_list:
            agents_obs["central_agent"] = np.concatenate(
                (
                    agents_obs["central_agent"],
                    np.array(list(agent_states[agent].values()), dtype='float32')
                ),
                dtype='float32'
            )
            
        return agents_obs
