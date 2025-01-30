"""
Independent Agents Policy
==========================

This is the default observation function. Here each agent has his oown observation space and is returned
without modifications considering only the agent_states provided in the EnergyPlusRunner class.
"""
from typing import Any, Dict, Tuple
from eprllib.MultiagentFunctions.MultiagentFunctions import MultiagentFunction
from eprllib.Utils.annotations import override
import numpy as np
import gymnasium as gym

class independent(MultiagentFunction):
    def __init__(
        self,
        multiagent_fn_config: Dict[str,Any] = {}
        ):
        """This class implements an independent observation space for each agent.

        Args:
            obs_fn_config (Dict[str,Any]): An emptly dict.
            
        """
        super().__init__(multiagent_fn_config)
    
    @override(MultiagentFunction)
    def get_agent_obs_dim(
        self,
        env_config: Dict[str,Any],
        agent: str = None
        ) -> gym.Space:
        """
        This method construct the observation space of the environment.

        Args:
            env_config (Dict): The environment configuration dictionary.

        Returns:
            space.Box: The observation space of the environment.
        """
        obs_space_len: int = 0
        
        # Variables
        if env_config["agents_config"][agent]["observation"]['variables'] is not None:
            obs_space_len += len(env_config["agents_config"][agent]["observation"]['variables'])
        
        # Internal variables
        if env_config["agents_config"][agent]["observation"]['internal_variables'] is not None:
            obs_space_len += len(env_config["agents_config"][agent]["observation"]['internal_variables'])
        
        # Meters
        if env_config["agents_config"][agent]["observation"]['meters'] is not None:
            obs_space_len += len(env_config["agents_config"][agent]["observation"]['meters'])
        
        # Simulation parameters
        sp_len = 0
        for value in env_config["agents_config"][agent]["observation"]['simulation_parameters'].values():
            if value:
                sp_len += 1
        obs_space_len += sp_len
        
        # Zone simulation parameters
        sp_len = 0
        for value in env_config["agents_config"][agent]["observation"]['zone_simulation_parameters'].values():
            if value:
                sp_len += 1
        obs_space_len += sp_len
        
        # One day weather prediction
        if env_config["agents_config"][agent]["observation"]['use_one_day_weather_prediction']:
            count_variables = 0
            for key in env_config["agents_config"][agent]["observation"]['prediction_variables'].keys():
                if env_config["agents_config"][agent]["observation"]['prediction_variables'][key]:
                    count_variables += 1
            obs_space_len += env_config["agents_config"][agent]["observation"]['prediction_hours']*count_variables
        
        # Actuators state
        if env_config["agents_config"][agent]["observation"]['use_actuator_state']:
            obs_space_len += len(env_config['agents_config'][agent]["action"]["actuators"])
        
        # Other obs
        if env_config["agents_config"][agent]["observation"]['other_obs'] is not None:
            obs_space_len += len(env_config["agents_config"][agent]["observation"]['other_obs'])
        
        return gym.spaces.Box(float("-inf"), float("inf"), (obs_space_len, ))
    
    @override(MultiagentFunction)
    def set_top_level_obs(
        self,
        env_config: Dict[str, Any],
        agent_states: Dict[str,Dict[str,Any]],
        dict_agents_obs: Dict[str,Any],
        infos: Dict[str, Dict[str, Any]],
        is_last_timestep: bool = False
    ) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]], bool]:
        
        # Agents in this timestep
        agent_list = [key for key in agent_states.keys()]
        agents_obs = {agent: [] for agent in agent_list}
        
        # Transform agent_state dict to np.array for each agent.
        for agent in agent_list:
            agents_obs[agent] = np.array(list(agent_states[agent].values()), dtype='float32')
            
        return agents_obs, infos, False
