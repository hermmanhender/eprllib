"""
Fully Shared Parameters Policy
================================

This module implements a fully shared parameters policy for the observation function.

The flat observation for each agent must be a numpy array and the size must be equal to all agents. To 
this end, a concatenation of variables is performed.

* AgentID one-hot enconde vector
* Agent state
* Actuators state (include other agents)
"""
# DEVELOP NOTES
# TODO: The number of actuators for the case of other agents is variable. Change the actuator state for agent action or see how
# could be possible to expand the observation space to addap it to different observations size.

from typing import Any, Dict, Tuple
from eprllib.MultiagentFunctions.MultiagentFunctions import MultiagentFunction
from eprllib.Utils.annotations import override
from eprllib.Utils.observation_utils import get_actuator_name
from eprllib.Utils.annotations import override

import numpy as np
import gymnasium as gym

class fully_shared_parameters(MultiagentFunction):
    def __init__(
        self,
        multiagent_fn_config: Dict[str,Any]
        ):
        """This class implements a fully shared parameters policy for the observation function.

        Args:
            obs_fn_config (Dict[str,Any]): The configuration dictionary for the observation function.
            This must to contain the key 'number_of_agents_total', that represent the maximal
            quantity to wich the policy is prepared. It is related with the unitary vector.
            
        """
        super().__init__(multiagent_fn_config)
        self.number_of_agents_total: int = multiagent_fn_config['number_of_agents_total']
        self.number_of_actuators_total: int = multiagent_fn_config['number_of_actuators_total']
        self.agent_ids: Dict[str, int] = None
        self.actuator_ids = None
    
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
        agent_list = [key for key in env_config["agents_config"].keys()]
        obs_space_len: int = 0
        
        if len(agent_list) < self.number_of_agents_total:
            raise ValueError("The number of agents must be greater than the number of agents in the environment configuration.")
        
        if self.number_of_agents_total > 1:
            obs_space_len += self.number_of_agents_total
        
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
        
        # Add the actuator state to all the agents obs_space_len
        # Check if at least one agent request the use_actuator_state
        use_actuator_state_flag = False
        for agent in agent_list:
            if env_config["agents_config"][agent]["observation"]['use_actuator_state']:
                use_actuator_state_flag = True
                break
        if use_actuator_state_flag:
            # Save the total amount of actuators in the environment to create a vector in set_agent_obs
            actuators = 0
            for agent in agent_list:
                actuators += len(env_config['agents_config'][agent]["action"]["actuators"])
            # chack that actuators is equal or minor to self.number_of_actuators_total
            if actuators > self.number_of_actuators_total:
                raise ValueError("The total amount of actuators in the environment is greater than the number of actuators in the environment configuration.")
            else:
                obs_space_len += self.number_of_actuators_total
        
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
        # Add the ID vectors if it's needed
        if self.agent_ids is None:
            id = 0
            self.agent_ids = {agent: None for agent in env_config['agents_config'].keys()}
            for agent in env_config['agents_config'].keys():
                self.agent_ids.update({agent: id})
                id += 1
            if len(self.agent_ids) > self.number_of_agents_total:
                print(f"The agents found were: {agent_list} with a total of {len(agent_list)}, that are greather than {self.number_of_agents_total}.")
                raise ValueError("The number of agents must be greater than the number of agents in the environment configuration.")
        
        if self.actuator_ids is None:
            id = 0
            self.actuator_ids = {}
            for agent in env_config['agents_config'].keys():
                for actuator_config in env_config["agents_config"][agent]["action"]["actuators"]:
                    self.actuator_ids.update({get_actuator_name(agent,actuator_config[0],actuator_config[1],actuator_config[2]): id})
                    id += 1
                    if len(self.actuator_ids) > self.number_of_actuators_total:
                        raise ValueError("The total amount of actuators in the environment is greater than the number of actuators in the environment configuration.")
        
        # agents in this timestep
        agent_list = [key for key in agent_states.keys()]
        # Add agent indicator for the observation for each agent
        agents_obs = {agent: np.array([], dtype='float32') for agent in agent_list}
        actuator_names = {agent: {} for agent in agent_list}
        
        for agent in agent_list:
            # Remove from agent_states and save the actuator items.
            for actuator_config in env_config["agents_config"][agent]["action"]["actuators"]:
                actuator_name = get_actuator_name(agent,actuator_config[0],actuator_config[1],actuator_config[2])
                actuator_names[agent].update({actuator_name: agent_states[agent].pop(actuator_name)})
            
            # Agent properties
            agent_id_vector = None
            
            # === AgentID one-hot enconde vector ===
            # If apply, add the igent ID vector for each agent obs
            if self.number_of_agents_total > 1:
                agent_id_vector = np.array([0]*self.number_of_agents_total)
                agent_id_vector[self.agent_ids[agent]] = 1
                
            
            if agent_id_vector is not None:
                agents_obs[agent] = np.concatenate(
                    (
                        agent_id_vector,
                        np.array(list(agent_states[agent].values()), dtype='float32')
                    ),
                    dtype='float32'
                )
            
        # if apply, add the actuator state as a vector of all agents.
        use_actuator_state_flag = False
        for agent in agent_list:
            if env_config["agents_config"][agent]["observation"]['use_actuator_state']:
                use_actuator_state_flag = True
                
        if use_actuator_state_flag:
            
            actuator_id_vector = np.array([-2]*self.number_of_actuators_total)
            
            for agent in agent_list:
                for actuator in actuator_names[agent]:
                    actuator_id_vector[self.actuator_ids[actuator]] = actuator_names[agent][actuator]
            
            for agent in agent_list:        
                agents_obs[agent] = np.concatenate(
                    (
                        agents_obs[agent],
                        actuator_id_vector,
                    ),
                    dtype='float32'
                )
            
        return agents_obs, infos, True
