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
from eprllib.ObservationFunctions.ObservationFunctions import ObservationFunction

import numpy as np
import gymnasium as gym

class FullySharedParameters(ObservationFunction):
    def __init__(
        self,
        obs_fn_config: Dict[str,Any]
        ):
        """This class implements a fully shared parameters policy for the observation function.

        Args:
            obs_fn_config (Dict[str,Any]): The configuration dictionary for the observation function.
            This must to contain the key 'number_of_agents_total', that represent the maximal
            quantity to wich the policy is prepared. It is related with the unitary vector.
            
        """
        super().__init__(obs_fn_config)
        self.number_of_agents_total: int = self.obs_fn_config['number_of_agents_total']
        self.number_of_actuators_total: int = self.obs_fn_config['number_of_actuators_total']
        self.agent_ids = None
        self.actuator_ids = None
    
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
        
        if len(agent_list) < self.number_of_agents_total:
            raise ValueError("The number of agents must be greater than the number of agents in the environment configuration.")
        
        for agent in agent_list:
            if self.number_of_agents_total > 1:
                obs_space_len[agent] += self.number_of_agents_total
            
            if env_config["agents_config"][agent]["observation"]['variables'] is not None:
                obs_space_len[agent] += len(env_config["agents_config"][agent]["observation"]['variables'])
                
            if env_config["agents_config"][agent]["observation"]['internal_variables'] is not None:
                obs_space_len[agent] += len(env_config["agents_config"][agent]["observation"]['internal_variables'])
                
            if env_config["agents_config"][agent]["observation"]['meters'] is not None:
                obs_space_len[agent] += len(env_config["agents_config"][agent]["observation"]['meters'])
                
                
            if env_config["agents_config"][agent]["observation"]['simulation_parameters'] is not None:
                sp_len = 0
                for value in env_config["agents_config"][agent]["observation"]['simulation_parameters'].values():
                    if value:
                        sp_len += 1
                obs_space_len[agent] += sp_len
            if env_config["agents_config"][agent]["observation"]['zone_simulation_parameters'] is not None:
                sp_len = 0
                for value in env_config["agents_config"][agent]["observation"]['zone_simulation_parameters'].values():
                    if value:
                        sp_len += 1
                obs_space_len[agent] += sp_len
                
            if env_config["agents_config"][agent]["observation"]['use_one_day_weather_prediction']:
                count_variables = 0
                for key in env_config["agents_config"][agent]["observation"]['prediction_variables'].keys():
                    if env_config["agents_config"][agent]["observation"]['prediction_variables'][key]:
                        count_variables += 1
                obs_space_len[agent] += env_config["agents_config"][agent]["observation"]['prediction_hours']*count_variables
            
            if env_config["agents_config"][agent]["observation"]['other_obs'] is not None:
                obs_space_len[agent] += len(env_config["agents_config"][agent]["observation"]['other_obs'])
        
        # check if all the agents has the same len in the obs_space_len
        if len(set(obs_space_len.values())) > 1:
            raise ValueError("The agents must have the same observation space length.")
        
        obs_space_len_shared = obs_space_len[agent_list[0]]
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
                obs_space_len_shared += self.number_of_actuators_total
                
        # construct the observation space.
        return gym.spaces.Box(float("-inf"), float("inf"), (obs_space_len_shared,))
        
    def set_agent_obs(
        self,
        env_config: Dict[str,Any],
        agent_states: Dict[str, Dict[str,Any]] = NotImplemented,
        ) -> Tuple[Dict[str,Any],Dict[str, Dict[str,Any]]]:
        # Add the ID vectors if it's needed
        id = 0
        if self.agent_ids is None:
            self.agent_ids = {agent: id for agent in [key for key in env_config['agents_config'].keys()]}
            id += 1
        
        id = 0
        if self.actuator_ids is None:
            for agent in env_config['agents_config'].keys():
                self.actuator_ids = {f"{agent}: {actuator_config[0]}: {actuator_config[1]}: {actuator_config[2]}": id for actuator_config in env_config["agents_config"][agent]["action"]["actuators"]}
                id += 1
        
        # agents in this timestep
        agent_list = [key for key in agent_states.keys()]
        # Add agent indicator for the observation for each agent
        agents_obs = {agent: [] for agent in agent_list}
        
        if len(agent_list) < self.number_of_agents_total:
            raise ValueError("The number of agents must be greater than the number of agents in the environment configuration.")
        
        
        for agent in agent_list:
            # Agent properties
            agent_id_vector = None
            
            # === AgentID one-hot enconde vector ===
            # If apply, add the igent ID vector for each agent obs
            if self.number_of_agents_total > 1:
                # 1. Label: NotImplementd yet.
                # 2. Values:
                agent_id_vector = np.array([0]*self.number_of_agents_total)
                agent_id_vector[self.agent_ids[agent]] = 1
                # 3. Infos: This is not useful for this part of the observation.
            
            # === Environment state ===
            # 1. Label: NotImplementd yet.
            # 2. Values: Transform the observation in a numpy array to meet the condition expected in a RLlib Environment
            ag_var = np.array(list(agent_states[agent].values()), dtype='float32')
            if agent_id_vector is not None:
                ag_var = np.concatenate(
                    (
                        agent_id_vector,
                        ag_var
                    ),
                    dtype='float32'
                )
            # if apply, add the actuator state of this agent
            use_actuator_state_flag = False
            for agent2 in agent_list:
                if env_config["agents_config"][agent2]["observation"]['use_actuator_state']:
                    use_actuator_state_flag = True
                    
            if use_actuator_state_flag:
                agent_list_copy = agent_list.copy()
                agent_list_copy.remove(agent)
                
                actuator_id_vector = np.array([-2]*self.number_of_actuators_total)
                
                for agent3 in agent_list_copy:
                    for actuator_config in env_config["agents_config"][agent3]["action"]["actuators"]:
                        actuator_name = f"{agent3}: {actuator_config[0]}: {actuator_config[1]}: {actuator_config[2]}"
                        actuator_id_vector[self.actuator_ids[actuator_name]] = agent_states[agent3][actuator_name]
                        
                ag_var = np.concatenate(
                    (
                        ag_var,
                        actuator_id_vector,
                    ),
                    dtype='float32'
                )
            
            agents_obs[agent] = ag_var
            
        return agents_obs
