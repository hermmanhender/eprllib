"""
Fully Shared Parameters Policy
================================

This module implements a fully shared parameters policy for the observation function.

The observations are classified into:

* Actuator states
* Environment state
    * Environment variables
    * Simulation parameters
    * Weather prediction
* Thermal zone state
    * Thermal zone variables
    * Static variables
    * Zone simulation parameters
    * Building properties
* Agent state
    * Object variables
    * Meters
    
The flat observation for each agent must be a numpy array and the size must be equal to all agents. To 
this end, a concatenation of variables is performed.

*Agent flat observation*

* AgentID one-hot enconde vector
* Environment state
* Thermal zone state where the agent belongs
* Agent state
* Other agents reduce observations
"""
from typing import Any, Dict, Tuple, Set, List
from eprllib.ObservationFunctions.ObservationFunctions import ObservationFunction

import numpy as np
import random
import gymnasium as gym

class FullySharedParameters(ObservationFunction):
    def __init__(
        self,
        config: Dict[str,Any]
        ):
        self.config = config
        super().__init__(config)
        self.number_of_agents_total: int = self.config['number_of_agents_total']
        self.number_of_thermal_zone_total: int = self.config['number_of_thermal_zone_total']
        self.observation_space_labels: Dict[str,List[str]] = None
    
    def get_agent_obs_dim(
        self,
        env_config: Dict[str,Any],
        _agent_ids: Set,
        _thermal_zone_ids: Set,
        ) -> gym.Space:
        """
        This method construct the observation space of the environment.

        Args:
            env_config (Dict): The environment configuration dictionary.

        Returns:
            space.Box: The observation space of the environment.
        """
        # Variable to save the obs_space dim.
        obs_space_len = 0
        
        # === AgentID one-hot enconde vector ===
        # Add agents id vector to space dim.
        if self.number_of_thermal_zone_total > 1:
            obs_space_len += self.number_of_thermal_zone_total
        
        # === Environment state ===
        # Environment variables.
        obs_space_len += len(env_config['variables_env'])
        # simulation_parameters: Count the keys in the dict that are True
        obs_space_len += len([key for key, value in env_config['simulation_parameters'].items() if value])
        # Add weather prediction.
        if env_config['use_one_day_weather_prediction']:
            obs_space_len += env_config['prediction_hours']*env_config['prediction_variables']
        
        # === Thermal zone state where the agent belongs ===
        # Thermal zone variables.
        obs_space_len += len(env_config['variables_thz'])
        # Add static_variables and building properties.
        for thermal_zone in _thermal_zone_ids:
            lenght_vector_sv = []
            lenght_vector_bp = []
            lenght_vector_sv.append(len([key for key in env_config['static_variables'][thermal_zone].keys()]))
            if env_config['use_building_properties']:
                lenght_vector_bp.append(len([key for key in env_config['building_properties'][thermal_zone].keys()]))
        # Check lenght vector elements are all the same, if don't a error happend.
        if len(set(lenght_vector_sv)) != 1:
            raise ValueError("The thermal zones have different number of static_variables.")
        # Add the lenght of the first thermal zone.
        obs_space_len += lenght_vector_sv[0]
        if env_config['use_building_properties']:
            # Check lenght vector elements are all the same, if don't a error happend.
            if len(set(lenght_vector_bp)) != 1:
                raise ValueError("The thermal zones have different number of building_properties.")
            obs_space_len += lenght_vector_bp[0]
        # zone_simulation_parameters: Count the keys in the dict that are True
        obs_space_len += len([key for key, value in env_config['zone_simulation_parameters'].items() if value])
    
        # === Agent state ===
        # Object variables and meters variables.
        # Check all the agents have the same lengt.
        for agent in _agent_ids:
            lenght_vector_obj = []
            lenght_vector_met = []
            lenght_vector_obj.append(len([key for key in env_config['variables_obj'][agent].keys()]))
            lenght_vector_met.append(len([key for key in env_config['meters'][agent].keys()]))
        # Check lenght vector elements are all the same, if don't a error happend.
        if len(set(lenght_vector_obj)) != 1:
            raise ValueError("The agents have different number of variables_obj.")
        if len(set(lenght_vector_met)) != 1:
            raise ValueError("The agents have different number of meters.")
        # Add the lenght of the first agent.
        obs_space_len += lenght_vector_obj[0]
        obs_space_len += lenght_vector_met[0]
        # actuator state.
        if env_config['use_actuator_state']:
            obs_space_len += 1
            
        # thermal_zone_indicator
        if env_config['use_thermal_zone_indicator']:
            obs_space_len += self.number_of_thermal_zone_total
            
        # agent type.
        if env_config['use_agent_type']:
            obs_space_len += 1
        
        # === Other agents reduce observations ===
        if self.number_of_agents_total > 1:
            for _ in range(self.number_of_agents_total):
                obs_space_len += self.number_of_agents_total
                # if apply, add the actuator state.
                if env_config['use_actuator_state']:
                    obs_space_len += 1
                # if apply, add the thermal zone indicator
                if env_config['use_thermal_zone_indicator']:
                    obs_space_len += self.number_of_thermal_zone_total
                # if apply, add the agent type.
                if env_config['use_agent_type']:
                    obs_space_len += 1
        
        # === Not observable variables === (discount them)
        if env_config['no_observable_variables']:
            obs_space_len -= len(env_config['no_observable_variables']['variables_env'])
            obs_space_len -= len(env_config['no_observable_variables']['variables_thz'])
            obs_space_len -= len(env_config['no_observable_variables']['simulation_parameters'])
            obs_space_len -= len(env_config['no_observable_variables']['zone_simulation_parameters'])
            
            for thermal_zone in _thermal_zone_ids:
                discount_len_vector = []
                discount_len_vector.append(len([key for key in env_config['no_observable_variables']['static_variables'][thermal_zone].keys()]))
            if len(set(lenght_vector_sv)) != 1:
                raise ValueError("The thermal zones in no_observable_variables have different number of static_variables.")    
            obs_space_len -= lenght_vector_sv[0]
            
            for agent in _agent_ids:
                discount_len_vector_obj = []
                discount_len_vector_met = []
                discount_len_vector_obj.append(len([key for key in env_config['no_observable_variables']['variables_obj'][agent].keys()]))
                discount_len_vector_met.append(len([key for key in env_config['no_observable_variables']['meters'][agent].keys()]))
            if len(set(discount_len_vector_obj)) != 1:
                raise ValueError("The agents in no_observable_variables have different number of variables_obj.")
            if len(set(discount_len_vector_met)) != 1:
                raise ValueError("The agents in no_observable_variables have different number of meters.")
            obs_space_len -= discount_len_vector_obj[0]
            obs_space_len -= discount_len_vector_met[0]
        
        # construct the observation space.
        return gym.spaces.Box(float("-inf"), float("inf"), (obs_space_len,))
        
    def set_agent_obs_and_infos(
        self,
        env_config: Dict[str,Any],
        _agent_ids: Set,
        _thermal_zone_ids: Set,
        actuator_states: Dict[str,Any] = NotImplemented,
        actuator_infos: Dict[str,Any] = NotImplemented,
        site_state: Dict[str,Any] = NotImplemented,
        site_infos: Dict[str,Any] = NotImplemented,
        thermal_zone_states: Dict[str, Dict[str,Any]] = NotImplemented,
        thermal_zone_infos: Dict[str, Dict[str,Any]] = NotImplemented,
        agent_states: Dict[str, Dict[str,Any]] = NotImplemented,
        agent_infos: Dict[str, Dict[str,Any]] = NotImplemented,
        ) -> Tuple[Dict[str,Any],Dict[Dict[str,Any]]]:
        
        # Add agent indicator for the observation for each agent
        agents_obs = {agent: [] for agent in _agent_ids}
        agents_infos = {agent: {} for agent in _agent_ids}
        agents_obs_labels = {agent: [] for agent in _agent_ids} # To assign labels to the obs vector
        # Agent_indicator vector (if any)
        # Agent env observation (thermal_zone related term)
        # Obs of others agents (actions, types) (point of view of the actual agent)
        
        for agent in _agent_ids:
            # Agent properties
            agent_thermal_zone = env_config['agents_config'][agent]['thermal_zone']
            agent_id_vector = None
            
            # === AgentID one-hot enconde vector === (this is applied latter but possisioning and the begining of the observation space)
            # If apply, add the igent ID vector for each agent obs
            if self.number_of_agents_total > 1:
                # 1. Label: NotImplementd yet.
                # 2. Values:
                agent_indicator = env_config['agents_config'][agent]['agent_indicator']
                agent_id_vector = np.array([0]*self.number_of_agents_total)
                agent_id_vector[agent_indicator-1] = 1
                # 3. Infos: This is not useful for this part of the observation.
            
            # === Environment state ===
            # 1. Label: NotImplementd yet.
            # 2. Values: Transform the observation in a numpy array to meet the condition expected in a RLlib Environment
            ag_var = np.array(list(site_state.values()), dtype='float32')
            if agent_id_vector != None:
                ag_var = np.concatenate(
                    (
                        agent_id_vector,
                        ag_var
                    ),
                    dtype='float32'
                )
            # 3. Infos: Add the site infos to the agent infos.
            ag_inf = site_infos
            
            # === Thermal zone state where the agent belongs ===
            # 1. Label: NotImplementd yet.
            # 2. Values: Transform the observation in a numpy array to meet the condition expected in a RLlib Environment
            ag_var = np.concatenate(
                (
                    ag_var,
                    np.array(list(thermal_zone_states[agent_thermal_zone].values()), dtype='float32')
                ),
                dtype='float32'
            )
            # 3. Infos: Add the thermal zone state infos to the agent infos.
            ag_inf.update(thermal_zone_infos[agent_thermal_zone])
            
            # === Agent state ===
            # 1. Label: NotImplementd yet.
            # 2. Values: Transform the observation in a numpy array to meet the condition expected in a RLlib Environment
            ag_var = np.concatenate(
                (
                    ag_var,
                    np.array(list(agent_states[agent].values()), dtype='float32')
                ),
                dtype='float32'
            )
            # if apply, add the actuator state of this agent
            if env_config['use_actuator_state']:
                ag_var = np.concatenate(
                    (
                        ag_var,
                        [actuator_states[agent]],
                    ),
                    dtype='float32'
                )
                ag_inf.update(actuator_infos[agent])
            # if apply, add the thermal zone indicator
            if env_config['use_thermal_zone_indicator']:
                if self.number_of_agents_total > 1:
                    thermal_zone_indicator = env_config['agents_config'][agent]['thermal_zone_indicator']
                    thermal_zone_id_vector = np.array([0]*self.number_of_thermal_zone_total)
                    thermal_zone_id_vector[thermal_zone_indicator-1] = 1
                    ag_var = np.concatenate(
                        (
                            ag_var,
                            thermal_zone_id_vector,
                        ),
                        dtype='float32'
                    )
            # if apply, add the agent type.
            if env_config['use_agent_type']:
                agent_type = env_config['agents_config'][agent]['actuator_type']
                ag_var = np.concatenate(
                    (
                        ag_var,
                        [agent_type],
                    ),
                    dtype='float32'
                )
            # 3. Infos: Add the agent state infos to the agent infos.
            ag_inf.update(agent_infos[agent])
            
        # === Other agents reduce observations ===
        # For this first the singular agent observation and infos are saved. After, if apply, the
        # others agents reduced observation is calculated and added at the end of the observation and
        # infos array of each agent.
            agents_infos[agent] = ag_inf
            agents_obs[agent] = ag_var
            
        # Create the general observation
        if self.number_of_agents_total > 1:
            
            first_agent = True
            other_agents_dict = {agent: [] for agent in _agent_ids}
            for agent in _agent_ids:
                # Identify the agent ID.
                agent_indicator = env_config['agents_config'][agent]['agent_indicator']
                ag_var = np.array([0]*self.number_of_agents_total)
                ag_var[agent_indicator-1] = 1
                
                # if apply, add the actuator state.
                if env_config['use_actuator_state']:
                    ag_var = np.concatenate(
                        (
                            ag_var,
                            [actuator_states[agent]],
                        ),
                        dtype='float32'
                    )
                # if apply, add the thermal zone indicator
                if env_config['use_thermal_zone_indicator']:
                    thermal_zone_indicator = env_config['agents_config'][agent]['thermal_zone_indicator']
                    thermal_zone_id_vector = np.array([0]*self.number_of_thermal_zone_total)
                    thermal_zone_id_vector[thermal_zone_indicator-1] = 1
                    ag_var = np.concatenate(
                        (
                            ag_var,
                            thermal_zone_id_vector,
                        ),
                        dtype='float32'
                    )
                # if apply, add the agent type.
                if env_config['use_agent_type']:
                    agent_type = env_config['agents_config'][agent]['actuator_type']
                    ag_var = np.concatenate(
                        (
                            ag_var,
                            [agent_type],
                        ),
                        dtype='float32'
                    )
                other_agents_dict.update({agent: Tuple(ag_var)})
                # Calculate the len of the other agents observation vector.
                if first_agent:
                    first_agent = False
                    length_agent_in_poll = np.array([0]*len(ag_var))
            
            # Add emptly vector for others agents observations that are not present.
            ag_pool = []
            if self.number_of_agents_total-len(ag_pool) > 0:
                for _ in range(self.number_of_agents_total-len(ag_pool)):
                    ag_pool.append(tuple(length_agent_in_poll))
            
            for agent in _agent_ids:
                obs_list = []
                # Add the other agents observations.
                for other_agent in _agent_ids:
                    if agent != other_agent:
                        ag_pool.append(other_agents_dict[other_agent])
            
                # Shuffle the others agents obs to improve the generalization of the model.
                shuffled_ag_pool = random.sample(ag_pool, len(ag_pool))  # This shuffles the list
                for embedding in shuffled_ag_pool:
                    obs_list.append(np.array(embedding, dtype='float32'))
                    
                # Concatenamos todas las observaciones en un solo NDArray
                obs = np.concatenate(obs_list, dtype='float32')
                agents_obs[agent] = np.concatenate(
                    (
                        agents_obs[agent],
                        obs,
                    ),
                    dtype='float32'
                )
                
        return agents_obs, agents_infos