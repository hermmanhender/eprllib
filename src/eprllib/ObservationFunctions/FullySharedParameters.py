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
        obs_fn_config: Dict[str,Any]
        ):
        super().__init__(obs_fn_config)
        self.number_of_agents_total: int = self.obs_fn_config['number_of_agents_total']
        self.number_of_thermal_zone_total: int = self.obs_fn_config['number_of_thermal_zone_total']
        self.agent_obs_extra_var: Dict[str,Dict[str,Any]] = self.obs_fn_config['agent_obs_extra_var']
        self.other_agent_obs_extra_var: Dict[str,Dict[str,Any]] = self.obs_fn_config['other_agent_obs_extra_var']
        self.observation_space_labels: Dict[str,List[str]] = NotImplemented
        self._agent_indicator: Dict[str,Any] = NotImplemented
    
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
        if self.number_of_agents_total > 1:
            obs_space_len += self.number_of_agents_total
        
        # === Environment state ===
        # Environment variables.
        if env_config['variables_env'] is not None:
            obs_space_len += len(env_config['variables_env'])
        # simulation_parameters: Count the keys in the dict that are True
        if env_config['simulation_parameters'] is not None:
            sp_len = 0
            for value in env_config['simulation_parameters'].values():
                if value:
                    sp_len += 1
            obs_space_len += sp_len
        # Add weather prediction.
        if env_config['use_one_day_weather_prediction']:
            count_variables = 0
            for key in env_config['prediction_variables'].keys():
                if env_config['prediction_variables'][key]:
                    count_variables += 1
            obs_space_len += env_config['prediction_hours']*count_variables
        
        # === Thermal zone state where the agent belongs ===
        # Thermal zone variables.
        if env_config['variables_thz'] is not None:
            obs_space_len += len(env_config['variables_thz'])
        
        # Add static_variables.
        if env_config['static_variables'] is not None:
            for thermal_zone in _thermal_zone_ids:
                lenght_vector_sv = []
                lenght_vector_sv.append(len([key for key in env_config['static_variables'][thermal_zone].keys()]))
            # Check lenght vector elements are all the same, if don't a error happend.
            if len(set(lenght_vector_sv)) != 1:
                raise ValueError("The thermal zones have different number of static_variables.")
            # Add the lenght of the first thermal zone.
            obs_space_len += lenght_vector_sv[0]
        
        if env_config['zone_simulation_parameters'] is not None:
            sp_len = 0
            for value in env_config['zone_simulation_parameters'].values():
                if value:
                    sp_len += 1
            obs_space_len += sp_len
    
        # === Agent state ===
        # Object variables and meters variables.
        # Check all the agents have the same lengt.
        if env_config['variables_obj'] is not None:
            for agent in _agent_ids:
                lenght_vector_obj = []
                lenght_vector_obj.append(len([key for key in env_config['variables_obj'][agent].keys()]))
            # Check lenght vector elements are all the same, if don't a error happend.
            if len(set(lenght_vector_obj)) != 1:
                raise ValueError("The agents have different number of variables_obj.")
            # Add the lenght of the first agent.
            obs_space_len += lenght_vector_obj[0]
        
        if env_config['meters'] is not None:
            for agent in _agent_ids:
                lenght_vector_met = []
                lenght_vector_met.append(len([key for key in env_config['meters'][agent]]))
            # Check lenght vector elements are all the same, if don't a error happend.
            if len(set(lenght_vector_met)) != 1:
                raise ValueError("The agents have different number of meters.")
            # Add the lenght of the first agent.
            obs_space_len += lenght_vector_met[0]
        
        # actuator state.
        if env_config['use_actuator_state']:
            obs_space_len += 1
            
        # variables defined in agent_obs_extra_var
        if self.obs_fn_config['agent_obs_extra_var'] is not None:
            agent_obs_extra_var = []
            for agent in _agent_ids:
                agent_obs_extra_var.append([key for key in self.obs_fn_config['agent_obs_extra_var'][agent].keys()])
            # check that all the elements in the list are the same
            if len(set(map(len, agent_obs_extra_var))) != 1:
                raise ValueError("The agents have different number of agent_obs_extra_var.")
            
            obs_space_len += agent_obs_extra_var[0]
        
        # === Other agents reduce observations ===
        if self.number_of_agents_total > 1:
            # multi-one hot-code vector
            obs_space_len += self.number_of_agents_total
        
            # if apply, add the actuator state.
            if env_config['use_actuator_state']:
                obs_space_len += self.number_of_agents_total
            
            for _ in range(self.number_of_agents_total):
                for agent in _agent_ids:
                    if self.obs_fn_config['other_agent_obs_extra_var'] is not None:
                        obs_space_len += len([key for key in self.obs_fn_config['other_agent_obs_extra_var'][agent].keys()])
        
        # === Not observable variables === (discount them)
        if env_config['no_observable_variables'] is not None:
            no_obs_keys = [key for key in env_config['no_observable_variables'].keys()]
            if 'variables_env' in no_obs_keys:
                obs_space_len -= len(env_config['no_observable_variables']['variables_env'])
                
            if 'variables_thz' in no_obs_keys:
                obs_space_len -= len(env_config['no_observable_variables']['variables_thz'])
            
            if 'simulation_parameters' in no_obs_keys:
                obs_space_len -= len(env_config['no_observable_variables']['simulation_parameters'])
            
            if 'zone_simulation_parameters' in no_obs_keys:
                obs_space_len -= len(env_config['no_observable_variables']['zone_simulation_parameters'])
            
            if 'static_variables' in no_obs_keys:
                for thermal_zone in _thermal_zone_ids:
                    discount_len_vector = []
                    discount_len_vector.append(len([key for key in env_config['no_observable_variables']['static_variables'][thermal_zone].keys()]))
                if len(set(lenght_vector_sv)) != 1:
                    raise ValueError("The thermal zones in no_observable_variables have different number of static_variables.")    
                obs_space_len -= lenght_vector_sv[0]
            
            if 'variables_obj' in no_obs_keys:
                for agent in _agent_ids:
                    discount_len_vector_obj = []
                    discount_len_vector_obj.append(len([key for key in env_config['no_observable_variables']['variables_obj'][agent]]))
                if len(set(discount_len_vector_obj)) != 1:
                    raise ValueError("The agents in no_observable_variables have different number of variables_obj.")
                obs_space_len -= discount_len_vector_obj[0]
                
            if 'meters' in no_obs_keys:
                for agent in _agent_ids:
                    discount_len_vector_met = []
                    discount_len_vector_met.append(len([key for key in env_config['no_observable_variables']['meters'][agent].keys()]))
                if len(set(discount_len_vector_met)) != 1:
                    raise ValueError("The agents in no_observable_variables have different number of meters.")
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
        ) -> Tuple[Dict[str,Any],Dict[str, Dict[str,Any]]]:
        
        # Add agent indicator for the observation for each agent
        agents_obs = {agent: [] for agent in _agent_ids}
        agents_infos = {agent: {} for agent in _agent_ids}
        
        # Agent_indicator vector (if any)
        # Agent env observation (thermal_zone related term)
        # Obs of others agents (actions, types) (point of view of the actual agent)
        
        for agent in _agent_ids:
            # Agent properties
            agent_thermal_zone = env_config['agents_config'][agent]['thermal_zone']
            agent_id_vector = None
            
            # === AgentID one-hot enconde vector ===
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
            if agent_id_vector is not None:
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
            
            # extra obs provided in the obs_fn_config dict.
            if self.obs_fn_config['agent_obs_extra_var'] is not None:
                agent_obs_extra_var = np.array([value for value in self.obs_fn_config['agent_obs_extra_var'][agent].values()])
                ag_var = np.concatenate(
                    (
                        ag_var,
                        agent_obs_extra_var,
                    ),
                    dtype='float32'
                )
            
            # 3. Infos: Add the agent state infos to the agent infos.
            ag_inf.update(agent_infos[agent])
            
            agents_obs[agent] = ag_var
            agents_infos[agent] = ag_inf
            
        
        # === Other agents reduce observations ===
        # For this first the singular agent observation and infos are saved. After, if apply, the
        # others agents reduced observation is calculated and added at the end of the observation and
        # infos array of each agent.
            
        # Create the general observation
        if self.number_of_agents_total > 1:
            
            # multi-one hot-code vector
            ag_var = np.array([0]*self.number_of_agents_total)
            for agent in _agent_ids:
                agent_indicator = env_config['agents_config'][agent]['agent_indicator']
                ag_var[agent_indicator-1] = 1
            # add to each agent
            for agent in _agent_ids:
                agents_obs[agent] = np.concatenate(
                    (
                        agents_obs[agent],
                        ag_var,
                    ),
                    dtype='float32'
                )
            
            # if apply, add the actuator state.
            if env_config['use_actuator_state']:
                ag_var = np.array([0]*self.number_of_agents_total)
                for agent in _agent_ids:
                    agent_indicator = env_config['agents_config'][agent]['agent_indicator']
                    ag_var[agent_indicator-1] = actuator_states[agent]
                for agent in _agent_ids:
                    agents_obs[agent] = np.concatenate(
                        (
                            agents_obs[agent],
                            ag_var,
                        ),
                        dtype='float32'
                    )
                
            # extra obs provided in the obs_fn_config dict.
            if self.obs_fn_config['other_agent_obs_extra_var'] is not None:
                other_agent_obs_extra_var = np.array([value for value in self.obs_fn_config['other_agent_obs_extra_var'][agent].values()])
                for agent in _agent_ids:
                    agents_obs[agent] = np.concatenate(
                        (
                            agents_obs[agent],
                            other_agent_obs_extra_var,
                        ),
                        dtype='float32'
                    )
                
                
        return agents_obs, agents_infos
