"""
Hierarchical Agents Connector
=============================

This module implements a hierarchical connector with two levels of hierarchy. In the top-level, a manager agent 
establishes goals and provides these as an augmentation to the observation of the low-level agents.

The low-level agents use a fully-shared-parameter policy.
"""

import numpy as np
import gymnasium as gym
from typing import Dict, Any, List, Tuple
from eprllib.AgentsConnectors.BaseConnector import BaseConnector
from eprllib.Utils.annotations import override

class HierarchicalTwoLevelsConnector(BaseConnector):
    def __init__(
        self, 
        connector_fn_config: Dict[str, Any] = {}
        ):
        """
        Initializes the hierarchical connector.

        Args:
            connector_fn_config (Dict[str, Any]): Configuration dictionary for the connector function.
                - sub_connector_fn (Callable): The function to create the sub-connector.
                - sub_connector_fn_config (Dict[str, Any]): Configuration for the sub-connector function.
                - top_level_agent (str): The identifier for the top-level agent.
                - top_level_temporal_scale (int): The temporal scale for the top-level agent.
        """
        super().__init__(connector_fn_config)
        
        self.sub_connector_fn: BaseConnector = connector_fn_config["sub_connector_fn"](connector_fn_config["sub_connector_fn_config"])
        self.top_level_agent: str = connector_fn_config["top_level_agent"]
        self.top_level_temporal_scale: int = connector_fn_config["top_level_temporal_scale"]
        
        self.timestep_runner: int = -1
        self.top_level_goal: int | List = None
        self.top_level_obs: Dict[str, Any] = None
        self.top_level_trayectory: Dict[str, List[float | int]] = {}
        
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
        
        if agent != self.top_level_agent: 
            observation_space_pre = self.sub_connector_fn.get_agent_obs_dim(env_config, agent)
            return gym.spaces.Box(float("-inf"), float("inf"), (observation_space_pre.shape[0] + 1, ))
        
        else:
            obs_space_len = 0
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
                obs_space_len += len(env_config['agents_config'][agent]["action"]["actuators"])
        
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
        """
        Set the multiagent observation.

        :param env_config: environment configuration
        :type env_config: Dict[str,Any]
        :param agent_states: agent states
        :type agent_states: Dict[str,Any]
        :param dict_agents_obs: dictionary of agents observations
        :type dict_agents_obs: Dict[str,Any]
        :return: multiagent observation
        :rtype: Dict[str,Any]
        """
        self.timestep_runner += 1
        # Save trayectories for future reward calculations
        for key in agent_states[self.top_level_agent].keys():
            if key not in self.top_level_trayectory.keys():
                self.top_level_trayectory[key] = []
            self.top_level_trayectory[key].append(agent_states[self.top_level_agent][key])
        # add the goal to the infos of the top level agent
        self.top_level_trayectory["goal"] = self.top_level_goal
        
        
        # Send the flat observation to the top_level_agent when the timestep is right or when the episode is ending.
        if self.timestep_runner % self.top_level_temporal_scale == 0 \
            or self.top_level_goal is None \
                or is_last_timestep:
            # Set the agents observation and infos to communicate with the EPEnv.
            self.top_level_obs = {self.top_level_agent: dict_agents_obs[self.top_level_agent]}
            top_level_infos = {self.top_level_agent: self.top_level_trayectory}
            self.top_level_trayectory = {}
            is_lowest_level = False
            return self.top_level_obs, top_level_infos, is_lowest_level
        
        else:
            self.top_level_obs = {self.top_level_agent: dict_agents_obs[self.top_level_agent]}
            return self.set_low_level_obs(
                env_config,
                agent_states,
                dict_agents_obs,
                infos,
                self.top_level_goal
            )
    
    @override(BaseConnector)
    def set_low_level_obs(
        self,
        env_config: Dict[str, Any],
        agent_states: Dict[str,Dict[str,Any]],
        dict_agents_obs: Dict[str,Any],
        infos: Dict[str, Dict[str, Any]],
        goals: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
        """
        Set the multiagent observation.

        :param env_config: environment configuration
        :type env_config: Dict[str,Any]
        :param agent_states: agent states
        :type agent_states: Dict[str,Any]
        :param dict_agents_obs: dictionary of agents observations
        :type dict_agents_obs: Dict[str,Any]
        :return: multiagent observation
        :rtype: Dict[str,Any]
        """
        del dict_agents_obs[self.top_level_agent]
        del infos[self.top_level_agent]
        del agent_states[self.top_level_agent]
        
        dict_agents_obs, infos_agents, is_lowest_level = self.sub_connector_fn.set_top_level_obs(
            env_config,
            agent_states,
            dict_agents_obs,
            infos,
        )
        
        self.top_level_goal = goals
        dict_agents_obs, infos_agents = self.add_goal(dict_agents_obs, infos_agents, self.top_level_goal)
        
        return dict_agents_obs, infos_agents, is_lowest_level
    
    def add_goal(
        self,
        dict_agents_obs: Dict[str,Any],
        infos: Dict[str, Dict[str, Any]],
        goals: Dict[str, Any]
    ) -> Tuple[Dict[str,Any],Dict[str, Dict[str, Any]]]:
        # Add the goal to the observation of all the other agents.
        if type(goals[self.top_level_agent]) == List: # This means a multi-discrete action_space
            if len(dict_agents_obs) != len(goals[self.top_level_agent]):
                raise ValueError("The MultiDiscrete space must contain a goal for each agent.")
            else:
                ix = 0
                for agent in dict_agents_obs.keys():
                    dict_agents_obs[agent] = np.concatenate(
                        (
                            dict_agents_obs[agent],
                            np.array([goals[self.top_level_agent][ix]], dtype='float32')
                        ),
                        dtype='float32'
                    )
                    infos[agent].update({'goal': goals[self.top_level_agent][ix]})
                    ix += 1
                    
        elif type(goals[self.top_level_agent]) in [int, np.int8, np.int32, np.int64]: # This means a discrete action_space
            for agent in dict_agents_obs.keys():
                dict_agents_obs[agent] = np.concatenate(
                    (
                        dict_agents_obs[agent],
                        np.array([goals[self.top_level_agent]], dtype='float32')
                    ),
                    dtype='float32'
                )
                infos[agent].update({'goal': goals[self.top_level_agent]})
        
        else:
            raise ValueError("The action space of the top_level_agent must be Discrete or MultiDiscrete spaces.")
        
        
        return dict_agents_obs, infos
    

class HierarchicalThreeLevelsConnector(BaseConnector):
    def __init__(
        self, 
        connector_fn_config: Dict[str, Any] = {}
        ):
        """
        Initializes the hierarchical connector.

        Args:
            connector_fn_config (Dict[str, Any]): Configuration dictionary for the connector function.
                - middle_level_connector_fn (Callable): The function to create the sub-connector.
                - middle_level_connector_fn_config (Dict[str, Any]): Configuration for the sub-connector function.
                - lower_level_connector_fn (Callable): The function to create the sub-connector.
                - lower_level_connector_fn_config (Dict[str, Any]): Configuration for the sub-connector function.
                - top_level_agent (str): The identifier for the top-level agent.
                - top_level_temporal_scale (int): The temporal scale for the top-level agent.
                - middle_level_agents (List[str]): The list of middle-level agents.
                - middle_level_temporal_scale (int): The temporal scale for the middle-level agents.
                - lower_level_agents (List[str]): The list of lower-level agents.
        """
        super().__init__(connector_fn_config)
        
        self.middle_level_connector_fn: BaseConnector = connector_fn_config["middle_level_connector_fn"](connector_fn_config["middle_level_connector_fn_config"])
        self.lower_level_connector_fn: BaseConnector = connector_fn_config["lower_level_connector_fn"](connector_fn_config["lower_level_connector_fn_config"])
        
        
        self.top_level_agent: str = connector_fn_config["top_level_agent"]
        self.top_level_temporal_scale: int = connector_fn_config["top_level_temporal_scale"]
        
        self.middle_level_agents: List[str] = connector_fn_config["middle_level_agents"]
        self.middle_level_temporal_scale: int = connector_fn_config["middle_level_temporal_scale"]
        
        self.lower_level_agents: List[str] = connector_fn_config["lower_level_agents"]
        
        self.timestep_runner: int = -1
        self.top_level_goal: int | List = None
        self.top_level_obs: Dict[str, Any] = None
        self.top_level_trayectory: Dict[str, List[float | int]] = {}
        self.middle_level_flag = False
        self.middle_level_objectives: Dict[str, Any] = None
        
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
        
        if agent in self.lower_level_agents: 
            observation_space_pre = self.lower_level_connector_fn.get_agent_obs_dim(env_config, agent)
            return gym.spaces.Box(float("-inf"), float("inf"), (observation_space_pre.shape[0] + 1, ))
        
        elif agent in self.middle_level_agents:
            observation_space_pre = self.middle_level_connector_fn.get_agent_obs_dim(env_config, agent)
            return gym.spaces.Box(float("-inf"), float("inf"), (observation_space_pre.shape[0] + env_config["agents_config"][agent]["trigger"]["trigger_fn_config"]["action_space_dim"]*2,))
        
        elif agent == self.top_level_agent:
            obs_space_len = 0
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
                obs_space_len += len(env_config['agents_config'][agent]["action"]["actuators"])
        
            return gym.spaces.Box(float("-inf"), float("inf"), (obs_space_len, ))
        
        else:
            raise ValueError(f"Agent {agent} not found in the environment configuration.")
        
    @override(BaseConnector)
    def set_top_level_obs(
        self,
        env_config: Dict[str, Any],
        agent_states: Dict[str,Dict[str,Any]],
        dict_agents_obs: Dict[str,Any],
        infos: Dict[str, Dict[str, Any]],
        is_last_timestep: bool = False
    ) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]], bool]:
        """
        Set the multiagent observation.

        :param env_config: environment configuration
        :type env_config: Dict[str,Any]
        :param agent_states: agent states
        :type agent_states: Dict[str,Any]
        :param dict_agents_obs: dictionary of agents observations
        :type dict_agents_obs: Dict[str,Any]
        :return: multiagent observation
        :rtype: Dict[str,Any]
        """
        self.timestep_runner += 1
        # Save trayectories for future reward calculations
        for key in agent_states[self.top_level_agent].keys():
            if key not in self.top_level_trayectory.keys():
                self.top_level_trayectory[key] = []
            self.top_level_trayectory[key].append(agent_states[self.top_level_agent][key])
        # add the goal to the infos of the top level agent
        self.top_level_trayectory["goal"] = self.top_level_goal
        
        
        # Send the flat observation to the top_level_agent when the timestep is right or when the episode is ending.
        if self.timestep_runner % self.top_level_temporal_scale == 0 \
            or self.top_level_goal is None \
                or is_last_timestep:
            # Set the agents observation and infos to communicate with the EPEnv.
            self.top_level_obs = {self.top_level_agent: dict_agents_obs[self.top_level_agent]}
            top_level_infos = {self.top_level_agent: self.top_level_trayectory}
            self.top_level_trayectory = {}
            is_lowest_level = False
            return self.top_level_obs, top_level_infos, is_lowest_level
        
        else:
            self.top_level_obs = {self.top_level_agent: dict_agents_obs[self.top_level_agent]}
            return self.set_low_level_obs(
                env_config,
                agent_states,
                dict_agents_obs,
                infos,
                self.top_level_goal
            )
    
    @override(BaseConnector)
    def set_low_level_obs(
        self,
        env_config: Dict[str, Any],
        agent_states: Dict[str,Dict[str,Any]],
        dict_agents_obs: Dict[str,Any],
        infos: Dict[str, Dict[str, Any]],
        goals: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
        """
        Set the multiagent observation.

        :param env_config: environment configuration
        :type env_config: Dict[str,Any]
        :param agent_states: agent states
        :type agent_states: Dict[str,Any]
        :param dict_agents_obs: dictionary of agents observations
        :type dict_agents_obs: Dict[str,Any]
        :return: multiagent observation
        :rtype: Dict[str,Any]
        """
        del dict_agents_obs[self.top_level_agent]
        del infos[self.top_level_agent]
        del agent_states[self.top_level_agent]
        
        if not self.middle_level_flag:
            
            for agent in self.middle_level_agents:
                del dict_agents_obs[agent]
                del infos[agent]
                del agent_states[agent]
        
            dict_agents_obs, infos_agents, is_lowest_level = self.middle_level_connector_fn.set_top_level_obs(
                env_config,
                agent_states,
                dict_agents_obs,
                infos,
            )
            self.middle_level_flag = True
            self.top_level_goal = goals
            dict_agents_obs, infos_agents = self.add_goal(dict_agents_obs, infos_agents, self.top_level_goal)
        
        else:
            
            for agent in self.lower_level_agents:
                del dict_agents_obs[agent]
                del infos[agent]
                del agent_states[agent]
                
            dict_agents_obs, infos_agents, is_lowest_level = self.lower_level_connector_fn.set_top_level_obs(
                env_config,
                agent_states,
                dict_agents_obs,
                infos,
            )
            self.middle_level_flag = False
            self.middle_level_objectives = goals
            dict_agents_obs, infos_agents = self.add_objectives(dict_agents_obs, infos_agents, self.middle_level_objectives)
        
        return dict_agents_obs, infos_agents, is_lowest_level
    
    def add_goal(
        self,
        dict_agents_obs: Dict[str,Any],
        infos: Dict[str, Dict[str, Any]],
        goals: Dict[str, Any]
    ) -> Tuple[Dict[str,Any],Dict[str, Dict[str, Any]]]:
        # Add the goal to the observation of all the other agents.
        if type(goals[self.top_level_agent]) == List: # This means a multi-discrete action_space
            if len(dict_agents_obs) != len(goals[self.top_level_agent]):
                raise ValueError("The MultiDiscrete space must contain a goal for each agent.")
            else:
                ix = 0
                for agent in dict_agents_obs.keys():
                    dict_agents_obs[agent] = np.concatenate(
                        (
                            dict_agents_obs[agent],
                            np.array([goals[self.top_level_agent][ix]], dtype='float32')
                        ),
                        dtype='float32'
                    )
                    infos[agent].update({'goal': goals[self.top_level_agent][ix]})
                    ix += 1
                    
        elif type(goals[self.top_level_agent]) in [int, np.int8, np.int32, np.int64]: # This means a discrete action_space
            for agent in dict_agents_obs.keys():
                dict_agents_obs[agent] = np.concatenate(
                    (
                        dict_agents_obs[agent],
                        np.array([goals[self.top_level_agent]], dtype='float32')
                    ),
                    dtype='float32'
                )
                infos[agent].update({'goal': goals[self.top_level_agent]})
        
        else:
            raise ValueError("The action space of the top_level_agent must be Discrete or MultiDiscrete spaces.")
        
        
        return dict_agents_obs, infos
    
    def add_objectives(
        self,
        dict_agents_obs: Dict[str,Any],
        infos: Dict[str, Dict[str, Any]],
        objectives: Dict[str, List[int | float]]
        ) -> Tuple[Dict[str,np.ndarray], Dict[str, Dict[str, Any]]]:
        """
        Processes a dictionary of agent objectives to generate a combined vector.
        
        Steps:
            1. Converts the dictionary into a matrix.
            2. Verifies that all agents have vectors of the same length.
            3. Computes the transpose of the matrix.
            4. Multiplies the transpose by the original matrix.
            5. Computes the column-wise mean of the resulting matrix.
            6. Extracts the diagonal of the resulting matrix.
            7. Integrates both vectors into a single concatenated vector.
            8. Concatenates the concatenated vector to the agent observations.
        
        Parameters:
            dict_agents_obs (dict): Dictionary with agent names as keys and observation vectors as values.
            infos (dict): Dictionary with agent names as keys and information dictionaries as values.
            objectives (dict): Dictionary with agent names as keys and objective vectors as values.
        
        Returns:
            dict_agents_obs (dict): Dictionary with agent names as keys and concatenated vectors as values.
            infos (dict): Dictionary with agent names as keys and information dictionaries as values.
        """
        # Convert the dictionary into a list of lists
        matrix = np.array(list(objectives.values()))
        
        # Verify that all lists have the same length
        if not all(len(vec) == len(matrix[0]) for vec in matrix):
            raise ValueError("All agents must have vectors of the same length.")
        
        # Compute the transpose of the matrix
        matrix_T = matrix.T
        
        # Multiply the transpose by the original matrix
        multiplied_matrix = np.dot(matrix_T, matrix)
        
        # Compute the column-wise mean
        column_means = np.mean(multiplied_matrix, axis=0)
        
        # Extract the diagonal
        diagonal_values = np.diag(multiplied_matrix)
        
        # Integrate both vectors into a single concatenated vector
        final_vector = np.concatenate((diagonal_values, column_means))
        
        # Concatenate the concatenated vector to the agent observations
        for agent in dict_agents_obs.keys():
            dict_agents_obs[agent] = np.concatenate(
                (
                    dict_agents_obs[agent],
                    final_vector
                ),
                dtype='float32'
            )
            infos[agent].update({'objective': final_vector})
            
        return dict_agents_obs, infos
