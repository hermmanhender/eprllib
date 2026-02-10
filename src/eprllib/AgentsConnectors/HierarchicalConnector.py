"""
Hierarchical Agents Connector
=============================

This module implements a hierarchical connector with two levels of hierarchy. In the top-level, a manager agent 
establishes goals and provides these as an augmentation to the observation of the low-level agents.

The low-level agents use a fully-shared-parameter policy.
"""

import numpy as np
from numpy.typing import NDArray
from numpy import float32
from gymnasium.spaces import Box
from gymnasium import Space
from typing import Dict, Any, List, Tuple, Optional
from eprllib.AgentsConnectors.BaseConnector import BaseConnector
from eprllib.Utils.annotations import override
from eprllib.Utils.connector_utils import (
    set_variables_in_obs,
    set_internal_variables_in_obs,
    set_meters_in_obs,
    set_simulation_parameters_in_obs,
    set_zone_simulation_parameters_in_obs,
    set_prediction_variables_in_obs,
    set_other_obs_in_obs,
    set_actuators_in_obs,
    set_user_occupation_forecast_in_obs
    )
from eprllib import logger

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
        self.top_level_goal: Optional[Dict[str, Any]] = None
        self.top_level_obs: Optional[Dict[str, Any]] = None
        self.top_level_trayectory: Dict[str, Any] = {}
        self.obs_indexed_top_level: Dict[str, Dict[str, int]] = {}
        self.obs_indexed_low_level: Dict[str, Dict[str, int]] = {}
        
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
        
        if agent != self.top_level_agent: 
            observation_space_pre: Space[Any]|Box = self.sub_connector_fn.get_agent_obs_dim(env_config, agent)
            self.obs_indexed_low_level = self.sub_connector_fn.obs_indexed
            obs_indexed_len = len(self.obs_indexed)
            self.obs_indexed_low_level[agent]["goal"] = obs_indexed_len
            
            assert type(observation_space_pre) == Box, "The observation space must not be None."
            
            return Box(float("-inf"), float("inf"), (observation_space_pre.shape[0] + 1, ))
        
        else:
            obs_space_len: int = 0
            self.obs_indexed[agent] = {}
        
            self.obs_indexed[agent], obs_space_len = set_variables_in_obs(env_config, agent, self.obs_indexed[agent])
            self.obs_indexed[agent], obs_space_len = set_internal_variables_in_obs(env_config, agent, self.obs_indexed[agent])
            self.obs_indexed[agent], obs_space_len = set_meters_in_obs(env_config, agent, self.obs_indexed[agent])
            self.obs_indexed[agent], obs_space_len = set_simulation_parameters_in_obs(env_config, agent, self.obs_indexed[agent], obs_space_len)
            self.obs_indexed[agent], obs_space_len = set_zone_simulation_parameters_in_obs(env_config, agent, self.obs_indexed[agent])
            self.obs_indexed[agent], obs_space_len = set_prediction_variables_in_obs(env_config, agent, self.obs_indexed[agent])
            self.obs_indexed[agent], obs_space_len = set_other_obs_in_obs(env_config, agent, self.obs_indexed[agent])
            self.obs_indexed[agent], obs_space_len = set_actuators_in_obs(env_config, agent, self.obs_indexed[agent])
            self.obs_indexed[agent], obs_space_len = set_user_occupation_forecast_in_obs(env_config, agent, self.obs_indexed[agent], obs_space_len)
        
            assert obs_space_len > 0, "The observation space length must be greater than 0."
            assert len(self.obs_indexed_top_level[agent]) == obs_space_len, "The observation space length must be equal to the number of indexed observations."
            # obs_space_len += 1
            logger.debug(f"HierarchicalTwoLevelsConnector: Observation space length for agent {agent}: {obs_space_len}")
        
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
        if agent != self.top_level_agent:
            return self.obs_indexed_low_level[agent]
        else:
            return self.obs_indexed_top_level[agent]
        
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
                msg = "HierarchicalTwoLevelsConnector: The MultiDiscrete space must contain a goal for each agent."
                logger.error(msg)
                raise ValueError(msg)
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
            msg = "HierarchicalTwoLevelsConnector: The action space of the top_level_agent must be Discrete or MultiDiscrete spaces."
            logger.error(msg)
            raise ValueError(msg)
        
        
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
        self.top_level_goal: Optional[Dict[str, Any]] = None
        self.top_level_obs: Optional[Dict[str, Any]] = None
        self.top_level_trayectory: Dict[str, Any] = {}
        self.middle_level_flag = False
        self.middle_level_objectives: Optional[Dict[str, Any]] = None
        self.obs_indexed_top_level: Dict[str, Dict[str, int]] = {}
        self.obs_indexed_low_level: Dict[str, Dict[str, int]] = {}
        self.obs_indexed_middle_level: Dict[str, Dict[str, int]] = {}
        
    @override(BaseConnector)
    def get_agent_obs_dim(
        self,
        env_config: Dict[str, Any],
        agent: Optional[str] = None
    ) -> Space[Any]:
        """
        Construct the observation space of the environment.

        Args:
            env_config (Dict[str, Any]): The environment configuration dictionary.
            agent (str, optional): The agent identifier.

        Returns:
            gym.Space: The observation space of the environment.
        """
        
        if agent in self.lower_level_agents:  
            observation_space_pre: Space[Any]|Box = self.lower_level_connector_fn.get_agent_obs_dim(env_config, agent)
            self.obs_indexed_low_level[agent] = self.lower_level_connector_fn.obs_indexed[agent]
            obs_indexed_len = len(self.obs_indexed)
            self.obs_indexed_low_level[agent]["goal"] = obs_indexed_len
            
            assert type(observation_space_pre) == Box, "The observation space must not be None."
            
            return Box(float("-inf"), float("inf"), (observation_space_pre.shape[0] + 1, ))
        
        elif agent in self.middle_level_agents:
            observation_space_pre: Space[Any]|Box = self.middle_level_connector_fn.get_agent_obs_dim(env_config, agent)
            self.obs_indexed_middle_level = self.middle_level_connector_fn.obs_indexed
            obs_indexed_len = len(self.obs_indexed)
            self.obs_indexed_middle_level[agent]["goal"] = obs_indexed_len
            
            assert type(observation_space_pre) == Box, "The observation space must not be None."
            
            return Box(float("-inf"), float("inf"), (observation_space_pre.shape[0] + 1, ))
        
        elif agent == self.top_level_agent:
            obs_space_len: int = 0

            assert agent != None, "The agent identifier must not be None."
            
            self.obs_indexed[agent], obs_space_len = set_variables_in_obs(env_config, agent, self.obs_indexed[agent])
            self.obs_indexed[agent], obs_space_len = set_internal_variables_in_obs(env_config, agent, self.obs_indexed[agent])
            self.obs_indexed[agent], obs_space_len = set_meters_in_obs(env_config, agent, self.obs_indexed[agent])
            self.obs_indexed[agent], obs_space_len = set_simulation_parameters_in_obs(env_config, agent, self.obs_indexed[agent], obs_space_len)
            self.obs_indexed[agent], obs_space_len = set_zone_simulation_parameters_in_obs(env_config, agent, self.obs_indexed[agent])
            self.obs_indexed[agent], obs_space_len = set_prediction_variables_in_obs(env_config, agent, self.obs_indexed[agent])
            self.obs_indexed[agent], obs_space_len = set_other_obs_in_obs(env_config, agent, self.obs_indexed[agent])
            self.obs_indexed[agent], obs_space_len = set_actuators_in_obs(env_config, agent, self.obs_indexed[agent])
            self.obs_indexed[agent], obs_space_len = set_user_occupation_forecast_in_obs(env_config, agent, self.obs_indexed[agent], obs_space_len)
            
            assert obs_space_len > 0, "The observation space length must be greater than 0."
            assert len(self.obs_indexed_top_level[agent]) == obs_space_len, "The observation space length must be equal to the number of indexed observations."
            # obs_space_len += 1
            logger.debug(f"HierarchicalThreeLevelsConnector: Observation space length for agent {agent}: {obs_space_len}")
        
            return Box(float("-inf"), float("inf"), (obs_space_len, ))
        
        else:
            msg = f"HierarchicalThreeLevelsConnector: Agent {agent} not found in the environment configuration."
            logger.error(msg)
            raise ValueError(msg)
        
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
        if agent in self.lower_level_agents:
            return self.obs_indexed_low_level[agent]
        elif agent in self.middle_level_agents:
            return self.obs_indexed_middle_level[agent]
        else:
            return self.obs_indexed_top_level[agent]
        
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
                msg = "HierarchicalThreeLevelsConnector: The MultiDiscrete space must contain a goal for each agent."
                logger.error(msg)
                raise ValueError(msg)
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
            msg = "HierarchicalThreeLevelsConnector: The action space of the top_level_agent must be Discrete or MultiDiscrete spaces."
            logger.error(msg)
            raise ValueError(msg)
        
        
        return dict_agents_obs, infos
    
    def add_objectives(
        self,
        dict_agents_obs: Dict[str,Any],
        infos: Dict[str, Dict[str, Any]],
        objectives: Dict[str, List[int | float]]
        ) -> Tuple[Dict[str,NDArray[float32]], Dict[str, Dict[str, Any]]]:
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
            msg = "HierarchicalThreeLevelsConnector: All agents must have vectors of the same length."
            logger.error(msg)
            raise ValueError(msg)
        
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
