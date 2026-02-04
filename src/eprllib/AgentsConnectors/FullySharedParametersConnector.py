"""
Fully Shared Parameters Agents Connector
=========================================

This module implements a fully shared parameters policy for the observation function.

The flat observation for each agent must be a numpy array and the size must be equal for all agents. To 
achieve this, a concatenation of variables is performed, which includes:

- AgentID one-hot encoded vector
- Agent state
- Actuators state (including other agents)

Notes:
------
- The number of actuators for other agents is variable. Consider changing the actuator state for agent action or 
  finding a way to expand the observation space to adapt to different observation sizes.
"""
import time
import numpy as np
from gymnasium.spaces import Box
from typing import Any, Dict, Tuple, Optional
from eprllib.AgentsConnectors.BaseConnector import BaseConnector
from eprllib.Utils.annotations import override
import eprllib.Utils.observation_utils as observation_utils
from eprllib.Utils.connector_utils import (
    set_variables_in_obs,
    set_internal_variables_in_obs,
    set_meters_in_obs,
    set_simulation_parameters_in_obs,
    set_zone_simulation_parameters_in_obs,
    set_prediction_variables_in_obs,
    set_other_obs_in_obs,
    set_user_occupation_forecast_in_obs
    )
from eprllib import logger

class FullySharedParametersConnector(BaseConnector):
    def __init__(
        self,
        connector_fn_config: Dict[str, Any]
    ):
        """
        This class implements a fully shared parameters policy for the observation function.

        Args:
            connector_fn_config (Dict[str, Any]): The configuration dictionary for the observation function.
            This must contain the key 'number_of_agents_total', which represents the maximum
            quantity for which the policy is prepared. It is related to the unitary vector.
        """
        super().__init__(connector_fn_config)
        self.number_of_agents_total: int = connector_fn_config['number_of_agents_total']
        self.number_of_actuators_total: int = connector_fn_config['number_of_actuators_total']
        self.agent_ids: Optional[Dict[str, Optional[int]]] = None
        self.actuator_ids: Optional[Dict[str, Optional[int]]] = None
    
    @override(BaseConnector)
    def get_agent_obs_dim(
        self,
        env_config: Dict[str,Any],
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
        agent_list = [key for key in env_config["agents_config"].keys()]
        obs_space_len: int = 0
        self.obs_indexed[agent] = {}
        
        if len(agent_list) > self.number_of_agents_total:
            raise ValueError("The number of agents must be greater than the number of agents in the environment configuration.")
        
        if self.number_of_agents_total > 1:
            agents_used = self.number_of_agents_total
            for agent_name in agent_list:
                self.obs_indexed[agent][agent_name] = obs_space_len
                obs_space_len += 1
                agents_used -= 1
            # Add the rest of the agents as zero vectors
            if agents_used > 0:
                for i in range(agents_used):
                    self.obs_indexed[agent][f"not_used_agent_{i+1}"] = obs_space_len
                    obs_space_len += 1
        assert obs_space_len == len(self.obs_indexed[agent]), f"The observation space length must be equal to the number of indexed observations. Obs indexed:{len(self.obs_indexed[agent])} != Obs space len:{obs_space_len}."
        self.obs_indexed[agent], obs_space_len = set_variables_in_obs(env_config, agent, self.obs_indexed[agent], obs_space_len)
        self.obs_indexed[agent], obs_space_len = set_internal_variables_in_obs(env_config, agent, self.obs_indexed[agent], obs_space_len)
        self.obs_indexed[agent], obs_space_len = set_meters_in_obs(env_config, agent, self.obs_indexed[agent], obs_space_len)
        self.obs_indexed[agent], obs_space_len = set_simulation_parameters_in_obs(env_config, agent, self.obs_indexed[agent], obs_space_len)
        self.obs_indexed[agent], obs_space_len = set_zone_simulation_parameters_in_obs(env_config, agent, self.obs_indexed[agent], obs_space_len)
        self.obs_indexed[agent], obs_space_len = set_prediction_variables_in_obs(env_config, agent, self.obs_indexed[agent], obs_space_len)
        self.obs_indexed[agent], obs_space_len = set_other_obs_in_obs(env_config, agent, self.obs_indexed[agent], obs_space_len)
        self.obs_indexed[agent], obs_space_len = set_user_occupation_forecast_in_obs(env_config, agent, self.obs_indexed[agent], obs_space_len)
        
        # Add the actuator state to all the agents obs_space_len
        # Check if at least one agent request the use_actuator_state
        use_actuator_state_flag = False
        for agent_name in agent_list:
            if env_config["agents_config"][agent_name]["observation"]['use_actuator_state']:
                use_actuator_state_flag = True
                break
        if use_actuator_state_flag:
            # Save the total amount of actuators in the environment to create a vector in set_agent_obs
            actuator_used = self.number_of_actuators_total
            for agent_name in agent_list:
                for actuator in range(len(env_config["agents_config"][agent_name]["action"]['actuators'])):
                    actuator_component_type = env_config["agents_config"][agent_name]["action"]['actuators'][actuator][0]
                    actuator_control_type = env_config["agents_config"][agent_name]["action"]['actuators'][actuator][1]
                    actuator_key = env_config["agents_config"][agent_name]["action"]['actuators'][actuator][2]
                    self.obs_indexed[agent][observation_utils.get_actuator_name(
                        agent,
                        actuator_component_type,
                        actuator_control_type,
                        actuator_key
                        )] = obs_space_len
                    obs_space_len += 1
                    actuator_used -= 1
            # Add the rest of the actuators as zero vectors
            if actuator_used > 0:
                for i in range(actuator_used):
                    self.obs_indexed[agent][f"not_used_actuator_{i+1}"] = obs_space_len
                    obs_space_len += 1
                    
            # chack that actuators is equal or minor to self.number_of_actuators_total
            if actuator_used < 0:
                logger.error("The total amount of actuators in the environment is greater than the number of actuators in the environment configuration.")
        
        assert obs_space_len > 0, "The observation space length must be greater than 0."
        assert len(self.obs_indexed[agent]) == obs_space_len, f"The observation space length must be equal to the number of indexed observations. Obs indexed:{len(self.obs_indexed)} != Obs space len:{obs_space_len}. The agent {agent} has the following indexed observations: {self.obs_indexed[agent]}."
        # obs_space_len += 1
        logger.debug(f"Observation space length for agent {agent}: {obs_space_len}")
        
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
        if self.obs_indexed == {}:
            self.get_agent_obs_dim(env_config, agent)
        return self.obs_indexed[agent]
    
    @override(BaseConnector)
    def set_top_level_obs(
        self,
        env_config: Dict[str, Any],
        agent_states: Dict[str,Dict[str,Any]],
        dict_agents_obs: Dict[str,Any],
        infos: Dict[str, Dict[str, Any]],
        is_last_timestep: bool = False
    ) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]], bool]:
        """This method construct the observation of the environment. 
        The observation is a concatenation of the following variables:
        - agent indicator
        - agent state
        - actuator state
        - internal variables
        - meters
        - simulation parameters
        - zone simulation parameters
        - weather prediction
        - other observation

        Args:
            env_config (Dict[str, Any]): Dict of an environment config builded.
            agent_states (Dict[str,Dict[str,Any]]): The agent actual states before filter.
            dict_agents_obs (Dict[str,Any]): The agent actual states after filter with ``ObservationFunction``.
            infos (Dict[str, Dict[str, Any]]): Agent infos dict.
            is_last_timestep (bool, optional): Flag to indicate the last timestep before finish the RunPeriod. Defaults to False.

        Raises:
            ValueError: _description_
            ValueError: _description_

        Returns:
            Tuple[Dict[str, Any], Dict[str, Dict[str, Any]], bool]: Returns a Tuple with the agents observations addapted to
            fully shared parameter policy, the infos dict updated, and a flag to tell that is the lowest-level agent (that means
            that the actions obtained after apply this observation into the policy shall be implemened in an actuator after 
            trigger the policy action in the ActionFunction of the agent.)
        """
        # Check that the lenght of all the dict_agents_obs key values are the same:
        if len(set([len(value) for value in dict_agents_obs.values()])) != 1:
            msg = "The lenght of all the dict_agents_obs key values must be the same when you use homogeneous fully shared parameters policy."
            logger.error(msg)
            raise ValueError(msg)
        
        # Add the ID vectors if it's needed
        if self.agent_ids is None:
            id = 0
            self.agent_ids = {agent: None for agent in env_config['agents_config'].keys()}
            for agent in env_config['agents_config'].keys():
                self.agent_ids.update({agent: id})
                id += 1
                if len(self.agent_ids) > self.number_of_agents_total:
                    msg = f"The agents found were: {env_config['agents_config'].keys()} with a total of {len(env_config['agents_config'].keys())}, that are greather than {self.number_of_agents_total}."
                    logger.error(msg)
                    raise ValueError(msg)

            if len(self.agent_ids) > self.number_of_agents_total:
                msg = f"The agents found were: {env_config['agents_config'].keys()} with a total of {len(env_config['agents_config'].keys())}, that are greather than {self.number_of_agents_total}."
                logger.error(msg)
                raise ValueError(msg)
        
        if self.actuator_ids is None:
            id = 0
            self.actuator_ids = {}
            for agent in env_config['agents_config'].keys():
                for actuator_config in env_config["agents_config"][agent]["action"]["actuators"]:
                    self.actuator_ids.update({observation_utils.get_actuator_name(agent,actuator_config[0],actuator_config[1],actuator_config[2]): id})
                    id += 1
                    if len(self.actuator_ids) > self.number_of_actuators_total:
                        msg = f"The actuators found were: {self.actuator_ids.keys()} with a total of {len(self.actuator_ids.keys())}, that are greather than {self.number_of_actuators_total}."
                        logger.error(msg)
                        raise ValueError(msg)
        
        # agents in this timestep
        agent_list = [key for key in dict_agents_obs.keys()]
        # Add agent indicator for the observation for each agent
        actuator_names: Dict[str, Dict[str, Any]] = {agent: {} for agent in agent_list}
        
        for agent in agent_list:
            # Remove from agent_states and save the actuator items.
            for actuator_config in env_config["agents_config"][agent]["action"]["actuators"]:
                actuator_name = observation_utils.get_actuator_name(agent,actuator_config[0],actuator_config[1],actuator_config[2])
                actuator_names[agent].update({actuator_name: agent_states[agent].get(actuator_name, -2)})
                if actuator_names[agent][actuator_name] == -2:
                    logger.info(f"Looking for actuator: {actuator_name}")
                    logger.info(f"Available keys in agent_states[{agent}]: {agent_states[agent].keys()}")
                    time.sleep(10)
                    
            
            # Agent properties
            agent_id_vector = None
            
            # === AgentID one-hot enconde vector ===
            # If apply, add the igent ID vector for each agent obs
            if self.number_of_agents_total > 1:
                agent_id_vector = np.array([0]*self.number_of_agents_total)
                agent_id_vector[self.agent_ids[agent]] = 1
                
            
            if agent_id_vector is not None:
                dict_agents_obs[agent] = np.concatenate(
                    (
                        agent_id_vector,
                        dict_agents_obs[agent]
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
                dict_agents_obs[agent] = np.concatenate(
                    (
                        dict_agents_obs[agent],
                        actuator_id_vector,
                    ),
                    dtype='float32'
                )
            
        return dict_agents_obs, infos, True
