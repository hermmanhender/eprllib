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

from typing import Any, Dict, Tuple
from eprllib.AgentsConnectors.BaseConnector import BaseConnector
from eprllib.Utils.annotations import override
from eprllib.Utils.observation_utils import get_actuator_name

import numpy as np
import gymnasium as gym

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
        self.agent_ids: Dict[str, int] = None
        self.actuator_ids = None
    
    @override(BaseConnector)
    def get_agent_obs_dim(
        self,
        env_config: Dict[str,Any],
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
            raise ValueError("The lenght of all the dict_agents_obs key values must be the same when you use homogeneous fully shared parameters policy.")
        
        # Add the ID vectors if it's needed
        if self.agent_ids is None:
            
            self.agent_ids = {agent: env_config['agents_config'][agent]["agent_id"] for agent in env_config['agents_config'].keys()}

            if len(self.agent_ids) > self.number_of_agents_total:
                print(f"The agents found were: {env_config['agents_config'].keys()} with a total of {len(env_config['agents_config'].keys())}, that are greather than {self.number_of_agents_total}.")
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
                
            
            if agent_id_vector is not None and any(agent_states[agent].values()):
                agents_obs[agent] = np.concatenate(
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
