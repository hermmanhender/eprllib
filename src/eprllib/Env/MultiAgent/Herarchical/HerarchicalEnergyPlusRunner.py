"""
EnergyPlus Runner
==================

This script contain the EnergyPlus Runner that execute EnergyPlus from its 
Python API in the version 24.2.0.
"""
import numpy as np
from queue import Queue
from typing import Any, Dict, List
from ctypes import c_void_p
from ray.rllib.utils.annotations import override
from eprllib.Env.MultiAgent.EnergyPlusRunner import EnergyPlusRunner
from eprllib.ActionFunctions.ActionFunctions import ActionFunction
from eprllib.ObservationFunctions.ObservationFunctions import ObservationFunction
from eprllib.Utils.env_config_utils import EP_API_add_path

# EnergyPlus Python API path adding
EP_API_add_path(version="24-2-0")
from pyenergyplus.api import EnergyPlusAPI

api = EnergyPlusAPI()

class HerarchicalEnergyPlusRunner(EnergyPlusRunner):
    """
    """
    def __init__(
        self,
        env_config: Dict[str, Any],
        episode: int,
        obs_queue: Queue,
        act_queue: Queue,
        infos_queue: Queue,
        agents: List,
        observation_fn: ObservationFunction,
        action_fn: Dict[str, ActionFunction],
        ) -> None:
        """
        The object has an intensive interaction with EnergyPlus Environment script, exchange information
        between two threads. For a good coordination queue events are stablished and different canals of
        information are defined.

        Args:
            episode (int): Episode number.
            env_config (Dict[str, Any]): Environment configuration defined in the call to the EnergyPlus Environment.
            obs_queue (Queue): Queue object definition.
            act_queue (Queue): Queue object definition.
            infos_queue (Queue): Queue object definition.
        
        Return:
            None.
        """
        super.__init__(
            env_config,
            episode,
            obs_queue,
            act_queue,
            infos_queue,
            agents,
            observation_fn,
            action_fn
        )
        
        self.top_level_agent_name: str = env_config["top_level_agent_name"]
        self.top_level_temporal_scale: int = env_config["top_level_temporal_scale"]
        self.timestep_runner: int = 0
        self.top_level_goal: int|List = None
    
    @override
    def _collect_obs(
        self,
        state_argument: c_void_p
        ) -> None:
        """
        EnergyPlus callback that collects output variables, meters and actuator actions
        values and enqueue them to the EnergyPlus Environment thread.
        """
        # To not perform observations when the callbacks and the 
        # warming period are not complete.
        if not self._init_callback(state_argument) or self.simulation_complete:
            return
        
        dict_agents_obs = {agent: {} for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        
        # Agents observe: site state, thermal zone state (only the one that it belong), specific object variables 
        # and meters, and others parameters assigned as True in the env_config.observation object.
        # Get the state of the actuators.
        agent_states: Dict[str, Dict[str, Any]] = {agent: {} for agent in self.agents}
        for agent in self.agents:
            agent_states[agent].update(self.get_variables_state(state_argument, agent))
            agent_states[agent].update(self.get_internal_variables_state(state_argument, agent))
            agent_states[agent].update(self.get_meters_state(state_argument, agent))
            agent_states[agent].update(self.get_simulation_parameters_values(state_argument, agent))
            agent_states[agent].update(self.get_zone_simulation_parameters_values(state_argument, agent))
            agent_states[agent].update(self.get_weather_prediction(state_argument, agent))
            agent_states[agent].update(self.get_actuators_state(state_argument, agent))
            agent_states[agent].update(self.get_other_obs(self.env_config, agent))
        
        # Send the flat observation to the top_level_agent when the timestep is right or when the episode is ending.
        if self.top_level_temporal_scale % self.timestep_runner == 0:
            # Set the agents observation and infos to communicate with the EPEnv.
            self.obs_queue.put({self.top_level_agent_name: np.array(list(agent_states[self.top_level_agent_name].values()))})
            self.obs_event.set()
            self.infos_queue.put({self.top_level_agent_name: self.infos[self.top_level_agent_name]})
            self.infos_event.set()
            # Wait for a goal selection
            event_flag = self.act_event.wait(self.env_config["timeout"])
            if not event_flag:
                print(f"The time waiting for a goal from the {self.top_level_agent_name} agent was surpased.")
                return
            # Get the action from the EnergyPlusEnvironment `step` method.
            self.top_level_goal = self.act_queue.get()
        
            
        # Delete the top_level_agent observation from the agent_states
        del agent_states[self.top_level_agent_name]
        # Add the goal to the observation of all the other agents.
        if self.top_level_goal is None:
            raise ValueError("The top_level_agent must be called in the first timestep.")
        
        if type(self.top_level_goal) == List: # This means a multi-discrete action_space
            if len(agent_states) != len(self.top_level_goal):
                raise ValueError("The MultiDiscrete space must contain a Discrete sub-space for each agent.")
            else:
                ix = 0
                for agent in agent_states.keys():
                    agent_states[agent].update({f"{agent}_goal": self.top_level_goal[ix]})
                    ix += 1
        if type(self.top_level_goal) == int: # This means a discrete action_space
            for agent in agent_states.keys():
                agent_states[agent].update({f"{agent}_goal": self.top_level_goal})
        else:
            raise ValueError("The action space of the top_level_agent must be Discrete or MultiDiscrete spaces.")
        
        
        dict_agents_obs = self.observation_fn.set_agent_obs(
            self.env_config,
            agent_states
        )
        
        # Set the agents observation and infos to communicate with the EPEnv.
        self.obs_queue.put(dict_agents_obs)
        self.obs_event.set()
        self.infos_queue.put(self.infos)
        self.infos_event.set()
        
        self.timestep_runner += 1
