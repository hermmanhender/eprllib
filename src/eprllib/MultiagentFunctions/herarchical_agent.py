"""
Herarchica agent implementation
================================

Two levels of herarchy are used. In the top-level, a manager agent establish goals
and gives this as an augmentation to the observation of the low-level agents.

The low level agents use a fully-shared-parameter policy.
"""
import numpy as np
import gymnasium as gym
from typing import Dict, Any, List, Tuple
from eprllib.MultiagentFunctions.MultiagentFunctions import MultiagentFunction
from eprllib.Utils.annotations import override

class herarchical_agent(MultiagentFunction):
    def __init__(
        self, 
        multiagent_fn_config: Dict[str, Any] = {}
        ):
        super().__init__(multiagent_fn_config)
        
        self.sub_multiagent_fn: MultiagentFunction = multiagent_fn_config["sub_multiagent_fn"](multiagent_fn_config["sub_multiagent_fn_config"])
        self.top_level_agent: str = multiagent_fn_config["top_level_agent"]
        self.top_level_temporal_scale: int = multiagent_fn_config["top_level_temporal_scale"]
        
        self.timestep_runner: int = 0
        self.top_level_goal: int|List = None
        self.top_level_trayectory: Dict[str,List[float|int]] = {}
        
    @override(MultiagentFunction)
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
        # Get the observation space dict.
        observation_space: Dict[str, gym.spaces.Box] = self.sub_multiagent_fn.get_agent_obs_dim(env_config) # this is a dictionary with gymnasium Discrete spaces inside for each agent.
        # increase the dimension for each Discrete space in 1.
        for agent in observation_space.keys():
            observation_space[agent] = gym.spaces.Box(float("-inf"), float("inf"), (observation_space[agent].shape + 1, ))
        
        return observation_space
        
    @override(MultiagentFunction)
    def set_top_level_obs(
        self,
        env_config: Dict[str, Any],
        agent_states: Dict[str,Dict[str,Any]],
        dict_agents_obs: Dict[str,Any],
        infos: Dict[str, Dict[str, Any]]
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
        # Save trayectories for future reward calculations
        for key in agent_states[self.top_level_agent].keys():
            if key not in self.top_level_trayectory.keys():
                self.top_level_trayectory[key] = []
            self.top_level_trayectory[key].append(agent_states[self.top_level_agent][key])
        
        # Send the flat observation to the top_level_agent when the timestep is right or when the episode is ending.
        if self.top_level_temporal_scale % self.timestep_runner == 0:
            # Set the agents observation and infos to communicate with the EPEnv.
            top_level_obs = {self.top_level_agent: np.array(list(agent_states[self.top_level_agent].values()))}
            top_level_infos = {self.top_level_agent: self.top_level_trayectory}
            self.top_level_trayectory = {}
            self.timestep_runner += 1
            return top_level_obs, top_level_infos, False
        else:
            del agent_states[self.top_level_agent]
            del infos[self.top_level_agent]
            return dict_agents_obs, infos, False
    
    @override(MultiagentFunction)
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
        del agent_states[self.top_level_agent]
        del infos[self.top_level_agent]
        
        # Add the goal to the observation of all the other agents.
        if type(goals[self.top_level_agent]) == List: # This means a multi-discrete action_space
            if len(agent_states) != len(goals[self.top_level_agent]):
                raise ValueError("The MultiDiscrete space must contain a goal for each agent.")
            else:
                ix = 0
                for agent in agent_states.keys():
                    agent_states[agent].update({f"{agent}_goal": goals[self.top_level_agent][ix]})
                    ix += 1
        if type(goals[self.top_level_agent]) == int: # This means a discrete action_space
            for agent in agent_states.keys():
                agent_states[agent].update({f"{agent}_goal": goals[self.top_level_agent]})
        else:
            raise ValueError("The action space of the top_level_agent must be Discrete or MultiDiscrete spaces.")
        
        dict_agents_obs, infos_agents, is_lowest_level = self.sub_multiagent_fn.set_top_level_obs(
            env_config,
            agent_states,
            dict_agents_obs,
            infos,
        )
        
        return dict_agents_obs, infos_agents, is_lowest_level
    