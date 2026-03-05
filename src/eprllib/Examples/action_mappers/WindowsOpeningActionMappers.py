"""
Windows Opening ActionMappers
================================

This module contains classes to implement window opening ActionMappers for controlling actuators in the environment.
"""
import gymnasium as gym
import numpy as np
from typing import Any, Dict, List, Tuple, Optional
from eprllib.Agents.ActionMappers.BaseActionMapper import BaseActionMapper
from eprllib.Utils.observation_utils import get_actuator_name
from eprllib.Utils.annotations import override

class WindowsOpeningDiscreteActionMapper(BaseActionMapper):

    @override(BaseActionMapper)
    def setup(self):
        # Here you have access to the self.action_mapper_config and self.agent_name
        
        # The name of the actuator that will be controlled by the agent. 
        # This will be obtained from the actuators_config in the environment configuration file.
        self.window_actuator: Optional[str] = None

        # Here we use the config dict to provide the action space dimension.
        self.action_space_dim: int = self.action_mapper_config.get("action_space_dim", 11)
    
    @override(BaseActionMapper)    
    def get_action_space_dim(self) -> gym.Space[Any]:
        """
        Get the action space of the environment.

        Returns:
            gym.Space: Action space of the environment.
        """
        return gym.spaces.Discrete(self.action_space_dim)
    
    @override(BaseActionMapper)
    def actuator_names(
        self, 
        actuators_config: Dict[str, Tuple[str,str,str]]
        ) -> None:
        """
        This method is used to transform the agent dict action to actuator dict action. Consider that
        one agent could manage more than one actuator. For that reason, it is important to transform the
        action dict to actuator dict actions.

        Args:
            action (Any): The action to be transformed.
            actuators (List[str]): List of actuators controlled by the agent.

        Returns:
            Dict[str, Any]: Transformed actions for the actuators.
        """
        
        # Here the name of the actuator is obtained from the actuators_config 
        # in the environment configuration file.
        self.window_actuator = get_actuator_name(
            self.agent_name,
            actuators_config["Window Opening Actuator"][0],
            actuators_config["Window Opening Actuator"][1],
            actuators_config["Window Opening Actuator"][2]
        )
        
        
    @override(BaseActionMapper)
    def _agent_to_actuator_action(self, action: Any, actuators: List[str]) -> Dict[str, Any]:
        """
        Transform the agent action to actuator action.

        Args:
            action (Any): The action to be transformed.
            actuators (List[str]): List of actuators controlled by the agent.

        Returns:
            Dict[str, Any]: Transformed actions for the actuators.
        """
        if self.window_actuator in actuators:
            return {self.window_actuator: action/10}
        else:
            return {}
    

class WindowsOpeningContinousActionMapper(BaseActionMapper):
    @override(BaseActionMapper)
    def setup(self):
        # Here you have access to the self.action_mapper_config and self.agent_name
        
        # The name of the actuator that will be controlled by the agent. 
        # This will be obtained from the actuators_config in the environment configuration file.
        self.window_actuator: Optional[str] = None
        
    
    @override(BaseActionMapper)    
    def get_action_space_dim(self) -> gym.Space[Any]:
        """
        Get the action space of the environment.

        Returns:
            gym.Space: Action space of the environment.
        """
        return gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
    
    
    @override(BaseActionMapper)
    def actuator_names(
        self, 
        actuators_config: Dict[str, Tuple[str,str,str]]
        ) -> None:
        """
        This method is used to transform the agent dict action to actuator dict action. Consider that
        one agent could manage more than one actuator. For that reason, it is important to transform the
        action dict to actuator dict actions.

        Args:
            action (Any): The action to be transformed.
            actuators (List[str]): List of actuators controlled by the agent.

        Returns:
            Dict[str, Any]: Transformed actions for the actuators.
        """
        
        # Here the name of the actuator is obtained from the actuators_config 
        # in the environment configuration file.
        self.window_actuator = get_actuator_name(
            self.agent_name,
            actuators_config["Window Opening Actuator"][0],
            actuators_config["Window Opening Actuator"][1],
            actuators_config["Window Opening Actuator"][2]
        )
        
        
    @override(BaseActionMapper)
    def _agent_to_actuator_action(self, action: Any, actuators: List[str]) -> Dict[str, Any]:
        """
        Transform the agent action to actuator action.

        Args:
            action (Any): The action to be transformed.
            actuators (List[str]): List of actuators controlled by the agent.

        Returns:
            Dict[str, Any]: Transformed actions for the actuators.
        """
        if self.window_actuator in actuators:
            return {self.window_actuator: action}
        else:
            return {}
    