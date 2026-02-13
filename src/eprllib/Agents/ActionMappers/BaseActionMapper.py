"""
Action Mapper Base Class
==========================

This module contains the base class to create action transformer functions and define
the action space dimension in the environment.

```ActionMapper`` functions are used to adapt the action given by the neural network, 
normally an integer for discrete spaces and a float for continuous spaces like Box. The actions
must be adapted to values required for the actuators in EnergyPlus. Each agent has the
capacity to control one actuator.

``ActionMapper`` must be defined in the ``EnvironmentConfig`` definition to create the environment and is
called in the ``Environment.Environment`` class and used in the ``EnvironmentRunner`` class
to transform the dict of agent actions to actuator values.
"""
from typing import Dict, Any, List
import gymnasium as gym
from eprllib import logger

class BaseActionMapper:
    """
    Base class to create action transformer functions.
    """
    action_mapper_config: Dict[str, Any] = {}
    
    def __init__(
        self,
        action_mapper_config: Dict[str, Any] = {}
    ):
        """
        This class is used to transform the actions of the agents before applying
        them in the environment.

        Args:
            action_mapper_config (Dict[str, Any]): Configuration for the action transformer function.
        """
        self.action_mapper_config = action_mapper_config
        
        logger.info(f"BaseActionMapper: The BaseActionMapper was correctly inicializated with {self.action_mapper_config} config.")
    
    def get_action_space_dim(self) -> gym.Space[Any]:
        """This method is used to get the action space of the environment.

        Raises:
            NotImplementedError: This method should be implemented in the child class.

        Returns:
            gym.Space: Action space of the environment.
        """
        msg = "BaseActionMapper: This method should be implemented in the child class."
        logger.error(msg)
        raise NotImplementedError
    
    def agent_to_actuator_action(self, action: Any, actuators: List[str]) -> Dict[str, Any]:
        """
        This method is used to transform the agent dict action to actuator dict action. Consider that
        one agent could manage more than one actuator. For that reason, it is important to transform the
        action dict to actuator dict actions.
        
        See ``eprllib.utils.observation_utils.get_actuator_name`` to see how actuators are named.

        Args:
            action (Any): The action to be transformed.
            actuators (List[str]): List of actuators controlled by the agent.

        Returns:
            Dict[str, Any]: Transformed actions for the actuators.
        """
        msg = "BaseActionMapper: This method should be implemented in the child class."
        logger.error(msg)
        raise NotImplementedError(msg)
    
    def get_actuator_action(self, action: float | int, actuator: str) -> int | float:
        """
        This method is used to get the actions of the actuators after transforming the
        agent dict action to actuator dict action with agent_to_actuator_action.

        Args:
            action (float | int): Action provided by the policy and transformed 
            by agent_to_actuator_action.
            actuator (str): The actuator that requires the action.

        Returns:
            int | float: The action of the actuator.
        """
        return action

    def action_to_goal(self, action: int | float) -> int | float:
        """
        This method is used to transform the action to a goal. The goal is used to define the reward
        in hierarchical agents.

        Args:
            action (int | float): The action to be transformed.

        Returns:
            int | float: The transformed action to goal.
        """
        return action
    