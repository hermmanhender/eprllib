"""
Trigger Base Class
==========================

This module contains the base class to create action transformer functions and define
the action space dimension in the environment.

Action transformer functions are used to adapt the action given by the neural network, 
normally an integer for discrete spaces and a float for continuous spaces like Box. The actions
must be adapted to values required for the actuators in EnergyPlus. Each agent has the
capacity to control one actuator.

ActionFunction must be defined in the EnvConfig definition to create the environment and is
called in the EnergyPlusEnvironment.EnergyPlusEnv_v0 class and used in the EnergyPlusRunner.EnergyPlusRunner class
to transform the dict of agent actions to actuator values.
"""
from typing import Dict, Any, List
import gymnasium as gym

class BaseTrigger:
    """
    Base class to create action transformer functions.
    """
    def __init__(
        self,
        trigger_fn_config: Dict[str, Any] = {}
    ):
        """
        This class is used to transform the actions of the agents before applying
        them in the environment.

        Args:
            trigger_fn_config (Dict[str, Any]): Configuration for the action transformer function.
        """
        self.trigger_fn_config = trigger_fn_config
    
    def get_action_space_dim(self) -> gym.Space:
        """This method is used to get the action space of the environment.

        Raises:
            NotImplementedError: This method should be implemented in the child class.

        Returns:
            gym.Space: Action space of the environment.
        """
        raise NotImplementedError("This method should be implemented in the child class.")
    
    def agent_to_actuator_action(self, action: Any, actuators: List[str]) -> Dict[str, Any]:
        """
        This method is used to transform the agent dict action to actuator dict action. Consider that
        one agent could manage more than one actuator. For that reason, it is important to transform the
        action dict to actuator dict actions.
        
        The actuators are named as: agent_{n}, where n corresponds with the order listed in the
        actuators list.

        Args:
            action (Any): The action to be transformed.
            actuators (List[str]): List of actuators controlled by the agent.

        Returns:
            Dict[str, Any]: Transformed actions for the actuators.
        """
        raise NotImplementedError("This method should be implemented in the child class.")
    
    def get_actuator_action(self, action: float | int, actuator: str) -> Any:
        """
        This method is used to get the actions of the actuators after transforming the
        agent dict action to actuator dict action with agent_to_actuator_action.

        Args:
            action (float | int): Action provided by the policy and transformed by agent_to_actuator_action.
            actuator: The actuator that requires the action.

        Returns:
            Dict[str, Any]: A dict of transformed action for each agent in the environment.
        """
        return action
