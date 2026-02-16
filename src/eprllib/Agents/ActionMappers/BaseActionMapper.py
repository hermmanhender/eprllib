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
from typing import Dict, Any, List, Optional
import gymnasium as gym
from eprllib.Utils.annotations import OverrideToImplementCustomLogic
from eprllib import logger

class BaseActionMapper:
    """
    Base class to create action transformer functions.
    """
    action_mapper_config: Dict[str, Any] = {}
    agent_name: str
    
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
    
        # Make sure, `setup()` is only called once, no matter what.
        if hasattr(self, "_is_setup") and self._is_setup:
            raise RuntimeError(
                "``BaseActionMapper.setup()`` called twice within your ActionMapper implementation "
                f"{self}! Make sure you are using the proper inheritance order "
                " and that you are NOT overriding the constructor, but "
                "only the ``setup()`` method of your subclass."
            )
        try:
            self.setup()
        except AttributeError as e:
            raise e

        self._is_setup:bool = True
        self._ist_initialized:bool = False

    
    def agent_to_actuator_action(self, action: Any, actuators: List[str]) -> Dict[str, Any]:
        """
        This method is used to call the agent_to_actuator_action method. Do not overwrite this method.

        Args:
            action (Any): The action to be transformed.
            actuators (List[str]): List of actuators controlled by the agent.

        Returns:
            Dict[str, Any]: Transformed actions for the actuators.
        """
        if not self._ist_initialized:
            self._actuator_names_assignation(actuators)
            self._ist_initialized = True
        actuator_dict_actions = self._agent_to_actuator_action(action, actuators)

        for actuator in actuator_dict_actions:
            if actuator_dict_actions[actuator] is None:
                msg = f"The actuator {actuator} is not in the list of actuators.\nThe actual list of actuators is: \n{actuator_dict_actions.keys()}"
                logger.error(msg)
                raise ValueError(msg)
        return actuator_dict_actions
    
    
    # ===========================
    # === OVERRIDABLE METHODS ===
    # ===========================
    
    @OverrideToImplementCustomLogic
    def setup(self):
        """
        Sets up the components of the module.

        This is called automatically during the __init__ method of this class.
        """
        pass
    
    @OverrideToImplementCustomLogic
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
    
    
    @OverrideToImplementCustomLogic
    def _agent_to_actuator_action(self, action: Any, actuators: List[str]) -> Dict[str, Any]:
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
    
    
    @OverrideToImplementCustomLogic
    def _actuator_names_assignation(self, actuators: List[str]) -> None:
        """
        This method is used to assign the names of the actuators to the agent.
        To avoid recalculate each timestep, this is only executed one time. To use well
        this method follow the example:
        
            ```
            from eprllib.Utils.observation_utils import get_actuator_name
            
            self.actuator_name = get_actuator_name(
                self.agent_name,
                self.actuator_config[0],
                self.actuator_config[1],
                self.actuator_config[2]
            )
            ```
        
        See ``eprllib.utils.observation_utils.get_actuator_name`` to see how actuators are named.
        
        Args:
            actuators (List[str]): List of actuators controlled by the agent.
        """
        msg = "BaseActionMapper: This method should be implemented in the child class."
        logger.error(msg)
        raise NotImplementedError(msg)
