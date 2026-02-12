"""
Windows Shading ActionMappers
===============================

This module contains classes to implement window shading ActionMappers for controlling actuators in the environment.
"""
import gymnasium as gym
from typing import Any, Dict, List, Tuple, Optional
from eprllib.Agents.ActionMappers.BaseActionMapper import BaseActionMapper
from eprllib.Utils.observation_utils import get_actuator_name
from eprllib.Utils.annotations import override
from eprllib.Utils.agent_utils import get_agent_name, config_validation
from eprllib import logger

class WindowsShadingActionMapper(BaseActionMapper):
    REQUIRED_KEYS: Dict[str, Any] = {
        "shading_actuator": Tuple[str, str, str],
    }
    
    def __init__(
        self,
        action_mapper_config: Dict[str, Any]
    ):
        """
        This class implements the window shading action function.

        Args:
            action_mapper_config (Dict[str, Any]): The configuration of the action function.
            It should contain the following keys:
                - shading_actuator (Tuple[str, str, str]): The configuration for the shading actuator.
        """
        # Validate the config.
        config_validation(action_mapper_config, self.REQUIRED_KEYS)
        
        super().__init__(action_mapper_config)
        
        self.agent_name: Optional[str] = None
        self.shading_actuator: Optional[str] = None
    
    @override(BaseActionMapper)    
    def get_action_space_dim(self) -> gym.Space[Any]:
        """
        Get the action space of the environment.

        Returns:
            gym.Space: Action space of the environment.
        """
        return gym.spaces.Discrete(11)
    
    @override(BaseActionMapper)
    def agent_to_actuator_action(self, action: Any, actuators: List[str]) -> Dict[str, Any]:
        """
        Transform the agent action to actuator action. Consider that one agent may manage more than one actuator.

        Args:
            action (Any): The agent action, normally an int or float.
            actuators (List[str]): The list of actuator names that the agent manages.

        Returns:
            Dict[str, Any]: Dictionary with the actions for the actuators.
        """
        if self.agent_name is None:
            self.agent_name = get_agent_name(actuators)
            self.shading_actuator = get_actuator_name(
                self.agent_name,
                self.action_mapper_config['shading_actuator'][0],
                self.action_mapper_config['shading_actuator'][1],
                self.action_mapper_config['shading_actuator'][2]
            )
        
        assert self.shading_actuator is not None, "Shading actuator name has not been initialized."
        
        actuator_dict_actions: Dict[str, Any] = {actuator: None for actuator in actuators}
        if action == 0:  # Shading device is off (applies to shades and blinds).
            actuator_dict_actions.update({self.shading_actuator: 0})
        else:  # 3.0: Exterior shade is on.
            actuator_dict_actions.update({self.shading_actuator: 3})
            
        # Check if there is an actuator_dict_actions value equal to None.
        for actuator in actuator_dict_actions:
            if actuator_dict_actions[actuator] is None:
                msg = f"The actuator {actuator} is not in the list of actuators: \n{actuators}.\nThe actuator dict is: \n{actuator_dict_actions}"
                logger.error(msg)
                raise ValueError(msg)
            
        return actuator_dict_actions
    
    @override(BaseActionMapper)
    def get_actuator_action(self, action: float | int, actuator: str) -> Any:
        """
        Get the actions of the actuators after transforming the agent action to actuator action.

        Args:
            action (float | int): Action provided by the policy and transformed by agent_to_actuator_action.
            actuator (str): The actuator that requires the action.

        Returns:
            Any: The action for the actuator.
        """
        return action

    @override(BaseActionMapper)
    def action_to_goal(self, action: int | float) -> int | float:
        """
        This method is used to transform the action to a goal. The goal is used to define the reward.

        Args:
            action (Any): The action to be transformed.

        Returns:
            Any: The transformed action.
        """
        return action
    