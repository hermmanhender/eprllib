"""
Windows Shading Triggers
========================

This module contains classes to implement window shading triggers for controlling actuators in the environment.
"""
import gymnasium as gym
from typing import Any, Dict, List, Tuple
from eprllib.Agents.Triggers.BaseTrigger import BaseTrigger
from eprllib.Utils.observation_utils import get_actuator_name
from eprllib.Utils.annotations import override
from eprllib.Utils.agent_utils import get_agent_name, config_validation
from eprllib import logger

class WindowsShadingTrigger(BaseTrigger):
    REQUIRED_KEYS = {
        "shading_actuator": Tuple[str, str, str],
    }
    
    def __init__(
        self,
        trigger_fn_config: Dict[str, Any]
    ):
        """
        This class implements the window shading action function.

        Args:
            trigger_fn_config (Dict[str, Any]): The configuration of the action function.
            It should contain the following keys:
                - shading_actuator (Tuple[str, str, str]): The configuration for the shading actuator.
        """
        # Validate the config.
        config_validation(trigger_fn_config, self.REQUIRED_KEYS)
        
        super().__init__(trigger_fn_config)
        
        self.agent_name = None
        self.shading_actuator = None
    
    @override(BaseTrigger)    
    def get_action_space_dim(self) -> gym.Space:
        """
        Get the action space of the environment.

        Returns:
            gym.Space: Action space of the environment.
        """
        return gym.spaces.Discrete(11)
    
    @override(BaseTrigger)
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
            self.agent_name = get_agent_name(self.shading_actuator)
            self.shading_actuator = get_actuator_name(
                self.agent_name,
                self.trigger_fn_config['shading_actuator'][0],
                self.trigger_fn_config['shading_actuator'][1],
                self.trigger_fn_config['shading_actuator'][2]
            )
        
        actuator_dict_actions = {actuator: None for actuator in actuators}
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
    
    @override(BaseTrigger)
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

    @override(BaseTrigger)
    def action_to_goal(self, action: int | float) -> int | float:
        """
        This method is used to transform the action to a goal. The goal is used to define the reward.

        Args:
            action (Any): The action to be transformed.

        Returns:
            Any: The transformed action.
        """
        return action
    