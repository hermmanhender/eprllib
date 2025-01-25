"""
Window Shading Control
=======================

The window shading control allow two actuators: Control Mode and Slat Angle.

The following settings are valid and must be used:
    • -1.0: No shading device.
    • 0.0: Shading device is off (applies to shades and blinds).
    • 1.0: Interior shade is on.
    • 2.0: Glazing is switched to a darker state (switchable glazings only).
    • 3.0: Exterior shade is on.
    • 4.0: Exterior screen is on.
    • 6.0: Interior blind is on.
    • 7.0: Exterior blind is on.
    • 8.0: Between-glass shade is on.
    • 9.0: Between-glass blind is on.

"""
import gymnasium as gym
from typing import Any, Dict, List
from eprllib.ActionFunctions.ActionFunctions import ActionFunction
from eprllib.Utils.observation_utils import (
    get_actuator_name,
)

class shading_actions(ActionFunction):
    def __init__(self, action_fn_config:Dict[str,Any]):
        """
        This class implements the Dual Set Point Thermostat action function.

        Args:
            action_fn_config (Dict[str,Any]): The configuration of the action function.
            It should contain the following keys:
                agent_name (str): The agent that use this action funtion.
                shading_actuator (Tuple[str,str,str]): The actuator configuration tuple that the agent control.
        """
        super().__init__(action_fn_config)
        
        self.shading_actuator = get_actuator_name(
            action_fn_config['agent_name'],
            action_fn_config['shading_actuator'][0],
            action_fn_config['shading_actuator'][1],
            action_fn_config['shading_actuator'][2]
        )
        
    def get_action_space_dim(self) -> gym.Space:
        """
        Get the action space of the environment.

        Returns:
            gym.Space: Action space of the environment.
        """
        
        return gym.spaces.Discrete(11)
    
    def agent_to_actuator_action(self, action: Any, actuators: List[str]) -> Dict[str,Any]:
        """
        This method is used to transform the agent action to actuator dict action. Consider that
        one agent may manage more than one actuator.
        
        Args:
        
            action (Any): The agent action, normally an int of float.
            actuators (List[str]): The list of actuator names that the agent manage.
        
        Return:
            Dict[str, Any]: Dictionary with the actions for the actuators.
        """
        actuator_dict_actions = {actuator: None for actuator in actuators}
        if action == 0: # Shading device is off (applies to shades and blinds).
            actuator_dict_actions.update({self.shading_actuator: 0})
        else: # 3.0: Exterior shade is on.
            actuator_dict_actions.update({self.shading_actuator: 3})
        return actuator_dict_actions
    
    def get_actuator_action(self, action:float|int, actuator: str) -> Any:
        """
        This method is used to get the actions of the actuators after transform the
        agent dict action to actuator dict action with agent_to_actuator_action.

        Args:
            action (float|int): Action provided by the policy and transformed by agent_to_actuator_action.
            actuator: The actuator that require the action.

        Returns:
            Dict[str, Any]: A dict of transformed action for each agent in the environment.
        """
        return action
