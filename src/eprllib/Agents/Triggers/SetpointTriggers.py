"""
Setpoint triggers
==================

This module contains classes to implement setpoint triggers for controlling actuators in the environment.
"""

import gymnasium as gym
from typing import Any, Dict, List, Tuple
from eprllib.Agents.Triggers.BaseTrigger import BaseTrigger
from eprllib.Utils.observation_utils import get_actuator_name
from eprllib.Utils.annotations import override
from eprllib.Utils.agent_utils import get_agent_name

class DualSetpointTriggerDiscreteAndAvailabilityTrigger(BaseTrigger):
    def __init__(
        self,
        trigger_fn_config: Dict[str, Any]
    ):
        """
        This class implements the Dual Set Point Thermostat action function.

        Args:
            trigger_fn_config (Dict[str, Any]): The configuration of the action function.
            It should contain the following keys:
                - temperature_range (Tuple[int, int]): The temperature range for the setpoints.
                - actuator_for_cooling (Tuple[str, str, str]): The configuration for the cooling actuator.
                - actuator_for_heating (Tuple[str, str, str]): The configuration for the heating actuator.
                - availability_actuator (Tuple[str, str, str]): The configuration for the availability actuator.
        """
        super().__init__(trigger_fn_config)
        
        self.agent_name = None
        self.temperature_range: Tuple[int, int] = trigger_fn_config['temperature_range']
        self.actuator_for_cooling = None
        self.actuator_for_heating = None
        self.availability_actuator = None
    
    @override(BaseTrigger)    
    def get_action_space_dim(self) -> gym.Space:
        """
        Get the action space of the environment.

        Returns:
            gym.Space: Action space of the environment.
        """
        
        return gym.spaces.Discrete(11)
    
    @override(BaseTrigger)
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
        if self.agent_name is None:
            self.agent_name = get_agent_name(actuators)
            self.actuator_for_cooling = get_actuator_name(
                self.agent_name,
                self.trigger_fn_config['actuator_for_cooling'][0],
                self.trigger_fn_config['actuator_for_cooling'][1],
                self.trigger_fn_config['actuator_for_cooling'][2]
            )
            self.actuator_for_heating = get_actuator_name(
                self.agent_name,
                self.trigger_fn_config['actuator_for_heating'][0],
                self.trigger_fn_config['actuator_for_heating'][1],
                self.trigger_fn_config['actuator_for_heating'][2]
            )
            self.availability_actuator = get_actuator_name(
                self.agent_name,
                self.trigger_fn_config['availability_actuator'][0],
                self.trigger_fn_config['availability_actuator'][1],
                self.trigger_fn_config['availability_actuator'][2]
            )
            
        actuator_dict_actions = {actuator: None for actuator in actuators}
        
        if action == 0:
            actuator_dict_actions.update({
                self.actuator_for_cooling: max(self.temperature_range),
                self.actuator_for_heating: min(self.temperature_range),
                self.availability_actuator: 0
            })
        
        else:
            actuator_dict_actions.update({
                self.actuator_for_cooling: min(self.temperature_range)+1 + (action-1)/(10-1) * (max(self.temperature_range) - min(self.temperature_range)-1),
                self.actuator_for_heating: min(self.temperature_range) + (action-1)/(10-1) * (max(self.temperature_range) - min(self.temperature_range)-1),
                self.availability_actuator: 1
            })

        # Check if there is an actuator_dict_actions value equal to None.
        for actuator in actuator_dict_actions:
            if actuator_dict_actions[actuator] is None:
                raise ValueError(f"The actuator {actuator} is not in the list of actuators.")
        
        return actuator_dict_actions
    
    @override(BaseTrigger)
    def get_actuator_action(self, action:float|int, actuator: str) -> Any:
        """
        This method is used to get the actions of the actuators after transform the 
        
        """
        return action


class AvailabilityTrigger(BaseTrigger):
    def __init__(
        self, 
        trigger_fn_config:Dict[str,Any]
        ):
        """
        This class implements the Dual Set Point Thermostat action function.

        Args:
            trigger_fn_config (Dict[str,Any]): The configuration of the action function.
            It should contain the following keys: agents_type (Dict[str, int]): A dictionary 
            mapping agent names to their types (1 for cooling, 2 for heating, 3 for Availability).
        """
        super().__init__(trigger_fn_config)
        
        self.agent_name = None
        self.availability_actuator = None
    
    @override(BaseTrigger)    
    def get_action_space_dim(self) -> gym.Space:
        """
        Get the action space of the environment.

        Returns:
            gym.Space: Action space of the environment.
        """
        return gym.spaces.Discrete(2)
    
    @override(BaseTrigger)
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
        if self.agent_name is None:
            self.agent_name = get_agent_name(actuators)
            self.availability_actuator = get_actuator_name(
                self.agent_name,
                self.trigger_fn_config['availability_actuator'][0],
                self.trigger_fn_config['availability_actuator'][1],
                self.trigger_fn_config['availability_actuator'][2]
            )

        
        actuator_dict_actions = {actuator: None for actuator in actuators}
        
        if action == 0:
            actuator_dict_actions.update({
                self.availability_actuator: 0
            })
        
        else:
            actuator_dict_actions.update({
                self.availability_actuator: 1
            })

        # Check if there is an actuator_dict_actions value equal to None.
        for actuator in actuator_dict_actions:
            if actuator_dict_actions[actuator] is None:
                raise ValueError(f"The actuator {actuator} is not in the list of actuators.")
        
        return actuator_dict_actions
    
    @override(BaseTrigger)
    def get_actuator_action(self, action:float|int, actuator: str) -> Any:
        """
        This method is used to get the actions of the actuators after transform the 
        
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
    