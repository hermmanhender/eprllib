"""
Exhaust Fan ActionMappers
============================

This module contains classes to implement ActionMappers for controlling exhaust fan actuators in the environment.
"""

import gymnasium as gym
from typing import Any, Dict, List, Tuple, Optional
from eprllib.Agents.ActionMappers.BaseActionMapper import BaseActionMapper
from eprllib.Utils.observation_utils import get_actuator_name
from eprllib.Utils.annotations import override
from eprllib.Utils.agent_utils import get_agent_name

class CentralAgentActionMapper(BaseActionMapper):
    REQUIRED_KEYS = {
        "exhaust_fan_actuator": Tuple[str, str, str],
        "window_actuator": Tuple[str, str, str],
        "hvac_availability_actuator": Tuple[str, str, str]
    }
    
    def __init__(
        self,
        action_mapper_config: Dict[str, Any]
    ):
        """
        This class implements the Exhaust Fan actions.

        Args:
            action_mapper_config (Dict[str, Any]): The configuration of the action function.
            It should contain the following keys:
                - modes (List[float]): The flow factor to modify the maximum flow of the exhaust fan. The order 
                in the list corresponds with mode of the Fan, usually mode 0 is off and mode 1 has the lower 
                flow factor. No more than 11 modes are allowed.
                - exhaust_fan_actuator (Tuple[str, str, str]): The configuration for the exhaust fan actuator.
        """
        # Validate the config.
        # config_validation(action_mapper_config, self.REQUIRED_KEYS)
        
        super().__init__(action_mapper_config)
        
        self.agent_name: str = "None"
        self.modes: List[float] = [0, 0.0600, 0.1313, 0.2093, 0.2973]
        self.exhaust_fan_actuator: Optional[str] = None
        self.window_actuator: Optional[str] = None
        self.hvac_availability_actuator: Optional[str] = None
        
        # Check that all the elements in self.modes list are floats in the range [0, 1].
        for mode in self.modes:
            if mode < 0 or mode > 1:
                raise ValueError("The modes must be in the range [0, 1].")
    
    @override(BaseActionMapper)    
    def get_action_space_dim(self) -> gym.Space[Any]:
        """
        Get the action space of the environment.

        Returns:
            gym.Space: Action space of the environment.
        """
        
        return gym.spaces.MultiDiscrete([len(self.modes), 5, 2])
    
    @override(BaseActionMapper)
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
        if self.agent_name is "None":
            self.agent_name = get_agent_name(actuators)
            self.exhaust_fan_actuator = get_actuator_name(
                self.agent_name,
                self.action_mapper_config['exhaust_fan_actuator'][0],
                self.action_mapper_config['exhaust_fan_actuator'][1],
                self.action_mapper_config['exhaust_fan_actuator'][2]
            )
            self.window_actuator = get_actuator_name(
                self.agent_name,
                self.action_mapper_config['window_actuator'][0],
                self.action_mapper_config['window_actuator'][1],
                self.action_mapper_config['window_actuator'][2]
            )
            self.hvac_availability_actuator = get_actuator_name(
                self.agent_name,
                self.action_mapper_config['hvac_availability_actuator'][0],
                self.action_mapper_config['hvac_availability_actuator'][1],
                self.action_mapper_config['hvac_availability_actuator'][2]
            )
            
        actuator_dict_actions: Dict[str, Any] = {actuator: None for actuator in actuators}
        
        assert isinstance(self.exhaust_fan_actuator, str), f"Exhaust fan actuator is not set. It should be a string, but got {type(self.exhaust_fan_actuator)}."
        assert isinstance(self.window_actuator, str), f"Window actuator is not set. It should be a string, but got {type(self.window_actuator)}."
        assert isinstance(self.hvac_availability_actuator, str), f"HVAC availability actuator is not set. It should be a string, but got {type(self.hvac_availability_actuator)}."
        
        actuator_dict_actions.update({
            self.exhaust_fan_actuator: self.modes[action[0]],
        })
        
        actuator_dict_actions.update({self.window_actuator: action[1] / 4})
        
        actuator_dict_actions.update({self.hvac_availability_actuator: action[2]})
        
        return actuator_dict_actions
    
    @override(BaseActionMapper)
    def get_actuator_action(self, action:float|int, actuator: str) -> Any:
        """
        This method is used to get the actions of the actuators after transform the 
        
        """
        return action
