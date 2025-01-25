"""
Setpoint Controls
==================

"""
import gymnasium as gym
from typing import Any, Dict, List, Tuple
from eprllib.ActionFunctions.ActionFunctions import ActionFunction
from eprllib.Utils.observation_utils import (
    get_actuator_name,
)

class discrete_dual_setpoint_and_availability(ActionFunction):
    def __init__(self, action_fn_config:Dict[str,Any]):
        """
        This class implements the Dual Set Point Thermostat action function.

        Args:
            action_fn_config (Dict[str,Any]): The configuration of the action function.
            It should contain the following keys: agents_type (Dict[str, int]): A dictionary 
            mapping agent names to their types (1 for cooling, 2 for heating, 3 for Availability).
        """
        super().__init__(action_fn_config)
        
        agent_name = action_fn_config['agent_name']
        self.temperature_range: Tuple[int,int] = action_fn_config['temperature_range']
        self.actuator_for_cooling = get_actuator_name(
            agent_name,
            action_fn_config['actuator_for_cooling'][0],
            action_fn_config['actuator_for_cooling'][1],
            action_fn_config['actuator_for_cooling'][2]
        )
        self.actuator_for_heating = get_actuator_name(
            agent_name,
            action_fn_config['actuator_for_heating'][0],
            action_fn_config['actuator_for_heating'][1],
            action_fn_config['actuator_for_heating'][2]
        )
        self.availability_actuator = get_actuator_name(
            agent_name,
            action_fn_config['availability_actuator'][0],
            action_fn_config['availability_actuator'][1],
            action_fn_config['availability_actuator'][2]
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
    
    def get_actuator_action(self, action:float|int, actuator: str) -> Any:
        """
        This method is used to get the actions of the actuators after transform the 
        
        """
        return action
    