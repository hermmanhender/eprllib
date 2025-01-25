"""
Window Opening Contros
=======================

"""
import gymnasium as gym
from typing import Any, Dict, List, Tuple
from eprllib.ActionFunctions.ActionFunctions import ActionFunction
from eprllib.Utils.observation_utils import (
    get_actuator_name,
)

class discrete_opening(ActionFunction):
    def __init__(self, action_fn_config:Dict[str,Any]):
        """
        This class implements the Dual Set Point Thermostat action function.

        Args:
            action_fn_config (Dict[str,Any]): The configuration of the action function.
            It should contain the following keys: agents_type (Dict[str, int]): A dictionary 
            mapping agent names to their types (1 for cooling, 2 for heating, 3 for Availability).
        """
        super().__init__(action_fn_config)
        
        self.window_actuator = get_actuator_name(
            action_fn_config['agent_name'],
            action_fn_config['window_actuator'][0],
            action_fn_config['window_actuator'][1],
            action_fn_config['window_actuator'][2]
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
        
        """
        actuator_dict_actions = {actuator: None for actuator in actuators}
        
        actuator_dict_actions.update({self.window_actuator: action/10})
        
        return actuator_dict_actions
    
    def get_actuator_action(self, action:float|int, actuator: str) -> Any:
        return action