"""
Dual Set Point Thermostat
=========================

This module contains the implementation of the Dual Set Point Thermostat action function.
The action function transforms the action values from [0,1] to the appropriate range for the
actuator type (cooling, heating, or flow rate). The transformation is linear for cooling and
heating, and linear for flow rate.
"""

from typing import Dict, Any
from eprllib.ActionFunctions.ActionFunctions import ActionFunction

class DualSetPointThermostat(ActionFunction):
    def __init__(self, action_fn_config:Dict[str,Any]):
        """
        This class implements the Dual Set Point Thermostat action function.

        Args:
            action_fn_config (Dict[str,Any]): The configuration of the action function.
            It should contain the following keys: agents_type (Dict[str, int]): A dictionary 
            mapping agent names to their types (1 for cooling, 2 for heating, 3 for flow rate).
        """
        super().__init__(action_fn_config)
        self.agents_type: Dict[str, Any] = action_fn_config['agents_type']
    
    def transform_action(self, action:Dict[str,int]) -> Dict[str, float|int]:
        # create a dict to save the action per agent.
        action_transformed = {agent: 0. for agent in self.agents_type.keys()}
        for agent, type in self.agents_type.items():
            if type == 1: # Cooling
                # transfort value between [0,1] to [23,27]
                action_transformed[agent] = 23 + (action[agent]/10)*(27-23)
            elif type == 2: # Heating
                # transfort value between [0,1] to [18,22]
                action_transformed[agent] = 18 + (action[agent]/10)*(22-18)
            elif type == 3: # Flow Rate
                action_transformed[agent] = action[agent]/20
            else:
                raise ValueError('Actuator type not valid.')
        
        return action_transformed