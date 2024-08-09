"""
Dual Set Point Thermostat
=========================

This module contains the implementation of the Dual Set Point Thermostat action function.
The action function transforms the action values from [0,1] to the appropriate range for the
actuator type (cooling, heating, or flow rate). The transformation is linear for cooling and
heating, and linear for flow rate.
"""

from typing import Dict, Set
from eprllib.ActionFunctions.ActionFunctions import ActionFunction

class DualSetPointThermostat(ActionFunction):
    def __init__(self, agents_config:Dict, _agent_ids:Set):
        super().__init__(agents_config, _agent_ids)
    
    def transform_action(self, action:Dict[str,int]) -> Dict[str, float|int]:
        action_transformed = {agent: 0. for agent in self._agent_ids}
        for agent in self._agent_ids:
            actuator_type = self.agents_config[agent]['actuator_type']
            if actuator_type == 1: # Cooling
                # transfort value between [0,1] to [23,27]
                action_transformed[agent] = 23 + (action[agent]/10)*(27-23)
            elif actuator_type == 2: # Heating
                # transfort value between [0,1] to [18,22]
                action_transformed[agent] = 18 + (action[agent]/10)*(22-18)
            elif actuator_type == 3: # Flow Rate
                action_transformed[agent] = action[agent]/20
            else:
                raise ValueError('Actuator type not valid.')
        
        return action_transformed