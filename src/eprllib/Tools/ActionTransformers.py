"""This script will be contain some action transformer methods to implement in
eprllib. Most of them are applied in the test section where examples to test the 
library are developed.
"""
from typing import Dict, Set, Any

class ActionTransformer:
    def __init__(
        self,
        agents_config: Dict,
        _agent_ids: Set,
    ):
        self.agents_config = agents_config
        self._agent_ids = _agent_ids
    
    def transform_action(self, action:Dict[str,float]) -> Dict[str, Any]:
        raise NotImplementedError

"""Example"""

class DualSetPointThermostat(ActionTransformer):
    def __init__(self, agents_config:Dict, _agent_ids:Set):
        super().__init__(agents_config, _agent_ids)
    
    def transform_action(self, action:Dict[str,float]) -> Dict[str, Any]:
        for agent in self._agent_ids:
            actuator_type = self.agents_config[agent]['actuator_type']
            if actuator_type == 1: # Cooling
                # transfort value between [0,1] to [16,27]
                action[agent] = 16 + action[agent]*11
            elif actuator_type == 2: # Heating
                # transfort value between [0,1] to [18,25]
                action[agent] = 18 + action[agent]*7
            else:
                raise ValueError('Actuator type not valid.')
            
        return action