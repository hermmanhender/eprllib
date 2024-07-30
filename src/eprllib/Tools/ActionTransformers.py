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