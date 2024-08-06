"""This modele define the basic policy class to use in eprllib.
"""
from typing import Optional, Any, List, Dict, Set, Tuple
from eprllib.Policies.Policy import PolicyClass

class FullySharedPolicy(PolicyClass):
    def __init__(
        self,
        *args,
        **kargs
    ):
        super.__init__()
    
    def compute_actions(self, ep_state:int|float|Any, action_dict:Dict, actuator_handles, api, **kargs):
        agents = self.get_acting_agents()
        for agent in agents:
            api.exchange.set_actuator_value(
                state=ep_state,
                actuator_handle=actuator_handles[agent],
                actuator_value=action_dict[agent]
            )
