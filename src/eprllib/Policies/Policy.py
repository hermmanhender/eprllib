"""This modele define the basic policy class to use in eprllib.
"""
from typing import Optional, Any, List, Dict, Set, Tuple

class PolicyClass:
    def __init__(
        self,
        *args,
        **kargs
    ):
        pass
    
    def get_acting_agents(action_dict:Dict) -> Set:
        """_description_
        """
        return {agent for agent in action_dict.keys()}
    
    def compute_actions(self, ep_state:int|float|Any, action_dict:Dict, **kargs):
        return NotImplementedError()