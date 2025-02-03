"""
Hierarchical trigger
=====================

This class uses a discrete action space. The size of the action space must be specified in the 
`trigger_fn_config` dictionary with the key "action_space_dim".
"""

from gymnasium import Space
from gymnasium.spaces import Discrete
from typing import Any, List
from eprllib.Agents.Triggers.BaseTrigger import BaseTrigger
from eprllib.Utils.annotations import override

class HierarchicalTriggerDiscrete(BaseTrigger):
    def __init__(
        self,
        trigger_fn_config: dict = {}
    ):
        super().__init__(trigger_fn_config)
        self.action_space_dim = trigger_fn_config.get('action_space_dim', False)
        if not self.action_space_dim:
            raise ValueError("The action space dimension must be provided.")
        if type(self.action_space_dim) != int:
            raise ValueError("The action space dimension must be an integer.")
    
    @override(BaseTrigger)
    def get_action_space_dim(self) -> Space:
        """This method is used to get the action space of the environment.

        Returns:
            gym.Space: Action space of the environment.
        """
        return Discrete(self.action_space_dim)

    @override(BaseTrigger)
    def agent_to_actuator_action(self, action: Any, actuators: List[str]):
        """
        This method is not used in top_level_agent.
        """
        raise ValueError("This method should not be called.")
    
    @override(BaseTrigger)
    def get_actuator_action(self, action: float | int, actuator: str):
        """
        This method is not used in top_level_agent.
        """
        raise ValueError("This method should not be called.")
    