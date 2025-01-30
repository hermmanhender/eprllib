"""
Herarchical action function implemented
========================================

This class used a discrete action space. The size of it must be specified in the 
``action_fn_config`` dict with the key "action_space_dim".
"""

from gymnasium import Space
from gymnasium.spaces import Discrete
from typing import Any, List
from eprllib.ActionFunctions.ActionFunctions import ActionFunction
from eprllib.Utils.annotations import override

class discrete_n_space(ActionFunction):
    def __init__(
        self,
        action_fn_config: dict = {}
    ):
        super.__init__(action_fn_config)
        self.action_space_dim = action_fn_config.get('action_space_dim', False)
        if not self.action_space_dim:
            raise ValueError("The action space dimension must be provided.")
        if type(self.action_space_dim) != int:
            raise ValueError("The action space dimension must be an integer.")
    
    @override(ActionFunction)
    def get_action_space_dim(self) -> Space:
        """This method is used to get the action space of the environment.

        Raises:
            NotImplementedError: This method should be implemented in the child class.

        Returns:
            gym.Space: Action space of the environment.
        """
        return Discrete(self.action_space_dim)

    @override(ActionFunction)
    def agent_to_actuator_action(self, action: Any, actuators: List[str]):
        """
        This method is used not used in top_level_agent.
        """
        raise ValueError("This method should not be called.")
    
    @override(ActionFunction)
    def get_actuator_action(self, action:float|int, actuator: str):
        """
        This method is used not used in top_level_agent.
        """
        raise ValueError("This method should not be called.")