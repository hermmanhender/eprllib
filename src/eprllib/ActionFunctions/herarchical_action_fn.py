"""
Herarchical action function implemented
========================================

This class used a discrete action space. The size of it must be specified in the 
``action_fn_config`` dict with the key "action_space_dim".
"""

from gymnasium import Space
from gymnasium.spaces import Discrete
from eprllib.ActionFunctions.ActionFunctions import HerarchicalActionFunction
from eprllib.Utils.annotations import override

class discrete_n_space(HerarchicalActionFunction):
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
    
    @override(HerarchicalActionFunction)
    def get_action_space_dim(self) -> Space:
        """This method is used to get the action space of the environment.

        Raises:
            NotImplementedError: This method should be implemented in the child class.

        Returns:
            gym.Space: Action space of the environment.
        """
        return Discrete(self.action_space_dim)
