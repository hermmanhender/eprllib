"""
Action Function Base Class
==========================

This module contain the base class to create action transformer functions.
Action transformer functions are used to transform the actions of the agents.
They are used in the EnvConfig class to create the environment.
"""
from typing import Dict, Set, Any

class RBActionFunction:
    """
    Base class to create action transformer functions.
    """
    def __init__(
        self,
        action_fn_config: Dict[str,Any] = {}
    ):
        """
        This class is used to transform the actions of the agents before applying
        them in the environment.

        Args:
            action_fn_config (Dict[str,Any]): Configuration for the action transformer function.
        """
        self.action_fn_config = action_fn_config
    
    def get_actions(self, infos:Dict[str,Dict[str,Any]]) -> Dict[str, Any]:
        """
        This method is used to transform the actions of the agents before applying.

        Args:
            infos:Dict[str,Dict[str,Any]]: Observations and information of the environment, provided by the environment.

        Returns:
            Dict[str, Any]: A dict of rule based actions for each agent in the environment.
        """
        return NotImplementedError
