"""
Action Function Base Class
==========================

This module contain the base class to create action transformer functions.
Action transformer functions are used to transform the actions of the agents.
They are used in the EnvConfig class to create the environment.
"""
from typing import Dict, Set, Any

class ActionFunction:
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
    
    def transform_action(self, action:Dict[str,float]) -> Dict[str, Any]:
        """
        This method is used to transform the actions of the agents before applying.

        Args:
            action (Dict[str,float]): Action provided by the policy.

        Returns:
            Dict[str, Any]: A dict of transformed action for each agent in the environment.
        """
        return action
