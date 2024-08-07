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
        agents_config: Dict,
        _agent_ids: Set,
    ):
        """
        This class is used to transform the actions of the agents before applying
        them in the environment.

        Args:
            agents_config (Dict): This tale the configuration of the agents in the EnvConfig class.
            _agent_ids (Set): Agent ids in the environment.
        """
        self.agents_config = agents_config
        self._agent_ids = _agent_ids
    
    def transform_action(self, action:Dict[str,float]) -> Dict[str, Any]:
        """
        This method is used to transform the actions of the agents before applying.

        Args:
            action (Dict[str,float]): Action provided by the policy.

        Returns:
            Dict[str, Any]: A dict of transformed action for each agent in the environment.
        """
        return action
