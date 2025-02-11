"""
Hierarchical trigger
=====================

This class uses a discrete action space. The size of the action space must be specified in the 
`trigger_fn_config` dictionary with the key "action_space_dim".
"""
import numpy as np
from gymnasium import Space
from gymnasium.spaces import Discrete, MultiDiscrete
from typing import Any, List
from eprllib.Agents.Triggers.BaseTrigger import BaseTrigger
from eprllib.Utils.annotations import override

class HierarchicalGoalTriggerDiscrete(BaseTrigger):
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

    @override(BaseTrigger)
    def action_to_goal(self, action: int | float) -> int | float:
        """
        This method is used to transform the action to a goal. The goal is used to define the reward.

        Args:
            action (Any): The action to be transformed.

        Returns:
            Any: The transformed action.
        """
        return action/self.action_space_dim
    
    
class HierarchicalObjectiveTriggerMultiDiscrete(BaseTrigger):
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
        return MultiDiscrete(np.array(list(10))*self.action_space_dim)

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

    @override(BaseTrigger)
    def action_to_goal(self, action: int | float) -> int | float:
        """
        This method is used to transform the action to a goal. In this case, the agents are using a MultiDiscrete action 
        space, that are transformed to a single vector.

        Args:
            action (Any): The action to be transformed.

        Returns:
            Any: The transformed action.
        """
        return action
    