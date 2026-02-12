"""
Hierarchical ActionMappers
============================

This class uses a discrete action space. The size of the action space must be specified in the 
`action_mapper_config` dictionary with the key "action_space_dim".
"""
from gymnasium import Space
from gymnasium.spaces import Discrete, MultiDiscrete
from typing import Any, List, Dict
from eprllib.Agents.ActionMappers.BaseActionMapper import BaseActionMapper
from eprllib.Utils.annotations import override
from eprllib.Utils.agent_utils import config_validation
from eprllib import logger

class HierarchicalGoalActionMapperDiscrete(BaseActionMapper):
    REQUIRED_KEYS: Dict[str, Any] = {
        "action_space_dim": int,
    }
    
    def __init__(
        self,
        action_mapper_config: Dict[str, Any] = {}
    ):
        # Validate the config.
        config_validation(action_mapper_config, self.REQUIRED_KEYS)
        
        super().__init__(action_mapper_config)
    
    @override(BaseActionMapper)
    def get_action_space_dim(self) -> Space[Any]:
        """This method is used to get the action space of the environment.

        Returns:
            gym.Space: Action space of the environment.
        """
        return Discrete(self.action_mapper_config['action_space_dim'])

    @override(BaseActionMapper)
    def agent_to_actuator_action(self, action: Any, actuators: List[str]):
        """
        This method is not used in top_level_agent.
        """
        msg = "This method should not be called."
        logger.error(msg)
        raise ValueError(msg)
    
    @override(BaseActionMapper)
    def get_actuator_action(self, action: float | int, actuator: str):
        """
        This method is not used in top_level_agent.
        """
        msg = f"The actuator {actuator} is not used in the top_level_agent."
        logger.error(msg)
        raise ValueError(msg)

    @override(BaseActionMapper)
    def action_to_goal(self, action: int | float) -> int | float:
        """
        This method is used to transform the action to a goal. The goal is used to define the reward.

        Args:
            action (Any): The action to be transformed.

        Returns:
            Any: The transformed action.
        """
        return action/(self.action_mapper_config['action_space_dim']-1)
    
    
class HierarchicalObjectiveActionMapperMultiDiscrete(BaseActionMapper):
    REQUIRED_KEYS = {
        "action_space_dim": int,
    }
    
    def __init__(
        self,
        action_mapper_config: Dict[str, Any] = {}
    ):
        # Validate the config.
        config_validation(action_mapper_config, self.REQUIRED_KEYS)
        
        super().__init__(action_mapper_config)
    
    @override(BaseActionMapper)
    def get_action_space_dim(self) -> Space[Any]:
        """This method is used to get the action space of the environment.

        Returns:
            gym.Space: Action space of the environment.
        """
        # TODO: Verify that the action space is the espected.
        # The action space is a MultiDiscrete space with the same dimension for each agent.
        return MultiDiscrete([self.action_mapper_config['action_space_dim']] * 10)

    @override(BaseActionMapper)
    def agent_to_actuator_action(self, action: Any, actuators: List[str]):
        """
        This method is not used in top_level_agent.
        """
        msg = "This method should not be called."
        logger.error(msg)
        raise ValueError(msg)
    
    @override(BaseActionMapper)
    def get_actuator_action(self, action: float | int, actuator: str):
        """
        This method is not used in top_level_agent.
        """
        msg = f"The actuator {actuator} is not used in the top_level_agent."
        logger.error(msg)
        raise ValueError(msg)

    @override(BaseActionMapper)
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
    