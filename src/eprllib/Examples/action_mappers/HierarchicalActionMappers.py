"""
Hierarchical ActionMappers
============================

This class uses a discrete action space. The size of the action space must be specified in the 
`action_mapper_config` dictionary with the key "action_space_dim".
"""
from gymnasium import Space
from gymnasium.spaces import Discrete
from typing import Any, List, Dict, Tuple
from eprllib.Agents.ActionMappers.BaseActionMapper import BaseActionMapper
from eprllib.Utils.annotations import override

class HierarchicalGoalActionMapperDiscrete(BaseActionMapper):
    
    @override(BaseActionMapper)
    def setup(self):
        # Here you have access to the self.action_mapper_config and self.agent_name

        # Here we use the config dict to provide the action space dimension.
        self.action_space_dim: int = self.action_mapper_config.get("action_space_dim", 11)
    
    
    @override(BaseActionMapper)
    def get_action_space_dim(self) -> Space[Any]:
        """This method is used to get the action space of the environment.

        Returns:
            gym.Space: Action space of the environment.
        """
        return Discrete(self.action_space_dim)


    @override(BaseActionMapper)
    def actuator_names(
        self, 
        actuators_config: Dict[str, Tuple[str,str,str]]
        ) -> None:
        """
        This method is used to transform the agent dict action to actuator dict action. Consider that
        one agent could manage more than one actuator. For that reason, it is important to transform the
        action dict to actuator dict actions.

        Args:
            action (Any): The action to be transformed.
            actuators (List[str]): List of actuators controlled by the agent.

        Returns:
            Dict[str, Any]: Transformed actions for the actuators.
        """
        
        # Here the name of the actuator is obtained from the actuators_config 
        # in the environment configuration file.
        pass
        
        
    @override(BaseActionMapper)
    def _agent_to_actuator_action(self, action: Any, actuators: List[str]) -> Dict[str, Any]:
        """
        This method is not used in top_level_agent.
        """
        return {}
    

    @override(BaseActionMapper)
    def action_to_goal(self, action: int | float) -> int | float:
        """
        This method is used to transform the action to a goal. The goal is used to define the reward.

        Args:
            action (Any): The action to be transformed.

        Returns:
            Any: The transformed action.
        """
        return action/(self.action_space_dim-1)
    