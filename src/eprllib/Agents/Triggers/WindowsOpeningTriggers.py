"""
Windows Opening Triggers
========================

This module contains classes to implement window opening triggers for controlling actuators in the environment.
"""
import gymnasium as gym
from typing import Any, Dict, List, Tuple # type: ignore
from eprllib.Agents.Triggers.BaseTrigger import BaseTrigger
from eprllib.Utils.observation_utils import get_actuator_name
from eprllib.Utils.annotations import override
from eprllib.Utils.agent_utils import get_agent_name, config_validation
from eprllib import logger

class WindowsOpeningTrigger(BaseTrigger):
    REQUIRED_KEYS: Dict[str, Any] = {
        "window_actuator": Tuple[str, str, str],
    }
    
    def __init__(
        self,
        trigger_fn_config: Dict[str, Any]
    ):
        """
        This class implements the window opening action function.

        Args:
            trigger_fn_config (Dict[str, Any]): The configuration of the action function.
            It should contain the following keys:
                - window_actuator (Tuple[str, str, str]): The configuration for the window actuator.
        """
        # Validate the config.
        config_validation(trigger_fn_config, self.REQUIRED_KEYS)
        
        super().__init__(trigger_fn_config)
        
        self.agent_name = None
        self.window_actuator = None
    
    @override(BaseTrigger)    
    def get_action_space_dim(self) -> gym.Space[Any]:
        """
        Get the action space of the environment.

        Returns:
            gym.Space: Action space of the environment.
        """
        return gym.spaces.Discrete(11)
    
    @override(BaseTrigger)
    def agent_to_actuator_action(self, action: Any, actuators: List[str]) -> Dict[str, Any]:
        """
        Transform the agent action to actuator action.

        Args:
            action (Any): The action to be transformed.
            actuators (List[str]): List of actuators controlled by the agent.

        Returns:
            Dict[str, Any]: Transformed actions for the actuators.
        """
        if self.agent_name is None:
            self.agent_name = get_agent_name(actuators)
            self.window_actuator = get_actuator_name(
                self.agent_name,
                self.trigger_fn_config['window_actuator'][0],
                self.trigger_fn_config['window_actuator'][1],
                self.trigger_fn_config['window_actuator'][2]
            )
        
        assert self.window_actuator is not None, "Window actuator name has not been initialized."
        
        actuator_dict_actions = {actuator: None for actuator in actuators}
        actuator_dict_actions.update({self.window_actuator: action / 10})
        
        # Check if there is an actuator_dict_actions value equal to None.
        for actuator in actuator_dict_actions:
            if actuator_dict_actions[actuator] is None:
                msg = f"The actuator {actuator} is not in the list of actuators: \n{actuators}.\nThe actuator dict is: \n{actuator_dict_actions}"
                logger.error(msg)
                raise ValueError(msg)
            
        return actuator_dict_actions
    
    @override(BaseTrigger)
    def get_actuator_action(self, action: float | int, actuator: str) -> Any:
        """
        Get the actuator action.

        Args:
            action (float | int): The action to be applied.
            actuator (str): The actuator to be controlled.

        Returns:
            Any: The action for the actuator.
        """
        return action

    @override(BaseTrigger)
    def action_to_goal(self, action: int | float) -> int | float:
        """
        This method is used to transform the action to a goal. The goal is used to define the reward.

        Args:
            action (Any): The action to be transformed.

        Returns:
            Any: The transformed action.
        """
        return action
    