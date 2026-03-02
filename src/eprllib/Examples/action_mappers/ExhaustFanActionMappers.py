"""
Exhaust Fan ActionMappers
=============================

This module contains classes to implement ActionMappers for controlling exhaust fan actuators in the environment.
"""
import gymnasium as gym
from typing import Any, Dict, List, Tuple, Optional
from eprllib.Agents.ActionMappers.BaseActionMapper import BaseActionMapper
from eprllib.Utils.observation_utils import get_actuator_name
from eprllib.Utils.annotations import override
from eprllib import logger

class ExhaustFanActionMapper(BaseActionMapper):
    
    @override(BaseActionMapper)
    def setup(self):
        
        # The name of the actuator that will be controlled by the agent. 
        # This will be obtained from the actuators_config in the environment configuration file.
        self.exhaust_fan_actuator: Optional[str] = None
        
        # Here we use the config dict to provide the action space dimension.
        self.action_space_dim: int = self.action_mapper_config.get("action_space_dim", 11)
        
        # Here we use the config dict to provide the modes of the exhaust fan.
        self.modes: List[float] = self.action_mapper_config['modes']
        
        #  Check if the lenght of the modes are larger than 11 (that is the action space for this class).
        if len(self.modes) > 11:
            msg = f"The lenght of the modes must be less than 11. The lenght of the modes is {len(self.modes)}."
            logger.error(msg)
            raise ValueError(msg)
        
        # Check that all the elements in self.modes list are floats in the range [0, 1].
        for mode in self.modes:
            if mode < 0 or mode > 1:
                msg = f"The mode {mode} is not in the range [0, 1]."
                logger.error(msg)
                raise ValueError(msg)
    
    
    @override(BaseActionMapper)    
    def get_action_space_dim(self) -> gym.Space[Any]:
        """
        Get the action space of the environment.

        Returns:
            gym.Space: Action space of the environment.
        """
        return gym.spaces.Discrete(self.action_space_dim)
    
    
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
        self.exhaust_fan_actuator = get_actuator_name(
            self.agent_name,
            actuators_config["Exhaust Fan Actuator"][0],
            actuators_config["Exhaust Fan Actuator"][1],
            actuators_config["Exhaust Fan Actuator"][2]
        )
        
        
    @override(BaseActionMapper)
    def _agent_to_actuator_action(self, action: Any, actuators: List[str]) -> Dict[str,Any]:
        """
        This method is used to transform the agent action to actuator dict action. Consider that
        one agent may manage more than one actuator.

        Args:
            action (Any): The agent action, normally an int of float.
            actuators (List[str]): The list of actuator names that the agent manage.

        Return:
            Dict[str, Any]: Dictionary with the actions for the actuators.
            
        Raises:
            ValueError: If the actuator is not in the list of actuators.
        """
        if self.exhaust_fan_actuator in actuators:
            return {self.exhaust_fan_actuator: self.modes[action]}
        else:
            return {}
