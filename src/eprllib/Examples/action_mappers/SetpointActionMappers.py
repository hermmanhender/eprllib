"""
Setpoint ActionMappers
========================

This module contains classes to implement setpoint ActionMappers for controlling actuators in the environment.
"""
import gymnasium as gym
from typing import Any, Dict, List, Tuple, Optional
from eprllib.Agents.ActionMappers.BaseActionMapper import BaseActionMapper
from eprllib.Utils.observation_utils import get_actuator_name
from eprllib.Utils.annotations import override


class DualSetpointDiscreteAndAvailabilityActionMapper(BaseActionMapper):
    
    @override(BaseActionMapper)
    def setup(self):
        # Here you have access to the self.action_mapper_config and self.agent_name
        
        # The name of the actuator that will be controlled by the agent. 
        # This will be obtained from the actuators_config in the environment configuration file.
        self.actuator_for_cooling: Optional[str] = None
        self.actuator_for_heating: Optional[str] = None
        self.availability_actuator: Optional[str] = None

        # Here we use the config dict to provide the action space dimension.
        self.action_space_dim: int = self.action_mapper_config.get("action_space_dim", 11)
        if self.action_space_dim < 3:
            raise ValueError("The action_space_dim must be greater than or equal to 3.")
        
        temperature_range: Tuple[float|int, float|int] = self.action_mapper_config['temperature_range']
        self.temperature_range_max: float|int = max(temperature_range)
        self.temperature_range_min: float|int = min(temperature_range)
        self.deadband: int = self.action_mapper_config.get('deadband', 2)
        
    
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
        self.actuator_for_cooling = get_actuator_name(
            self.agent_name,
            actuators_config["Cooling Actuator"][0],
            actuators_config["Cooling Actuator"][1],
            actuators_config["Cooling Actuator"][2]
        )
        self.actuator_for_heating = get_actuator_name(
            self.agent_name,
            actuators_config["Heating Actuator"][0],
            actuators_config["Heating Actuator"][1],
            actuators_config["Heating Actuator"][2]
        )
        self.availability_actuator = get_actuator_name(
            self.agent_name,
            actuators_config["Availability Actuator"][0],
            actuators_config["Availability Actuator"][1],
            actuators_config["Availability Actuator"][2]
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
        """ 
        assert self.actuator_for_cooling in actuators, "Cooling actuator name has not been initialized."
        assert self.actuator_for_heating in actuators, "Heating actuator name has not been initialized."
        assert self.availability_actuator in actuators, "Availability actuator name has not been initialized."
        
        if action == 0:
            return {
                self.actuator_for_heating: float(self.temperature_range_min),
                self.actuator_for_cooling: float(self.temperature_range_min + self.deadband),
                self.availability_actuator: 0
            }
        
        else:
            return {
                self.actuator_for_heating: float(self.temperature_range_min + ((action-1)/(self.action_space_dim-2))*(self.temperature_range_max - self.temperature_range_min - self.deadband)),
                self.actuator_for_cooling: float(self.temperature_range_min + ((action-1)/(self.action_space_dim-2))*(self.temperature_range_max - self.temperature_range_min - self.deadband) + self.deadband),
                self.availability_actuator: 1
            }

