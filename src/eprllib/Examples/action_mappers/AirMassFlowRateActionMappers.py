"""
Air Mass Flow Rate ActionMappers
===================================

An actuator called “Ideal Loads Air System” is available with control types called “Air Mass Flow
Rate” (supply air), “Outdoor Air Mass Flow Rate,” “Air Temperature,” and “Air Humidity Ratio.”
These are available in models that use the ideal loads air system, formerly known as purchased
air. The units are kg/s for mass flow rate, C for temperature and kgWater/kgDryAir for humidity
ratio. The unique identifier is the user-defined name of the ZoneHVAC:IdealLoadsAirSystem input
object.
For Air Temperature and Air Humidity Ratio, the overrides are absolute. They are applied after
all other limits have been checked. For mass flow rate, the overrides are not absolute,the internal
controls will still apply the capacity and flow rate limits if defined in the input object. The EMS
override will be ignored if the ideal loads system is off (the availability schedule value is zero or it
has been forced “off” by an availability manager). If both the Air Mass Flow Rate and Outdoor Air
Mass Flow Rate are overridden, the Outdoor Air Mass Flow Rate will not be allowed to be greater
than the override value for Air Mass Flow Rate.
"""
import gymnasium as gym
import numpy as np
from typing import Any, Dict, List, Tuple, Optional
from eprllib.Agents.ActionMappers.BaseActionMapper import BaseActionMapper
from eprllib.Utils.observation_utils import get_actuator_name
from eprllib.Utils.annotations import override

class AirMassFlowRateActionMapper(BaseActionMapper):
    
    @override(BaseActionMapper)
    def setup(self) -> None:
        """
        This class implements the “Air Mass Flow Rate” (supply air) action function.

        Args:
            action_mapper_config (Dict[str, Any]): The configuration of the action function.
            It should contain the following keys:
                - air_mass_flow_rate_range (Tuple[int, int]): The range for the air mass flow rate.
                - ideal_loads_air_system_actuator (Tuple[str, str, str]): The configuration for the actuator.
                
        """
        # The name of the actuator that will be controlled by the agent. 
        # This will be obtained from the actuators_config in the environment configuration file.
        self.air_mass_flow_rate_actuator: Optional[str] = None
        
        # Here we use the config dict to provide the range of the air mass flow rate.
        air_mass_flow_rate_range: Tuple[float|int, float|int] = self.action_mapper_config['air_mass_flow_rate_range']
        self.air_mass_flow_rate_range_max: float|int = max(air_mass_flow_rate_range)
        self.air_mass_flow_rate_range_min: float|int = min(air_mass_flow_rate_range)
        
    
    @override(BaseActionMapper)    
    def get_action_space_dim(self) -> gym.Space[Any]:
        """
        Get the action space of the environment.

        Returns:
            gym.Space: Action space of the environment.
        """
        return gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
    
    
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
        
        self.air_mass_flow_rate_actuator = get_actuator_name(
            self.agent_name,
            actuators_config['Air Mass Flow Rate Actuator'][0],
            actuators_config['Air Mass Flow Rate Actuator'][1],
            actuators_config['Air Mass Flow Rate Actuator'][2]
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
        if self.air_mass_flow_rate_actuator not in actuators:
            return {}
        
        else:
            if action <= 0.0:
                return {self.air_mass_flow_rate_actuator: float(self.air_mass_flow_rate_range_min)}
            
            elif action >= 1.0:
                return {self.air_mass_flow_rate_actuator: float(self.air_mass_flow_rate_range_max)}
            
            else:
                return {self.air_mass_flow_rate_actuator: float(self.air_mass_flow_rate_range_min + (action)*(self.air_mass_flow_rate_range_max - self.air_mass_flow_rate_range_min))}
