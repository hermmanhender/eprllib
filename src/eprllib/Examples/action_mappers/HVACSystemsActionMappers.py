"""
HVAC Systems actuators ActionMappers
======================================

System Node Setpoints
----------------------

A series of actuators is available for all the setpoints that can be placed on system nodes. System
nodes are used to define air and plant loops, and a natural application of EMS is to control the
setpoints at these nodes. The node actuators are all called “System Node Setpoint.” There are
nine control types:

* Temperature Setpoint, (°C)
* Temperature Minimum Setpoint (°C)
* Temperature Maximum Setpoint (°C)
* Humidity Ratio Setpoint (kgWater/kgDryAir)
* Humidity Ratio Minimum Setpoint (kgWater/kgDryAir)
* Humidity Ratio Maximum Setpoint (kgWater/kgDryAir)
* Mass Flow Rate Setpoint (kg/s)
* Mass Flow Rate Minimum Available Setpoint (kg/s)
* Mass Flow Rate Maximum Available Setpoint (kg/s)

Using these actuators is natural with an EMS. Typically, the controller would place the setpoint
on the outlet node. Then the component’s low-level controller should operate to meet the leaving
setpoint. Setting the setpoints on nodes should be a common application for the EMS.

Although all nine possible setpoints are available as EMS actuators, it does not follow that
EnergyPlus can use all of them. Most components can use only one or two setpoints. If a component
cannot control to meet the setpoints on a node, the actuator will do nothing.

"""
import gymnasium as gym
import numpy as np
from typing import Any, Dict, List, Tuple, Optional
from eprllib.Agents.ActionMappers.BaseActionMapper import BaseActionMapper
from eprllib.Utils.observation_utils import get_actuator_name
from eprllib.Utils.annotations import override

class SystemNodeSetpointsTemperatureSetpoint(BaseActionMapper):
    
    @override(BaseActionMapper)
    def setup(self):
        """
        Sets up the components of the module.

        This is called automatically during the __init__ method of this class.
        """
        # The name of the actuator that will be controlled by the agent. 
        # This will be obtained from the actuators_config in the environment configuration file.
        self.hvac_system_actuator: Optional[str] = None
        
        # Here we use the config dict to provide the range of the air mass flow rate.
        air_mass_flow_rate_range: Tuple[float|int, float|int] = self.action_mapper_config['variable_range']
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
        
        # Here the name of the actuator is obtained from the actuators_config 
        # in the environment configuration file.
        self.hvac_system_actuator = get_actuator_name(
            self.agent_name,
            actuators_config["HVAC System Actuator"][0],
            actuators_config["HVAC System Actuator"][1],
            actuators_config["HVAC System Actuator"][2]
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
        if self.hvac_system_actuator not in actuators:
            return {}
        else:
            if action <= 0.0:
                return {self.hvac_system_actuator: float(self.air_mass_flow_rate_range_min)}
            
            elif action >= 1.0:
                return {self.hvac_system_actuator: float(self.air_mass_flow_rate_range_max)}
            
            else:
                return {self.hvac_system_actuator: float(self.air_mass_flow_rate_range_min + \
                    (action) * (self.air_mass_flow_rate_range_max - self.air_mass_flow_rate_range_min))}
