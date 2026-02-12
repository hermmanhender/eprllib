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
from eprllib.Utils.agent_utils import get_agent_name, config_validation
from eprllib import logger

class SystemNodeSetpointsTemperatureSetpoint(BaseActionMapper):
    REQUIRED_KEYS: Dict[str, Any] = {
        "variable_range": Tuple[int|float, int|float],
        "actuator_config": Tuple[str, str, str]
    }
    
    def __init__(
        self,
        action_mapper_config: Dict[str, Any]
    ):
        """
        This class implements the “Air Mass Flow Rate” (supply air) action function.

        Args:
            action_mapper_config (Dict[str, Any]): The configuration of the action function.
            It should contain the following keys:
                - variable_range (Tuple[int, int]): The range for variable.
                - actuator_config (Tuple[str, str, str]): The configuration for the actuator.
                
        """
        # Validate the config.
        config_validation(action_mapper_config, self.REQUIRED_KEYS)
        
        super().__init__(action_mapper_config)
        
        self.agent_name: Optional[str] = None
        air_mass_flow_rate_range: Tuple[float|int, float|int] = action_mapper_config['variable_range']
        self.air_mass_flow_rate_range_max: float|int = max(air_mass_flow_rate_range)
        self.air_mass_flow_rate_range_min: float|int = min(air_mass_flow_rate_range)
        self.ideal_loads_air_system_actuator = None
    
    @override(BaseActionMapper)    
    def get_action_space_dim(self) -> gym.Space[Any]:
        """
        Get the action space of the environment.

        Returns:
            gym.Space: Action space of the environment.
        """
        return gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
    
    @override(BaseActionMapper)
    def agent_to_actuator_action(self, action: Any, actuators: List[str]) -> Dict[str,Any]:
        """
        This method is used to transform the agent action to actuator dict action. Consider that
        one agent may manage more than one actuator.

        Args:
            action (Any): The agent action, normally an int of float.
            actuators (List[str]): The list of actuator names that the agent manage.

        Return:
            Dict[str, Any]: Dictionary with the actions for the actuators.
        """
        if self.agent_name is None:
            self.agent_name = get_agent_name(actuators)
            self.ideal_loads_air_system_actuator = get_actuator_name(
                self.agent_name,
                self.action_mapper_config['actuator_config'][0],
                self.action_mapper_config['actuator_config'][1],
                self.action_mapper_config['actuator_config'][2]
            )
            
        actuator_dict_actions: Dict[str, Any] = {actuator: None for actuator in actuators}
        
        assert self.ideal_loads_air_system_actuator is not None, "ideal_loads_air_system_actuator actuator name has not been initialized."
        
        if action <= 0.0:
            actuator_dict_actions.update({
                self.ideal_loads_air_system_actuator: float(self.air_mass_flow_rate_range_min)
            })
        
        elif action >= 1.0:
            actuator_dict_actions.update({
                self.ideal_loads_air_system_actuator: float(self.air_mass_flow_rate_range_max)
            })
        
        else:
            actuator_dict_actions.update({
                self.ideal_loads_air_system_actuator: float(self.air_mass_flow_rate_range_min + (action)*(self.air_mass_flow_rate_range_max - self.air_mass_flow_rate_range_min))
            })

        # Check if there is an actuator_dict_actions value equal to None.
        for actuator in actuator_dict_actions:
            if actuator_dict_actions[actuator] is None:
                msg = f"The actuator {actuator} is not in the list of actuators: \n{actuators}.\nThe actuator dict is: \n{actuator_dict_actions}"
                logger.error(msg)
                raise ValueError(msg)
        
        return actuator_dict_actions
    
    @override(BaseActionMapper)
    def get_actuator_action(self, action:float|int, actuator: str) -> Any:
        """
        This method is used to get the actions of the actuators after transform the 
        agent action to actuator action.

        Args:
            action (float|int): The action to be transformed.
            actuator (str): The actuator name.

        Returns:
            Any: The transformed action.
        """
        return action
