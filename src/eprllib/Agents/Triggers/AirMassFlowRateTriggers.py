"""
Air Mass Flow Rate triggers
==================

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
from eprllib.Agents.Triggers.BaseTrigger import BaseTrigger
from eprllib.Utils.observation_utils import get_actuator_name
from eprllib.Utils.annotations import override
from eprllib.Utils.agent_utils import get_agent_name, config_validation
from eprllib import logger

class AirMassFlowRateTrigger(BaseTrigger):
    REQUIRED_KEYS: Dict[str, Any] = {
        "air_mass_flow_rate_range": Tuple[int|float, int|float],
        "ideal_loads_air_system_actuator": Tuple[str, str, str]
    }
    
    def __init__(
        self,
        trigger_fn_config: Dict[str, Any]
    ):
        """
        This class implements the “Air Mass Flow Rate” (supply air) action function.

        Args:
            trigger_fn_config (Dict[str, Any]): The configuration of the action function.
            It should contain the following keys:
                - air_mass_flow_rate_range (Tuple[int, int]): The range for the air mass flow rate.
                - ideal_loads_air_system_actuator (Tuple[str, str, str]): The configuration for the actuator.
                
        """
        # Validate the config.
        config_validation(trigger_fn_config, self.REQUIRED_KEYS)
        
        super().__init__(trigger_fn_config)
        
        self.agent_name: Optional[str] = None
        air_mass_flow_rate_range: Tuple[float|int, float|int] = trigger_fn_config['air_mass_flow_rate_range']
        self.air_mass_flow_rate_range_max: float|int = max(air_mass_flow_rate_range)
        self.air_mass_flow_rate_range_min: float|int = min(air_mass_flow_rate_range)
        self.ideal_loads_air_system_actuator = None
    
    @override(BaseTrigger)    
    def get_action_space_dim(self) -> gym.Space[Any]:
        """
        Get the action space of the environment.

        Returns:
            gym.Space: Action space of the environment.
        """
        return gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
    
    @override(BaseTrigger)
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
                self.trigger_fn_config['ideal_loads_air_system_actuator'][0],
                self.trigger_fn_config['ideal_loads_air_system_actuator'][1],
                self.trigger_fn_config['ideal_loads_air_system_actuator'][2]
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
    
    @override(BaseTrigger)
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
