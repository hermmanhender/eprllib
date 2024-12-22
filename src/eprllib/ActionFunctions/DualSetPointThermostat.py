"""
Dual Set Point Thermostat
=========================

This module contains the implementation of the Dual Set Point Thermostat action function.
The action function transforms the action values from [0,1] to the appropriate range for the
actuator type (cooling, heating, or flow rate). The transformation is linear for cooling and
heating, and linear for flow rate.
"""

from typing import Dict, Any,Tuple
from eprllib.ActionFunctions.ActionFunctions import ActionFunction
import gymnasium as gym

class DualSetPointThermostat(ActionFunction):
    def __init__(self, action_fn_config:Dict[str,Any]):
        """
        This class implements the Dual Set Point Thermostat action function.

        Args:
            action_fn_config (Dict[str,Any]): The configuration of the action function.
            It should contain the following keys: agents_type (Dict[str, int]): A dictionary 
            mapping agent names to their types (1 for cooling, 2 for heating, 3 for Availability).
        """
        super().__init__(action_fn_config)
        self.agents_type: Dict[str, Any] = action_fn_config['agents_type']
        self.heating_range: Tuple[int,int] = action_fn_config['heating_range']
        self.cooling_range: Tuple[int,int] = action_fn_config['cooling_range']
    
    
    def get_action_space_dim(self) -> gym.Space:
        """
        Get the action space of the environment.

        Returns:
            gym.Space: Action space of the environment.
        """
        return gym.spaces.Discrete(11)
    
    
    def transform_action(self, action:float|int, agent_id) -> Any:
        
        # === Cooling ===
        if self.agents_type[agent_id] == 1: 
            # transfort value between [0,1] to [23,29]
            action = min(self.cooling_range[0],self.cooling_range[1]) \
                + (action/10)*(max(self.cooling_range[0],self.cooling_range[1]) \
                    - min(self.cooling_range[0],self.cooling_range[1]))
        
        # === Heating ===
        elif self.agents_type[agent_id] == 2:
            # transfort value between [0,1] to [18,22]
            action = min(self.heating_range[0],self.heating_range[1]) \
                + (action/10)*(max(self.heating_range[0],self.heating_range[1]) \
                    - min(self.heating_range[0],self.heating_range[1]))
            
        # === Availability ===
        elif self.agents_type[agent_id] == 3:
            if action == 0:
                action = 0
            else:
                action = 1
            
        # === Opening window ===
        elif self.agents_type[agent_id] in [4,5,6,7]:
            action = action/10
        
        # === Window Shading Control ===
        elif self.agents_type[agent_id] == 8:
            # –1.0: No shading device.
            # 0.0: Shading device is off (applies to shades and blinds).
            # 1.0: Interior shade is on.
            # 2.0: Glazing is switched to a darker state (switchable glazings only).
            # 3.0: Exterior shade is on.
            # 4.0: Exterior screen is on.
            # 6.0: Interior blind is on.
            # 7.0: Exterior blind is on.
            # 8.0: Between-glass shade is on.
            # 9.0: Between-glass blind is on.
            if action == 0:
                action = 0
            else:
                action = 1
        else:
            raise ValueError(f"Actuator type not valid. Actuator values are: {self.agents_type} and the valid options are: (1,2,3,4,5,6,7)")
        
        return action
    