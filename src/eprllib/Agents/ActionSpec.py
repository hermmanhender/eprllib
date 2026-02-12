"""
Specification for the action space and actuators
===========================================================
This module defines the `ActionSpec` class, which is used to specify the 
configuration of action space and actuators for agents in reinforcement 
learning environments.
"""
from typing import Dict, List, Tuple, Any, Optional
from eprllib import logger

class ActionSpec:
    """
    ActionSpec is the base class for an action specification to safe configuration of the object.
    """
    actuators: Optional[List[Tuple[str, str, str]]] = None
    
    def __init__(
        self,
        actuators: Optional[List[Tuple[str, str, str]]] = None,
    ):
        """
        Construction method.
        
        Args:
            actuators (List[Tuple[str, str, str]]): Actuators are the way that users modify the program at 
            runtime using custom logic and calculations. Not every variable inside EnergyPlus can be 
            actuated. This is intentional, because opening that door could allow the program to run at 
            unrealistic conditions, with flow imbalances or energy imbalances, and many other possible problems.
            Instead, a specific set of items are available to actuate, primarily control functions, 
            flow requests, and environmental boundary conditions. These actuators, when used in conjunction 
            with the runtime API and data exchange variables, allow a user to read data, make decisions and 
            perform calculations, then actuate control strategies for subsequent time steps.
            Actuator functions are similar, but not exactly the same, as for variables. An actuator
            handle/ID is still looked up, but it takes the actuator type, component name, and control
            type, since components may have more than one control type available for actuation. The
            actuator can then be “actuated” by calling a set-value function, which overrides an internal
            value, and informs EnergyPlus that this value is currently being externally controlled. To
            allow EnergyPlus to resume controlling that value, there is an actuator reset function as well.
            One agent can manage several actuators.
        
        Raises:
            ValueError: If the actuators are not defined as a list of tuples of 3 elements.
            KeyError: If an invalid key is provided when setting an item.
            
        """
        if actuators is None:
            print("No actuators provided.")
            self.actuators = []
        else: 
            self.actuators = actuators
    
    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any) -> None:
        valid_keys = self.__dict__.keys()
        if key not in valid_keys:
            msg = f"ActionSpec: Invalid key: {key}."
            logger.error(msg)
            raise KeyError(msg)
        setattr(self, key, value)
        
    def build(self) -> Dict[str, Any]:
        """
        This method is used to build the ActionSpec object.
        """
        # Check that the actuators are defined as a list of tuples of 3 elements.
        for actuator in self.actuators:
            if len(actuator) != 3:
                msg = f"ActionSpec: The actuators must be defined as a list of tuples of 3 elements but {len(actuator)} was given."
                logger.error(msg)
                raise ValueError(msg)
        
        return vars(self)
    