"""
Action Function Base Class
==========================

This module contain the base class to create action transformer functions and define
the action space dimention in the environment.

Action transformer functions are used to adapt the action given for the neural network, 
normally an integral for discrete spaces and a float for continuos spaces as Box. The actions
must be adecuated to values required for the actuators in EnergyPlus. Each agent has the
capacity to control one actuator.

ActionFunction must be define in the EnvConfig definition to create the environment and is
called in the EnergyPlusEnvironment.EnergyPlusEnv_v0 class and used in the EnergyPlusRunner.EnergyPlusRunner class
to transform the dict of agent actions to actuator values.
"""
from typing import Dict, Any, List, Tuple
import gymnasium as gym

class ActionFunction:
    """
    Base class to create action transformer functions.
    """
    def __init__(
        self,
        action_fn_config: Dict[str,Any] = {}
    ):
        """
        This class is used to transform the actions of the agents before applying
        them in the environment.

        Args:
            action_fn_config (Dict[str,Any]): Configuration for the action transformer function.
        """
        self.action_fn_config = action_fn_config
    
    def get_action_space_dim(self) -> gym.Space:
        """This method is used to get the action space of the environment.

        Raises:
            NotImplementedError: This method should be implemented in the child class.

        Returns:
            gym.Space: Action space of the environment.
        """
        return NotImplementedError("This method should be implemented in the child class.")
    
    def agent_to_actuator_action(self, action: Any, actuators: List[str]) -> Dict[str,Any]:
        """
        This method is used to transform the agent dict action to actuator dict action. Consider that
        one agent could manage more than one actuator. For that reason it is important to transformt the
        action dict to actuator dict actions.
        
        The actuators are named as: agent_{n}, where n correspond with the order listed in the
        configuration. For example, in the following code the action provide for the policy will
        correspond to the "battery_controller" agent, but this action dict must to be transform
        in two actions, one for the "battery_controller_0" actuator (that allows draw the battery), and
        one for the "battery_controller_1" actuator (that charge the battery):
        
        ```
        from eprllib.Env.EnvConfig import EnvConfig
        eprllib_config = EnvConfig()
        eprllib_config.agents(
            agents_config = {
                "battery_controler": {
                    'ep_actuator_config': [
                        ("Electrical Storage", "Power Draw Rate", "Battery"),
                        ("Electrical Storage", "Power Charge Rate", "Battery")
                        ],
                    'thermal_zone': 'Thermal Zone',
                    'agent_indicator': 1,
                },}
        )
        ```

        Args:
            action (Dict[str, Any]): Action provided by the policy.

        Raises:
            NotImplementedError: This method should be implemented in the child class.

        Returns:
            Dict[str, Any]: A dict of transformed action for each agent in the environment.
        """
        raise NotImplementedError("This method should be implemented in the child class.")
    
    def get_actuator_action(self, action:float|int, actuator: str) -> Any:
        """
        This method is used to get the actions of the actuators after transform the
        agent dict action to actuator dict action with agent_to_actuator_action.

        Args:
            action (float|int): Action provided by the policy and transformed by agent_to_actuator_action.
            actuator: The actuator that require the action.

        Returns:
            Dict[str, Any]: A dict of transformed action for each agent in the environment.
        """
        return action

class ActionSpec:
    """
    ActionSpec is the base class for an action specification to safe configuration of the object.
    """
    def __init__(
        self,
        actuators: List[Tuple[str,str,str]] = NotImplemented,
        action_fn: ActionFunction = None,
        action_fn_config: Dict[str, Any] = {},
        ):
        """
        _Description_
        
        Args:
            actuators (List[Tuple[str,str,str]]): Actuators are the way that users modify the program at 
            runtime using custom logic and calculations. Not every variable inside EnergyPlus can be 
            actuated. This is intentional, because opening that door could allow the program to run at 
            unrealistic conditions, with flowimbalances or energy imbalances, and many other possible problems.
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
            
            action_fn (ActionFunction): In the definition of the action space, usualy is use the discrete form of the 
            gym spaces. In general, we don't use actions from 0 to n directly in the EnergyPlus simulation. With the 
            objective to transform appropiately the discret action into a value action for EP we define the action_fn. 
            This function take the arguments agent_id and action. You can find examples in eprllib.ActionFunctions.
            
            action_fn_config (Dict[str, Any]):
        """
        if actuators == NotImplemented:
            raise NotImplementedError("actuators must be defined.")
        self.actuators = actuators
        self.action_fn = action_fn
        self.action_fn_config = action_fn_config
    
    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)
