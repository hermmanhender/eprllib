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
from typing import Dict, Any
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
    
    def transform_action(self, action:float|int, agent_id) -> Any:
        """
        This method is used to transform the actions of the agents before applying.

        Args:
            action (float|int): Action provided by the policy.
            agent_id: The agent that require the action transform.

        Returns:
            Dict[str, Any]: A dict of transformed action for each agent in the environment.
        """
        return action
