"""
Exhaust Fan ActionMappers
=============================

This module contains classes to implement ActionMappers for controlling exhaust fan actuators in the environment.
"""
import gymnasium as gym
import numpy as np
from typing import Any, Dict, List, Tuple, Optional
from eprllib.Agents.ActionMappers.BaseActionMapper import BaseActionMapper
from eprllib.Utils.observation_utils import get_actuator_name
from eprllib.Utils.annotations import override
from eprllib.Utils.agent_utils import get_agent_name, config_validation
from eprllib import logger

class ExhaustFanActionMapper(BaseActionMapper):
    REQUIRED_KEYS: Dict[str, Any] = {
        "modes": List[float|int],
        "exhaust_fan_actuator": Tuple[str, str, str]
    }
    
    def __init__(
        self,
        action_mapper_config: Dict[str, Any]
    ):
        """
        This class implements the Exhaust Fan actions.

        Args:
            action_mapper_config (Dict[str, Any]): The configuration of the action function.
            It should contain the following keys:
                - modes (List[float]): The flow factor to modify the maximum flow of the exhaust fan. The order 
                in the list corresponds with mode of the Fan, usually mode 0 is off and mode 1 has the lower 
                flow factor. No more than 11 modes are allowed.
                - exhaust_fan_actuator (Tuple[str, str, str]): The configuration for the exhaust fan actuator.
                The first element is the agent name, the second is the actuator name, and the third is the actuator type.
        Raises:
            ValueError: If the configuration is not valid, or if the modes are not in the range [0, 1], or if the length of the modes is larger than 11.
        """
        # Validate the config.
        config_validation(action_mapper_config, self.REQUIRED_KEYS)
        
        super().__init__(action_mapper_config)
        
        self.agent_name = None
        self.modes: List[float] = action_mapper_config['modes']
        self.exhaust_fan_actuator: Optional[str] = None
        
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
        return gym.spaces.Discrete(11)
    
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
            
        Raises:
            ValueError: If the actuator is not in the list of actuators.
        """
        if self.agent_name is None:
            self.agent_name = get_agent_name(actuators)
            self.exhaust_fan_actuator = get_actuator_name(
                self.agent_name,
                self.action_mapper_config['exhaust_fan_actuator'][0],
                self.action_mapper_config['exhaust_fan_actuator'][1],
                self.action_mapper_config['exhaust_fan_actuator'][2]
            )
            
        actuator_dict_actions: Dict[str, Any] = {actuator: None for actuator in actuators}
        
        # TODO: This can be not beneficial for the selection of an action, because introduce noise
        # in the effect of the actions. The best way of avoid this (I think) it to introduce a mask
        # for the actions that reduce the probability of choose the actions that are not in the modes.
        if action >= len(self.modes):
            action_list = [_ for _ in range(len(self.modes))]
            action = np.random.choice(action_list)
        
        # The actuator name is optional in the __init__ but it is initialized in the first call to this method.
        # We need to assert that it is not None to avoid a type error.
        assert self.exhaust_fan_actuator is not None, "Exhaust fan actuator name has not been initialized."

        actuator_dict_actions.update({
            self.exhaust_fan_actuator: self.modes[action],
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
            action (float|int): The action of the agent.
            actuator (str): The name of the actuator.

        Returns:
            Any: The action of the actuator.
        """
        return action
