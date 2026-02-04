"""
Setpoint triggers
==================

This module contains classes to implement setpoint triggers for controlling actuators in the environment.
"""
import gymnasium as gym
import numpy as np
from typing import Any, Dict, List, Tuple, Optional
from eprllib.Agents.Triggers.BaseTrigger import BaseTrigger
from eprllib.Utils.observation_utils import get_actuator_name
from eprllib.Utils.annotations import override
from eprllib.Utils.agent_utils import get_agent_name, config_validation
from eprllib import logger

class DualSetpointTriggerDiscreteAndAvailabilityTrigger(BaseTrigger):
    REQUIRED_KEYS: Dict[str, Any] = {
        "temperature_range": Tuple[int|float, int|float],
        "actuator_for_cooling": Tuple[str, str, str],
        "actuator_for_heating": Tuple[str, str, str],
        "availability_actuator": Tuple[str, str, str],
        "action_space_dim": int
    }
    
    def __init__(
        self,
        trigger_fn_config: Dict[str, Any]
    ):
        """
        This class implements the Dual Set Point Thermostat action function.

        Args:
            trigger_fn_config (Dict[str, Any]): The configuration of the action function.
            It should contain the following keys:
                - temperature_range (Tuple[int, int]): The temperature range for the setpoints.
                - actuator_for_cooling (Tuple[str, str, str]): The configuration for the cooling actuator.
                - actuator_for_heating (Tuple[str, str, str]): The configuration for the heating actuator.
                - availability_actuator (Tuple[str, str, str]): The configuration for the availability actuator.
                - action_space_dim (int): The dimension of the action space. Must be greater than or equal to 3.
        """
        # Validate the config.
        config_validation(trigger_fn_config, self.REQUIRED_KEYS)
        
        super().__init__(trigger_fn_config)
        
        self.agent_name: Optional[str] = None
        temperature_range: Tuple[float|int, float|int] = trigger_fn_config['temperature_range']
        self.temperature_range_max: float|int = max(temperature_range)
        self.temperature_range_min: float|int = min(temperature_range)
        self.temperature_range_avg: float = (self.temperature_range_max + self.temperature_range_min) / 2
        self.actuator_for_cooling = None
        self.actuator_for_heating = None
        self.availability_actuator = None
        # self.band_gap_range_len: int = trigger_fn_config['band_gap_range_len'] + 1
        self.action_space_dim: int = trigger_fn_config['action_space_dim']
        if self.action_space_dim < 3:
            raise ValueError("The action_space_dim must be greater than or equal to 3.")
    
    @override(BaseTrigger)    
    def get_action_space_dim(self) -> gym.Space[Any]:
        """
        Get the action space of the environment.

        Returns:
            gym.Space: Action space of the environment.
        """
        return gym.spaces.Discrete(self.action_space_dim)
    
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
            self.actuator_for_cooling = get_actuator_name(
                self.agent_name,
                self.trigger_fn_config['actuator_for_cooling'][0],
                self.trigger_fn_config['actuator_for_cooling'][1],
                self.trigger_fn_config['actuator_for_cooling'][2]
            )
            self.actuator_for_heating = get_actuator_name(
                self.agent_name,
                self.trigger_fn_config['actuator_for_heating'][0],
                self.trigger_fn_config['actuator_for_heating'][1],
                self.trigger_fn_config['actuator_for_heating'][2]
            )
            self.availability_actuator = get_actuator_name(
                self.agent_name,
                self.trigger_fn_config['availability_actuator'][0],
                self.trigger_fn_config['availability_actuator'][1],
                self.trigger_fn_config['availability_actuator'][2]
            )
            
        actuator_dict_actions: Dict[str, Any] = {actuator: None for actuator in actuators}
        
        assert self.actuator_for_cooling is not None, "Cooling actuator name has not been initialized."
        assert self.actuator_for_heating is not None, "Heating actuator name has not been initialized."
        assert self.availability_actuator is not None, "Availability actuator name has not been initialized."
        
        if action == 0:
            actuator_dict_actions.update({
                self.actuator_for_cooling: self.temperature_range_avg + 1,
                self.actuator_for_heating: self.temperature_range_avg,
                self.availability_actuator: 0
            })
        
        else:
            actuator_dict_actions.update({
                self.actuator_for_cooling: self.temperature_range_avg + (((action-1)/(self.action_space_dim-2))*(self.temperature_range_max - 1 - self.temperature_range_avg)) + 1,
                self.actuator_for_heating: self.temperature_range_avg - (((action-1)/(self.action_space_dim-2))*(self.temperature_range_avg - self.temperature_range_min)),
                self.availability_actuator: 1
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

class DualSetpointTriggerDiscreteAndAvailabilityTrigger_v2(BaseTrigger):
    REQUIRED_KEYS: Dict[str, Any] = {
        "temperature_range": Tuple[int|float, int|float],
        "actuator_for_cooling": Tuple[str, str, str],
        "actuator_for_heating": Tuple[str, str, str],
        "availability_actuator": Tuple[str, str, str],
        "action_space_dim": int
    }
    
    def __init__(
        self,
        trigger_fn_config: Dict[str, Any]
    ):
        """
        This class implements the Dual Set Point Thermostat action function.

        Args:
            trigger_fn_config (Dict[str, Any]): The configuration of the action function.
            It should contain the following keys:
                - temperature_range (Tuple[int, int]): The temperature range for the setpoints.
                - actuator_for_cooling (Tuple[str, str, str]): The configuration for the cooling actuator.
                - actuator_for_heating (Tuple[str, str, str]): The configuration for the heating actuator.
                - availability_actuator (Tuple[str, str, str]): The configuration for the availability actuator.
                - action_space_dim (int): The dimension of the action space. Must be greater than or equal to 3.
        """
        # Validate the config.
        config_validation(trigger_fn_config, self.REQUIRED_KEYS)
        
        super().__init__(trigger_fn_config)
        
        self.agent_name: Optional[str] = None
        temperature_range: Tuple[float|int, float|int] = trigger_fn_config['temperature_range']
        self.temperature_range_max: float|int = max(temperature_range)
        self.temperature_range_min: float|int = min(temperature_range)
        self.actuator_for_cooling = None
        self.actuator_for_heating = None
        self.availability_actuator = None
        self.deadband: int = trigger_fn_config.get('deadband', 2)
        self.action_space_dim: int = trigger_fn_config['action_space_dim']
        if self.action_space_dim < 3:
            raise ValueError("The action_space_dim must be greater than or equal to 3.")
    
    @override(BaseTrigger)    
    def get_action_space_dim(self) -> gym.Space[Any]:
        """
        Get the action space of the environment.

        Returns:
            gym.Space: Action space of the environment.
        """
        return gym.spaces.Discrete(self.action_space_dim)
    
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
            self.actuator_for_cooling = get_actuator_name(
                self.agent_name,
                self.trigger_fn_config['actuator_for_cooling'][0],
                self.trigger_fn_config['actuator_for_cooling'][1],
                self.trigger_fn_config['actuator_for_cooling'][2]
            )
            self.actuator_for_heating = get_actuator_name(
                self.agent_name,
                self.trigger_fn_config['actuator_for_heating'][0],
                self.trigger_fn_config['actuator_for_heating'][1],
                self.trigger_fn_config['actuator_for_heating'][2]
            )
            self.availability_actuator = get_actuator_name(
                self.agent_name,
                self.trigger_fn_config['availability_actuator'][0],
                self.trigger_fn_config['availability_actuator'][1],
                self.trigger_fn_config['availability_actuator'][2]
            )
            
        actuator_dict_actions: Dict[str, Any] = {actuator: None for actuator in actuators}
        
        assert self.actuator_for_cooling is not None, "Cooling actuator name has not been initialized."
        assert self.actuator_for_heating is not None, "Heating actuator name has not been initialized."
        assert self.availability_actuator is not None, "Availability actuator name has not been initialized."
        
        if action == 0:
            actuator_dict_actions.update({
                self.actuator_for_heating: float(self.temperature_range_min),
                self.actuator_for_cooling: float(self.temperature_range_min + self.deadband),
                self.availability_actuator: 0
            })
        
        else:
            actuator_dict_actions.update({
                self.actuator_for_heating: float(self.temperature_range_min + ((action-1)/(self.action_space_dim-2))*(self.temperature_range_max - self.temperature_range_min - self.deadband)),
                self.actuator_for_cooling: float(self.temperature_range_min + ((action-1)/(self.action_space_dim-2))*(self.temperature_range_max - self.temperature_range_min - self.deadband) + self.deadband),
                self.availability_actuator: 1
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


class DualSetpointTriggerContinuosAndAvailabilityTrigger_v2(BaseTrigger):
    REQUIRED_KEYS: Dict[str, Any] = {
        "actuator_for_cooling": Tuple[str, str, str],
        "actuator_for_heating": Tuple[str, str, str],
        "availability_actuator": Tuple[str, str, str],
    }
    
    def __init__(
        self,
        trigger_fn_config: Dict[str, Any]
    ):
        """
        This class implements the Dual Set Point Thermostat action function.

        Args:
            trigger_fn_config (Dict[str, Any]): The configuration of the action function.
            It should contain the following keys:
                - temperature_range (Tuple[int, int]): The temperature range for the setpoints.
                - actuator_for_cooling (Tuple[str, str, str]): The configuration for the cooling actuator.
                - actuator_for_heating (Tuple[str, str, str]): The configuration for the heating actuator.
                - availability_actuator (Tuple[str, str, str]): The configuration for the availability actuator.
                - action_space_dim (int): The dimension of the action space. Must be greater than or equal to 3.
        """
        # Validate the config.
        config_validation(trigger_fn_config, self.REQUIRED_KEYS)
        
        super().__init__(trigger_fn_config)
        
        
        temperature_range: Tuple[float|int, float|int] = trigger_fn_config.get('temperature_range', (16, 32))
        self.temperature_range_max: float = float(max(temperature_range))
        self.temperature_range_min: float = float(min(temperature_range))
        
        self.deadband: int = trigger_fn_config.get('deadband', 2)
        if (self.temperature_range_max - self.temperature_range_min) < self.deadband:
            raise ValueError(f"The temperature range must be at least {self.deadband} degrees wide to accommodate the deadband.")
        
        self.agent_name: Optional[str] = None
        self.actuator_for_cooling = None
        self.actuator_for_heating = None
        self.availability_actuator = None
    
    @override(BaseTrigger)    
    def get_action_space_dim(self) -> gym.Space[Any]:
        """
        Get the action space of the environment.

        Returns:
            gym.Space: Action space of the environment.
        """
        return gym.spaces.Dict({
            "availability": gym.spaces.Discrete(2), # 0: OFF, 1: ON
            "temperature": gym.spaces.Box(low=self.temperature_range_min, high=self.temperature_range_max, shape=(1,), dtype=np.float32)
        })
    
    
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
            self.actuator_for_cooling = get_actuator_name(
                self.agent_name,
                self.trigger_fn_config['actuator_for_cooling'][0],
                self.trigger_fn_config['actuator_for_cooling'][1],
                self.trigger_fn_config['actuator_for_cooling'][2]
            )
            self.actuator_for_heating = get_actuator_name(
                self.agent_name,
                self.trigger_fn_config['actuator_for_heating'][0],
                self.trigger_fn_config['actuator_for_heating'][1],
                self.trigger_fn_config['actuator_for_heating'][2]
            )
            self.availability_actuator = get_actuator_name(
                self.agent_name,
                self.trigger_fn_config['availability_actuator'][0],
                self.trigger_fn_config['availability_actuator'][1],
                self.trigger_fn_config['availability_actuator'][2]
            )
            
        actuator_dict_actions: Dict[str, Any] = {actuator: None for actuator in actuators}
        
        assert self.actuator_for_cooling is not None, "Cooling actuator name has not been initialized."
        assert self.actuator_for_heating is not None, "Heating actuator name has not been initialized."
        assert self.availability_actuator is not None, "Availability actuator name has not been initialized."
        
        # On-Off action
        if action["availability"] == 0:
            actuator_dict_actions.update({
                self.availability_actuator: 0
            })
        else:
            actuator_dict_actions.update({
                self.availability_actuator: 1
            })
        
        # Temperature actions
        actuator_dict_actions.update({
            self.actuator_for_cooling: action["temperature"][0] + self.deadband,
            self.actuator_for_heating: action["temperature"][0],
        })

        # Check if there is an actuator_dict_actions value equal to None.
        for actuator in actuator_dict_actions:
            if actuator_dict_actions[actuator] is None:
                msg = f"The actuator {actuator} is not in the list of actuators: \n{actuators}.\nThe actuator dict is: \n{actuator_dict_actions}"
                logger.error(msg)
                raise ValueError(msg)
        
        return actuator_dict_actions
    
    @override(BaseTrigger)
    def get_actuator_action(self, action:int|float, actuator: str) -> float|int:
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
    
class DualSetpointContinuosAndConstantFlowTrigger(BaseTrigger):
    REQUIRED_KEYS: Dict[str, Any] = {
        "actuator_for_cooling": Tuple[str, str, str],
        "actuator_for_heating": Tuple[str, str, str],
        # "availability_actuator": Tuple[str, str, str],
        # "air_mass_flow_rate": int|float,
        # "ideal_loads_air_system_actuator": Tuple[str, str, str],
        # "air_temperature_actuator": Tuple[str, str, str],
        # "air_temperature": int|float
    }
    
    def __init__(
        self,
        trigger_fn_config: Dict[str, Any]
    ):
        """
        This class implements the Dual Set Point Thermostat action function.

        Args:
            trigger_fn_config (Dict[str, Any]): The configuration of the action function.
            It should contain the following keys:
                - temperature_range (Tuple[int, int]): The temperature range for the setpoints.
                - actuator_for_cooling (Tuple[str, str, str]): The configuration for the cooling actuator.
                - actuator_for_heating (Tuple[str, str, str]): The configuration for the heating actuator.
                - availability_actuator (Tuple[str, str, str]): The configuration for the availability actuator.
                - action_space_dim (int): The dimension of the action space. Must be greater than or equal to 3.
        """
        # Validate the config.
        config_validation(trigger_fn_config, self.REQUIRED_KEYS)
        
        super().__init__(trigger_fn_config)
        
        self.agent_name: Optional[str] = None
        
        temperature_range: Tuple[float|int, float|int] = trigger_fn_config.get('temperature_range', (16, 32))
        
        self.temperature_range_max: float|int = max(temperature_range)
        self.temperature_range_min: float|int = min(temperature_range)
        
        self.deadband: int = trigger_fn_config.get('deadband', 2)
        if (self.temperature_range_max - self.temperature_range_min) < self.deadband:
            raise ValueError(f"The temperature range must be at least {self.deadband} degrees wide to accommodate the deadband.")
        
        self.actuator_for_cooling = None
        self.actuator_for_heating = None
        
        # Create a .csv file to save the actions values during execution
        # self.actions_file_path: Optional[str] = trigger_fn_config.get('actions_file_path', None)
        # if self.actions_file_path is not None:
        #     self.actions_file = open(self.actions_file_path, 'w')

        
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
            self.actuator_for_cooling = get_actuator_name(
                self.agent_name,
                self.trigger_fn_config['actuator_for_cooling'][0],
                self.trigger_fn_config['actuator_for_cooling'][1],
                self.trigger_fn_config['actuator_for_cooling'][2]
            )
            self.actuator_for_heating = get_actuator_name(
                self.agent_name,
                self.trigger_fn_config['actuator_for_heating'][0],
                self.trigger_fn_config['actuator_for_heating'][1],
                self.trigger_fn_config['actuator_for_heating'][2]
            )
            
        actuator_dict_actions: Dict[str, Any] = {actuator: None for actuator in actuators}
        
        assert self.actuator_for_cooling is not None, "Cooling actuator name has not been initialized."
        assert self.actuator_for_heating is not None, "Heating actuator name has not been initialized."
        action = float(action) # Try to convert the action in to a float number.
        assert type(action) == float, f"The action must be a float but is {type(action)}."
        
        # Save the action to the file.
        # self.actions_file.write(f"{action}\n")
        # self.actions_file.flush()
        
        if action < 0.0:
            actuator_dict_actions.update({
                self.actuator_for_heating: float(self.temperature_range_min),
                self.actuator_for_cooling: float(self.temperature_range_min + self.deadband)
            })
        
        elif action > 1.0:
            actuator_dict_actions.update({
                self.actuator_for_heating: float(self.temperature_range_max - self.deadband),
                self.actuator_for_cooling: float(self.temperature_range_max)
            })
        
        else:
            actuator_dict_actions.update({
                self.actuator_for_heating: float(self.temperature_range_min + (action)*(self.temperature_range_max - self.temperature_range_min - self.deadband)),
                self.actuator_for_cooling: float(self.temperature_range_min + (action)*(self.temperature_range_max - self.temperature_range_min - self.deadband) + self.deadband)
            })

        # Check if there is an actuator_dict_actions value equal to None.
        for actuator in actuator_dict_actions:
            if actuator_dict_actions[actuator] is None:
                msg = f"The actuator {actuator} is not in the list of actuators: \n{actuators}.\nThe actuator dict is: \n{actuator_dict_actions}"
                logger.error(msg)
                raise ValueError(msg)
        
        return actuator_dict_actions
    
    @override(BaseTrigger)
    def get_actuator_action(self, action:int|float, actuator: str) -> float|int:
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
    

class AvailabilityTrigger(BaseTrigger):
    REQUIRED_KEYS = {
        "availability_actuator": Tuple[str, str, str]
    }
    
    def __init__(
        self, 
        trigger_fn_config:Dict[str,Any]
        ):
        """
        This class implements the Dual Set Point Thermostat action function.

        Args:
            trigger_fn_config (Dict[str,Any]): The configuration of the action function.
            It should contain the following keys: agents_type (Dict[str, int]): A dictionary 
            mapping agent names to their types (1 for cooling, 2 for heating, 3 for Availability).
        """
        # Validate the config.
        config_validation(trigger_fn_config, self.REQUIRED_KEYS)
        
        super().__init__(trigger_fn_config)
        
        self.agent_name: Optional[str] = None
        self.availability_actuator: Optional[str] = None
    
    @override(BaseTrigger)    
    def get_action_space_dim(self) -> gym.Space[Any]:
        """
        Get the action space of the environment.

        Returns:
            gym.Space: Action space of the environment.
        """
        return gym.spaces.Discrete(2)
    
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
            self.availability_actuator = get_actuator_name(
                self.agent_name,
                self.trigger_fn_config['availability_actuator'][0],
                self.trigger_fn_config['availability_actuator'][1],
                self.trigger_fn_config['availability_actuator'][2]
            )

        
        actuator_dict_actions: Dict[str, Any] = {actuator: None for actuator in actuators}
        
        assert self.availability_actuator is not None, "Availability actuator name has not been initialized."
        
        if action == 0:
            actuator_dict_actions.update({
                self.availability_actuator: 0
            })
        
        else:
            actuator_dict_actions.update({
                self.availability_actuator: 1
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

    @override(BaseTrigger)
    def action_to_goal(self, action: int | float) -> int | float:
        """
        This method is used to transform the action to a goal. The goal is used to define the reward.

        Args:
            action (Any): The action to be transformed.

        Returns:
            Any: The transformed action.
        """
        return action
    