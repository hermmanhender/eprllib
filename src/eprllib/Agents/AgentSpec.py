"""
Defining agents
================

This module implements the classes to define agents. Agents are defined by the ``AgentSpec`` class. This class
contains the observation, filter, action, trigger, and reward specifications. The observation is defined by the
``ObservationSpec`` class. The filter is defined by the ``FilterSpec`` class. The action is defined by the ``ActionSpec`` class.
The trigger is defined by the ``TriggerSpec`` class. The reward is defined by the ``RewardSpec`` class.

The ``AgentSpec`` class has a method called ``build`` that is used to build the ``AgentSpec`` object. This method is used to
validate the properties of the object and to return the object as a dictionary. It is used internally when you build
the environment to provide it to RLlib.
"""

from typing import Dict, Any, List, Tuple

from eprllib.Agents.Rewards.BaseReward import BaseReward
from eprllib.Agents.Triggers.BaseTrigger import BaseTrigger
from eprllib.Agents.Filters.BaseFilter import BaseFilter
from eprllib.Agents.Filters.DefaultFilter import DefaultFilter
from eprllib.Utils.Utils import validate_properties

class RewardSpec:
    """
    RewardSpec is the base class for an reward specification to safe configuration of the object.
    """
    def __init__(
        self,
        reward_fn: BaseReward = NotImplemented,
        reward_fn_config: Dict[str, Any] = {},
        ):
        """
        Construction method.
        
        Args:
            reward_fn (BaseReward): The reward funtion take the arguments EnvObject (the GymEnv class) and the infos 
            dictionary. As a return, gives a float number as reward. See eprllib.Agents.Rewards for examples.
            
        """
        if reward_fn == NotImplemented:
            raise NotImplementedError("reward_fn must be implemented")
            
        self.reward_fn = reward_fn
        self.reward_fn_config = reward_fn_config
    
    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)
    
    def build(self) -> Dict:
        """
        This method is used to build the ActionSpec object.
        """
        # Check that the variables defined in RewardSpec are the allowed in the RewardSpec base
        # class.
        # expected_types = {
        #     'reward_fn': BaseReward,
        #     'reward_fn_config': dict,
        # }
        
        # is_valid, errors = validate_properties(self, expected_types)
        # if is_valid:
        #     print("All properties have correct types")
        # else:
        #     print("Validation errors:")
        #     for error in errors:
        #         print(f"- {error}")
            
        return vars(self)
        
class ObservationSpec:
    """
    ObservationSpec is the base class for an observation specification to safe configuration of the object.
    """
    def __init__(
        self,
        variables: List[Tuple[str, str]] = None,
        internal_variables: List[str] = None,
        meters: List[str] = None,
        simulation_parameters: Dict[str, bool] = {},
        zone_simulation_parameters: Dict[str, bool] = {},
        use_one_day_weather_prediction: bool = False,
        prediction_hours: int = 24,
        prediction_variables: Dict[str, bool] = {},
        use_actuator_state: bool = False,
        other_obs: Dict[str, float | int] = {}
    ):
        """
        Construction method.
        
        Args:
            variables (List[Tuple[str, str]]): Variables represent time series output variables in the simulation. There are thousands
            of variables made available based on the specific configuration. A user typically requests
            variables to be in their output files by adding Output:Variable objects to the input file. It
            is important to note that if the user does not request these variables, they are not tracked,
            and thus not available on the API.
            
            internal_variables (List[str]): Internal variables form a category of built-in data accessible for eprllib.
            They are internal in that they access information about the input file from inside EnergyPlus. Internal variables 
            are automatically made available. The EDD file lists the specific internal variable types, their unique
            identifying names, and the units. The rest of this section provides information about specific internal variables.
            To see the EDD file, run the simulation to generate the output files and configure an object in Output:EnergyManagementSystem
            with "Internal Variable Availability Dictionary Reporting" to Verbose.
            
            meters (List[str]): Meters represent groups of variables which are collected together, much like a meter on
            a building which represents multiple energy sources. Meters are handled the same way as
            variables, except that meters do not need to be requested prior running a simulation. From
            an API standpoint, a client must simply get a handle to a meter by name, and then access
            the meter value by using a get-value function on the API.
            
            simulation_parameters (Dict[str, bool]): A number of parameters are made available as they vary through the
            simulation, including the current simulation day of week, day of year, hour, and many other
            things. These do not require a handle, but are available through direct function calls. The keys (all by default False) are:        
                'actual_date_time', 'actual_time', 'current_time', 'day_of_month', 'day_of_week', 'day_of_year', 'holiday_index',
                'hour', 'minutes', 'month', 'num_time_steps_in_hour', 'year', 'is_raining', 'sun_is_up', 'today_weather_albedo_at_time',
                'today_weather_beam_solar_at_time', 'today_weather_diffuse_solar_at_time', 'today_weather_horizontal_ir_at_time',
                'today_weather_is_raining_at_time', 'today_weather_is_snowing_at_time', 'today_weather_liquid_precipitation_at_time',
                'today_weather_outdoor_barometric_pressure_at_time', 'today_weather_outdoor_dew_point_at_time', 
                'today_weather_outdoor_dry_bulb_at_time', 'today_weather_outdoor_relative_humidity_at_time', 'today_weather_sky_temperature_at_time',
                'today_weather_wind_direction_at_time', 'today_weather_wind_speed_at_time', 'tomorrow_weather_albedo_at_time',
                'tomorrow_weather_beam_solar_at_time', 'tomorrow_weather_diffuse_solar_at_time', 'tomorrow_weather_horizontal_ir_at_time',
                'tomorrow_weather_is_raining_at_time', 'tomorrow_weather_is_snowing_at_time', 'tomorrow_weather_liquid_precipitation_at_time',
                'tomorrow_weather_outdoor_barometric_pressure_at_time', 'tomorrow_weather_outdoor_dew_point_at_time', 'tomorrow_weather_outdoor_dry_bulb_at_time',
                'tomorrow_weather_outdoor_relative_humidity_at_time', 'tomorrow_weather_sky_temperature_at_time', 'tomorrow_weather_wind_direction_at_time',
                'tomorrow_weather_wind_speed_at_time'
            
            zone_simulation_parameters (Dict[str, bool]):  A number of parameters are made available as they vary through the
            simulation for thermal zones. The keys (all by default False) are:
                'system_time_step', 'zone_time_step', 'zone_time_step_number'
            
            use_one_day_weather_prediction (bool): We use the internal variables of EnergyPlus to provide with a 
            prediction of the weather time ahead. You can specify the `prediction_hours` and the prediction variables
            listed on `prediction_variables`.
            
            prediction_hours (int): Default is 24
            
            prediction_variables (Dict[str, bool]): See `use_one_day_weather_prediction`. The variables are (by default 
            all False):
                'albedo', 'beam_solar', 'diffuse_solar', 'horizontal_ir', 'is_raining', 'is_snowing', 'liquid_precipitation',
                'outdoor_barometric_pressure', 'outdoor_dew_point', 'outdoor_dry_bulb', 'outdoor_relative_humidity',
                'sky_temperature', 'wind_direction', 'wind_speed'.
            
            use_actuator_state (bool): define if the actuator state will be used as an observation for the agent.
            
            other_obs (Dict[str, float | int]): Custom observation dictionary.
        """
        counter = 0
        # Variables
        self.variables = variables
        if self.variables is not None:
            counter += len(variables)
    
        # Internal variables
        self.internal_variables = internal_variables
        if self.internal_variables is not None:
            counter += len(internal_variables)
        
        # Meters
        self.meters = meters
        if meters is not None:
            counter += len(meters)
        
        # Simulation parameters
        self.simulation_parameters: Dict[str, bool] = {
            'actual_date_time': False,
            'actual_time': False,
            'current_time': False,
            'day_of_month': False,
            'day_of_week': False,
            'day_of_year': False,
            'holiday_index': False,
            'hour': False,
            'minutes': False,
            'month': False,
            'num_time_steps_in_hour': False,
            'year': False,
            'is_raining': False,
            'sun_is_up': False,
            'today_weather_albedo_at_time': False,
            'today_weather_beam_solar_at_time': False,
            'today_weather_diffuse_solar_at_time': False,
            'today_weather_horizontal_ir_at_time': False,
            'today_weather_is_raining_at_time': False,
            'today_weather_is_snowing_at_time': False,
            'today_weather_liquid_precipitation_at_time': False,
            'today_weather_outdoor_barometric_pressure_at_time': False,
            'today_weather_outdoor_dew_point_at_time': False,
            'today_weather_outdoor_dry_bulb_at_time': False,
            'today_weather_outdoor_relative_humidity_at_time': False,
            'today_weather_sky_temperature_at_time': False,
            'today_weather_wind_direction_at_time': False,
            'today_weather_wind_speed_at_time': False,
            'tomorrow_weather_albedo_at_time': False,
            'tomorrow_weather_beam_solar_at_time': False,
            'tomorrow_weather_diffuse_solar_at_time': False,
            'tomorrow_weather_horizontal_ir_at_time': False,
            'tomorrow_weather_is_raining_at_time': False,
            'tomorrow_weather_is_snowing_at_time': False,
            'tomorrow_weather_liquid_precipitation_at_time': False,
            'tomorrow_weather_outdoor_barometric_pressure_at_time': False,
            'tomorrow_weather_outdoor_dew_point_at_time': False,
            'tomorrow_weather_outdoor_dry_bulb_at_time': False,
            'tomorrow_weather_outdoor_relative_humidity_at_time': False,
            'tomorrow_weather_sky_temperature_at_time': False,
            'tomorrow_weather_wind_direction_at_time': False,
            'tomorrow_weather_wind_speed_at_time': False,
        }
        # Check that the keys introduced in the Dict are admissible values.
        admissible_values = list(self.simulation_parameters.keys())
        for key in simulation_parameters.keys():
            if key not in admissible_values:
                raise ValueError(f"The key '{key}' is not admissible in the simulation_parameters. The admissible values are: {admissible_values}")
        # Update the boolean values in the self.simulation_parameters Dict.
        self.simulation_parameters.update(simulation_parameters)
        # Count the variables introduced.
        counter += sum([1 for value in simulation_parameters.values() if value])
        
        # Zone simulation parameters.
        self.zone_simulation_parameters: Dict[str, bool] = {
            'system_time_step': False,
            'zone_time_step': False,
            'zone_time_step_number': False,
        }
        
        # Check that the keys introduced in the Dict are admissible values.
        admissible_values = list(self.zone_simulation_parameters.keys())
        for key in zone_simulation_parameters.keys():
            if key not in admissible_values:
                raise ValueError(f"The key '{key}' is not admissible in the zone_simulation_parameters. The admissible values are: {admissible_values}")
        # Update the boolean values in the self.zone_simulation_parameters Dict.
        self.zone_simulation_parameters.update(zone_simulation_parameters)
        # Count the variables introduced.
        counter += sum([1 for value in zone_simulation_parameters.values() if value])
        
        # Prediction weather.
        self.prediction_variables: Dict[str, bool] = {
            'albedo': False,
            'beam_solar': False,
            'diffuse_solar': False,
            'horizontal_ir': False,
            'is_raining': False,
            'is_snowing': False,
            'liquid_precipitation': False,
            'outdoor_barometric_pressure': False,
            'outdoor_dew_point': False,
            'outdoor_dry_bulb': False,
            'outdoor_relative_humidity': False,
            'sky_temperature': False,
            'wind_direction': False,
            'wind_speed': False,
        }
        
        self.use_one_day_weather_prediction = use_one_day_weather_prediction
        if self.use_one_day_weather_prediction:
            admissible_values = list(self.prediction_variables.keys())
            for key in prediction_variables.keys():
                if key not in admissible_values:
                    raise ValueError(f"The key '{key}' is not admissible in the prediction_variables. The admissible values are: {admissible_values}")
            # Update the boolean values in the self.prediction_variables Dict.
            self.prediction_variables.update(prediction_variables)
        
            if prediction_hours <= 0 or prediction_hours > 24:
                self.prediction_hours = 24
                raise ValueError(f"The variable 'prediction_hours' must be between 1 and 24. It is taken the value of {prediction_hours}. The value of 24 is used.")
            self.prediction_hours = prediction_hours
            # Count prediction parameters.
            counter += sum([1 for value in prediction_variables.values() if value])
        
        # Actuator value in the observation.
        self.use_actuator_state = use_actuator_state
        if self.use_actuator_state:
            counter += 1
        
        # Custom observation dict.
        self.other_obs = other_obs
        counter += len(self.other_obs)
        
        # Check that at least one parameter were defined in the observation.
        if counter == 0:
            raise ValueError("At least one variable/meter/actuator/parameter must be defined in the observation.")
    
    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)
        
    
    def build(self) -> Dict:
        """
        This method is used to build the ObservationSpec object.
        """
        # Check that the variables defined in ObservationSpec are the allowed in the ObservationSpec base
        # class.
        # expected_types = {
        #     'variables': (list, None),
        #     'internal_variables': (list, None),
        #     'meters': (list, None),
        #     'simulation_parameters': dict,
        #     'zone_simulation_parameters': dict,
        #     'use_one_day_weather_prediction': bool,
        #     'prediction_hours': int,
        #     'prediction_variables': dict,
        #     'use_actuator_state': bool,
        #     'other_obs': dict
        # }
        
        # is_valid, errors = validate_properties(self, expected_types)
        # if is_valid:
        #     print("All properties have correct types")
        # else:
        #     print("Validation errors:")
        #     for error in errors:
        #         print(f"- {error}")
            
        return vars(self)

class FilterSpec:
    """
    FilterSpec is the base class for a filter specification to safe configuration of the object.
    """
    def __init__(
        self,
        filter_fn: BaseFilter = None,
        filter_fn_config: Dict[str, Any] = {}
    ):
        """
        Construction method.
        
        Args:
            filter_fn (BaseFilter): The filter function takes the arguments agent_id, observation and returns the
            observation filtered. See ``eprllib.Agents.Filters`` for examples.
            
            filter_fn_config (Dict[str, Any]): The configuration of the filter function.
        """
        if filter_fn is None:
            print("No filter provided. Default filter will be used.")
            self.filter_fn = DefaultFilter
            self.filter_fn_config = {}
        else:
            self.filter_fn = filter_fn
            self.filter_fn_config = filter_fn_config
    
    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)
        
    def build(self) -> Dict:
        """
        This method is used to build the FilterSpec object.
        """
        # Check that the variables defined in FilterSpec are the allowed in the FilterSpec base
        # class.
        # expected_types = {
        #     'filter_fn': (BaseFilter, None),
        #     'filter_fn_config': dict
        # }
        
        # is_valid, errors = validate_properties(self, expected_types)
        # if is_valid:
        #     print("All properties have correct types")
        # else:
        #     print("Validation errors:")
        #     for error in errors:
        #         print(f"- {error}")
            
        return vars(self)

class ActionSpec:
    """
    ActionSpec is the base class for an action specification to safe configuration of the object.
    """
    def __init__(
        self,
        actuators: List[Tuple[str, str, str]] = NotImplemented,
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
        """
        if actuators == NotImplemented:
            raise NotImplementedError("actuators must be defined.")
        
        self.actuators = actuators
    
    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)
        
    def build(self) -> Dict:
        """
        This method is used to build the ActionSpec object.
        """
        # Check that the variables defined in ActionSpec are the allowed in the ActionSpec base
        # class.
        # expected_types = {
        #     'actuators': list
        # }
        
        # is_valid, errors = validate_properties(self, expected_types)
        # if is_valid:
        #     print("All properties have correct types")
        # else:
        #     print("Validation errors:")
        #     for error in errors:
        #         print(f"- {error}")
            
        return vars(self)

class TriggerSpec:
    """
    TriggerSpec is the base class for a trigger specification to safe configuration of the object.
    """
    def __init__(
        self,
        trigger_fn: BaseTrigger = NotImplemented,
        trigger_fn_config: Dict[str, Any] = {},
    ):
        """
        Construction method.
        
        Args:
            trigger_fn (BaseTrigger): The trigger function takes the arguments agent_id, observation and returns the
            observation filtered. See ``eprllib.Agents.Triggers`` for examples.
            
            trigger_fn_config (Dict[str, Any]): The configuration of the trigger function.
        """
        if trigger_fn == NotImplemented:
            raise NotImplementedError("trigger_fn must be defined.")
        
        self.trigger_fn = trigger_fn
        self.trigger_fn_config = trigger_fn_config
    
    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)
    
    def build(self) -> Dict:
        """
        This method is used to build the TriggerSpec object.
        """
        # Check that the variables defined in TriggerSpec are the allowed in the TriggerSpec base
        # class.
        # expected_types = {
        #     'trigger_fn': BaseTrigger,
        #     'trigger_fn_config': dict
        # }
        
        # is_valid, errors = validate_properties(self, expected_types)
        # if is_valid:
        #     print("All properties have correct types")
        # else:
        #     print("Validation errors:")
        #     for error in errors:
        #         print(f"- {error}")
            
        return vars(self)

class AgentSpec:
    """
    AgentSpec is the base class for an agent specification to safe configuration of the object.
    """
    def __init__(
        self,
        observation: ObservationSpec = NotImplemented,
        filter: FilterSpec = None,
        action: ActionSpec = NotImplemented,
        trigger: TriggerSpec = None,
        reward: RewardSpec = NotImplemented,
        **kwargs):
        """
        Contruction method for the AgentSpec class.

        Args:
            observation (ObservationSpec, optional): Defines the observation of the agent using
            the ObservationSpec class or a Dict. Defaults to NotImplemented.
            filter (FilterSpec, optional): Defines the filter for this agent using FilterSpec or a Dict. Defaults to None.
            action (ActionSpec, optional): Defines the action characteristics of the agent using ActionSpec or a Dict. Defaults to NotImplemented.
            trigger (TriggerSpec, optional): Defines the trigger for the agent using TriggerSpec or a Dict. Defaults to None.
            reward (RewardSpec, optional): Defines the reward elements of the agent using RewardSpec or a Dict. Defaults to NotImplemented.

        Raises:
            NotImplementedError: _description_
            NotImplementedError: _description_
            NotImplementedError: _description_
        """
        
        if observation == NotImplemented:
            raise NotImplementedError("observation must be deffined.")
        if action == NotImplemented:
            raise NotImplementedError("action must be deffined.")
        if reward == NotImplemented:
            raise NotImplementedError("reward must be deffined.")
        
        self.observation = observation
        if filter is None:
            self.filter = FilterSpec()
        else:
            self.filter = filter
        self.action = action
        if trigger is None:
            self.trigger = TriggerSpec()
        else:
            self.trigger = trigger
        self.reward = reward
        
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)
        
    def build(self) -> Dict:
        """
        This method is used to build the AgentSpec object.
        """
        # Check that the variables defined in AgentSpec are the allowed in the AgentSpec base
        # class.
        # expected_types = {
        #     'observation': (dict, ObservationSpec),
        #     'filter': (dict, FilterSpec, None),
        #     'action': (dict, ActionSpec),
        #     'trigger': (dict, TriggerSpec, None),
        #     'reward': (dict, RewardSpec)
        # }
        
        # is_valid, errors = validate_properties(self, expected_types)
        # if is_valid:
        #     print("All properties have correct types")
        # else:
        #     print("Validation errors:")
        #     for error in errors:
        #         print(f"- {error}")
        
        # if isinstance(self.observation, ObservationSpec):
        #     self.observation = self.observation.build()

        # if isinstance(self.filter, FilterSpec):
        #     self.filter = self.filter.build()

        # if isinstance(self.action, ActionSpec):
        #     self.action = self.action.build()

        # if isinstance(self.trigger, TriggerSpec):
        #     self.trigger = self.trigger.build()

        # if isinstance(self.reward, RewardSpec):
        #     self.reward = self.reward.build()
            
        return vars(self)
