"""
Agent Spec
===========

This module implement the base for an agent specification to safe configuration of the object.
"""

from typing import Dict, Any, List, Tuple

from eprllib.RewardFunctions.RewardFunctions import RewardFunction
from eprllib.ActionFunctions.ActionFunctions import ActionFunction

class RewardSpec:
    """
    RewardSpec is the base class for an reward specification to safe configuration of the object.
    """
    def __init__(
        self,
        reward_fn: RewardFunction = NotImplemented,
        reward_fn_config: Dict[str, Any] = {},
        ):
        """
        _Description_
        
        Args:
            reward_fn (RewardFunction): The reward funtion take the arguments EnvObject (the GymEnv class) and the infos 
            dictionary. As a return, gives a float number as reward. See eprllib.RewardFunctions for examples.
            
        """
        if reward_fn == NotImplemented:
            raise NotImplementedError("reward_fn must be defined.")
        
        self.reward_fn = reward_fn
        self.reward_fn_config = reward_fn_config
        
    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)
        

class ObservationSpec:
    """
    ObservationSpec is the base class for an observation specification to safe configuration of the object.
    """
    def __init__(
        self,
        variables: List[Tuple[str,str]] = None,
        internal_variables: List[str] = None,
        meters: List[str] = None,
        simulation_parameters: Dict[str, bool] = {},
        zone_simulation_parameters: Dict[str, bool] = {},
        use_one_day_weather_prediction: bool = False,
        prediction_hours: int = 24,
        prediction_variables: Dict[str, bool] = {},
        use_actuator_state: bool = False,
        other_obs: Dict[str, float|int] = {}
        ):
        """
        Construction method.
        
        Args:
            variables (List[Tuple[str,str]]): Variables represent time series output variables in the simulation. There are thousands
            of variables made available based on the specific configuration. A user typically requests
            variables to be in their output files by adding Output:Variable objects to the input file. It
            is important to note that if the user does not request these variables, they are not tracked,
            and thus not available on the API.
            
            internal_variables (List[str]): Internal variables form a category of built-in data accessible for eprllib.
            They are internal in that they access information about the input file from inside EnergyPlus. Internal variables 
            are automatically made available. The EDD file lists the specific internal variable types, their unique
            identifying names, and the units. The rest of this section provides information about specific internal variables.
            To see the EDD file, run the simulation to generate the output files and conigure an object in Output:EnergyManagementSystem
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
            prediction of the weathertime ahead. You can specify the `prediction_hours` and the prediction variables
            listed on `prediction_variables`.
            
            prediction_hours (int): Default is 24
            
            prediction_variables (Dict[str, bool]): See `use_one_day_weather_prediction`. The variables are (by default 
            all False):
                'albedo', 'beam_solar', 'diffuse_solar', 'horizontal_ir', 'is_raining', 'is_snowing', 'liquid_precipitation',
                'outdoor_barometric_pressure', 'outdoor_dew_point', 'outdoor_dry_bulb', 'outdoor_relative_humidity',
                'sky_temperature', 'wind_direction', 'wind_speed'.
            
            use_actuator_state (bool): define if the actuator state will be used as an observation for the agent.
            
            other_obs (Dict[str, float|int]):
        """
        counter = 0
        # Variables
        if variables is not None:
            self.variables= variables
            counter += len(variables)
        
        # Internal variables
        if internal_variables is not None:
            self.internal_variables = internal_variables
            counter += len(internal_variables)
        
        # Meters
        if meters is not None:
            self.meters = meters
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
        admissible_values = [values for values in self.simulation_parameters.keys()]
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
        admissible_values = [values for values in self.zone_simulation_parameters.keys()]
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
            admissible_values = [values for values in self.prediction_variables.keys()]
            for key in prediction_variables.keys():
                if key not in admissible_values:
                    raise ValueError(f"The key '{key}' is not admissible in the prediction_variables. The admissible values are: {admissible_values}")
            # Update the boolean values in the self.simulation_parameters Dict.
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

class ActionSpec:
    """
    ActionSpec is the base class for an action specification to safe configuration of the object.
    """
    def __init__(
        self,
        actuators: List[Tuple[str,str,str]] = NotImplemented,
        action_fn: ActionFunction = NotImplemented,
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
        if action_fn == NotImplemented:
            raise NotImplementedError("action_fn must be deffined.")
        
        self.actuators = actuators
        self.action_fn = action_fn
        self.action_fn_config = action_fn_config
    
    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

class AgentSpec:
    """
    AgentSpec is the base class for an agent specification to safe configuration of the object.
    """
    def __init__(
        self,
        observation: ObservationSpec = NotImplemented,
        action: ActionSpec = NotImplemented,
        reward: RewardSpec = NotImplemented,
        **kwargs):
        
        if observation == NotImplemented:
            raise NotImplementedError("observation must be deffined.")
        if action == NotImplemented:
            raise NotImplementedError("action must be deffined.")
        if reward == NotImplemented:
            raise NotImplementedError("reward must be deffined.")
        
        self.observation = observation
        self.action = action
        self.reward = reward
        
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)
