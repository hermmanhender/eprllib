"""
Specification for the observation space and parameters
===========================================================
This module defines the `ObservationSpec` class, which is used to specify the configuration of 
observation space and parameters for agents in reinforcement learning environments.

It ensures that the observation space is properly defined and adheres to the expected interface.
"""
from typing import Dict, List, Tuple, Optional, Any
from eprllib.Agents import (
    SIMULATION_PARAMETERS,
    ZONE_SIMULATION_PARAMETERS, 
    PREDICTION_VARIABLES,
    PREDICTION_HOURS,
    VALID_USER_TYPES,
    VALID_ZONE_TYPES
)
from eprllib import logger

class ObservationSpec:
    """
    ObservationSpec is the base class for an observation specification to safe configuration of the object.
    """
    variables: Optional[List[Tuple[str, str]]] = None
    internal_variables: Optional[List[Tuple[str, str]]] = None
    meters: Optional[List[str]] = None
    simulation_parameters: Dict[str, bool] = {}
    zone_simulation_parameters: Dict[str, bool] = {}
    use_one_day_weather_prediction: bool = False
    weather_prediction_hours: int = PREDICTION_HOURS
    prediction_variables: Dict[str, bool] = {}
    use_actuator_state: bool = False
    other_obs: Dict[str, float | int] = {}
    user_occupation_function: bool = False
    user_occupation_forecast: bool = False
    user_type: str = VALID_USER_TYPES[0]
    zone_type: str = VALID_ZONE_TYPES[0]
    occupation_schedule: Optional[Tuple[str, str, str]] = None
    occupation_prediction_hours: int = 24
    confidence_level: float = 0.95
    lambdaa: float = 0.05
    
    def __init__(
        self,
        variables: Optional[List[Tuple[str, str]]] = None,
        internal_variables: Optional[List[Tuple[str, str]]] = None,
        meters: Optional[List[str]] = None,
        simulation_parameters: Dict[str, bool] = {},
        zone_simulation_parameters: Dict[str, bool] = {},
        use_one_day_weather_prediction: bool = False,
        weather_prediction_hours: int = PREDICTION_HOURS,
        prediction_variables: Dict[str, bool] = {},
        use_actuator_state: bool = False,
        other_obs: Dict[str, float | int] = {},
        # history_len: int = 1,
        user_occupation_function: bool = False,
        user_occupation_forecast: bool = False,
        user_type: str = VALID_USER_TYPES[0],
        zone_type: str = VALID_ZONE_TYPES[0],
        occupation_schedule: Optional[Tuple[str, str, str]] = None,
        occupation_prediction_hours: int = 24,
        confidence_level: float = 0.95,
        lambdaa: float = 0.05
    ) -> None:
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
            prediction of the weather time ahead. You can specify the `weather_prediction_hours` and the prediction variables
            listed on `prediction_variables`.
            
            weather_prediction_hours (int): Default is 24
            
            prediction_variables (Dict[str, bool]): See `use_one_day_weather_prediction`. The variables are (by default 
            all False):
                'albedo', 'beam_solar', 'diffuse_solar', 'horizontal_ir', 'is_raining', 'is_snowing', 'liquid_precipitation',
                'outdoor_barometric_pressure', 'outdoor_dew_point', 'outdoor_dry_bulb', 'outdoor_relative_humidity',
                'sky_temperature', 'wind_direction', 'wind_speed'.
            
            use_actuator_state (bool): Define if the actuator state will be used as an observation for the agent.
            
            other_obs (Dict[str, float | int]): Custom observation dictionary.
            
            history_len (int): History length for each agent. DEPRECATED.
            
            user_occupation_function (bool): Define if the user occupation function will be used. Default is False.
            
            user_occupation_forecast (bool): Define if the user occupation forecast will be used. Default is False.
            
            user_type (str): Type of user. Default is 'Residential'
            
            zone_type (str): Type of zone. Default is 'Office'
            
            occupation_schedule (Tuple[str, str, str]): Occupation schedule for the user occupation forecast. Default is None.
            
            occupation_prediction_hours (int): Occupation prediction hours for the user occupation forecast. Default is 24.
            
            confidence_level (float): Confidence level for the user occupation forecast. Default is 0.95.
            
            lambdaa (float): Decay value of the confidence level. Default is 0.05.

        """
        # Variables
        self.variables = variables
        # Internal variables
        self.internal_variables = internal_variables
        # Meters
        self.meters = meters
        # Simulation parameters
        self.simulation_parameters = SIMULATION_PARAMETERS.copy()
        
        # Update the boolean values in the self.simulation_parameters Dict.
        self.simulation_parameters.update(simulation_parameters)
        
        # Zone simulation parameters.
        self.zone_simulation_parameters = ZONE_SIMULATION_PARAMETERS.copy()
        
        # Update the boolean values in the self.zone_simulation_parameters Dict.
        self.zone_simulation_parameters.update(zone_simulation_parameters)
        
        # Prediction weather.
        self.prediction_variables = PREDICTION_VARIABLES.copy()
        self.use_one_day_weather_prediction = use_one_day_weather_prediction
        if self.use_one_day_weather_prediction:
            # Update the boolean values in the self.prediction_variables Dict.
            self.prediction_variables.update(prediction_variables)
        self.weather_prediction_hours = weather_prediction_hours
            
        # Actuator value in the observation.
        self.use_actuator_state = use_actuator_state
        
        # Custom observation dict.
        self.other_obs = other_obs
        
        # History length for each agent.
        # self.history_len = history_len
        
        # User occupation forecast profile.
        self.user_occupation_forecast = user_occupation_forecast
        self.user_occupation_function = user_occupation_function
        if self.user_occupation_forecast: # This ensure that if forecast is used, the occupancy function is used as well.
            self.user_occupation_function = True
        
        # User type and zone type.
        self.user_type = user_type
        
        if user_type not in VALID_USER_TYPES:
            msg = f"ObservationSpec: User type '{user_type}' is not valid. Options: {VALID_USER_TYPES}"
            logger.error(msg)
            raise ValueError(msg)
        
        self.zone_type = zone_type
        if zone_type not in VALID_ZONE_TYPES:
            msg = f"ObservationSpec: Zone type '{zone_type}' is not valid. Options: {VALID_ZONE_TYPES}"
            logger.error(msg)
            raise ValueError(msg)
        
        self.confidence_level = confidence_level
        
        # Occupation forecast.
        self.occupation_schedule = occupation_schedule
        self.occupation_prediction_hours = occupation_prediction_hours
        self.lambdaa = lambdaa
        
    def __getitem__(self, key:str) -> Any:
        return getattr(self, key)

    def __setitem__(self, key:str, value:Any) -> None:
        valid_keys = self.__dict__.keys()
        if key not in valid_keys:
            msg = f"ObservationSpec: Invalid key: {key}."
            logger.error(msg)
            raise KeyError(msg)
        setattr(self, key, value)
        
    
    def build(self) -> Dict[str, Any]:
        """
        This method is used to build the ObservationSpec object.
        """
        # Check that the keys introduced in the Dict are admissible values.
        admissible_values = list(SIMULATION_PARAMETERS.keys())
        for key in self.simulation_parameters.keys():
            if key not in admissible_values:
                msg = f"ObservationSpec: The key '{key}' is not admissible in the simulation_parameters. The admissible values are: {admissible_values}"
                logger.error(msg)
                raise ValueError(msg)
        
        # Check that the keys introduced in the Dict are admissible values.
        admissible_values = list(ZONE_SIMULATION_PARAMETERS.keys())
        for key in self.zone_simulation_parameters.keys():
            if key not in admissible_values:
                msg = f"ObservationSpec: The key '{key}' is not admissible in the zone_simulation_parameters. The admissible values are: {admissible_values}"
                logger.error(msg)
                raise ValueError(msg)
        
        if self.use_one_day_weather_prediction:
            admissible_values = list(PREDICTION_VARIABLES.keys())
            for key in self.prediction_variables.keys():
                if key not in admissible_values:
                    msg = f"ObservationSpec: The key '{key}' is not admissible in the prediction_variables. The admissible values are: {admissible_values}"
                    logger.error(msg)
                    raise ValueError(msg)
        
        if self.weather_prediction_hours <= 0 or self.weather_prediction_hours > 24:
            self.weather_prediction_hours = PREDICTION_HOURS
            logger.warning(f"ObservationSpec: The variable 'weather_prediction_hours' must be between 1 and 24. It is taken the value of {self.weather_prediction_hours}. The value of 24 is used.")
        
        if self.occupation_prediction_hours <= 0 or self.occupation_prediction_hours > 24:
                self.occupation_prediction_hours = 24
                logger.warning(f"ObservationSpec: The variable 'occupation_prediction_hours' must be between 1 and 24. It is taken the value of {self.occupation_prediction_hours}. The value of 24 is used.")
        
        if self.user_occupation_forecast:
            if self.occupation_schedule is None:
                msg = "ObservationSpec: occupation_schedule must be provided if user_occupation_forecast is True."
                logger.error(msg)
                raise ValueError(msg)
            else:
                assert isinstance(self.occupation_schedule, tuple), "occupation_schedule must be a tuple."
        # Check that at least one variable/meter/actuator/parameter is defined.
        counter = 0
        if self.variables is not None:
            counter += len(self.variables)
        if self.internal_variables is not None:
            counter += len(self.internal_variables)
        if self.meters is not None:
            counter += len(self.meters)
        counter += sum([1 for value in self.simulation_parameters.values() if value])
        counter += sum([1 for value in self.zone_simulation_parameters.values() if value])
        counter += sum([1 for value in self.prediction_variables.values() if value])
        if self.use_actuator_state:
            counter += 1
        counter += len(self.other_obs)
        
        if counter == 0:
            msg = "ObservationSpec: At least one variable/meter/actuator/parameter must be defined in the observation."
            logger.error(msg)
            raise ValueError(msg)
        
        # if self.history_len <= 0:
        #     self.history_len = 1
        #     logger.warning(f"The variable 'history_len' must be greater than 0. It is taken the value of 1.")
        
        return vars(self)
            