"""
Specification for the observation space and parameters
===========================================================
This module defines the `ObservationSpec` class, which is used to specify the configuration of observation space and parameters for agents in reinforcement learning environments.
It ensures that the observation space is properly defined and adheres to the expected interface.
"""
import logging
import sys
from typing import Dict, List, Tuple
from eprllib.Agents import (
    SIMULATION_PARAMETERS, ZONE_SIMULATION_PARAMETERS, 
    PREDICTION_VARIABLES, PREDICTION_HOURS
)

logger = logging.getLogger("ray.rllib")

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
        prediction_hours: int = PREDICTION_HOURS,
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
        self.prediction_hours = prediction_hours
            
        # Actuator value in the observation.
        self.use_actuator_state = use_actuator_state
        
        # Custom observation dict.
        self.other_obs = other_obs
        
    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        valid_keys = self.__dict__.keys()
        if key not in valid_keys:
            logger.error(f"Invalid key: {key}")
            raise KeyError(f"Invalid key: {key}")
        setattr(self, key, value)
        
    
    def build(self) -> Dict:
        """
        This method is used to build the ObservationSpec object.
        """
        # Check that the keys introduced in the Dict are admissible values.
        admissible_values = list(SIMULATION_PARAMETERS.keys())
        for key in self.simulation_parameters.keys():
            if key not in admissible_values:
                msg = f"The key '{key}' is not admissible in the simulation_parameters. The admissible values are: {admissible_values}"
                logger.error(msg)
                raise ValueError(msg)
        
        # Check that the keys introduced in the Dict are admissible values.
        admissible_values = list(ZONE_SIMULATION_PARAMETERS.keys())
        for key in self.zone_simulation_parameters.keys():
            if key not in admissible_values:
                msg = f"The key '{key}' is not admissible in the zone_simulation_parameters. The admissible values are: {admissible_values}"
                logger.error(msg)
                raise ValueError(msg)
        
        if self.use_one_day_weather_prediction:
            admissible_values = list(PREDICTION_VARIABLES.keys())
            for key in self.prediction_variables.keys():
                if key not in admissible_values:
                    msg = f"The key '{key}' is not admissible in the prediction_variables. The admissible values are: {admissible_values}"
                    logger.error(msg)
                    raise ValueError(msg)
        
        if self.prediction_hours <= 0 or self.prediction_hours > 24:
            self.prediction_hours = PREDICTION_HOURS
            logger.warning(f"The variable 'prediction_hours' must be between 1 and 24. It is taken the value of {self.prediction_hours}. The value of 24 is used.")
        
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
            raise ValueError("At least one variable/meter/actuator/parameter must be defined in the observation.")
        
        return vars(self)
            