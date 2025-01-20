"""
Observaton Functions
=====================

Work in progress...
"""
from typing import Any, Dict, Tuple, List
import gymnasium as gym

class ObservationSpec:
    """
    ObservationSpec is the base class for an observation specification to safe configuration of the object.
    """
    def __init__(
        self,
        variables: List[Tuple] = None,
        internal_variables: List[Tuple] = None,
        meters: List = None,
        simulation_parameters: Dict[str, bool] = {},
        zone_simulation_parameters: Dict[str, bool] = {},
        use_one_day_weather_prediction: bool = False,
        prediction_hours: int = 24,
        prediction_variables: Dict[str, bool] = {},
        use_actuator_state: bool = False,
        other_obs: Dict[str, float|int] = None
        ):
        # User must define in concordance with the model.
        self.variables= variables
        self.internal_variables = internal_variables
        self.meters = meters
        
        # With validation lists.
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
        self.simulation_parameters.update(simulation_parameters)
        self.zone_simulation_parameters: Dict[str, bool] = {
            'system_time_step': False,
            'zone_time_step': False,
            'zone_time_step_number': False,
        }
        self.zone_simulation_parameters.update(zone_simulation_parameters)
        
        # Prediction weather.
        self.use_one_day_weather_prediction = use_one_day_weather_prediction
        self.prediction_hours = prediction_hours
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
        self.prediction_variables.update(prediction_variables)
        
        # Actuator value in the observation.
        self.use_actuator_state = use_actuator_state
        
        # Custom observation dict.
        self.other_obs = other_obs
    
    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)


class ObservationFunction:
    def __init__(
        self,
        obs_fn_config: Dict[str,Any]
        ):
        self.obs_fn_config = obs_fn_config
    
    def get_agent_obs_dim(
        self,
        env_config: Dict[str,Any],
        ) -> Dict[str, gym.Space]:
        return NotImplementedError("You must implement this method.")
        
    def set_agent_obs(
        self,
        env_config: Dict[str,Any],
        agent_states: Dict[str, Dict[str,Any]] = NotImplemented,
        ) -> Dict[str,Any]:
        
        return NotImplementedError("You must implement this method.")
    