"""
Agents
=======

This module contains classes for representing and manipulating agents in the
environment. The agents are responsible for taking actions in the environment
following a specified policy that responses to the current state of the
environment.

In this module, you will find:

- ``AgentSpec``: The main class for defining agents, including their observation, 
  filter, action, action_mapper, and reward specifications.
- ``ObservationSpec``: Defines the observation space for the agent.
- ``FilterSpec``: Defines filters to preprocess observations before they are fed to 
  the agent. NOTE: This object must to be coordinated with ``AgentsConnectors``.
- ``ActionSpec``: Defines the action space and actuators for the agent.
- ``ActionMapperSpec``: Defines ActionMappers that determine how the agent should 
transform policy actions into actutators an actions.
- ``RewardSpec``: Defines the reward function for the agent.

Additionally, you will find base classes and some applications for Filters, Rewards, 
and ActionMappers, which are essential parts of an agent in ``eprllib``.
"""

from typing import Dict, Any

SIMULATION_PARAMETERS: Dict[str, bool] = {
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

ZONE_SIMULATION_PARAMETERS: Dict[str, bool] = {
    'system_time_step': False,
    'zone_time_step': False,
    'zone_time_step_number': False,
}

PREDICTION_VARIABLES: Dict[str, bool] = {
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

PREDICTION_HOURS: int = 24

# --- CONFIGURING USER PROFILES BY ZONE ---
# Each profile now distinguishes between day and night zones.
# The sum of occupants in both zones at a given hour reflects the distribution of people.

OCCUPATION_PROFILES: Dict[str, Dict[str, Any]] = {
    "Office schedule": {
        "total_people": 1,
        "zone_daytime": {
            "weekdays": [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
            "weekends": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        },
        "zone_nightly": {
            "weekdays": [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
            "weekends": [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
        },
    },
    "Single with an office job": {
        "total_people": 1,
        "zone_daytime": {
            "weekdays": [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0],
            "weekends": [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        },
        "zone_nightly": {
            "weekdays": [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
            "weekends": [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
        },
    },
    "Typical family, office job": { # 2 adults, 2 children
        "total_people": 1,
        "zone_daytime": {
            "weekdays": [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
            "weekends": [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
        },
        "zone_nightly": {
            "weekdays": [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
            "weekends": [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        },
    },
    "Always present": {
        "total_people": 1,
        "zone_daytime": {
            "weekdays": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            "weekends": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        },
        "zone_nightly": {
            "weekdays": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            "weekends": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        },
    },
    # More profiles can be added here with the same structure.
}

# Validation list for allowed user types.
VALID_USER_TYPES = list(OCCUPATION_PROFILES.keys())
VALID_ZONE_TYPES = ["daytime", "nightly"]
