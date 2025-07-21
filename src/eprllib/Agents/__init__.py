"""
Agents
=======

This module contains classes for representing and manipulating agents in the
environment. The agents are responsible for taking actions in the environment
following a specified policy.

In this module, you will find:

- ``AgentSpec``: The main class for defining agents, including their observation, 
  filter, action, trigger, and reward specifications.
- ``ObservationSpec``: Defines the observation space for the agent.
- ``FilterSpec``: Defines filters to preprocess observations before they are fed to 
  the agent.
- ``ActionSpec``: Defines the action space and actuators for the agent.
- ``TriggerSpec``: Defines triggers that determine when the agent should take an action.
- ``RewardSpec``: Defines the reward function for the agent.

Additionally, you will find base classes and some applications for Filters, Rewards, 
and Triggers, which are essential parts of an agent in eprllib.
"""

SIMULATION_PARAMETERS = {
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

ZONE_SIMULATION_PARAMETERS = {
    'system_time_step': False,
    'zone_time_step': False,
    'zone_time_step_number': False,
}

PREDICTION_VARIABLES = {
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

PREDICTION_HOURS = 24