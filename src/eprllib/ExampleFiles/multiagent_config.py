"""This file provide the entire parameters configuration of the 
multi-agent environment approach.
"""
from eprllib.tools.rewards import dalamagkidis_2007

env_config = {
    # The path to the EnergyPlus model in the format of epJSON file.
    'epjson': 'path/to/epjson_file.json',
    # The path to the EnergyPlus weather file in the format of epw file.
    'epw': 'path/to/epw_file.epw',
    # The path to the EnergyPlus weather file for training in the format of epw file.
    'epw_training': 'path/to/epw_file.epw',
    # The path to the output directory for the EnergyPlus simulation.
    'output': 'path/to/output_directory',
    # For dubugging is better to print in the terminal the outputs of the EnergyPlus simulation process.
    'ep_terminal_output': False,
    # For evaluation process 'is_test=True' and for trainig False.
    'is_test': False,
    # The action space for all the agents.
    'action_space': 'gym.spaces.Discrete() type',
    # Dictionary of tuples with the EnergyPlus variables to observe and their corresponding names.
    # The tuples has the format of (name_of_the_variable, Thermal_Zone_name). You can check the
    # EnergyPlus output file to get the names of the variables.
    'ep_variables': {
        'Site Outdoor Air Drybulb Temperature': ('Site Outdoor Air Drybulb Temperature', 'Environment'),
        'Zone Mean Air Temperature': ('Zone Mean Air Temperature', 'Thermal Zone: Living'),
        'Zone Air Relative Humidity': ('Zone Air Relative Humidity', 'Thermal Zone: Living'),
        'Zone Thermal Comfort Fanger Model PPD': ('Zone Thermal Comfort Fanger Model PPD', 'Living Occupancy'),
    },
    # Dictionary of names of meters from EnergyPlus to observe.
    'ep_meters': {
        'Electricity': 'Electricity:Zone:THERMAL ZONE: LIVING',
        'NaturalGas': 'NaturalGas:Zone:THERMAL ZONE: LIVING',
        'Heating': 'Heating:DistrictHeatingWater',
        'Cooling': 'Cooling:DistrictCooling',
    },
    # Dictionary of tuples with the EnergyPlus actuators to control and their corresponding 
    # names.
    # The tuples has the format of (type_of_actuator, variable, name or zone). You can check
    # the documentation of EnergyPlus to get more information about the actuators.
    'ep_actuators': {
        'Heating Setpoint': ('Schedule:Compact', 'Schedule Value', 'HVACTemplate-Always 19'),
        'Cooling Setpoint': ('Schedule:Compact', 'Schedule Value', 'HVACTemplate-Always 25'),
        'Air Mass Flow Rate': ('Ideal Loads Air System', 'Air Mass Flow Rate', 'Thermal Zone: Living Ideal Loads Air System'),
    },
    # Dictionary of the types of the EnergyPlus actuators to control.
    # The types has to be the same as the tuples in the 'ep_actuators' dictionary.
    # The classification of the actuators follow the next correspondency of names:
    #   1: Cooling set point
    #   2: Heating set point
    #   3: Acondicionated Air Flow Rate
    #   4: North Window Opening
    #   5: East Window Opening
    #   6: South Window Opening
    #   7: West Window Opening
    #   8: North Window Shading
    #   9: East Window Shading
    #   10: South Window Shading
    #   11: West Window Shading
    #   12: Fan Flow Rate
    'ep_actuators_type': {
        'Heating Setpoint': 2,
        'Cooling Setpoint': 1,
        'Air Mass Flow Rate': 3,
    },
    # The time variables to observe in the EnergyPlus simulation. The format is a list of
    # the names described in the EnergyPlus epJSON format documentation 
    # (https://energyplus.readthedocs.io/en/latest/schema.html) related with
    # temporal variables. All the options are listed bellow.
    'time_variables': [
        # Gets a simple sum of the values of the date/time function. Could be used in 
        # random seeding.
        'actual_date_time',
        # Gets a simple sum of the values of the time part of the date/time function. 
        # Could be used in random seeding.
        'actual_time',
        # Get the current time of day in hours, where current time represents the end 
        # time of the current time step.
        'current_time',
        # Get the current day of month (1-31)
        'day_of_month',
        # Get the current day of the week (1-7) 
        'day_of_week',
        # Get the current day of the year (1-366)
        'day_of_year',
        # Gets a flag for the current day holiday type: 0 is no holiday, 1 is holiday 
        # type #1, etc.
        'holiday_index',
        # Get the current hour of the simulation (0-23)
        'hour',
        # Get the current minutes into the hour (1-60)
        'minutes',
        # Get the current month of the simulation (1-12)
        'month',
        # Returns the number of zone time steps in an hour, which is currently a 
        # constant value throughout a simulation.
        'num_time_steps_in_hour',
        # Gets the current system time step value in EnergyPlus. The system time 
        # step is variable and fluctuates during the simulation.
        'system_time_step',
        # Get the “current” year of the simulation, read from the EPW. All simulations 
        # operate at a real year, either user specified or automatically selected by 
        # EnergyPlus based on other data (start day of week + leap year option).
        'year',
        # Gets the current zone time step value in EnergyPlus. The zone time step is 
        # variable and fluctuates during the simulation.
        'zone_time_step',
        # The current zone time step index, from 1 to the number of zone time steps 
        # per hour.
        'zone_time_step_number',
    ],
    # The weather variables are related with weather values in the present timestep for the
    # agent. The following list provide all the options avialable.
    # To weather predictions see the 'weather_prob_days' config that is follow in this file.
    'weather_variables': [
        # Gets a flag for whether the it is currently raining. The C API returns an 
        # integer where 1 is yes and 0 is no, this simply wraps that with a bool 
        # conversion.
        'is_raining',
        # Gets a flag for whether the sun is currently up. The C API returns an integer 
        # where 1 is yes and 0 is no, this simply wraps that with a bool conversion.
        'sun_is_up',
        # Gets the specified weather data at the specified hour and time step index 
        # within that hour.
        'today_weather_albedo_at_time',
        'today_weather_beam_solar_at_time',
        'today_weather_diffuse_solar_at_time',
        'today_weather_horizontal_ir_at_time',
        'today_weather_is_raining_at_time',
        'today_weather_is_snowing_at_time',
        'today_weather_liquid_precipitation_at_time',
        'today_weather_outdoor_barometric_pressure_at_time',
        'today_weather_outdoor_dew_point_at_time',
        'today_weather_outdoor_dry_bulb_at_time',
        'today_weather_outdoor_relative_humidity_at_time',
        'today_weather_sky_temperature_at_time',
        'today_weather_wind_direction_at_time',
        'today_weather_wind_speed_at_time',
        'tomorrow_weather_albedo_at_time',
        'tomorrow_weather_beam_solar_at_time',
        'tomorrow_weather_diffuse_solar_at_time',
        'tomorrow_weather_horizontal_ir_at_time',
        'tomorrow_weather_is_raining_at_time',
        'tomorrow_weather_is_snowing_at_time',
        'tomorrow_weather_liquid_precipitation_at_time',
        'tomorrow_weather_outdoor_barometric_pressure_at_time',
        'tomorrow_weather_outdoor_dew_point_at_time',
        'tomorrow_weather_outdoor_dry_bulb_at_time',
        'tomorrow_weather_outdoor_relative_humidity_at_time',
        'tomorrow_weather_sky_temperature_at_time',
        'tomorrow_weather_wind_direction_at_time',
        'tomorrow_weather_wind_speed_at_time',
    ],
    # The information variables are important to provide information for the reward
    # function. The observation is pass trough the agent as a NDArray but the info
    # is a dictionary. In this way, we can identify clearly the value of a variable
    # with the key name. All the variables used in the reward function must to be in
    # the infos_variables list. The name of the variables must to corresponde with the
    # names defined in the earlier lists. 
    'infos_variables': [],
    # There are occasions where some variables are consulted to use in training but are
    # not part of the observation space. For that variables, you can use the following 
    # list. An strategy, for example, to use the Fanger PPD value in the reward function
    # but not in the observation space is to aggregate the PPD into the 'infos_variables' and
    # in the 'no_observable_variables' list.
    "no_observable_variables": [],
    # timeout define the time that the environment wait for an observation and the time that
    # the environment wait to apply an action in the EnergyPlus simulation. After that time,
    # the episode is finished. If your environment is time consuming, you can increase this
    # limit. By default the value is 10 seconds.
    "timeout": 10,
    # Sometimes is useful to cut the simulation RunPeriod into diferent episodes. By default, 
    # an episode is a entire RunPeriod EnergyPlus simulation. If you set the 'cut_episode_len'
    # in 1 you will truncate the, for example, annual simulation into 365 episodes.
    # TODO: For now, this only works with 6 timestep per hour configuration.
    'cut_episode_len': None,
    # We use the internal variables of EnergyPlus to provide with a prediction of the weather
    # time ahead.
    # The variables to predict are:
    #   - Dry Bulb Temperature in °C with squer desviation of 2.05 °C, 
    #   - Relative Humidity in % with squer desviation of 20%, 
    #   - Wind Direction in degree with squer desviation of 40°, 
    #   - Wind Speed in m/s with squer desviation of 3.41 m/s, 
    #   - Barometric pressure in Pa with a standart deviation of 1000 Pa, 
    #   - Liquid Precipitation Depth in mm with desviation of 0.5 mm.
    # This are predicted from the next hour into the 24 hours ahead for each 'weather_prob_days'
    # defined.
    # TODO: Provide a way to enable the user select the types of variables to predict and the
    # accuracy of the predictions.
    "weather_prob_days": 2,
    # In the definition of the action space, usualy is use the discrete form of the gym spaces.
    # In general, we don't use actions from 0 to n directly in the EnergyPlus simulation. With
    # the objective to transform appropiately the discret action into a value action for EP we
    # define the action_transformer funtion. This function take the arguments agent_id and
    # action. You can find examples in eprllib.tools.action_transformers .
    'action_transformer': None,
    # The reward funtion take the arguments EnvObject (the GymEnv class) and the infos dictionary.
    # As a return, gives a float number as reward. See eprllib.tools.rewards
    'reward_function': dalamagkidis_2007,
    # Reward function config dictionary
    'reward_function_config': {
        'cut_reward_len_timesteps': 1,
        'comfort_reward': True,
        'energy_reward': True,
        'co2_reward': True,
        'w1': 0.80,
        'w2': 0.01,
        'w3': 0.20,
        'energy_ref': 6805274,
        'co2_ref': 870,
        'occupancy_name': 'occupancy',
        'ppd_name': 'ppd',
        'T_interior_name': 'Ti',
        'cooling_name': 'cooling',
        'heating_name': 'heating',
        'co2_name': 'co2'
    },
    # The episode config define important aspects about the building to be simulated
    # in the episode. All this parameters are mandatory.
    # TODO: Allow the user decide if this variables are or not mandatory.
    'episode_config': {
        # Net Conditioned Building Area [m2]
        'building_area': 20.75,
        'aspect_ratio': 1.35,
        # Conditioned Window-Wall Ratio, Gross Window-Wall Ratio [%]
        'window_area_relation_north': 56.67,
        'window_area_relation_east': 18.19,
        'window_area_relation_south': 2.59,
        'window_area_relation_west': 0,
        'inercial_mass': "auto",
        'construction_u_factor': "auto",
        # User-Specified Maximum Total Cooling Capacity [W]
        'E_cool_ref': 2500,
        # User-Specified Maximum Sensible Heating Capacity [W]
        'E_heat_ref': 2500,
    },
}