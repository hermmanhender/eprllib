# Configurations for eprllib

To configurate the environment and to adapt the EnergyPlus model into eprllib,
there are some parameters that you can configurate. This are discussed in the 
following sections.

Note that all the configurable parameters must to be integrate in one dictionary
as is show in the final section of this page.

## EnergyPlus options

This group of parameters configurate the path to the epJSON and epw files,
the output folder and the status of the impressions in the terminal during
the simulation.

    'epjson': (str) "path_to_epjson_file.epJSON",
    "epw_training": (str) "path_to_epw_file_for_training.epw",
    "epw": (str) "path_to_epw_file_for_evaluation.epw",
    'output': (str) 'path_to_outputs_folder'),
    'ep_terminal_output': (bool) False

## Experiment options


## Environment options


## Optionals


## Example config dictionary

```
env_config={
    # === ENERGYPLUS OPTIONS === #
    'epjson': "C:/EnergyPlusV23-2-0/ExampleFiles/RefBldgSmallOfficeNew2004_Chicago.idf",
    "epw_training": "C:/EnergyPlusV23-2-0/WeatherData/USA_IL_Chicago-OHare.Intl.AP.725300_TMY3.epw",
    "epw": "C:/EnergyPlusV23-2-0/WeatherData/USA_IL_Chicago-OHare.Intl.AP.725300_TMY3.epw",
    'output': TemporaryDirectory("output","eprllib",'path_to_outputs_folder'),
    'ep_terminal_output': False,
    
    # === EXPERIMENT OPTIONS === #
    'is_test': False,
    
    # === ENVIRONMENT OPTIONS === #
    'action_space': Discrete(4),
    'action_transformer': action_transformers.thermostat_dual,
    'reward_function': rewards.PPD_Energy_reward,
    "ep_variables":{
        "To": ("Site Outdoor Air Drybulb Temperature", "Environment"),
        "Ti": ("Zone Mean Air Temperature", "Core_ZN"),
        "v": ("Site Wind Speed", "Environment"),
        "d": ("Site Wind Direction", "Environment"),
        "RHo": ("Site Outdoor Air Relative Humidity", "Environment"),
        "RHi": ("Zone Air Relative Humidity", "Core_ZN"),
        "pres": ("Site Outdoor Air Barometric Pressure", "Environment"),
        "occupancy": ("Zone People Occupant Count", "Core_ZN"),
        "ppd": ("Zone Thermal Comfort Fanger Model PPD", "Core_ZN People")
    },
    "ep_meters": {
        "heating_meter": "Heating:Electricity: Core_ZN",
        "cooling_meter": "Cooling:Electricity: Core_ZN",
    },
    "ep_actuators": {
        "cooling_setpoint": ("Zone Temperature Control", "Cooling Setpoint", "Core_ZN"),
        "heating_serpoint": ("Zone Temperature Control", "Heating Setpoint", "Core_ZN"),
    },
    'time_variables': [
        'hour',
        'day_of_year',
        'day_of_the_week',
        ],
    'weather_variables': [
        'is_raining',
        'sun_is_up',
        "today_weather_beam_solar_at_time",
        ],
    "infos_variables": ["ppd", 'heating_meter', 'cooling_meter'],
    "no_observable_variables": ["ppd"],
    
    # === OPTIONAL === #
    "timeout": 10,
    'beta_reward': 0.5,
    "weather_prob_days": 2
}
```
