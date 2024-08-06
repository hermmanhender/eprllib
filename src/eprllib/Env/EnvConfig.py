"""This module is used to define the basic configuration
of an environment.
"""
from typing import Optional, List, Dict, Set, Any
from tempfile import TemporaryDirectory
import time

class EnvConfig:
    def __init__(self):
        """
        _description_

        """
        # directories
        self.epjson_path = NotImplemented
        self.epw_path = NotImplemented
        date = time.asctime()
        self.output_path = TemporaryDirectory(prefix=f"{date}_eprllib_")

        # agents
        self.agents_config: Dict[str,Dict[str,Any]] = NotImplementedError(
            """The agents must to be configured. For example:

            OfficeModel = EnvConfig().agents({
                    'Agent 1 in Room 1': {
                        'ep_actuator_config': ("Ideal Loads Air System", "Air Mass Flow Rate", "Thermal Zone: Living Ideal Loads Air System"),
                        'thermal_zone': 'Thermal Zone: Living',
                        'actuator_type': 3,
                        'agent_indicator': 1,
                    },
                }
            )
            """
        )
        # functionalities
        self.episode_fn = None
        self.episode_config: Dict = None
    
    # We use the internal variables of EnergyPlus to provide with a prediction of the weather
    # time ahead.
    # The variables to predict are:
    #   - Dry Bulb Temperature in °C with squer desviation of 2.05 °C, 
    #   - Relative Humidity in % with squer desviation of 20%, 
    #   - Wind Direction in degree with squer desviation of 40°, 
    #   - Wind Speed in m/s with squer desviation of 3.41 m/s, 
    #   - Barometric pressure in Pa with a standart deviation of 1000 Pa, 
    #   - Liquid Precipitation Depth in mm with desviation of 0.5 mm.
    # This are predicted from the next hour into the 24 hours ahead defined.
    'use_one_day_weather_prediction': True,
    # Dictionary of tuples with the EnergyPlus variables to observe and their corresponding names.
    # The tuples has the format of (name_of_the_variable, Thermal_Zone_name). You can check the
    # EnergyPlus output file to get the names of the variables.
    "ep_environment_variables":[
        "Site Outdoor Air Drybulb Temperature",
        "Site Outdoor Air Barometric Pressure",
        "Site Wind Speed",
        "Site Wind Direction",
        "Site Outdoor Air Relative Humidity",
    ],
    "ep_thermal_zones_variables": [
        "Zone Mean Air Temperature",
        "Zone Air Relative Humidity",
        "Zone People Occupant Count",
    ],
    "ep_object_variables": {
        'Thermal Zone: Living': {
            "Zone Thermal Comfort Fanger Model PPD_Living Occupancy": ("Zone Thermal Comfort Fanger Model PPD", "Living Occupancy"),
        },
        'Thermal Zone: Room1': {
            "Zone Thermal Comfort Fanger Model PPD_Room1 Occupancy": ("Zone Thermal Comfort Fanger Model PPD", "Room1 Occupancy"),
        },
        'Thermal Zone: Room2': {
            "Zone Thermal Comfort Fanger Model PPD_Room2 Occupancy": ("Zone Thermal Comfort Fanger Model PPD", "Room2 Occupancy"),
        },
    },
    # Dictionary of names of meters from EnergyPlus to observe.
    "ep_meters": [
        "Electricity:Zone:THERMAL ZONE: LIVING",
        "Heating:DistrictHeatingWater",
        "Cooling:DistrictCooling",
    ],
    # The time variables to observe in the EnergyPlus simulation. The format is a list of
    # the names described in the EnergyPlus epJSON format documentation 
    # (https://energyplus.readthedocs.io/en/latest/schema.html) related with
    # temporal variables. All the options are listed bellow.
    'time_variables': [
        'hour',
        'day_of_year',
        'day_of_week',
        ],
    # The weather variables are related with weather values in the present timestep for the
    # agent. The following list provide all the options avialable.
    # To weather predictions see the 'weather_prob_days' config that is follow in this file.
    'weather_variables': [
        'is_raining',
        'sun_is_up',
        "today_weather_horizontal_ir_at_time",
        ],
    # The information variables are important to provide information for the reward
    # function. The observation is pass trough the agent as a NDArray but the info
    # is a dictionary. In this way, we can identify clearly the value of a variable
    # with the key name. All the variables used in the reward function must to be in
    # the infos_variables list. The name of the variables must to corresponde with the
    # names defined in the earlier lists.
    "infos_variables": {
        'Thermal Zone: Living': [
            "Zone Thermal Comfort Fanger Model PPD_Living Occupancy",
            "Heating:DistrictHeatingWater",
            "Cooling:DistrictCooling",
            'Zone People Occupant Count',
            "Zone Mean Air Temperature",
        ],
        'Thermal Zone: Room1': [
            "Zone Thermal Comfort Fanger Model PPD_Room1 Occupancy",
            "Heating:DistrictHeatingWater",
            "Cooling:DistrictCooling",
            'Zone People Occupant Count',
            "Zone Mean Air Temperature",
        ],
        'Thermal Zone: Room2': [
            "Zone Thermal Comfort Fanger Model PPD_Room2 Occupancy",
            "Heating:DistrictHeatingWater",
            "Cooling:DistrictCooling",
            'Zone People Occupant Count',
            "Zone Mean Air Temperature",
        ],
    },
    # There are occasions where some variables are consulted to use in training but are
    # not part of the observation space. For that variables, you can use the following 
    # list. An strategy, for example, to use the Fanger PPD value in the reward function
    # but not in the observation space is to aggregate the PPD into the 'infos_variables' and
    # in the 'no_observable_variables' list.
    "no_observable_variables": {
        'Thermal Zone: Living': [
            "Zone Thermal Comfort Fanger Model PPD_Living Occupancy",
        ],
        'Thermal Zone: Room1': [
            "Zone Thermal Comfort Fanger Model PPD_Room1 Occupancy",
        ],
        'Thermal Zone: Room2': [
            "Zone Thermal Comfort Fanger Model PPD_Room2 Occupancy",
        ],
    },
    
    # Sometimes is useful to cut the simulation RunPeriod into diferent episodes. By default, 
    # an episode is a entire RunPeriod EnergyPlus simulation. If you set the 'cut_episode_len'
    # in 1 (day) you will truncate the, for example, annual simulation into 365 episodes.
    'cut_episode_len': None,
    # timeout define the time that the environment wait for an observation and the time that
    # the environment wait to apply an action in the EnergyPlus simulation. After that time,
    # the episode is finished. If your environment is time consuming, you can increase this
    # limit. By default the value is 10 seconds.
    "timeout": 10,
    # The reward funtion take the arguments EnvObject (the GymEnv class) and the infos dictionary.
    # As a return, gives a float number as reward. See eprllib.tools.rewards
    'reward_function': normalize_reward_function,
    'reward_function_config': {
        'Thermal Zone: Living': {
            # cut_reward_len_timesteps: Este parámetro permite que el agente no reciba una recompensa en cada paso de tiempo, 
            # en cambio las variables para el cálculo de la recompensa son almacenadas en una lista para luego utilizar una 
            # recompensa promedio cuando se alcanza la cantidad de pasos de tiempo indicados por 'cut_reward_len_timesteps'.
            'cut_reward_len_timesteps': 1,
            # Parámetros para la exclusión de términos de la recompensa
            'comfort_reward': True,
            'energy_reward': True,
            'beta_reward': 0.5,
            'occupancy_name': 'Zone People Occupant Count',
            'ppd_name': 'Zone Thermal Comfort Fanger Model PPD_Living Occupancy',
            'T_interior_name': 'Zone Mean Air Temperature',
            'cooling_name': "Cooling:DistrictCooling",
            'heating_name': "Heating:DistrictHeatingWater",
            'cooling_energy_ref': 1500000,
            'heating_energy_ref': 1500000,
        },
        'Thermal Zone: Room1': {
            'cut_reward_len_timesteps': 1,
            'comfort_reward': True,
            'energy_reward': True,
            'beta_reward': 0.5,
            'occupancy_name': 'Zone People Occupant Count',
            'ppd_name': 'Zone Thermal Comfort Fanger Model PPD_Room1 Occupancy',
            'T_interior_name': 'Zone Mean Air Temperature',
            'cooling_name': "Cooling:DistrictCooling",
            'heating_name': "Heating:DistrictHeatingWater",
            'cooling_energy_ref': 1500000,
            'heating_energy_ref': 1500000,
        },
        'Thermal Zone: Room2': {
            'cut_reward_len_timesteps': 1,
            'comfort_reward': True,
            'energy_reward': True,
            'beta_reward': 0.5,
            'occupancy_name': 'Zone People Occupant Count',
            'ppd_name': 'Zone Thermal Comfort Fanger Model PPD_Room2 Occupancy',
            'T_interior_name': 'Zone Mean Air Temperature',
            'cooling_name': "Cooling:DistrictCooling",
            'heating_name': "Heating:DistrictHeatingWater",
            'cooling_energy_ref': 1500000,
            'heating_energy_ref': 1500000,
        },
    },
    # For dubugging is better to print in the terminal the outputs of the EnergyPlus simulation process.
    'ep_terminal_output': True,
    # In the definition of the action space, usualy is use the discrete form of the gym spaces.
    # In general, we don't use actions from 0 to n directly in the EnergyPlus simulation. With
    # the objective to transform appropiately the discret action into a value action for EP we
    # define the action_transformer funtion. This function take the arguments agent_id and
    # action. You can find examples in eprllib.tools.action_transformers .
    'action_transformer': DualSetPointThermostat,
}
    
    def directories(
        self,
        epjson_path:Optional[str] = None,
        epw_path:Optional[str] = None,
        output_path:Optional[str] = None,
        ) -> None:
        """_description_

        Args:
            epjson_path (str): The path to the EnergyPlus model in the format of epJSON file.
            epw_path (str): The path to the EnergyPlus weather file in the format of epw file.
            output_path (str): The path to the output directory for the EnergyPlus simulation.
        
        Return: The EnvConfig modified.
        """
        if epjson_path != None:
            self.epjson_path = epjson_path
        if epw_path != None:
            self.epw_path = epw_path
        if output_path != None:
            self.output_path = output_path
        
    def agents(
        self,
        agents_config: Dict[str,Dict[str,Any]] = None,
    ) -> None:
        """_description_

        Args:
            agents_config (Dict[str,Dict[str,Any]]): This dictionary contain the names of
            the agents involved in the environment. The mandatory components of the agent
            are: ep_actuator_configuration, thermal_zone, actuator_type, agent_indicator.
        
        Example:
            OfficeModel = EnvConfig().agents(
                agents_config={
                    'Agent 1 in Room 1': {
                        'ep_actuator_config': ("Schedule:Compact", "Schedule Value", "HVACHeatingLiving"),
                        'thermal_zone': 'Thermal Zone: Living',
                        'actuator_type': 2,
                        'agent_indicator': 2,
                    }
                }
            )
        
        Return:
            EnvConfig: The environment with modifications.
        """
        if agents_config != None:
            self.agents_config = agents_config
    
    def observation_options(
        self,
        use_actuator_state: bool = None,
        use_agent_indicator: bool = None,
        use_agent_type: bool = None,
        use_building_properties: bool = None,
        buildig_properties: Dict[str,Dict[str,float]] = None
    ) -> None:
        """_description_

        Args:
            use_actuator_state (bool): define if the actuator state will be used as an observation for the agent.
            use_agent_indicator (bool): define if agent indicator will be used as an observation for the agent. This is recommended True for muilti-agent usage and False for single agent case.
            use_agent_type (bool): define if the agent/actuator type will be used. This is recommended for different types of agents actuating in the same environment.
            use_building_properties (bool): # define if the building properties will be used as an observation for the agent. This is recommended if different buildings/thermal zones will be used with the same policy.
            buildig_properties (Dict[str,Dict[str,float]]): # The episode config define important aspects about the building to be simulated in the episode.
        
        Example:
            OfficeModel = EnvConfig().observation_options(
                use_building_properties = True,
                buildig_properties = {
                    {
                    'Thermal Zone: Living': {
                        'building_area': 20.75,
                        'aspect_ratio': 1.102564103,
                        'window_area_relation_north': 0.52173913,
                        'window_area_relation_east': 0.181267806,
                        'window_area_relation_south': 0,
                        'window_area_relation_west': 0.,
                        'E_cool_ref': 1500000.,
                        'E_heat_ref': 1500000.,
                    }
                }
            )
        
        Return:
            EnvConfig: The environment with modifications.
        """
        if use_actuator_state != None:
            self.use_actuator_state = use_actuator_state
        if use_agent_indicator != None:
            self.use_agent_indicator = use_agent_indicator
        if use_agent_type != None:
            self.use_agent_type = use_agent_type
        if use_building_properties != None:
            self.use_building_properties = use_building_properties
            if buildig_properties == None:
                NotImplementedError(
                    """The implmentation of building_properties is mandatory
                    when you set 'use_building_properties=True'. Set this to False or 
                    proporcionate a Dict[str,Dict[str,float]].
                    """
                )
            else:
                self.buildig_properties = buildig_properties

    def functionalities(
        self,
        episode_fn = None,
        episode_config: Dict = None,
    ) -> None:
        """_description_

        Args:
            episode_fn (): This method define the properties of the episode, taking the env_config dict and returning it with modifications.
            episode_config (Dict): 
        
        Example:
            OfficeModel = EnvConfig().observation_options(
                'episode_config_fn': random_weather_config,
                'episode_config':{
                    'epw_files_folder_path': 'C:/Users/grhen/Documents/GitHub/eprllib_experiments/AJEA2024/archivos/epw',
                }
            )
        
        Return:
            EnvConfig: The environment with modifications.
        """
        if episode_fn != None:
            self.episode_fn = episode_fn

            if episode_config == None:
                NotImplementedError(
                    """If you set a episode_fn you need to specify the episode_fn_config. If the function don't use a config, set this parameter to False.
                    """
                )
            else:
                self.episode_config = episode_config
    