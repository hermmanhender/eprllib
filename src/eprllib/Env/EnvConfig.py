"""
Environment Configuration
=========================

This module contain the class and methods used to configure the environment.
"""

from typing import Optional, List, Dict, Tuple, Any
from eprllib.ActionFunctions.ActionFunctions import ActionFunction
from eprllib.RewardFunctions.RewardFunctions import RewardFunction
from eprllib.EpisodeFunctions.EpisodeFunctions import EpisodeFunction
from eprllib.ObservationFunctions.ObservationFunctions import ObservationFunction

class EnvConfig:
    def __init__(self):
        """
        This is the main object that it is used to relate the EnergyPlus model and the RLlib policy training execution.
        
        **generals**
        
        * epjson_path (str): Path to the IDF/epJSON file. If you pretend to modify the EnergyPlus model with `eprllib.EpisodeFunctions`
        it is recommended to use the epJSON format.
        * epw_path (str): Path to the EnergyPlus Weather (epw) file.
        * output_path (str): Path to the output directory. If not specified, the default is the current directory.
        * ep_terminal_output (bool): Indicate if the EnergyPlus outputs during simulation must print or not in the Terminal. Default is True.
        * timeout (float): Time to wait for an observation of the environment. If you habe a slow model, you can increase the value of 
        this parameter. Default is 10.0

        **agents**
        
        Actuators/agents are the way that users modify the program at runtime using custom logic
        and calculations. Not every variable inside EnergyPlus can be actuated. This is intentional,
        because opening that door could allow the program to run at unrealistic conditions, with flow
        imbalances or energy imbalances, and many other possible problems. Instead, a specific set of
        items are available to actuate, primarily control functions, flow requests, and environmental
        boundary conditions. These actuators, when used in conjunction with the runtime API and
        data exchange variables, allow a user to read data, make decisions and perform calculations,
        then actuate control strategies for subsequent time steps.
        Actuator functions are similar, but not exactly the same, as for variables. An actuator
        handle/ID is still looked up, but it takes the actuator type, component name, and control
        type, since components may have more than one control type available for actuation. The
        actuator can then be “actuated” by calling a set-value function, which overrides an internal
        value, and informs EnergyPlus that this value is currently being externally controlled. To
        allow EnergyPlus to resume controlling that value, there is an actuator reset function as well.
        
        * agents_config (Dict[str,Dict[str,Any]]):

        **observations**
        
        * observation_fn (ObservationFunction): 
        * observation_fn_config (Dict[str, Any]): Deafult is an emptly dict.
        
        Variables represent time series output variables in the simulation. There are thousands
        of variables made available based on the specific configuration. A user typically requests
        variables to be in their output files by adding Output:Variable objects to the input file. It
        is important to note that if the user does not request these variables, they are not tracked,
        and thus not available on the API.
        
        * variables_env: List = []
        * variables_thz: List = []
        * variables_obj: Dict[str,Tuple[str,str]] = {}  
        
        Meters represent groups of variables which are collected together, much like a meter on
        a building which represents multiple energy sources. Meters are handled the same way as
        variables, except that meters do not need to be requested prior running a simulation. From
        an API standpoint, a client must simply get a handle to a meter by name, and then access
        the meter value by using a get-value function on the API.
        
        * meters: Dict[str,Tuple[str,str]] = {} # {'agent_ID':('variable','key_object')}
        
        The name “internal variable” is used here as it is what these variables were
        called in the original EMS implementation. Another name for these variables could be “static”
        variables. Basically, these variables represent data that does not change throughout a simulation 
        period. Examples include calculated zone volume or autosized equipment values. These
        values are treated just like meters, you use one function to access a handle ID, and then use
        this handle to lookup the value.
        
        * static_variables: Dict[str,Tuple[str,str]] = {} # {'thermal_zone_ID':('variable','key_object')}
                
        A number of parameters are made available as they vary through the
        simulation, including the current simulation day of week, day of year, hour, and many other
        things. These do not require a handle, but are available through direct function calls.
        
        * simulation_parameters: Dict[str,bool] = {
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
        * zone_simulation_parameters: Dict[str,bool] = {
            'system_time_step': False,
            'zone_time_step': False,
            'zone_time_step_number': False,
        }
        
        * infos_variables: Dict[str,List|Dict[str,List]] = NotImplemented # TODO: add actuators, weather_prediction, building_properties
        {
            'variables_env': [],
            'variables_thz': [],
            'variables_obj': {'agent_ID': []},
            'meters': {'agent_ID': []},
            'static_variables': {'thermal_zone_ID': []},
            'simulation_parameters': [],
            'zone_simulation_parameters': []
        
        * no_observable_variables: Dict[str,List|Dict[str,List]] = NotImplemented # TODO: add actuators, weather_prediction, building_properties
        {'variables_env': []
        'variables_thz': []
        'variables_obj': {'agent_ID': []}
        'meters': {'agent_ID': []}
        'static_variables': {'thermal_zone_ID': []}
        'simulation_parameters': []
        'zone_simulation_parameters': []
        
        * use_actuator_state: bool = False
        * use_one_day_weather_prediction: bool = False
        * prediction_hours: int = 24
        * prediction_variables: Dict[str,bool] = {
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

        **actions**
        
        * action_fn: ActionFunction = NotImplemented
        * action_fn_config: Dict[str, Any] = {}
        
        **rewards**
        
        * reward_fn: RewardFunction = NotImplemented

        **episodes**
        
        * cut_episode_len: int = 0
        * episode_fn: EpisodeFunction = EpisodeFunction({})
        
        """
        # generals
        self.epjson_path: str = NotImplemented
        self.epw_path: str = NotImplemented
        self.output_path: str = NotImplemented
        self.ep_terminal_output: bool = True
        self.timeout: float = 10.0

        # agents
        self.agents_config: Dict[str,Dict[str,Any]] = NotImplemented

        # observations
        self.observation_fn: ObservationFunction = NotImplemented
        self.variables_env: List = []
        self.variables_thz: List = []
        self.variables_obj: Dict[str,Tuple[str,str]] = {} # {'agent_ID':('variable','key_object')}
        self.meters: Dict[str,List[str]] = {} # {'agent_ID':['meter']}
        self.static_variables: Dict[str,Tuple[str,str]] = {} # {'thermal_zone_ID':('variable','key_object')}
        self.simulation_parameters: Dict[str,bool] = {
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
        self.zone_simulation_parameters: Dict[str,bool] = {
            'system_time_step': False,
            'zone_time_step': False,
            'zone_time_step_number': False,
        }
        self.infos_variables: Dict[str,List|Dict[str,List]] = NotImplemented # TODO: add actuators, weather_prediction, building_properties
        self.no_observable_variables: Dict[str,List|Dict[str,List]] = NotImplemented # TODO: add actuators, weather_prediction, building_properties
        self.use_actuator_state: bool = False
        self.use_one_day_weather_prediction: bool = False
        self.prediction_hours: int = 24
        self.prediction_variables: Dict[str,bool] = {
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

        # actions
        self.action_fn: ActionFunction = NotImplemented
        
        # rewards
        self.reward_fn: RewardFunction = NotImplemented

        # episodes
        self.cut_episode_len: int = 0
        self.episode_fn: EpisodeFunction = EpisodeFunction({})
    
    def generals(
        self, 
        epjson_path:str = NotImplemented,
        epw_path:str = NotImplemented,
        output_path:str = NotImplemented,
        ep_terminal_output:Optional[bool] = True,
        timeout:Optional[float] = 10.0,
        evaluation:bool = False,
        ):
        """
        This method is used to modify the general configuration of the environment.

        Args:
            epjson_path (str): The path to the EnergyPlus model in the format of epJSON file.
            epw_path (str): The path to the EnergyPlus weather file in the format of epw file.
            output_path (str): The path to the output directory for the EnergyPlus simulation.
            ep_terminal_output (bool): For dubugging is better to print in the terminal the outputs 
            of the EnergyPlus simulation process.
            timeout (float): timeout define the time that the environment wait for an observation 
            and the time that the environment wait to apply an action in the EnergyPlus simulation. 
            After that time, the episode is finished. If your environment is time consuming, you 
            can increase this limit. By default the value is 10 seconds.
            number_of_agents_total (int): The total amount of agents allow to interact in the cooperative
            policy. The value must be equal or greater than the number of agents configured in the agents
            section.
        """
        self.epjson_path = epjson_path
        self.epw_path = epw_path
        self.output_path = output_path
        self.ep_terminal_output = ep_terminal_output
        self.timeout = timeout
        self.evaluation = evaluation
        
    def agents(
        self,
        agents_config:Dict[str,Dict[str,Any]] = NotImplemented,
        ):
        """
        This method is used to modify the agents configuration of the environment.

        Args:
            multi_agent_method (str): This parameter define the method to be used in the multi-agent
            policy. The options are: "fully_shared" (default), "centralize", "independent", and "custom".
            For a single agent case, this parameter is not used.
            agents_config (Dict[str,Dict[str,Any]]): This dictionary contain the names of the agents 
            involved in the environment. The mandatory components of the agent are: ep_actuator_config, 
            thermal_zone, thermal_zone_indicator, actuator_type, agent_indicator.
        """
        if agents_config == NotImplemented:
            raise NotImplementedError("agents_config must be defined.")

        self.agents_config = agents_config
    
    def observations(
        self,
        observation_fn: ObservationFunction = NotImplemented,
        variables_env: List[str] = None,
        variables_thz: List[str] = None,
        variables_obj: Dict[str,Tuple[str,str]] = None, # {'agent_ID':('variable','key_object')}
        meters: Dict[str,str] = None, # {'agent_ID':'meter'}
        static_variables: Dict[str,Tuple[str,str]] = None,
        simulation_parameters: Dict[str,bool] = None,
        zone_simulation_parameters: Dict[str,bool] = None,
        infos_variables: Dict[str,List[str]]|bool = NotImplemented,
        no_observable_variables: Dict[str,List[str]] = None,
        use_actuator_state: Optional[bool] = False,
        use_one_day_weather_prediction: Optional[bool] = False,
        prediction_hours: int = 24,
        prediction_variables: Dict[str,bool]|bool = False,
        ):
        """
        This method is used to modify the observations configuration of the environment.

        Args:
            use_actuator_state (bool): define if the actuator state will be used as an observation for the agent.
            use_one_day_weather_prediction (bool): We use the internal variables of EnergyPlus to provide with a 
            prediction of the weathertime ahead. The variables to predict are:
            
            * Dry Bulb Temperature in °C with squer desviation of 2.05 °C, 
            * Relative Humidity in % with squer desviation of 20%, 
            * Wind Direction in degree with squer desviation of 40°, 
            * Wind Speed in m/s with squer desviation of 3.41 m/s, 
            * Barometric pressure in Pa with a standart deviation of 1000 Pa, 
            * Liquid Precipitation Depth in mm with desviation of 0.5 mm.
            
            This are predicted from the next hour into the 24 hours ahead defined.
            ep_environment_variables (List[str]):
            ep_thermal_zones_variables (List[str]): 
            ep_object_variables (Dict[str,Dict[str,Tuple[str,str]]]): 
            ep_meters (List[str]): names of meters from EnergyPlus to observe.
            time_variables (List[str]): The time variables to observe in the EnergyPlus simulation. The format is a 
            list of the names described in the EnergyPlus epJSON format documentation (https://energyplus.readthedocs.io/en/latest/schema.html) 
            related with temporal variables. All the options are listed bellow.
            weather_variables (List[str]): The weather variables are related with weather values in the present timestep 
            for the agent. The following list provide all the options avialable. To weather predictions see the 'weather_prob_days' 
            config that is follow in this file.
            infos_variables (Dict[str,List[str]]): The information variables are important to provide information for the 
            reward function. The observation is pass trough the agent as a NDArray but the info is a dictionary. In this 
            way, we can identify clearly the value of a variable with the key name. All the variables used in the reward 
            function must to be in the infos_variables list. The name of the variables must to corresponde with the names 
            defined in the earlier lists.
            no_observable_variables (Dict[str,List[str]]): There are occasions where some variables are consulted to use in 
            training but are not part of the observation space. For that variables, you can use the following  list. An strategy, 
            for example, to use the Fanger PPD value in the reward function but not in the observation space is to aggregate the 
            PPD into the 'infos_variables' and in the 'no_observable_variables' list.
        """
        if observation_fn == NotImplemented:
            raise NotImplementedError("observation_function must be defined.")
        self.observation_fn = observation_fn
        
        # TODO: Al least one variable must to be defined.
        self.use_actuator_state = use_actuator_state
        self.use_one_day_weather_prediction = use_one_day_weather_prediction
        if prediction_hours <= 0 or prediction_hours > 24:
            self.prediction_hours = 24
            raise ValueError(f"The variable 'prediction_hours' must be between 1 and 24. It is taken the value of {prediction_hours}. The value of 24 is used.")
        else:
            self.prediction_hours = prediction_hours
        if self.use_one_day_weather_prediction:
            admissible_values = [values for values in self.prediction_variables.keys()]
            for key in prediction_variables.keys():
                if key not in admissible_values:
                    raise ValueError(f"The key '{key}' is not admissible in the prediction_variables. The admissible values are: {admissible_values}")
            # Update the boolean values in the self.simulation_parameters Dict.
            self.prediction_variables.update(prediction_variables)
            
        self.variables_env = variables_env
        self.variables_thz = variables_thz
        self.variables_obj = variables_obj
        self.meters = meters
        
        self.static_variables = static_variables
        if not simulation_parameters:
            pass
        else:
            # Check that the keys introduced in the Dict are admissible values.
            admissible_values = [values for values in self.simulation_parameters.keys()]
            for key in simulation_parameters.keys():
                if key not in admissible_values:
                    raise ValueError(f"The key '{key}' is not admissible in the simulation_parameters. The admissible values are: {admissible_values}")
            # Update the boolean values in the self.simulation_parameters Dict.
            self.simulation_parameters.update(simulation_parameters)
        
        if not zone_simulation_parameters:
            pass
        else:
            # Check that the keys introduced in the Dict are admissible values.
            admissible_values = [values for values in self.zone_simulation_parameters.keys()]
            for key in zone_simulation_parameters.keys():
                if key not in admissible_values:
                    raise ValueError(f"The key '{key}' is not admissible in the zone_simulation_parameters. The admissible values are: {admissible_values}")
            # Update the boolean values in the self.zone_simulation_parameters Dict.
            self.zone_simulation_parameters.update(zone_simulation_parameters)
        
        if infos_variables == NotImplemented:
            raise NotImplementedError("infos_variables must be defined. The variables defined here are used in the reward function.")
        self.infos_variables = infos_variables
        self.no_observable_variables = no_observable_variables
    
    def actions(
        self,
        action_fn: ActionFunction = NotImplemented,
        ):
        """
        This method is used to modify the actions configuration of the environment.
        
        Args:
            action_space (spaces.Space[ActType]): The action space is the space of the possible actions that the agent
            can take in the environment. In the general case, we use the Discrete form of the gym spaces. This space is
            used to sample the actions from the agent. If not is specified here, the policy defined in RLlib must contain
            the action space.
            action_fn (ActionFunction): In the definition of the action space, usualy is use the discrete form of the 
            gym spaces. In general, we don't use actions from 0 to n directly in the EnergyPlus simulation. With the 
            objective to transform appropiately the discret action into a value action for EP we define the action_fn. 
            This function take the arguments agent_id and action. You can find examples in eprllib.ActionFunctions.
        """
        if action_fn == NotImplemented:
            raise NotImplementedError("action_fn must be defined.")
        self.action_fn = action_fn

    def rewards(
        self,
        reward_fn: RewardFunction = NotImplemented,
        ):
        """
        This method is used to modify the rewards configuration of the environment.

        Args:
            reward_fn (RewardFunction): The reward funtion take the arguments EnvObject (the GymEnv class) and the infos 
            dictionary. As a return, gives a float number as reward. See eprllib.RewardFunctions for examples.
        """
        if reward_fn == NotImplemented:
            raise NotImplementedError("reward_fn must be defined.")
        self.reward_fn = reward_fn

    def episodes(
        self,
        episode_fn: EpisodeFunction = EpisodeFunction({}),
        cut_episode_len: int = 0,
        ):
        """
        This method configure special functions to improve the use of eprllib.

        Args:
            episode_fn (): This method define the properties of the episode, taking the env_config dict and returning it 
            with modifications.
            episode_config (Dict): NotDescribed
            cut_episode_len (int): Sometimes is useful to cut the simulation RunPeriod into diferent episodes. By default, 
            an episode is a entire RunPeriod EnergyPlus simulation. If you set the 'cut_episode_len' in 1 (day) you will 
            truncate the, for example, annual simulation into 365 episodes. If ypu set to 0, no cut will be apply.
        """
        self.episode_fn = episode_fn
        self.cut_episode_len = cut_episode_len
