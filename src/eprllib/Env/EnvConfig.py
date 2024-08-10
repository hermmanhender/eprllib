"""
Environment Configuration
=========================

This module contain the class and methods used to configure the environment.
"""

from typing import Optional, List, Dict, Tuple, Any
from eprllib.ActionFunctions.ActionFunctions import ActionFunction
from eprllib.RewardFunctions.RewardFunctions import RewardFunction
from eprllib.EpisodeFunctions.EpisodeFunctions import EpisodeFunction

def env_config_to_dict(EnvConfig) -> Dict:
    """
    Convert an EnvConfig object into a dict before to be used in the env_config parameter of RLlib environment config.
    """
    return vars(EnvConfig)

class EnvConfig:
    def __init__(self):
        """
        This is the main object that it is used to relate the EnergyPlus model and the RLlib policy training execution.
        """
        # generals
        self.epjson_path: str = ''
        self.epw_path: str = ''
        self.output_path: str = ''
        self.ep_terminal_output: bool = True
        self.timeout: float = 10.0

        # agents
        self.agents_config: Dict[str,Dict[str,Any]] = {}

        # observations
        self.ep_environment_variables: List|bool = False
        self.ep_thermal_zones_variables: List|bool = False
        self.ep_object_variables: Dict[str,Dict[str,Tuple[str,str]]]|bool = False
        self.ep_meters: List|bool = False
        self.time_variables: List|bool = False
        self.weather_variables: List|bool = False
        self.infos_variables: Dict[str,List]|bool = False
        self.no_observable_variables: Dict[str,List]|bool = False
        self.use_actuator_state: bool = False
        self.use_agent_indicator: bool = False
        self.use_agent_type: bool = False
        self.use_building_properties: bool = False
        self.building_properties: Dict[str,Dict[str,float]] = {}
        self.use_one_day_weather_prediction: bool = False

        # actions
        self.action_fn: ActionFunction = ActionFunction

        # rewards
        self.reward_fn: RewardFunction = RewardFunction

        # functionalities
        self.cut_episode_len: int = 1
        self.episode_fn: EpisodeFunction = EpisodeFunction
    
    def generals(
        self, 
        epjson_path:str,
        epw_path:str,
        output_path:str,
        ep_terminal_output:Optional[bool] = True,
        timeout:Optional[float] = 10.0
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
        """
        self.epjson_path = epjson_path
        self.epw_path = epw_path
        self.output_path = output_path
        self.ep_terminal_output = ep_terminal_output
        self.timeout = timeout
        
    def agents(
        self, 
        agents_config:Dict[str,Dict[str,Any]]
        ):
        """
        This method is used to modify the agents configuration of the environment.

        Args:
            agents_config (Dict[str,Dict[str,Any]]): This dictionary contain the names of the agents 
            involved in the environment. The mandatory components of the agent are: ep_actuator_configuration, 
            thermal_zone, actuator_type, agent_indicator.
        """
        self.agents_config = agents_config
    
    def observations(
        self,
        ep_environment_variables: List[str]|bool = False,
        ep_thermal_zones_variables: List[str]|bool = False,
        ep_object_variables: Dict[str,Dict[str,Tuple[str,str]]]|bool = False,
        ep_meters: List[str]|bool = False,
        time_variables: List[str]|bool = False,
        weather_variables: List[str]|bool = False,
        infos_variables: Dict[str,List[str]]|bool = False,
        no_observable_variables: Dict[str,List[str]]|bool = False,
        use_actuator_state: Optional[bool] = False,
        use_agent_indicator: Optional[bool] = False,
        use_agent_type: Optional[bool] = False,
        use_building_properties: Optional[bool] = False,
        building_properties: Optional[Dict[str,Dict[str,float]]] = {},
        use_one_day_weather_prediction: Optional[bool] = False,
        ):
        """
        This method is used to modify the observations configuration of the environment.

        Args:
            use_actuator_state (bool): define if the actuator state will be used as an observation for the agent.
            use_agent_indicator (bool): define if agent indicator will be used as an observation for the agent. 
            This is recommended True for muilti-agent usage and False for single agent case.
            use_agent_type (bool): define if the agent/actuator type will be used. This is recommended for different 
            types of agents actuating in the same environment.
            use_building_properties (bool): # define if the building properties will be used as an observation for 
            the agent. This is recommended if different buildings/thermal zones will be used with the same policy.
            building_properties (Dict[str,Dict[str,float]]): # The episode config define important aspects about the 
            building to be simulated in the episode.
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
        # TODO: Al least one variable must to be defined.
        self.use_actuator_state = use_actuator_state
        self.use_agent_indicator = use_agent_indicator
        self.use_agent_type = use_agent_type
        self.use_building_properties = use_building_properties
        self.building_properties = building_properties
        self.use_one_day_weather_prediction = use_one_day_weather_prediction
        self.ep_environment_variables = ep_environment_variables
        self.ep_thermal_zones_variables = ep_thermal_zones_variables
        self.ep_object_variables = ep_object_variables
        self.ep_meters = ep_meters
        self.time_variables = time_variables
        self.weather_variables = weather_variables
        self.infos_variables = infos_variables
        self.no_observable_variables = no_observable_variables
    
    def actions(
        self,
        action_fn: ActionFunction = ActionFunction,
        ):
        """
        This method is used to modify the actions configuration of the environment.
        
        Args:
            action_fn (ActionFunction): In the definition of the action space, usualy is use the discrete form of the 
            gym spaces. In general, we don't use actions from 0 to n directly in the EnergyPlus simulation. With the 
            objective to transform appropiately the discret action into a value action for EP we define the action_fn. 
            This function take the arguments agent_id and action. You can find examples in eprllib.ActionFunctions.
        """
        self.action_fn = action_fn

    def rewards(
        self,
        reward_fn: RewardFunction = RewardFunction,
        ):
        """
        This method is used to modify the rewards configuration of the environment.

        Args:
            reward_fn (RewardFunction): The reward funtion take the arguments EnvObject (the GymEnv class) and the infos 
            dictionary. As a return, gives a float number as reward. See eprllib.RewardFunctions for examples.
        """
        self.reward_fn = reward_fn

    def functionalities(
        self,
        episode_fn: EpisodeFunction = EpisodeFunction,
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
