"""This module is used to define the basic configuration
of an environment.
"""
from typing import Optional, List, Dict, Set, Tuple, Any
from eprllib.Tools.ActionTransformers import ActionTransformer
from eprllib.Tools.Rewards import RewardFunction

def env_config_to_dict(EnvConfig) -> Dict:
    """Convert an EnvConfig object into a dict before to be used
    in the env_config parameter of RLlib environment config.
    """
    return vars(EnvConfig)

class EnvConfig:
    def __init__(self):
        """The EnvConfig class is used to define the basic configuration
        of an environment.
        """
        # generals
        self.epjson_path:str = None
        self.epw_path:str = None
        self.output_path:str = None
        self.ep_terminal_output: bool = True
        self.timeout: float = 10.0

        # agents
        self.agents_config: Dict[str,Dict[str,Any]] = None

        # observations
        self.use_actuator_state: bool = None
        self.use_agent_indicator: bool = None
        self.use_agent_type: bool = None
        self.use_building_properties: bool = None
        self.buildig_properties: Dict[str,Dict[str,float]] = None
        self.use_one_day_weather_prediction: bool = False
        self.ep_environment_variables: List = None
        self.ep_thermal_zones_variables: List = None
        self.ep_object_variables: Dict[str,Dict[str,Tuple[str,str]]] = None
        self.ep_meters: List = None

        # actions
        self.action_transformer: ActionTransformer = None

        # rewards
        self.reward_fn: RewardFunction = None
        self.reward_fn_config: Dict[str,Dict[str,Any]] = None

        # functionalities
        self.cut_episode_len: int = None
        self.episode_fn = None
        self.episode_config: Dict = None
    
    def generals(
        self,
        epjson_path:Optional[str] = None,
        epw_path:Optional[str] = None,
        output_path:Optional[str] = None,
        ep_terminal_output:bool = None,
        timeout: float = None,
    ) -> None:
        """This method is used to modify the general configuration of the environment.

        Args:
            epjson_path (str): The path to the EnergyPlus model in the format of epJSON file.
            epw_path (str): The path to the EnergyPlus weather file in the format of epw file.
            output_path (str): The path to the output directory for the EnergyPlus simulation.
            ep_terminal_output (bool): For dubugging is better to print in the terminal the outputs 
            of the EnergyPlus simulation process.
            timeout (float): timeout define the time that the environment wait for an observation and 
            the time that the environment wait to apply an action in the EnergyPlus simulation. After 
            that time, the episode is finished. If your environment is time consuming, you can increase 
            this limit. By default the value is 10 seconds.
        
        Return:
            The EnvConfig modified.
        """
        if epjson_path != None:
            self.epjson_path = epjson_path
        if epw_path != None:
            self.epw_path = epw_path
        if output_path != None:
            self.output_path = output_path
        if ep_terminal_output != None:
            self.ep_terminal_output = ep_terminal_output
        if timeout != None:
            self.timeout = timeout
        
    def agents(
        self,
        agents_config: Dict[str,Dict[str,Any]] = None,
    ) -> None:
        """This method is used to modify the agents configuration of the environment.

        Args:
            agents_config (Dict[str,Dict[str,Any]]): This dictionary contain the names of the agents involved in the environment. The mandatory components of the agent are: ep_actuator_configuration, thermal_zone, actuator_type, agent_indicator.
        
        Return:
            EnvConfig: The environment with modifications.
        """
        if agents_config == None:
            raise NotImplementedError(
                """The agents must to be configured.

                Example:
                    OfficeModel = EnvConfig().agents({
                            'Agent 1 in Room 1': {
                                'ep_actuator_config': (
                                    "Ideal Loads Air System", 
                                    "Air Mass Flow Rate", 
                                    "Thermal Zone: Living Ideal Loads Air System"
                                ),
                                'thermal_zone': 'Thermal Zone: Living',
                                'actuator_type': 3,
                                'agent_indicator': 1,
                            },
                        }
                    )
                """
            )
        self.agents_config = agents_config
    
    def observations(
        self,
        use_actuator_state: bool = None,
        use_agent_indicator: bool = None,
        use_agent_type: bool = None,
        use_building_properties: bool = None,
        buildig_properties: Dict[str,Dict[str,float]] = None,
        use_one_day_weather_prediction: bool = None,
        ep_environment_variables: List[str] = None,
        ep_thermal_zones_variables: List[str] = None,
        ep_object_variables: Dict[str,Dict[str,Tuple[str,str]]] = None,
        ep_meters: List[str] = None,
        time_variables: List[str] = None,
        weather_variables: List[str] = None,
        infos_variables: Dict[str,List[str]] = None,
        no_observable_variables: Dict[str,List[str]] = None
    ) -> None:
        """This method is used to modify the observations configuration of the environment.

        Args:
            use_actuator_state (bool): define if the actuator state will be used as an observation for the agent.
            use_agent_indicator (bool): define if agent indicator will be used as an observation for the agent. This is recommended True for muilti-agent usage and False for single agent case.
            use_agent_type (bool): define if the agent/actuator type will be used. This is recommended for different types of agents actuating in the same environment.
            use_building_properties (bool): # define if the building properties will be used as an observation for the agent. This is recommended if different buildings/thermal zones will be used with the same policy.
            buildig_properties (Dict[str,Dict[str,float]]): # The episode config define important aspects about the building to be simulated in the episode.
            use_one_day_weather_prediction (bool): We use the internal variables of EnergyPlus to provide with a prediction of the weathertime ahead. The variables to predict are:
               - Dry Bulb Temperature in °C with squer desviation of 2.05 °C, 
               - Relative Humidity in % with squer desviation of 20%, 
               - Wind Direction in degree with squer desviation of 40°, 
               - Wind Speed in m/s with squer desviation of 3.41 m/s, 
               - Barometric pressure in Pa with a standart deviation of 1000 Pa, 
               - Liquid Precipitation Depth in mm with desviation of 0.5 mm.
            This are predicted from the next hour into the 24 hours ahead defined.
            ep_environment_variables (List[str]):
            ep_thermal_zones_variables (List[str]): 
            ep_object_variables (Dict[str,Dict[str,Tuple[str,str]]]): 
            ep_meters (List[str]): names of meters from EnergyPlus to observe.
            time_variables (List[str]): The time variables to observe in the EnergyPlus simulation. The format is a list of the names described in the EnergyPlus epJSON format documentation (https://energyplus.readthedocs.io/en/latest/schema.html) related with temporal variables. All the options are listed bellow.
            weather_variables (List[str]): The weather variables are related with weather values in the present timestep for the agent. The following list provide all the options avialable. To weather predictions see the 'weather_prob_days' config that is follow in this file.
            infos_variables (Dict[str,List[str]]): The information variables are important to provide information for the reward function. The observation is pass trough the agent as a NDArray but the info is a dictionary. In this way, we can identify clearly the value of a variable with the key name. All the variables used in the reward function must to be in the infos_variables list. The name of the variables must to corresponde with the names defined in the earlier lists.
            no_observable_variables (Dict[str,List[str]]): There are occasions where some variables are consulted to use in training but are not part of the observation space. For that variables, you can use the following  list. An strategy, for example, to use the Fanger PPD value in the reward function but not in the observation space is to aggregate the PPD into the 'infos_variables' and in the 'no_observable_variables' list.
        
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
                raise NotImplementedError(
                    """The implmentation of building_properties is mandatory
                    when you set 'use_building_properties=True'. Set this to False or 
                    proporcionate a Dict[str,Dict[str,float]].
                    """
                )
            else:
                self.buildig_properties = buildig_properties
        if use_one_day_weather_prediction != None:
            self.use_one_day_weather_prediction = use_one_day_weather_prediction

        if ep_environment_variables != None:
            self.ep_environment_variables = ep_environment_variables
        if ep_thermal_zones_variables != None:
            self.ep_thermal_zones_variables = ep_thermal_zones_variables
        if ep_object_variables != None:
            self.ep_object_variables = ep_object_variables
        if ep_meters != None:
            self.ep_meters = ep_meters
        if time_variables != None:
            self.time_variables = time_variables
        if weather_variables != None:
            self.weather_variables = weather_variables
        if infos_variables != None:
            self.infos_variables = infos_variables
        if no_observable_variables != None:
            self.no_observable_variables = no_observable_variables
    
    def actions(
        self,
        action_transformer: ActionTransformer = None
    ) -> None:
        """This method is used to modify the actions configuration of the environment.
        
        Args:
            action_transformer (ActionTransformer): In the definition of the action space, usualy is use 
            the discrete form of the gym spaces. In general, we don't use actions from 0 to n directly in 
            the EnergyPlus simulation. With the objective to transform appropiately the discret action 
            into a value action for EP we define the action_transformer funtion. This function take the 
            arguments agent_id and action. You can find examples in eprllib.Tools.ActionTransformers.
        """
        if action_transformer != None:
            self.action_transformer = action_transformer

    def rewards(
        self,
        reward_fn: RewardFunction = None,
        reward_fn_config: Dict[str,Dict[str,Any]] = None
    ) -> None:
        """This method is used to modify the rewards configuration of the environment.

        Args:
            reward_fn (RewardFunction): The reward funtion take the arguments EnvObject (the GymEnv class) 
            and the infos dictionary. As a return, gives a float number as reward. See eprllib.tools.rewards
            reward_fn_config (Dict[str,Dict[str,Any]]): 
        """
        if reward_fn != None:
            self.reward_fn = reward_fn

            if reward_fn_config == None:
                raise NotImplementedError(
                    """If you set a reward_fn you need to specify the reward_fn_config. If the function don't 
                    use a config, set this parameter to False."""
                )
            else:
                self.reward_fn_config = reward_fn_config

    def functionalities(
        self,
        episode_fn = None,
        episode_config: Dict = None,
        cut_episode_len: int = None,
    ) -> None:
        """This method configure special functions to improve the use of eprllib.

        Args:
            episode_fn (): This method define the properties of the episode, taking the env_config dict and 
            returning it with modifications.
            episode_config (Dict): 
            cut_episode_len (int): Sometimes is useful to cut the simulation RunPeriod into diferent episodes. 
            By default, an episode is a entire RunPeriod EnergyPlus simulation. If you set the 'cut_episode_len' 
            in 1 (day) you will truncate the, for example, annual simulation into 365 episodes.
        
        Return:
            EnvConfig: The environment with modifications.
        """
        if episode_fn != None:
            self.episode_fn = episode_fn

            if episode_config == None:
                raise NotImplementedError(
                    """If you set a episode_fn you need to specify the episode_fn_config. If the 
                    function don't use a config, set this parameter to False.
                    """
                )
            else:
                self.episode_config = episode_config
        if cut_episode_len != None:
            self.cut_episode_len = cut_episode_len
