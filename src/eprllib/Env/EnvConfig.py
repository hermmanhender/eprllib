"""
Environment Configuration
==========================

This module contain the class and methods used to configure the environment.
"""

from typing import Optional, Dict, Any
from eprllib.EpisodeFunctions.EpisodeFunctions import EpisodeFunction
from eprllib.MultiagentFunctions.MultiagentFunctions import MultiagentFunction
from eprllib.MultiagentFunctions.independent import independent
from eprllib.Agents.AgentSpec import AgentSpec

class EnvConfig:
    def __init__(self):
        """
        This is the main object that it is used to relate the EnergyPlus model and the RLlib policy training execution.
        
        """
        # generals
        self.epjson_path: str = NotImplemented
        self.epw_path: str = NotImplemented
        self.output_path: str = NotImplemented
        self.ep_terminal_output: bool = True
        self.timeout: float = 10.0
        self.evaluation: bool = False

        # agents
        self.agents_config: Dict[str, AgentSpec] = NotImplemented
        self.multiagent_fn: MultiagentFunction = independent
        self.multiagent_fn_config: Dict[str, Any] = {}

        # episodes
        self.episode_fn: EpisodeFunction = EpisodeFunction
        self.episode_fn_config: Dict[str,Any] = {}
        self.cut_episode_len: int = 0
    
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
        self.ep_terminal_output = ep_terminal_output
        self.timeout = timeout
        self.evaluation = evaluation
        
        if epjson_path == NotImplemented:
            raise NotImplementedError("epjson_path must be defined.")
        if epjson_path.endswith(".epJSON") or epjson_path.endswith(".idf"):
            pass
        else:
            raise ValueError("The epjson_path must be a path to a epJSON or idf file.")
        if epw_path == NotImplemented:
            raise NotImplementedError("epw_path must be defined.")
        if epw_path.endswith(".epw"):
            pass
        else:
            raise ValueError("The epw_path must be a path to a epw file.")
        if output_path == NotImplemented:
            raise NotImplementedError("output_path must be defined.")
        
        self.epjson_path = epjson_path
        self.epw_path = epw_path
        self.output_path = output_path
        
    def agents(
        self,
        agents_config:Dict[str,AgentSpec] = NotImplemented,
        multiagent_fn: MultiagentFunction = NotImplemented,
        multiagent_fn_config: Dict[str, Any] = {},
        ):
        """
        This method is used to modify the agents configuration of the environment.

        Args:
            agents_config (Dict[str,Dict[str,Any]]): This dictionary contain the names of the agents 
            involved in the environment. The mandatory components of the agent are: ep_actuator_config, 
            thermal_zone, thermal_zone_indicator, actuator_type, agent_indicator.
            
        """
        if agents_config == NotImplemented:
            raise NotImplementedError("agents_config must be defined.")

        self.agents_config = agents_config
        
        if multiagent_fn == NotImplemented:
            print("The multiagent function is not defined. The default function (independent learning) will be used.")
        else:
            self.multiagent_fn = multiagent_fn
            self.multiagent_fn_config = multiagent_fn_config

    def episodes(
        self,
        episode_fn: EpisodeFunction = NotImplemented,
        episode_fn_config: Dict[str,Any] = {},
        cut_episode_len: int = 0,
        ):
        """
        This method configure special functions to improve the use of eprllib.

        Args:
            episode_fn (): This method define the properties of the episode, taking the env_config dict and returning it 
            with modifications.
            
            episode_fn_config (Dict): NotDescribed
            
            cut_episode_len (int): Sometimes is useful to cut the simulation RunPeriod into diferent episodes. By default, 
            an episode is a entire RunPeriod EnergyPlus simulation. If you set the 'cut_episode_len' in 1 (day) you will 
            truncate the, for example, annual simulation into 365 episodes. If ypu set to 0, no cut will be apply.
        """
        self.cut_episode_len = cut_episode_len
        
        if episode_fn == NotImplemented:
            print("The episode function is not defined. The default episode function will be used.")
        else:
            self.episode_fn = episode_fn
            self.episode_fn_config = episode_fn_config

    def build(self) -> Dict:
        """
        Convert an EnvConfig object into a dict before to be used in the env_config parameter of RLlib environment config.
        """
        # Check that the variables defined in EnvConfig are the allowed in the EnvConfig base
        # class.
        # if env_config_validation(MyEnvConfig):
        return vars(self)