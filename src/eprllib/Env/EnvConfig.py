"""
Environment Configuration
==========================

This module contains the class and methods used to configure the environment.
"""

from typing import Optional, Dict, Any
from eprllib.Episodes.BaseEpisode import BaseEpisode
from eprllib.Episodes.DefaultEpisode import DefaultEpisode
from eprllib.AgentsConnectors.BaseConnector import BaseConnector
from eprllib.AgentsConnectors.DefaultConnector import DefaultConnector
from eprllib.Agents.AgentSpec import AgentSpec
from eprllib.Utils.Utils import validate_properties

class EnvConfig:
    def __init__(self):
        """
        This is the main object that is used to relate the EnergyPlus model and the RLlib policy training execution.
        """
        # General configuration
        self.epjson_path: str = NotImplemented
        self.epw_path: str = NotImplemented
        self.output_path: str = NotImplemented
        self.ep_terminal_output: bool = True
        self.timeout: float = 10.0
        self.evaluation: bool = False

        # Agents configuration
        self.agents_config: Dict[str, AgentSpec | Dict] = NotImplemented
        self.connector_fn: BaseConnector = DefaultConnector
        self.connector_fn_config: Dict[str, Any] = {}

        # Episodes configuration
        self.episode_fn: BaseEpisode = DefaultEpisode
        self.episode_fn_config: Dict[str, Any] = {}
        self.cut_episode_len: int = 0
    
    def generals(
        self, 
        epjson_path: str = NotImplemented,
        epw_path: str = NotImplemented,
        output_path: str = NotImplemented,
        ep_terminal_output: Optional[bool] = True,
        timeout: Optional[float] = 10.0,
        evaluation: bool = False,
    ):
        """
        This method is used to modify the general configuration of the environment.

        Args:
            epjson_path (str): The path to the EnergyPlus model in the format of epJSON file.
            epw_path (str): The path to the weather file in the format of epw file.
            output_path (str): The path to the output directory.
            ep_terminal_output (Optional[bool]): Flag to enable or disable terminal output from EnergyPlus.
            timeout (Optional[float]): Timeout for the simulation.
            evaluation (bool): Flag to indicate if the environment is in evaluation mode.
        """
        self.epjson_path = epjson_path
        self.epw_path = epw_path
        self.output_path = output_path
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
        
    def agents(
        self,
        agents_config:Dict[str,AgentSpec|Dict] = NotImplemented,
        connector_fn: BaseConnector = NotImplemented,
        connector_fn_config: Dict[str, Any] = {},
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
        
        if connector_fn == NotImplemented:
            print("The multiagent function is not defined. The default function (independent learning) will be used.")
        else:
            self.connector_fn = connector_fn
            self.connector_fn_config = connector_fn_config

    def episodes(
        self,
        episode_fn: BaseEpisode = None,
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
        
        if episode_fn is None:
            print("The episode function is not defined. The default episode function will be used.")
        else:
            self.episode_fn = episode_fn
            self.episode_fn_config = episode_fn_config

    def build(self) -> Dict:
        """
        Convert an EnvConfig object into a dict before to be used in the env_config parameter of RLlib environment config.
        Also this method chek that all the variables are well defined and add some constant parameters to use after in 
        the program (like agents unique ID).
        """
        # Check that the variables defined in EnvConfig are the allowed in the EnvConfig base
        # class.
        # if env_config_validation(MyEnvConfig):
        
        
        # generals
        # Chech that the variables defined in EnvConfig are the allowed in the EnvConfig base
        # class.
        expected_types = {
            'epjson_path': str,
            'epw_path': str,
            'output_path': str,
            'ep_terminal_output': bool,
            'timeout': float,
            'evaluation': bool,
            'agents_config': Dict[str,Dict|AgentSpec],
            'connector_fn': BaseConnector,
            'connector_fn_config': Dict[str,Any],
            'episode_fn': BaseEpisode,
            'episode_fn_config': Dict[str,Any],
            'cut_episode_len': int
        }
        
        is_valid, errors = validate_properties(self, expected_types)
        if is_valid:
            print("All properties have correct types")
        else:
            print("Validation errors:")
            for error in errors:
                print(f"- {error}")
                
        
        if self.epjson_path.endswith(".epJSON") or self.epjson_path.endswith(".idf"):
            pass
        else:
            raise ValueError("The epjson_path must be a path to a epJSON or idf file.")
        
        if self.epw_path.endswith(".epw"):
            pass
        else:
            raise ValueError("The epw_path must be a path to a epw file.")

        # agents
        ix = 0
        for agent, config in self.agents_config.items():
            if type(config) == AgentSpec:
                self.agents_config[agent] = config.build()
            
            self.agents_config[agent].update({'agent_id': ix})
            ix += 1
        
        return vars(self)