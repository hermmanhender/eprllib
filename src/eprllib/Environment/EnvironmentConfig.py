"""
Environment Configuration
==========================

This module contains the class and methods used to configure the environment.
"""
from typing import Optional, Dict, Any
from ray.rllib.env.env_context import EnvContext
from ray.rllib.utils.typing import EnvType
from eprllib.Episodes.BaseEpisode import BaseEpisode
from eprllib.Episodes.DefaultEpisode import DefaultEpisode
from eprllib.AgentsConnectors.BaseConnector import BaseConnector
from eprllib.AgentsConnectors.DefaultConnector import DefaultConnector
from eprllib.Agents.AgentSpec import AgentSpec

class EnvironmentConfig:
    
    def from_dict(cls, config_dict: dict) -> "EnvironmentConfig":
        """Creates an EnvironmentConfig from a legacy python config dict.

        Args:
            config_dict: The legacy formatted python config dict for some algorithm.

        Returns:
            A new EnvironmentConfig object that matches the given python config dict.
        """
        pass
    
    def __init__(self):
        """
        This is the main object that is used to relate the EnergyPlus model and the RLlib policy training execution.
        """
        # General configuration
        self.epjson_path: str = None
        self.epw_path: str = None
        self.output_path: str = None
        self.ep_terminal_output: bool = True
        self.timeout: float | int = 10.0
        self.evaluation: bool = False

        # Agents configuration
        self.agents_config: Dict[str, AgentSpec | Dict] = None
        self.connector_fn: BaseConnector = DefaultConnector
        self.connector_fn_config: Dict[str, Any] = {}

        # Episodes configuration
        self.episode_fn: BaseEpisode = DefaultEpisode
        self.episode_fn_config: Dict[str, Any] = {}
        self.cut_episode_len: int = 0
    
    def to_dict(self) -> "EnvironmentConfig":
        """Converts all settings into a legacy config dict for backward compatibility.

        Returns:
            A complete AlgorithmConfigDict, usable in backward-compatible Tune/RLlib
            use cases.
        """
        pass
    
    def update_from_dict(
        self,
        config_dict,#: PartialAlgorithmConfigDict,
    ) -> "EnvironmentConfig":
        """Modifies this AlgorithmConfig via the provided python config dict.

        Warns if `config_dict` contains deprecated keys.
        Silently sets even properties of `self` that do NOT exist. This way, this method
        may be used to configure custom Policies which do not have their own specific
        AlgorithmConfig classes, e.g.
        `ray.rllib.examples.policy.random_policy::RandomPolicy`.

        Args:
            config_dict: The old-style python config dict (PartialAlgorithmConfigDict)
                to use for overriding some properties defined in there.

        Returns:
            This updated AlgorithmConfig object.
        """
        pass
    
    def build(self) -> Dict:
        """
        Convert an EnvConfig object into a dict before to be used in the env_config parameter of RLlib environment config.
        Also this method chek that all the variables are well defined and add some constant parameters to use after in 
        the program (like agents unique ID).
        """
        # === GENERALS === #
        # epjson_path
        if self.epjson_path is None:
            pass
        elif isinstance(self.epjson_path, str):
            if self.epjson_path.endswith(".epJSON"):
                pass
            elif self.epjson_path.endswith(".idf"):
                print("The epjson_path is an IDF file. Consider converting to epJSON.")
                pass
            else:
                raise ValueError("The epjson_path must be a path to a epJSON or idf file.")
        else:
            raise ValueError("The epjson_path must be a string.")
        
        # epw_path
        if self.epw_path is None:
            pass
        elif isinstance(self.epw_path, str):
            if self.epw_path.endswith(".epw"):
                pass
            else:
                raise ValueError("The epw_path must be a path to a epw file.")
        else:
            raise ValueError("The epw_path must be a string.")
        
        # output_path
        if self.output_path is None:
            print("The output_path is not defined. The default output path will be used.")
        elif isinstance(self.output_path, str):
            pass
        else:
            raise ValueError("The output_path must be a string.")
        
        # ep_terminal_output
        if isinstance(self.ep_terminal_output, bool):
            pass
        else:
            raise ValueError("The ep_terminal_output must be a boolean.")
        
        # timeout
        if isinstance(self.timeout, (float, int)):
            pass
        else:
            raise ValueError("The timeout must be a float or an integer.")
        
        # evaluation
        if isinstance(self.evaluation, bool):
            pass
        else:
            raise ValueError("The evaluation must be a boolean.")
                
        # === AGENTS === #
        if self.agents_config is None:
            raise ValueError("agents_config must be defined.")
            
        ix = 0
        for agent, config in self.agents_config.items():
            if isinstance(config, AgentSpec):
                self.agents_config[agent] = config.build()
            elif isinstance(config, dict):
                pass
            else:
                raise ValueError(f"The agent {agent} must be an instance of AgentSpec or a dictionary.")
            
            self.agents_config[agent].update({'agent_id': ix})
            ix += 1
        # print(f"Agents config: {self.agents_config}")
        
        # === CONNECTOR === #
        if self.connector_fn is None:
            print("The multiagent function is not defined. The default connector will be used.")
            self.connector_fn = DefaultConnector
            self.connector_fn_config = {}
        elif isinstance(self.connector_fn, BaseConnector):
            pass
        else:
            raise ValueError(f"The connector_fn must be an instance of BaseConnector but {type(self.connector_fn)} was given.")
        if isinstance(self.connector_fn_config, dict):
            pass
        else:
            raise ValueError(f"The connector_fn_config must be a dictionary but {type(self.connector_fn_config)} was given.")
        
        # === EPISODES === #
        if self.episode_fn is None:
            print("The episode function is not defined. The default episode function will be used.")
        elif isinstance(self.episode_fn, BaseEpisode):
            pass
        else:
            raise ValueError(f"The episode_fn must be an instance of BaseEpisode but {type(self.episode_fn)} was given.")
            
        if isinstance(self.episode_fn_config, dict):
            pass
        else:
            raise ValueError(f"The episode_fn_config must be a dictionary but {type(self.episode_fn_config)} was given.")
            
        if isinstance(self.cut_episode_len, int):
            pass
        else:
            raise ValueError(f"The cut_episode_len must be an integer but {type(self.cut_episode_len)} was given.")
            
        return vars(self)
    
    def validate_env(env: EnvType, env_context: EnvContext) -> None:
        """Env validator function for this Algorithm class.

        Override this in child classes to define custom validation
        behavior.

        Args:
            env: The (sub-)environment to validate. This is normally a
                single sub-environment (e.g. a gym.Env) within a vectorized
                setup.
            env_context: The EnvContext to configure the environment.

        Raises:
            Exception: in case something is wrong with the given environment.
        """
        pass
    
    
    def generals(
        self, 
        epjson_path: str = None,
        epw_path: str = None,
        output_path: Optional[str] = None,
        ep_terminal_output: Optional[bool] = True,
        timeout: Optional[float | int] = 10.0,
        evaluation: bool = False,
    ):
        """
        This method is used to modify the general configuration of the environment.

        Args:
            epjson_path (str): The path to the EnergyPlus model in the format of epJSON file.
            epw_path (str): The path to the weather file in the format of epw file.
            output_path (str): The path to the output directory.
            ep_terminal_output (Optional[bool]): Flag to enable or disable terminal output from EnergyPlus.
            timeout (Optional[float | int]): Timeout for the simulation.
            evaluation (bool): Flag to indicate if the environment is in evaluation mode.
        """
        self.epjson_path = epjson_path
        self.epw_path = epw_path
        self.output_path = output_path
        self.ep_terminal_output = ep_terminal_output
        self.timeout = timeout
        self.evaluation = evaluation
        
    def agents(
        self,
        agents_config:Dict[str,AgentSpec|Dict] = None,
        ):
        """
        This method is used to modify the agents configuration of the environment.

        Args:
            agents_config (Dict[str,Dict[str,Any]]): This dictionary contain the names of the agents 
            involved in the environment. The mandatory components of the agent are: ep_actuator_config, 
            thermal_zone, thermal_zone_indicator, actuator_type, agent_indicator.
            
        """
        self.agents_config = agents_config

    def connector(
        self,
        connector_fn: BaseConnector = None,
        connector_fn_config: Dict[str, Any] = {},
        ):
        """
        This method is used to modify the agents connector configuration of the environment.

        Args:
            connector_fn (BaseConnector): This method is used to define the interaction between the agents and the 
            environment. By default, the independent learning is used.
            
            connector_fn_config (Dict): NotDescribed
            
        """
        self.connector_fn = connector_fn
        self.connector_fn_config = connector_fn_config
    
    def episodes(
        self,
        episode_fn: BaseEpisode = None,
        episode_fn_config: Dict[str,Any] = {},
        cut_episode_len: int = 0,
        ):
        """
        This method configure episode functions to improve the use of eprllib.

        Args:
            episode_fn (): This method define the properties of the episode, taking the env_config dict and returning it 
            with modifications.
            
            episode_fn_config (Dict): NotDescribed
            
            cut_episode_len (int): Sometimes is useful to cut the simulation RunPeriod into diferent episodes. By default, 
            an episode is a entire RunPeriod EnergyPlus simulation. If you set the 'cut_episode_len' in 1 (day) you will 
            truncate the, for example, annual simulation into 365 episodes. If ypu set to 0, no cut will be apply.
        """
        self.episode_fn = episode_fn
        self.episode_fn_config = episode_fn_config
        self.cut_episode_len = cut_episode_len

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)
        