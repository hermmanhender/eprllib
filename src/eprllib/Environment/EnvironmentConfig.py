"""
Environment Configuration
==========================

This module contains the class and methods used to configure the environment.
"""
import logging
from typing import Optional, Dict, Any
from tempfile import TemporaryDirectory
from eprllib.Episodes.BaseEpisode import BaseEpisode
from eprllib.Episodes.DefaultEpisode import DefaultEpisode
from eprllib.AgentsConnectors.BaseConnector import BaseConnector
from eprllib.AgentsConnectors.DefaultConnector import DefaultConnector
from eprllib.AgentsConnectors.IndependentConnector import IndependentConnector
from eprllib.Agents.AgentSpec import AgentSpec
from eprllib.Environment import TIMEOUT, CUT_EPISODE_LEN

logger = logging.getLogger("ray.rllib")

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
        self.timeout: float | int = TIMEOUT
        self.evaluation: bool = False

        # Agents configuration
        self.agents_config: Dict[str, AgentSpec | Dict] = None
        self.connector_fn: BaseConnector = DefaultConnector
        self.connector_fn_config: Dict[str, Any] = {}

        # Episodes configuration
        self.episode_fn: BaseEpisode = DefaultEpisode
        self.episode_fn_config: Dict[str, Any] = {}
        self.cut_episode_len: int = CUT_EPISODE_LEN
    
    def to_dict(self) -> Dict:
        """Converts all settings into a legacy config dict for backward compatibility.

        Returns:
            A complete EnvironmentConfigDict, usable in backward-compatible Tune/RLlib
            use cases.
        """
        try:
            return self._build()
        except Exception as e:
            logger.error(f"Error building EnvironmentConfig: {e}")
            raise ValueError("Failed to build EnvironmentConfig. Please check the configuration settings.") from e
    
    def update_from_dict(
        self,
        config_dict,#: PartialAlgorithmConfigDict,
    ) -> "EnvironmentConfig":
        """Modifies this EnvironmentConfig via the provided python config dict.

        Args:
            config_dict: The dict to use for updating this EnvironmentConfig.

        Returns:
            This updated EnvironmentConfig object.
        """
        # TODO: Implement this method to update the EnvironmentConfig from a dict.
        pass
    
    def _build(self) -> Dict:
        """
        Convert an EnvConfig object into a dict before to be used in the env_config parameter of RLlib environment config.
        Also this method chek that all the variables are well defined and add some constant parameters to use after in 
        the program (like agents unique ID).
        """
        logger.info("Building the environment configuration...")
        # === GENERALS === #
        # epjson_path
        if self.epjson_path is None:
            logger.warning("The epjson_path is not defined. Is spected to be defined in the Episode class used. If don't, an error will be raised in the future.")
            pass
        elif isinstance(self.epjson_path, str):
            if self.epjson_path.endswith(".epJSON"):
                pass
            elif self.epjson_path.endswith(".idf"):
                logger.info("The epjson_path is an IDF file. Consider converting to epJSON.")
                pass
            else:
                msg = f"The epjson_path is not a valid epJSON or IDF file: {self.epjson_path}"
                logger.error(msg)
                raise ValueError(msg)
        else:
            msg = "The epjson_path is not a string. Consider converting to epJSON."
            logger.warning(msg)
            raise ValueError(msg)
        
        # epw_path
        if self.epw_path is None:
            logger.warning("The epw_path is not defined. Is spected to be defined in the Episode class used. If don't, an error will be raised in the future.")
            pass
        elif isinstance(self.epw_path, str):
            if self.epw_path.endswith(".epw"):
                pass
            else:
                msg = f"The epw_path is not a valid epw file: {self.epw_path}"
                logger.error(msg)
                raise ValueError(msg)
        else:
            msg = "The epw_path is not a string. Consider converting to epw."
            logger.warning(msg)
            raise ValueError(msg)
        
        # output_path
        if self.output_path is None:
            self.output_path = TemporaryDirectory("eprllib_output").name
            logger.warning(f"The output_path is not defined. {self.output_path} will be used.")
            
        elif isinstance(self.output_path, str):
            pass
        else:
            msg = f"The output_path is not a string: {self.output_path}"
            logger.error(msg)
            raise ValueError(msg)
        
        # ep_terminal_output
        if isinstance(self.ep_terminal_output, bool):
            pass
        else:
            msg = f"The ep_terminal_output is not a boolean: {self.ep_terminal_output}"
            logger.error(msg)
            raise ValueError(msg)
        
        # timeout
        if isinstance(self.timeout, (float, int)):
            pass
        else:
            msg = f"The timeout is not a float or an integer: {self.timeout}"
            logger.error(msg)
            raise ValueError(msg)
        
        # evaluation
        if isinstance(self.evaluation, bool):
            pass
        else:
            msg = f"The evaluation is not a boolean: {self.evaluation}"
            logger.error(msg)
            raise ValueError(msg)
        logger.debug("Generals config built successfully.")
        
        # === AGENTS === #
        if self.agents_config is None:
            msg = "The agents_config is not defined. At least one agent must be defined."
            logger.error(msg)
            raise ValueError(msg)
            
        ix = 0
        for agent, config in self.agents_config.items():
            if isinstance(config, AgentSpec):
                self.agents_config[agent] = config.build()
            elif isinstance(config, dict):
                pass
            else:
                msg = f"The agent {agent} must be an instance of AgentSpec or a dictionary."
                logger.error(msg)
                raise ValueError(msg)
            
            self.agents_config[agent].update({'agent_id': ix})
            ix += 1
        logger.debug("Agents config built successfully.")
        
        # === CONNECTOR === #
        if self.connector_fn is None:
            if ix > 1:
                # TODO: If FullySharedConnector is better than IndependentConnector in the future, change this.
                self.connector_fn = IndependentConnector
                self.connector_fn_config = {}
                logger.warning(f"The connector function is not defined. The connector {self.connector_fn.__name__} will be used with the following configuration: {self.connector_fn_config}.")
            if ix == 1:
                self.connector_fn = DefaultConnector
                self.connector_fn_config = {}
                logger.warning(f"The connector function is not defined. The connector {self.connector_fn.__name__} will be used.")
            else:
                msg = "The connector function is not defined. At least one agent must be defined."
                logger.error(msg)
                raise ValueError(msg)
            
        
        elif issubclass(self.connector_fn, BaseConnector):
            pass
        
        else:
            msg = f"The connector_fn must be an instance of BaseConnector but {type(self.connector_fn)} was given."
            logger.error(msg)
            raise ValueError(msg)
        
        if isinstance(self.connector_fn_config, dict):
            pass
        else:
            msg = f"The connector_fn_config must be a dictionary but {type(self.connector_fn_config)} was given."
            logger.error(msg)
            raise ValueError(msg)
        logger.debug("Connector config built successfully.")
        
        # === EPISODES === #
        if self.episode_fn is None:
            msg = "The episode function is not defined. The default episode function will be used."
            logger.info(msg)
            self.episode_fn = DefaultEpisode
            self.episode_fn_config = {}
            
        elif issubclass(self.episode_fn, BaseEpisode):
            pass
        else:
            msg = f"The episode_fn must be an instance of BaseEpisode but {type(self.episode_fn)} was given."
            logger.error(msg)
            raise ValueError(msg)
            
        if isinstance(self.episode_fn_config, dict):
            pass
        else:
            msg = f"The episode_fn_config must be a dictionary but {type(self.episode_fn_config)} was given."
            logger.error(msg)
            raise ValueError(msg)
            
        if isinstance(self.cut_episode_len, int):
            pass
        else:
            msg = f"The cut_episode_len must be an integer but {type(self.cut_episode_len)} was given."
            logger.error(msg)
            raise ValueError(msg)
            
        return vars(self)
    
    
    def generals(
        self, 
        epjson_path: str = None,
        epw_path: str = None,
        output_path: Optional[str] = None,
        ep_terminal_output: Optional[bool] = True,
        timeout: Optional[float | int] = TIMEOUT,
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
        