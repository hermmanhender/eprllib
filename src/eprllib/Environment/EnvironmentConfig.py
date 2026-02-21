"""
Environment Configuration
==========================

This module contains the class and methods used to configure the environment.
"""
from typing import Optional, Dict, Any, Type
from tempfile import TemporaryDirectory

from eprllib.Episodes.BaseEpisode import BaseEpisode
from eprllib.Episodes.DefaultEpisode import DefaultEpisode
from eprllib.Connectors.BaseConnector import BaseConnector
from eprllib.Connectors.DefaultConnector import DefaultConnector
from eprllib.Connectors.IndependentConnector import IndependentConnector
from eprllib.Agents.AgentSpec import AgentSpec
from eprllib.Environment import TIMEOUT, CUT_EPISODE_LEN
from eprllib import logger

class EnvironmentConfig:
    """
    This class is used to configure the environment.
    """
    epjson_path: Optional[str] = None
    epw_path: Optional[str] = None
    output_path: Optional[str] = None
    ep_terminal_output: bool = True
    timeout: float | int = TIMEOUT
    evaluation: bool = False
    agents_config: Optional[Dict[str, AgentSpec|Dict[str, Any]]] = None
    connector_fn: Optional[Type[BaseConnector]] = None
    connector_fn_config: Dict[str, Any] = {}
    episode_fn: Optional[Type[BaseEpisode]] = None
    episode_fn_config: Dict[str, Any] = {}
    cut_episode_len: int = CUT_EPISODE_LEN
    
    def __init__(self):
        """
        This is the main object that is used to relate the EnergyPlus model and the RLlib policy training execution.
        """
        # General configuration
        self.epjson_path: Optional[str] = None
        self.epw_path: Optional[str] = None
        self.output_path: Optional[str] = None
        self.ep_terminal_output: bool = True
        self.timeout: float | int = TIMEOUT
        self.evaluation: bool = False

        # Agents configuration
        self.agents_config: Optional[Dict[str, AgentSpec|Dict[str, Any]]] = None
        self.connector_fn: Optional[Type[BaseConnector]] = None
        self.connector_fn_config: Dict[str, Any] = {}

        # Episodes configuration
        self.episode_fn: Optional[Type[BaseEpisode]] = None
        self.episode_fn_config: Dict[str, Any] = {}
        self.cut_episode_len: int = CUT_EPISODE_LEN
    
    def to_dict(self) -> Dict[str, Any]:
        """Converts all settings into a legacy config dict for backward compatibility.

        Returns:
            A complete EnvironmentConfigDict, usable in backward-compatible Tune/RLlib
            use cases.
        """
        try:
            return self._build()
        except Exception as e:
            logger.error(f"EnvironmentConfig: Error building EnvironmentConfig: {e}")
            raise ValueError("Failed to build EnvironmentConfig. Please check the configuration settings.") from e
    
    def from_dict(cls, config_dict: Dict[str, Any]) -> "EnvironmentConfig":
        """Creates an EnvironmentConfig from a legacy python config dict.

        Args:
            config_dict: The legacy formatted python config dict for some algorithm.

        Returns:
            A new EnvironmentConfig object that matches the given python config dict.
        """
        pass
    
    def update_from_dict(
        self,
        config_dict: Dict[str, Any],#: PartialAlgorithmConfigDict,
    ) -> "EnvironmentConfig":
        """Modifies this EnvironmentConfig via the provided python config dict.

        Args:
            config_dict: The dict to use for updating this EnvironmentConfig.

        Returns:
            This updated EnvironmentConfig object.
        """
        # TODO: Implement this method to update the EnvironmentConfig from a dict.
        pass
    
    def _build(self) -> Dict[str, Any]:
        """
        Convert an EnvConfig object into a dict before to be used in the env_config parameter of RLlib environment config.
        Also this method chek that all the variables are well defined and add some constant parameters to use after in 
        the program (like agents unique ID).
        """
        logger.info("EnvironmentConfig: Building the environment configuration...")
        # === GENERALS === #
        # epjson_path
        if self.epjson_path is None:
            logger.warning("EnvironmentConfig: The epjson_path is not defined. Is spected to be defined in the Episode class used. If don't, an error will be raised in the future.")
            pass
        elif self.epjson_path.endswith(".epJSON"):
            pass
        elif self.epjson_path.endswith(".idf"):
            logger.info("EnvironmentConfig: The epjson_path is an IDF file. Consider converting to epJSON.")
            pass
        else:
            msg = "EnvironmentConfig: The epjson_path is not a string. Consider converting to epJSON."
            logger.warning(msg)
            raise ValueError(msg)
        
        # epw_path
        if self.epw_path is None:
            logger.warning("EnvironmentConfig: The epw_path is not defined. Is spected to be defined in the Episode class used. If don't, an error will be raised in the future.")
            pass
        elif self.epw_path.endswith(".epw"):
            pass
        else:
            msg = "EnvironmentConfig: The epw_path is not a string. Consider converting to epw."
            logger.warning(msg)
            raise ValueError(msg)
        
        # output_path
        if self.output_path is None:
            self.output_path = TemporaryDirectory("eprllib_output").name
            logger.warning(f"EnvironmentConfig: The output_path is not defined. {self.output_path} will be used.")
        
        # === AGENTS === #
        if self.agents_config is None:
            msg = "EnvironmentConfig: The agents_config is not defined. At least one agent must be defined."
            logger.error(msg)
            raise ValueError(msg)
            
        ix = 0
        for agent in self.agents_config:
            config = self.agents_config[agent]
            if isinstance(config, AgentSpec):
                config = config.build()
                self.agents_config[agent] = config
            assert isinstance(config, Dict), f"The agent_config for '{agent}' must be a dictionary."
            config.update({'agent_id': ix})
            ix += 1
        logger.debug("EnvironmentConfig: Agents config built successfully.")
        
        # === CONNECTOR === #
        if self.connector_fn is None:
            if ix > 1:
                # TODO: If FullySharedConnector is better than IndependentConnector in the future, change this.
                self.connector_fn = IndependentConnector
                self.connector_fn_config = {}
                logger.warning(f"EnvironmentConfig: The connector function is not defined. The connector {self.connector_fn.__name__} will be used with the following configuration: {self.connector_fn_config}.")
            elif ix == 1:
                self.connector_fn = DefaultConnector
                self.connector_fn_config = {}
                logger.warning(f"EnvironmentConfig: The connector function is not defined. The connector {self.connector_fn.__name__} will be used.")
        
        logger.debug("EnvironmentConfig: Connector config built successfully.")
        
        # === EPISODES === #
        if self.episode_fn is None:
            msg = "EnvironmentConfig: The episode function is not defined. The default episode function will be used."
            logger.info(msg)
            self.episode_fn = DefaultEpisode
            self.episode_fn_config = {}
            
        return vars(self)
    
    
    def generals(
        self, 
        epjson_path: Optional[str] = None,
        epw_path: Optional[str] = None,
        output_path: Optional[str] = None,
        ep_terminal_output: bool = True,
        timeout: float|int = TIMEOUT,
        evaluation: bool = False,
        ) -> None:
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
        agents_config: Optional[Dict[str,AgentSpec|Dict[str,Any]]] = None,
        ) -> None:
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
        connector_fn: Type[BaseConnector] = BaseConnector,
        connector_fn_config: Dict[str, Any] = {},
        ) -> None:
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
        episode_fn: Optional[Type[BaseEpisode]] = None,
        episode_fn_config: Dict[str,Any] = {},
        cut_episode_len: int = 0,
        ) -> None:
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

    def __getitem__(self, key: str):
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any):
        setattr(self, key, value)
        