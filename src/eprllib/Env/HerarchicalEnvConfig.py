"""
Herarchical Environment Configuration
======================================

This module contain the class and methods used to configure the herarchical environment.
"""

from typing import Optional, Dict, Any
from eprllib.Utils.annotations import override
from eprllib.ObservationFunctions.ObservationFunctions import ObservationFunction
from eprllib.Env.EnvConfig import EnvConfig
        
class HerarchicalEnvConfig(EnvConfig):
    def __init__(self):
        super.__init__()
        
        self.top_level_temporal_scale: int = 48
        self.top_level_agent_name: str = NotImplemented
        
    @override(EnvConfig)
    def generals(
        self, 
        epjson_path:str = NotImplemented,
        epw_path:str = NotImplemented,
        output_path:str = NotImplemented,
        ep_terminal_output:Optional[bool] = True,
        timeout:Optional[float] = 10.0,
        evaluation:bool = False,
        observation_fn: ObservationFunction = NotImplemented,
        observation_fn_config: Dict[str, Any] = {},
        top_level_temporal_scale: int = NotImplemented,
        top_level_agent_name: str = NotImplemented
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
        
        if observation_fn == NotImplemented:
            print("The observation function is not defined. The default observation function (independient) will be used.")
        else:
            self.observation_fn = observation_fn
            self.observation_fn_config = observation_fn_config
            
        if top_level_temporal_scale == NotImplemented:
            print(f"The top_level_temporal_scale was not defined in the HerarchicalEnvConfig.general method. The default of 48 timesteps is used.")

        if top_level_agent_name == NotImplemented:
            raise NotImplementedError(f"One of the agents defined in the agent_config argument in the HerarchicalEnvConfig.agents method must be assigned as top_level_agent_name.")
        elif type(top_level_agent_name) != str:
            raise ValueError(f"The agent name must be an string. The input was: {top_level_agent_name}")
        else:
            self.top_level_agent_name = top_level_agent_name
          
    def build_herarchical(self) -> Dict:
        """
        Convert an EnvConfig object into a dict before to be used in the env_config parameter of RLlib environment config.
        """
        # Check the parameters defined.
        if self.top_level_agent_name not in self.agents_config.keys():
            raise ValueError(f"The agent {self.top_level_agent_name} must be defined.")
        
        # After check the herarchical conditions, build the configuration.
        return self.build()