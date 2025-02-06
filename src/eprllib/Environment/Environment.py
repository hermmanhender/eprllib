"""
Environment
============

This module contains the class and methods used to configure the environment.
"""

from typing import Optional
from eprllib.Environment.EnvironmentConfig import EnvironmentConfig

class Environment:
    #: The AlgorithmConfig instance of the Algorithm.
    env_config: Optional[EnvironmentConfig] = None
    
    def from_checkpoint(
        cls,
        path: str
    ) -> "Environment":
        return NotImplementedError("Not implemented yet.")
    
    def __init__(
        self,
        env_config: Optional[EnvironmentConfig] = None,
        **kwargs,
    ):
        """
        Initializes an Environment instance.
        
        Args:
            env_config (Optional[EnvironmentConfig]): The EnvironmentConfig instance to use.
            **kwargs: Additional keyword arguments to pass to the EnvironmentConfig instance.
            
        """
        if isinstance(env_config, dict):
            default_config = self.get_default_config()
        
        # Validate and freeze our AlgorithmConfig object (no more changes possible).
        # env_config.validate_env()
    
    def get_default_config(cls) -> EnvironmentConfig:
        return EnvironmentConfig()
    