"""
Specification for the ActionMapper functions
===========================================================
This module defines the ``ActionMapperSpec`` class, which is used to specify
the configuration of ActionMapper functions for agents in reinforcement 
learning environments.
"""
from typing import Dict, Any, Type, Optional
from eprllib.Agents.ActionMappers.BaseActionMapper import BaseActionMapper
from eprllib import logger

class ActionMapperSpec:
    """
    ActionMapperSpec is the base class for a ActionMapper specification to safe configuration of the object.
    """
    action_mapper: Optional[Type[BaseActionMapper]] = None
    action_mapper_config: Dict[str, Any] = {}
    
    def __init__(
        self,
        action_mapper: Optional[Type[BaseActionMapper]] = None,
        action_mapper_config: Dict[str, Any] = {},
    ) -> None:
        """
        Construction method.
        
        Args:
            action_mapper (BaseActionMapper): The ActionMapper function takes the arguments agent_id, observation and returns the
            observation filtered. See ``eprllib.Agents.ActionMappers`` for examples.
            
            action_mapper_config (Dict[str, Any]): The configuration of the ActionMapper function.
        
        Returns:
            None
            
        Raises:
            ValueError: If the action_mapper is not defined.
            ValueError: If the action_mapper_config is not a dictionary.
            ValueError: If the action_mapper is not a subclass of BaseActionMapper.
        """
        self.action_mapper = action_mapper
        self.action_mapper_config = action_mapper_config
        
        logger.info(f"ActionMapperSpec: The ActionMapperSpec was correctly inicializated with {self.action_mapper} class and {self.action_mapper_config} config.")
    
    def __getitem__(self, key:str) -> Any:
        return getattr(self, key)

    def __setitem__(self, key:str, value:Any) -> None:
        valid_keys = self.__dict__.keys()
        if key not in valid_keys:
            msg = f"ActionMapperSpec: Invalid key: {key}."
            logger.error(msg)
            raise KeyError(msg)
        setattr(self, key, value)
    
    def build(self) -> Dict[str, Any]:
        """
        This method is used to build the ActionMapperSpec object.
        """
        if self.action_mapper == None:
            msg = "ActionMapperSpec: The ActionMapper class must be defined."
            logger.error(msg)
            raise ValueError(msg)
        
        logger.info(f"ActionMapperSpec: The ActionMapper class is {self.action_mapper}.")
        
        return vars(self)
