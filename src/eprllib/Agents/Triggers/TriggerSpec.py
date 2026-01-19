"""
Specification for the trigger functions
===========================================================
This module defines the `TriggerSpec` class, which is used to specify the configuration of trigger 
functions for agents in reinforcement learning environments.
"""
from typing import Dict, Any, Type, Optional
from eprllib.Agents.Triggers.BaseTrigger import BaseTrigger
from eprllib import logger

class TriggerSpec:
    """
    TriggerSpec is the base class for a trigger specification to safe configuration of the object.
    """
    def __init__(
        self,
        trigger_fn: Optional[Type[BaseTrigger]] = None,
        trigger_fn_config: Dict[str, Any] = {},
    ):
        """
        Construction method.
        
        Args:
            trigger_fn (BaseTrigger): The trigger function takes the arguments agent_id, observation and returns the
            observation filtered. See ``eprllib.Agents.Triggers`` for examples.
            
            trigger_fn_config (Dict[str, Any]): The configuration of the trigger function.
            
        Raises:
            ValueError: If the trigger function is not a subclass of BaseTrigger or if the configuration is not a dictionary.
            NotImplementedError: If the trigger function is NotImplemented.
            TypeError: If the trigger function is not a subclass of BaseTrigger.
        """
        self.trigger_fn = trigger_fn
        self.trigger_fn_config = trigger_fn_config
    
    def __getitem__(self, key:str) -> Any:
        return getattr(self, key)

    def __setitem__(self, key:str, value:Any) -> None:
        valid_keys = self.__dict__.keys()
        if key not in valid_keys:
            msg = f"Invalid key: {key}."
            logger.error(msg)
            raise KeyError(msg)
        setattr(self, key, value)
    
    def build(self) -> Dict[str, Any]:
        """
        This method is used to build the TriggerSpec object.
        """
        if self.trigger_fn == None:
            msg = "The trigger function must be defined."
            logger.error(msg)
            raise ValueError(msg)
            
        return vars(self)
