"""
Specification for the trigger functions
===========================================================
This module defines the `TriggerSpec` class, which is used to specify the configuration of trigger 
functions for agents in reinforcement learning environments.
"""
from typing import Dict, Any
from eprllib.Agents.Triggers.BaseTrigger import BaseTrigger
from eprllib import logger

class TriggerSpec:
    """
    TriggerSpec is the base class for a trigger specification to safe configuration of the object.
    """
    def __init__(
        self,
        trigger_fn: BaseTrigger = NotImplemented,
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
    
    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        valid_keys = self.__dict__.keys()
        if key not in valid_keys:
            msg = f"Invalid key: {key}."
            logger.error(msg)
            raise KeyError(msg)
        setattr(self, key, value)
    
    def build(self) -> Dict:
        """
        This method is used to build the TriggerSpec object.
        """
        if self.trigger_fn == NotImplemented:
            msg = "The trigger function must be defined."
            logger.error(msg)
            raise ValueError(msg)
        
        if not issubclass(self.trigger_fn, BaseTrigger):
            msg = f"The trigger function must be based on BaseTrigger class but {type(self.trigger_fn)} was given."
            logger.error(msg)
            raise ValueError(msg)

        if not isinstance(self.trigger_fn_config, dict):
            msg = f"The configuration for the trigger function must be a dictionary but {type(self.trigger_fn_config)} was given."
            logger.error(msg)
            raise ValueError(msg)
            
        return vars(self)
