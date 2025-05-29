"""
Specification for the filter functions
===========================================
This module defines the `FilterSpec` class, which is used to specify the configuration of filter functions for agents in reinforcement learning environments.
It ensures that the filter function is properly defined and adheres to the expected interface.
"""
import logging
from typing import Dict, Any
from eprllib.Agents.Filters.BaseFilter import BaseFilter
from eprllib.Agents.Filters.DefaultFilter import DefaultFilter

logger = logging.getLogger("ray.rllib")

class FilterSpec:
    """
    FilterSpec is the base class for a filter specification to safe configuration of the object.
    """
    def __init__(
        self,
        filter_fn: BaseFilter = None,
        filter_fn_config: Dict[str, Any] = {}
    ):
        """
        Construction method.
        
        Args:
            filter_fn (BaseFilter): The filter function takes the arguments agent_id, observation and returns the
            observation filtered. See ``eprllib.Agents.Filters`` for examples.
            
            filter_fn_config (Dict[str, Any]): The configuration of the filter function.
        """
        self.filter_fn = filter_fn
        self.filter_fn_config = filter_fn_config            
    
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
        This method is used to build the FilterSpec object.
        """
        if self.filter_fn is None:
            msg = "No filter function provided. Using DefaultFilter."
            logger.warning(msg)
            self.filter_fn = DefaultFilter
            self.filter_fn_config = {}
            
        if not issubclass(self.filter_fn, BaseFilter):
            msg = f"The filter function must be based on BaseFilter class but {type(self.filter_fn)} was given."
            logger.error(msg)
            raise ValueError(msg)

        if not isinstance(self.filter_fn_config, dict):
            msg = f"The configuration for the filter function must be a dictionary but {type(self.filter_fn_config)} was given."
            logger.error(msg)
            raise ValueError(msg)
            
        return vars(self)
    