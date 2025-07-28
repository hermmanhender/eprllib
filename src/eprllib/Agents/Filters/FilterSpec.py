"""
Specification for the filter functions
===========================================
This module defines the `FilterSpec` class, which is used to specify the configuration of filter functions for agents in reinforcement learning environments.
It ensures that the filter function is properly defined and adheres to the expected interface.
"""
from typing import Dict, Any, Optional # type: ignore
from eprllib.Agents.Filters.BaseFilter import BaseFilter
from eprllib.Agents.Filters.DefaultFilter import DefaultFilter
from eprllib import logger

class FilterSpec:
    """
    FilterSpec is the base class for a filter specification to safe configuration of the object.
    """
    def __init__(
        self,
        filter_fn: Optional[BaseFilter] = None,
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
    
    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any) -> None:
        valid_keys = self.__dict__.keys()
        if key not in valid_keys:
            msg = f"Invalid key: {key}."
            logger.error(msg)
            raise KeyError(msg)
        setattr(self, key, value)
        
    def build(self) -> Dict[str, Any]:
        """
        This method is used to build the FilterSpec object.
        """
        if self.filter_fn is None:
            msg = "No filter function provided. Using DefaultFilter."
            logger.warning(msg)
            self.filter_fn = DefaultFilter
            self.filter_fn_config = {}
        return vars(self)
    