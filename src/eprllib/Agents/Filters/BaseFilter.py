"""
Base Filter
============

This module contains the base class for defining filter functions used in agent specifications.
Filters are used to preprocess observations before they are fed to the agent. The `BaseFilter`
class provides the basic structure and methods that can be extended to create custom filters.

This class can not be used directly in eprllib, but as a base to create new filters. All the filters
must be based in this class.
"""
from eprllib import logger
from typing import Any, Dict
from numpy import float64
from numpy.typing import NDArray

class BaseFilter:
    """
    Base class for defining filter functions used in agent specifications.
    Filters are used to preprocess observations before they are fed to the agent.
    """
    def __init__(
        self,
        filter_fn_config: Dict[str, Any]
    ):
        """
        Initializes the BaseFilter class.

        Args:
            filter_fn_config (Dict[str, Any]): Configuration dictionary for the filter function.
        """
        self.filter_fn_config = filter_fn_config
        
    def get_filtered_obs(
        self,
        env_config: Dict[str, Any],
        agent_states: Dict[str, Any],
    ) -> NDArray[float64]:
        """
        Returns the filtered observations for the agent based on the environment configuration
        and agent states. This method processes the raw observations according to the filter
        configuration specified in filter_fn_config.

        Args:
            env_config (Dict[str, Any]): Configuration dictionary for the environment. Can include settings 
            that affect how the observations are filtered.
            
            agent_states (Dict[str, Any], optional): Dictionary containing the states of the agent.

        Returns:
            NDarray: Filtered observations as a numpy array of float64 values.
        """
        msg = "BaseFilter: This method should be implemented in a subclass."
        logger.error(msg)
        raise NotImplementedError(msg)