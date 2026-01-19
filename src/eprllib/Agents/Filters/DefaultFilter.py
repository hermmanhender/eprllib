"""
Default Filter
===============

This module contains the default filter class for preprocessing observations before they are fed to the agent.
The `DefaultFilter` class extends the `BaseFilter` class and provides a basic implementation that can be used
as-is or extended to create custom filters.
"""
import numpy as np
from typing import Any, Dict
from numpy.typing import NDArray
from eprllib.Agents.Filters.BaseFilter import BaseFilter
from eprllib.Utils.annotations import override
from eprllib import logger

class DefaultFilter(BaseFilter):
    """
    Default filter class for preprocessing observations.

    This class extends the `BaseFilter` class and provides a basic implementation that can be used
    as-is or extended to create custom filters. The `get_filtered_obs` method returns the agent
    states as a numpy array of float32 values.
    """
    def __init__(
        self,
        filter_fn_config: Dict[str, Any]
    ):
        """
        Initializes the DefaultFilter class.

        Args:
            filter_fn_config (Dict[str, Any]): Configuration dictionary for the filter function.
            This configuration can include settings that affect how the observations are filtered.
        """
        super().__init__(filter_fn_config)
        
    @override(BaseFilter)
    def get_filtered_obs(
        self,
        env_config: Dict[str, Any],
        agent_states: Dict[str, Any],
    ) -> NDArray[np.float32]:
        """
        Returns the filtered observations for the agent based on the environment configuration
        and agent states. This method processes the raw observations according to the filter
        configuration specified in filter_fn_config.

        Args:
            env_config (Dict[str, Any]): Configuration dictionary for the environment. Can include 
            settings that affect how the observations are filtered.
            
            agent_states (Dict[str, Any], optional): Dictionary containing the states of the agent.

        Returns:
            NDarray: Filtered observations as a numpy array of float32 values.
            
        Raises:
            TypeError: If agent_states is not a dictionary.
            ValueError: If agent_states is empty or contains non-numeric values.
        """
        # Check if the agent_states dictionary is empty
        if not agent_states:
            msg = "agent_states dictionary is empty"
            logger.error(msg)
            raise ValueError(msg)
        
        # Check if all values in the agent_states dictionary are numeric
        if not all(isinstance(value, (int, float)) for value in agent_states.values()):
            msg = "All values in agent_states must be numeric"
            logger.error(msg)
            raise ValueError(msg)
        
        return np.array(list(agent_states.values()), dtype='float32')
    