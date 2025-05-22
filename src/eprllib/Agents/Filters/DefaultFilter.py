"""
Default Filter
===============

This module contains the default filter class for preprocessing observations before they are fed to the agent.
The `DefaultFilter` class extends the `BaseFilter` class and provides a basic implementation that can be used
as-is or extended to create custom filters.
"""
import numpy as np
from typing import Any, Dict
from eprllib.Agents.Filters.BaseFilter import BaseFilter
from eprllib.Utils.annotations import override

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
    ) -> np.ndarray:
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
        """
        return np.array(list(agent_states.values()), dtype='float32')