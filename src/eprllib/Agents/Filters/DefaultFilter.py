"""
Default Filter
===============

This module contains the default filter class for preprocessing observations before they are fed to the agent.
The ``DefaultFilter`` class extends the `BaseFilter` class and provides a basic implementation that can be used
as-is or extended to create custom filters.
"""
import numpy as np
from typing import Any, Dict
from numpy.typing import NDArray
from eprllib.Agents.Filters.BaseFilter import BaseFilter
from eprllib.Utils.annotations import override

class DefaultFilter(BaseFilter):
    """
    Default filter class for preprocessing observations.

    This class extends the `BaseFilter` class and provides a basic implementation that can be used
    as-is or extended to create custom filters. The `get_filtered_obs` method returns the agent
    states as a numpy array of float64 values.
    """
    @override(BaseFilter)
    def setup(self):
        """
        Sets up the components of the module.

        This is called automatically during the __init__ method of this class.
        """
        pass
        
    @override(BaseFilter)
    def _get_filtered_obs(
        self,
        env_config: Dict[str, Any],
        agent_states: Dict[str, Any],
    ) -> NDArray[np.float64]:
        """
        Returns the filtered observations for the agent based on the environment configuration
        and agent states. This method processes the raw observations according to the filter
        configuration specified in filter_fn_config.

        Args:
            env_config (Dict[str, Any]): Configuration dictionary for the environment. Can include 
            settings that affect how the observations are filtered.
            
            agent_states (Dict[str, Any], optional): Dictionary containing the states of the agent.

        Returns:
            NDarray: Filtered observations as a numpy array of float64 values.
            
        Raises:
            TypeError: If agent_states is not a dictionary.
            ValueError: If agent_states is empty or contains non-numeric values.
        """
        return np.array(list(agent_states.values()), dtype='float64')
    