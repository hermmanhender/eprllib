"""
Base Filter
============

This module contains the base class for defining filter functions used in agent specifications.
Filters are used to preprocess observations before they are fed to the agent. The `BaseFilter`
class provides the basic structure and methods that can be extended to create custom filters.
"""

from typing import Any, Dict
import numpy as np
import gymnasium as gym
from eprllib.AgentsConnectors.BaseConnector import BaseConnector

class BaseFilter:
    def __init__(
        self,
        filter_fn_config: Dict[str, Any]
    ):
        """
        Base class for defining a filter function configuration.

        Args:
            filter_fn_config (Dict[str, Any]): Configuration dictionary for the filter function.
        """
        self.filter_fn_config = filter_fn_config
        
    def set_agent_obs(
        self,
        env_config: Dict[str, Any],
        agent_states: Dict[str, Any] = NotImplemented,
    ) -> Dict[str, Any]:
        """
        Sets the observation for the agent.

        Args:
            env_config (Dict[str, Any]): Configuration dictionary for the environment.
            agent_states (Dict[str, Any], optional): Dictionary containing the states of the agent. Defaults to NotImplemented.

        Returns:
            Dict[str, Any]: Dictionary containing the observations for the agent.
        """
        return np.array(list(agent_states.values()), dtype='float32')