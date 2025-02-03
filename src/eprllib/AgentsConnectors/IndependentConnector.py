"""
Independent Agents Connector
=============================

This module implements the default observation function where each agent has its own observation space 
and it is returned without modifications, considering only the agent_states provided in the BaseRunner class.
"""

from typing import Any, Dict
from eprllib.AgentsConnectors.DefaultConnector import DefaultConnector

class IndependentConnector(DefaultConnector):
    def __init__(
        self,
        connector_fn_config: Dict[str, Any] = {}
    ):
        """
        Initializes the independent connector.

        Args:
            connector_fn_config (Dict[str, Any]): Configuration dictionary for the multi-agent function.
        """
        super().__init__(connector_fn_config)