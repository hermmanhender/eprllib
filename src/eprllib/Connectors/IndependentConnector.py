"""
Independent Agents Connector
=============================

This module implements the default observation function where each agent has its own observation space 
and it is returned without modifications, considering only the agent_states provided in the BaseRunner class.
"""
from eprllib.Connectors.DefaultConnector import DefaultConnector
from eprllib import logger
from eprllib.Utils.annotations import override

class IndependentConnector(DefaultConnector):
    """
    Connector for Independent Agents.
    """
    @override(DefaultConnector)
    def setup(self) -> None:
        """
        This method can be overridden in subclasses to perform setup tasks.
        """
        logger.info(f"IndependentConnector: The IndependentConnector was correctly inicializated with {self.connector_fn_config} config.")
        pass
    