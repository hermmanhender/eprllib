"""
Connectors
==================

This module contains connectors for agents, which are responsible for combining and transforming
agents' observations to provide flexible configurations for communication between agents.

The module includes the following classes:

- ``BaseConnector``: The base class for creating connector functions.

These connectors are used to provide different configurations for multi-agent environments, enabling various
communication and observation strategies.
"""

from eprllib.Connectors.BaseConnector import BaseConnector
from eprllib.Connectors.DefaultConnector import DefaultConnector

__all__ = ["BaseConnector", "DefaultConnector"]