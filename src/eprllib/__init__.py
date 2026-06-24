"""
eprllib
========

``eprllib`` was born out of the need to bridge the gap between building modeling with
**EnergyPlus** and Deep Reinforcement Learning (**DRL**). Traditionally, integrating these two
disciplines has been complex and laborious. ``eprllib`` aims to simplify this process,
offering an intuitive and flexible interface for developing intelligent agents that
interact with building simulations.
"""
# Version management.
from .version import __version__, EP_VERSION, ep_version_list
from .Agents.ActionSpec import ActionSpec
from .Agents.AgentSpec import AgentSpec
from .Agents.ObservationSpec import ObservationSpec

from .Connectors.BaseConnector import BaseConnector
from .Connectors.DefaultConnector import DefaultConnector

from .Environment.EnvironmentConfig import EnvironmentConfig
from .Environment.EnvironmentRunner import EnvironmentRunner
from .Environment.MultiAgentEnvironment import MultiAgentEnvironment
from .Environment.SingleAgentEnvironment import SingleAgentEnvironment

from .Episodes.BaseEpisode import BaseEpisode
from .Episodes.DefaultEpisode import DefaultEpisode


__all__ = [
    "__version__",
    "ep_version_list",
    "EP_VERSION",
    "ActionSpec", "AgentSpec", "ObservationSpec",
    "BaseConnector", "DefaultConnector",
    "EnvironmentConfig", "EnvironmentRunner", "MultiAgentEnvironment", "SingleAgentEnvironment",
    "BaseEpisode", "DefaultEpisode",
    ]