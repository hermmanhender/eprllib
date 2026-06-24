"""
Agents
=======

This module contains classes for representing and manipulating agents in the 
environment. The agents are responsible for taking actions in the environment 
following a specified policy that responses to the current state of the environment.

In this module, you will find:

- ``AgentSpec``: The main class for defining agents, including their observation, 
    filter, action, action_mapper, and reward specifications.
- ``ObservationSpec``: Defines the observation space for the agent.
- ``FilterSpec``: Defines filters to preprocess observations before they are fed to the agent.
- ``ActionSpec``: Defines the action space and actuators for the agent.
- ``ActionMapperSpec``: Defines ActionMappers that determine how the agent 
    should transform policy actions into actutators an actions.
- ``RewardSpec``: Defines the reward function for the agent.

.. note:: ``Filter`` must to be coordinated with ``Connector``.

Additionally, you will find base classes and some applications for ``Filters``, ``Rewards``, and ``ActionMappers``, 
which are essential parts of an agent in ``eprllib``.

"""

from .ActionMappers.BaseActionMapper import BaseActionMapper
from .ActionMappers.ActionMapperSpec import ActionMapperSpec
from .Filters.BaseFilter import BaseFilter
from .Filters.FilterSpec import FilterSpec
from .Rewards.BaseReward import BaseReward
from .Rewards.RewardSpec import RewardSpec


__all__ = [
    "ActionMapperSpec", "BaseActionMapper",
    "FilterSpec","BaseFilter",
    "RewardSpec", "BaseReward",
]