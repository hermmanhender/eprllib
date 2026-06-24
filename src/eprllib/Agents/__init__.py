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

from eprllib.Agents.ActionMappers import ActionMapperSpec, BaseActionMapper
from eprllib.Agents.Filters import FilterSpec, BaseFilter
from eprllib.Agents.Rewards import RewardSpec, BaseReward
from eprllib.Agents import AgentSpec, ObservationSpec, ActionSpec


__all__ = [
    "ActionMapperSpec", "BaseActionMapper",
    "FilterSpec","BaseFilter",
    "RewardSpec", "BaseReward",
    "AgentSpec", "ObservationSpec", "ActionSpec"
]