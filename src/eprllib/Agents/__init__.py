"""
Agents
=======

This module contains classes for representing and manipulating agents in the
environment. The agents are responsible for taking actions in the environment
following a specified policy.

In this module, you will find:

- ``AgentSpec``: The main class for defining agents, including their observation, 
  filter, action, trigger, and reward specifications.
- ``ObservationSpec``: Defines the observation space for the agent.
- ``FilterSpec``: Defines filters to preprocess observations before they are fed to 
  the agent.
- ``ActionSpec``: Defines the action space and actuators for the agent.
- ``TriggerSpec``: Defines triggers that determine when the agent should take an action.
- ``RewardSpec``: Defines the reward function for the agent.

Additionally, you will find base classes and some applications for Filters, Rewards, 
and Triggers, which are essential parts of an agent in eprllib.
"""