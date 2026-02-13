"""
Reward functions in eprllib
============================

The reward functions are called in the ``step()`` method in the environment iteratively for
each agent present in the respective timestep.

This module contains the reward functions used in reinforcement learning and 
implemented in ``eprllib``.

You can implement your own reward function by creating a new class that inherits from
:class:`~eprllib.Agents.Rewards.BaseReward` and overriding the ``get_reward`` method.

The module includes the following classes:

- :class:`~eprllib.Agents.Rewards.BaseReward`: The base class for creating reward functions.
- :class:`~eprllib.Agents.Rewards.RewardSpec`: which is used to specify the configuration of reward functions.

These classes are used in the ``EnvironmentConfig`` class to specify the rewards that can be
performed on the environment.
"""
