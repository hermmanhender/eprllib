"""
Reward functions in eprllib
============================

The reward functions are called in the ``step()`` method in the environment iteratively for
each agent present in the respective timestep.

This module contains the reward functions used in reinforcement learning and 
implemented in eprllib.

You can implement your own reward function by creating a new class that inherits from
:class:`~eprllib.Agents.Rewards.BaseReward` and overriding the ``calculate_reward`` method.

The module includes the following classes:

- BaseReward: The base class for creating reward functions.
- ComfortRewards: Contains classes to calculate rewards based on comfort.
- EnergyRewards: Contains classes to calculate rewards based on energy.
- CombinedRewards: Contains classes that combine comfort and energy methods to calculate rewards.

These classes are used to provide feedback to the agents, guiding them to optimize their policies based on the defined reward criteria.
"""
