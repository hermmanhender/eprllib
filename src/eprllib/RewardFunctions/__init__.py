"""
Reward functions in eprllib
============================

The reward functions are called in the ``step()`` method in the environment iteratively for
each agent present in the respective timestep.

In this module contain the reward functions used in the reinforcement learning and 
implemented in eprllib.

You can implement your own reward function by creating a new class that inherits from
:class:`~eprllib.RewardFunctions.RewardFunction` and overriding the ``calculate_reward`` method.
"""
