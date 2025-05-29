"""
Reward functions in eprllib
============================

The reward functions are called in the ``step()`` method in the environment iteratively for
each agent present in the respective timestep.

This module contains the reward functions used in reinforcement learning and 
implemented in eprllib.

You can implement your own reward function by creating a new class that inherits from
:class:`~eprllib.Agents.Rewards.BaseReward` and overriding the ``get_reward`` method.

The module includes the following classes:

- :class:`~eprllib.Agents.Rewards.BaseReward`: The base class for creating reward functions.
- :class:`~eprllib.Agents.Rewards.ASHRAE55SimpleModel`: Contains classes to calculate rewards based on ASHRAE-55 comfort model.
- :class:`~eprllib.Agents.Rewards.CEN15251`: Contains classes to calculate rewards based on CEN-15251 comfort model.
- :class:`~eprllib.Agents.Rewards.EnergyRewards`: Contains classes to calculate rewards based on energy.

Also, it is possible to combine the previous rewards methods for multi-objective reward funtions that
are normaly used in building control (e.g. use comfort and energy optimization). Implementations in 
eprllib are:

- :class:`~eprllib.Agents.Rewards.EnergyAndAshrae55SimpleModel`
- :class:`~eprllib.Agents.Rewards.EnergyAndCEN15251`

These classes are used to provide feedback to the agents, guiding them to optimize their policies based on 
the defined reward criteria.
"""
