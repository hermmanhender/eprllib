"""
Environments
============

The environments in RLlib can be implemented as single-agent, multi-agent, or
external environments. In ``eprllib``, we use a multi-agent approach, allowing multiple 
agents to run in the environment, but also supporting a single agent if only one is specified.

The standard configuration uses a policy with fully shared parameters, but future 
versions aim to add flexibility to the policy.

You can configure the environment with the :class:`~eprllib.Environment.EnvironmentConfig.EnvironmentConfig` class.

The module includes the following classes and functions:

- ``Environment``: The base class for creating multi-agent environments.
- ``EnvironmentRunner``: The base class for running EnergyPlus simulations.
- ``EnvironmentConfig``: The class used to configure the environment.
"""

TIMEOUT = 10.0
CUT_EPISODE_LEN = 0