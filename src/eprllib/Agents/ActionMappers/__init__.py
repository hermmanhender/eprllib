"""
ActionMappers
=========

This module defines the classes and functions used to perform actions on the
environment.

``ActionMapper`` functions are used to adapt the action given by the neural network, 
normally an integer for discrete spaces and a float for continuous spaces like Box. The actions
must be adapted to values required for the actuators in EnergyPlus. Each agent has the
capacity to control one actuator.

``ActionMapper`` must be defined in the ``EnvironmentConfig`` definition to create the environment and is
called in the ``Environment.Environment`` class and used in the ``Environment.EnvironmentRunner`` class
to transform the dict of agent actions to actuator values.

The module includes the following classes:

    - ``BaseActionMapper``: The base class for creating ``ActionMapper`` functions.
    - ``ActionMapperSpec``: which is used to specify the configuration of ``ActionMapper`` functions in the agent definition.

These classes are used in the ``EnvironmentConfig`` class to specify the actions that can 
be performed on the environment.
"""
