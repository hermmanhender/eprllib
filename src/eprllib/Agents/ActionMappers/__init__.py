"""
ActionMappers
=========

This module defines the classes and functions used to perform actions on the
environment. These ActionMappers are responsible for transforming agent actions into
actuator commands within the environment.

The module includes the following classes:

- BaseActionMapper: The base class for creating action transformer functions.
- DualSetpointActionMapperDiscreteAndAvailabilityActionMapper: Implements the Dual Set Point Thermostat action function.
- WindowsOpeningActionMapper: Implements the window opening action function.
- WindowsShadingActionMapper: Implements the window shading action function.

These classes are used in the `EnvConfig` class to specify the actions that can be performed on the environment.
"""
