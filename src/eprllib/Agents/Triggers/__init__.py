"""
Triggers
=========

This module defines the classes and functions used to perform actions on the
environment. These triggers are responsible for transforming agent actions into
actuator commands within the environment.

The module includes the following classes:

- BaseTrigger: The base class for creating action transformer functions.
- DualSetpointTriggerDiscreteAndAvailabilityTrigger: Implements the Dual Set Point Thermostat action function.
- WindowsOpeningTrigger: Implements the window opening action function.
- WindowsShadingTrigger: Implements the window shading action function.

These classes are used in the `EnvConfig` class to specify the actions that can be performed on the environment.
"""
