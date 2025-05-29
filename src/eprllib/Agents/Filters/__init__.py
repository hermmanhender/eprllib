"""
Filters
========

This module contains classes for defining and applying filters to preprocess observations before they are fed to the agent.
Filters are essential for modifying the raw observations from the environment to a format that is suitable for the agent's
policy.

In this module, you will find:

- ``BaseFilter``: The base class for defining filter functions. It provides the basic structure and methods that can be 
extended to create custom filters.
- ``DefaultFilter``: A default implementation of the ``BaseFilter`` class that can be used as-is or extended to create 
custom filters.
- ``FullySharedParametersFilter``: A filter class for the fully-shared-parameters policy, which builds the observation 
vector for the agent by removing the actuator state from the agent state vector.

These filters can be used to preprocess observations in various ways, depending on the specific requirements of the 
agent's policy.
"""