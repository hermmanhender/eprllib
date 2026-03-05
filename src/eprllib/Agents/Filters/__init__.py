"""
Filters
========

This module contains classes for defining and applying filters to preprocess observations before they are fed to the agent.
Filters are essential for modifying the raw observations from the environment to a format that is suitable for the agent's
policy.

In this module, you will find:

- ``BaseFilter``: The base class for defining filter functions. It provides the basic structure and methods that can be 
extended to create custom filters.
- ``FilterSpec``: A class used to specify the configuration of filter functions in the agent definition. It ensures that 
the filter function is properly defined and adheres to the expected interface.

These filters can be used to preprocess observations in various ways, depending on the specific requirements of the 
agent's policy.
"""