"""
Agents connectors
==================

This module contains connectors for agents, which are responsible for combining and transforming 
agents' observations to provide flexible configurations for communication between agents. 

The module includes the following classes:

- BaseConnector: The base class for creating connector functions.
- DefaultConnector: Implements the default connector class that allows the combination of agents' observations.
- CentralizedConnector: Implements a centralized connector where a central agent takes the observations of all agents.
- FullySharedParametersConnector: Implements a fully shared parameters policy for the observation function.
- HierarchicalConnector: Implements a hierarchical connector with two levels of hierarchy.
- IndependentConnector: Implements the default observation function where each agent has its own observation space.

These connectors are used to provide different configurations for multi-agent environments, enabling various 
communication and observation strategies.
"""