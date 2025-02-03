Key Concepts
=============

Work in progres...[]

``eprllib`` serves the environment to use later on RLlib. To conigurate the environment the class ``EnvConfig`` is
used. Two methods in this class allow the correct configuration of the general parameters of a EnergyPlus model
and the agents specifics variables: observations, actions, rewards, filters, and triggers.

Aditionaly, four classes are stablished to provide flexibility in the environment configuration:

    * ``Rewards API``
    * ``Triggers API``
    * ``Filters API``
    * ``AgentsConnector API``
    * ``Episodes API``

Work in progres...

.. image:: Images/general_overview.png
    :width: 600

