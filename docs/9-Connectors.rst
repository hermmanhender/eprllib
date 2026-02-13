Connectors API
==============

Introduction
------------

In eprllib, **Connectors** define how agents interact with the environment and with each other. They provide a flexible mechanism to implement various interaction patterns, such as cooperation, competition, or hierarchical control. This document provides a detailed explanation of the AgentConnectors API in eprllib.

.. image:: Images/connectors.png
    :width: 600
    :alt: Connectors diagram
    :align: center
    :figclass: align-center
    :caption: Connectors diagram.

Connector: Defining Agent Interactions
--------------------------------------

The ``Connector`` API is used to define how agents interact with the environment and with each other. It is specified in the ``EnvironmentConfig`` class using the ``agents()`` method.

*   ``connector_fn``: A function that defines how agents interact with the environment.
*   ``connector_fn_config``: A dictionary of parameters that will be passed to the connector function.

Connector Functions (connector_fn)
----------------------------------

Connector functions are responsible for managing the flow of information between agents and the environment. They take into account the current state of the environment, the agent's observations, and the agent's actions to manage the interaction.

*   **DefaultConnector:**

    The ``DefaultConnector`` is a built-in connector function that is provided as a standard option. It implements a simple interaction pattern where:

    *   Each agent receives its own observations from the environment.
    *   Each agent sends its actions to the environment.
    *   Each agent receives its own reward from the environment.

*   **Creating Custom Connector Functions:**

    You can create custom connector functions to implement more complex interaction patterns. A connector function should:

    *   Take the current state of the environment, the agent's observations, and the agent's actions as input.
    *   Manage the flow of information between agents and the environment.
    *   Return the next observations, rewards, done flags, and info dictionaries for each agent.

Connector Function Configuration (connector_fn_config)
------------------------------------------------------

Connector functions can be configured using the ``connector_fn_config`` parameter in the ``agents()`` method of ``EnvironmentConfig``. This allows you to customize the behavior of the connector function without modifying its code.

*   **Configuring DefaultConnector:**

    The ``DefaultConnector`` does not have any configuration parameters.

*   **Defining Custom Configuration Parameters:**

    When creating custom connector functions, you can define your own configuration parameters to control their behavior.

Integrating Connectors with EnvironmentConfig
---------------------------------------------

Once you have defined your connector function, you need to integrate it into the environment configuration using the ``EnvironmentConfig`` class. The ``agents()`` method of ``EnvironmentConfig`` is used to specify the connector function and its configuration.

.. code-block:: python

    from eprllib.Environment.EnvironmentConfig import EnvironmentConfig
    from eprllib.Connectors.DefaultConnector import DefaultConnector
    from eprllib.Agents.AgentSpec import AgentSpec, ObservationSpec, ActionSpec, RewardSpec, FilterSpec, TriggerSpec
    from eprllib.Agents.Filters.DefaultFilter import DefaultFilter
    from eprllib.Agents.ActionMappers.SetpointActionMappers import DualSetpointDiscreteAndAvailabilityActionMapper

    # Create the EnvironmentConfig object
    env_config = EnvironmentConfig()

    # Integrate the connector into the environment configuration
    env_config.agents(
        connector_fn=DefaultConnector,
        connector_fn_config={},
        agents_config={
            "HVAC": AgentSpec(
                observation=ObservationSpec(
                    variables=[
                        ("Site Outdoor Air Drybulb Temperature", "Environment"),
                        ("Zone Mean Air Temperature", "Thermal Zone"),
                    ],
                    meters=[
                        "Electricity:Building",
                    ],
                ),
                action=ActionSpec(
                    actuators=[
                        ("Schedule:Compact", "Schedule Value", "heating_setpoint"),
                        ("Schedule:Compact", "Schedule Value", "cooling_setpoint"),
                        ("Schedule:Constant", "Schedule Value", "HVAC_OnOff"),
                    ],
                ),
                filter=FilterSpec(
                    filter_fn=DefaultFilter,
                    filter_fn_config={},
                ),
                action_mapper=ActionMapperSpec(
                    action_mapper=DualSetpointDiscreteAndAvailabilityActionMapper,
                    action_mapper_config={
                        'temperature_range': (18, 28),
                        'actuator_for_cooling': ("Schedule:Compact", "Schedule Value", "cooling_setpoint"),
                        'actuator_for_heating': ("Schedule:Compact", "Schedule Value", "heating_setpoint"),
                        'availability_actuator': ("Schedule:Constant", "Schedule Value", "HVAC_OnOff"),
                    },
                ),
                reward=RewardSpec(
                    reward_fn=lambda agent_name, thermal_zone, beta, people_name, cooling_name, heating_name, cooling_energy_ref, heating_energy_ref, **kwargs: 0,
                    reward_fn_config={
                        "thermal_zone": "Thermal Zone",
                        "beta": 0.001,
                        'people_name': "People",
                        'cooling_name': "Cooling:DistrictCooling",
                        'heating_name': "Heating:DistrictHeatingWater",
                        'cooling_energy_ref': None,
                        'heating_energy_ref': None,
                    },
                ),
            ),
        }
    )

Integration with AgentSpec
--------------------------

Connectors interact with the agents defined by ``AgentSpec``. They are responsible for:

*   **Managing Observations:** Providing the correct observations to each agent based on its ``ObservationSpec``.
*   **Managing Actions:** Receiving actions from each agent and applying them to the environment.
*   **Managing Rewards:** Providing the correct rewards to each agent based on its ``RewardSpec``.

Interaction Patterns
--------------------

Connectors allow you to implement various interaction patterns between agents. Here are some examples:

*   **Cooperative Agents:**

    In a cooperative setting, agents work together to achieve a common goal. A custom connector can be used to:

    *   Share observations between agents.
    *   Combine actions from multiple agents before applying them to the environment.
    *   Distribute rewards among agents based on their contributions.

*   **Hierarchical Agents:**

    In a hierarchical setting, some agents control other agents. A custom connector can be used to:

    *   Receive high-level commands from a master agent.
    *   Distribute these commands to subordinate agents.
    *   Aggregate information from subordinate agents and provide it to the master agent.

*   **Competitive Agents:**

    In a competitive setting, agents compete against each other. A custom connector can be used to:

    *   Manage the interactions between competing agents.
    *   Calculate rewards based on the relative performance of the agents.

*   **Custom Patterns:**

    You can create custom connectors to implement any interaction pattern you need.

Examples
--------

Here's a complete example of how to define and use the ``DefaultConnector``:

.. code-block:: python

    from eprllib.Environment.EnvironmentConfig import EnvironmentConfig
    from eprllib.Connectors.DefaultConnector import DefaultConnector
    from eprllib.Agents.AgentSpec import AgentSpec, ObservationSpec, ActionSpec, RewardSpec, FilterSpec, TriggerSpec
    from eprllib.Agents.Filters.DefaultFilter import DefaultFilter
    from eprllib.Agents.ActionMappers.SetpointActionMappers import DualSetpointDiscreteAndAvailabilityActionMapper

    # Create the EnvironmentConfig object
    env_config = EnvironmentConfig()

    # Integrate the connector into the environment configuration
    env_config.agents(
        connector_fn=DefaultConnector,
        connector_fn_config={},
        agents_config={
            "HVAC": AgentSpec(
                observation=ObservationSpec(
                    variables=[
                        ("Site Outdoor Air Drybulb Temperature", "Environment"),
                        ("Zone Mean Air Temperature", "Thermal Zone"),
                    ],
                    meters=[
                        "Electricity:Building",
                    ],
                ),
                action=ActionSpec(
                    actuators=[
                        ("Schedule:Compact", "Schedule Value", "heating_setpoint"),
                        ("Schedule:Compact", "Schedule Value", "cooling_setpoint"),
                        ("Schedule:Constant", "Schedule Value", "HVAC_OnOff"),
                    ],
                ),
                filter=FilterSpec(
                    filter_fn=DefaultFilter,
                    filter_fn_config={},
                ),
                action_mapper=ActionMapperSpec(
                    action_mapper=DualSetpointDiscreteAndAvailabilityActionMapper,
                    action_mapper_config={
                        'temperature_range': (18, 28),
                        'actuator_for_cooling': ("Schedule:Compact", "Schedule Value", "cooling_setpoint"),
                        'actuator_for_heating': ("Schedule:Compact", "Schedule Value", "heating_setpoint"),
                        'availability_actuator': ("Schedule:Constant", "Schedule Value", "HVAC_OnOff"),
                    },
                ),
                reward=RewardSpec(
                    reward_fn=lambda agent_name, thermal_zone, beta, people_name, cooling_name, heating_name, cooling_energy_ref, heating_energy_ref, **kwargs: 0,
                    reward_fn_config={
                        "thermal_zone": "Thermal Zone",
                        "beta": 0.001,
                        'people_name': "People",
                        'cooling_name': "Cooling:DistrictCooling",
                        'heating_name': "Heating:DistrictHeatingWater",
                        'cooling_energy_ref': None,
                        'heating_energy_ref': None,
                    },
                ),
            ),
        }
    )

By understanding these concepts, you'll be able to effectively define and use connectors in eprllib for your building energy optimization and control projects.
