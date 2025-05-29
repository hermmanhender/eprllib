Filters API
===========

Introduction
------------

In eprllib, **Filters** are used to process an agent's observations before they are used for decision-making. They provide a mechanism to modify or transform the raw observations, allowing for more robust and efficient agent behavior. This document provides a detailed explanation of the Filters API in eprllib.

.. image:: Images/filters.png
    :width: 600
    :alt: Filters diagram
    :align: center
    :figclass: align-center
    :caption: Filters diagram.

FilterSpec: Defining Observation Filters
----------------------------------------

The ``FilterSpec`` class is used to define how an agent's observations are filtered. It specifies the filter function and its configuration. It allows you to define:

*   ``filter_fn``: A function that filters the observations.
*   ``filter_fn_config``: A dictionary of parameters that will be passed to the filter function.

.. code-block:: python

    from eprllib.Agents.AgentSpec import FilterSpec
    from eprllib.Agents.Filters.DefaultFilter import DefaultFilter

    filter_spec = FilterSpec(
        filter_fn=DefaultFilter,
        filter_fn_config={},
    )

Filter Functions (filter_fn)
----------------------------

Filter functions are responsible for modifying the agent's observations. They take the raw observations as input and return the filtered observations as output.

*   **DefaultFilter:**

    The ``DefaultFilter`` is a built-in filter function that is provided as a standard option. It does not modify the observations. It is used when no filtering is required.

*   **Creating Custom Filter Functions:**

    You can create custom filter functions to implement more complex filtering logic. A filter function should:

    *   Take the raw observations as input.
    *   Return the filtered observations as output.

Filter Function Configuration (filter_fn_config)
------------------------------------------------

Filter functions can be configured using the ``filter_fn_config`` parameter in ``FilterSpec``. This allows you to customize the behavior of the filter function without modifying its code.

*   **Configuring DefaultFilter:**

    The ``DefaultFilter`` does not have any configuration parameters.

*   **Defining Custom Configuration Parameters:**

    When creating custom filter functions, you can define your own configuration parameters to control their behavior.

Integrating Filters with AgentSpec
----------------------------------

Once you have defined your filter using ``FilterSpec``, you need to integrate it into the agent definition using the ``AgentSpec`` class. The ``filter`` parameter of ``AgentSpec`` is used to specify the filter for the agent.

.. code-block:: python

    from eprllib.Agents.AgentSpec import AgentSpec, FilterSpec
    from eprllib.Agents.Filters.DefaultFilter import DefaultFilter

    # Define the filter
    filter_spec = FilterSpec(
        filter_fn=DefaultFilter,
        filter_fn_config={},
    )

    # Define the agent and integrate the filter
    agent_spec = AgentSpec(
        # ... other agent parameters ...
        filter=filter_spec,
    )

Relationship with ObservationSpec
---------------------------------

Filters operate on the observations defined in ``ObservationSpec``. The order of operations is as follows:

1.  **Observations are gathered:** The agent's observations are collected from the environment based on the configuration in ``ObservationSpec``.
2.  **Observations are filtered:** The raw observations are passed to the filter function defined in ``FilterSpec``.
3.  **Filtered observations are used:** The filtered observations are then used by the agent for decision-making.

Examples
--------

Here's a complete example of how to define and use filters:

.. code-block:: python

    from eprllib.Agents.AgentSpec import AgentSpec, ObservationSpec, ActionSpec, RewardSpec, FilterSpec, TriggerSpec
    from eprllib.Agents.Filters.DefaultFilter import DefaultFilter
    from eprllib.Agents.Triggers.SetpointTriggers import DualSetpointTriggerDiscreteAndAvailabilityTrigger

    # Define the filter
    filter_spec = FilterSpec(
        filter_fn=DefaultFilter,
        filter_fn_config={},
    )

    # Define the agent
    agent_spec = AgentSpec(
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
        filter=filter_spec,
        trigger=TriggerSpec(
            trigger_fn=DualSetpointTriggerDiscreteAndAvailabilityTrigger,
            trigger_fn_config={
                "agent_name": "HVAC",
                'temperature_range': (18, 28),
                'actuator_for_cooling': ("Schedule:Compact", "Schedule Value", "cooling_setpoint"),
                'actuator_for_heating': ("Schedule:Compact", "Schedule Value", "heating_setpoint"),
                'availability_actuator': ("Schedule:Constant", "Schedule Value", "HVAC_OnOff"),
            },
        ),
        reward=RewardSpec(
            reward_fn=lambda agent_name, thermal_zone, beta, people_name, cooling_name, heating_name, cooling_energy_ref, heating_energy_ref, **kwargs: 0,
            reward_fn_config={
                "agent_name": "HVAC",
                "thermal_zone": "Thermal Zone",
                "beta": 0.001,
                'people_name': "People",
                'cooling_name': "Cooling:DistrictCooling",
                'heating_name': "Heating:DistrictHeatingWater",
                'cooling_energy_ref': None,
                'heating_energy_ref': None,
            },
        ),
    )

By understanding these concepts, you'll be able to effectively define and use filters in eprllib for your building energy optimization and control projects.
