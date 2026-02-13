ActionMapper API
================

Introduction
------------

In eprllib, ``ActionMappers`` determine how the policy actions are executed into EnergyPlus actuators. They 
provide a mechanism to control the timing of actions, allowing for more complex and nuanced agent behavior. This 
document provides a detailed explanation of the Triggers API in eprllib.

.. image:: Images/triggers.png
    :width: 600
    :alt: Triggers diagram
    :align: center

ActionMapperSpec: Defining ActionMappers
----------------------------------------

The ``ActionMapperSpec`` class is used to define how an agent's actions are transform and adapted to actuators. 
It specifies the ActionMapper function and its configuration. It allows you to define:

*   ``action_mapper_fn``: An ActionMapper function.
*   ``action_mapper_config``: A dictionary of parameters that will be passed to the ActionMapper function.

.. code-block:: python

    from eprllib.Agents.ActionMappers.ActionMapperSpec import ActionMapperSpec
    from eprllib.Agents.ActionMappers.SetpointActionMappers import DualSetpointDiscreteAndAvailabilityActionMapper

    action_mapper = ActionMapperSpec(
        action_mapper_fn=DualSetpointDiscreteAndAvailabilityActionMapper,
        action_mapper_config={
            'temperature_range': (18, 28),
            'actuator_for_cooling': ("Schedule:Compact", "Schedule Value", "cooling_setpoint"),
            'actuator_for_heating': ("Schedule:Compact", "Schedule Value", "heating_setpoint"),
            'availability_actuator': ("Schedule:Constant", "Schedule Value", "HVAC_OnOff"),
        },
    )

ActionMapper Functions (action_mapper)
--------------------------------------

ActionMapper functions are responsible for determining when an agent's actions should be executed. They take into account the current state of the environment and the agent's observations to make this decision.

*   **DualSetpointDiscreteAndAvailabilityActionMapper:**

    The ``DualSetpointDiscreteAndAvailabilityActionMapper`` is a built-in trigger function that is commonly used for HVAC control. It triggers actions based on:

    *   **Temperature Range:** The desired temperature range for the zone.
    *   **Cooling Actuator:** The actuator used to control cooling.
    *   **Heating Actuator:** The actuator used to control heating.
    *   **Availability Actuator:** The actuator used to control the availability of the HVAC system.

*   **Creating Custom ActionMapper Functions:**

    You can create custom trigger functions to implement more complex triggering logic. A trigger function should:

    *   Take the current state of the environment and the agent's observations as input.
    *   Return a boolean value indicating whether the action should be triggered.

ActionMapper Function Configuration (action_mapper_config)
----------------------------------------------------------

ActionMapper functions can be configured using the ``action_mapper_config`` parameter in ``ActionMapperSpec``. 
This allows you to customize the behavior of the ActionMapper function without modifying its code.

*   **Configuring DualSetpointDiscreteAndAvailabilityActionMapper:**

    The ``DualSetpointDiscreteAndAvailabilityActionMapper`` can be configured with the following parameters:

    *   ``temperature_range``: A tuple defining the desired temperature range (min, max).
    *   ``actuator_for_cooling``: The actuator used to control cooling.
    *   ``actuator_for_heating``: The actuator used to control heating.
    *   ``availability_actuator``: The actuator used to control the availability of the HVAC system.

*   **Defining Custom Configuration Parameters:**

    When creating custom trigger functions, you can define your own configuration parameters to control their behavior.

Integrating ActionMapper with AgentSpec
---------------------------------------

Once you have defined your trigger using ``ActionMapperSpec``, you need to integrate it into the agent definition using 
the ``AgentSpec`` class. The ``action_mapper`` parameter of ``AgentSpec`` is used to specify the ``ActionMapper`` for the agent.

.. code-block:: python

    from eprllib.Agents.AgentSpec import AgentSpec
    from eprllib.Agents.ActionMappers.ActionMapperSpec import ActionMapperSpec
    from eprllib.Agents.ActionMappers.SetpointActionMappers import DualSetpointDiscreteAndAvailabilityActionMapper

    # Define the action mapper
    action_mapper_spec = ActionMapperSpec(
        action_mapper_fn=DualSetpointDiscreteAndAvailabilityActionMapper,
        action_mapper_config={
            'temperature_range': (18, 28),
            'actuator_for_cooling': ("Schedule:Compact", "Schedule Value", "cooling_setpoint"),
            'actuator_for_heating': ("Schedule:Compact", "Schedule Value", "heating_setpoint"),
            'availability_actuator': ("Schedule:Constant", "Schedule Value", "HVAC_OnOff"),
        },
    )

    # Define the agent and integrate the trigger
    agent_spec = AgentSpec(
        # ... other agent parameters ...
        action_mapper=action_mapper_spec,
    )


By understanding these concepts, you'll be able to effectively define and use triggers in eprllib for your building energy optimization and control projects.
