Agents API
===========

The agents in eprllib are defined by its elementary componentes:

1. The capability of **observation**.
2. The **action** or control habilities.
3. And the **reward**, compensation or penalty by its actions given an observation.

.. image:: Images/agents.png
    :width: 600

The image shows the main paramenters inside an agent. All of them must be defined. The ``AgentSpec`` class
help to construct the agents. To use it just import the class with:

.. code-block:: python

    from eprllib.Agent.AgentSpec import AgentSpec

For example, for a simple thermostat definition that take the Zone Mean Air Temperature and put the HVAC on or off:

.. code-block:: python

    from eprllib.Agent.AgentSpec import (
        AgentSpec,
        ActionSpec,
        ObservationSpec,
        RewardSpec
    )
    from eprllib.ActionFunctions.setpoint_control import availability
    from eprllib.RewardFunctions.energy_and_cen15251 import reward_fn

    HVAC_agent = AgentSpec(
        observation = ObservationSpec(
            variables = [
                ("Zone Mean Air Temperature", "Thermal Zone")
            ]
        ),
        action = ActionSpec(
            actuators = [
                ("Schedule:Constant", "Schedule Value", "HVAC_availability"),
            ],
            action_fn = SetpointAgentActions,
            action_fn_config = {
                'agent_name': 'HVAC_agent',
                'availability_actuator': ("Schedule:Constant", "Schedule Value", "HVAC_availability")
            }
        ),
        reward = RewardSpec(
            reward_fn = reward_fn,
            reward_fn_config = {
                "agent_name": "HVAC_agent",
                "thermal_zone": "Thermal Zone",
                "beta": 0.1,
                'people_name': "Room1 Occupancy",
                'cooling_name': "Cooling:DistrictCooling",
                'heating_name': "Heating:DistrictHeatingWater",
                'cooling_energy_ref': 1500000,
                'heating_energy_ref': 1500000,
        )
    )

The agent defined as before is called inside the method ``EnvConfig.agents()`` in the argument 
``agents_config`` as the value for the dictionary.

.. code-block:: python

    from eprllib.Env.EnvConfig import EnvConfig

    EnvironmentConfig = EnvConfig()
    EnvironmentConfig.agents(
        agents_config = {
            'HVAC_agent': HVAC_agent,
        }
    )

See the sections of Actions, Observations and Rewards to learn how to configurate each of them.

AgentsConnector API
--------------------

A ``AgentsConnector`` API is provided to allow agents cooperate or acting in differents ways, like in a hierarchy or 
in a cooperative manner.

Work in progres...