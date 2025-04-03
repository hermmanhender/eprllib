Key Concepts
=============

Introduction
------------

eprllib is designed to facilitate the development of Reinforcement Learning (RL) agents for building energy optimization and control. It provides a structured framework for defining environments, agents, and their interactions, while seamlessly integrating with the powerful RLlib library. This document outlines the key concepts and components that make up eprllib.

The Environment
---------------

The environment in eprllib represents the building simulation, powered by EnergyPlus. It's where the RL agents interact and learn.

*   **EnvironmentConfig:**

    The ``EnvironmentConfig`` class is the central configuration object for defining the EnergyPlus environment. It allows you to specify:

    *   **General Parameters:**
        *   Path to the EnergyPlus model file (`epjson_path`).
        *   Path to the weather file (`epw_path`).
        *   Output directory (`output_path`).
        *   Other general settings (e.g., `ep_terminal_output`, `timeout`, `evaluation`).
    *   **Agent Specifications:**
        *   Details about the agents that will interact with the environment (see the "Agents" section below).
    *   **Episode Specifications:**
        *   Details about the episodes that will interact with the environment (see the "Episodes" section below).

    .. code-block:: python

        from eprllib.Env.EnvironmentConfig import EnvironmentConfig

        env_config = EnvironmentConfig()
        env_config.generals(
            epjson_path="path/to/your/model.epJSON",
            epw_path="path/to/your/weather.epw",
            output_path="path/to/output",
            ep_terminal_output=False,
            timeout=10,
            evaluation=False,
        )

*   **Environment:**

    The ``Environment`` class is the base class for creating RL environments in eprllib. It handles the interaction with EnergyPlus and provides the necessary methods for RLlib to interact with the environment.

*   **EnergyPlus Integration:**

    eprllib communicates with EnergyPlus through its Python API. This allows eprllib to:

    *   Read sensor data from the EnergyPlus simulation.
    *   Set actuator values in the EnergyPlus simulation.
    *   Control the simulation flow (e.g., advance time steps).

    .. image:: Images/overview.png
        :width: 600
        :alt: Overview of eprllib and EnergyPlus interaction
        :align: center
        :figclass: align-center
        :caption: Overview of eprllib and EnergyPlus interaction.

Agents
------

Agents are the decision-making entities in the RL process. In eprllib, agents interact with the EnergyPlus environment to learn optimal control strategies.

*   **AgentSpec:**

    The ``AgentSpec`` class defines the specifications for an agent. It includes:

    *   **ObservationSpec:** Defines what the agent can observe in the environment.
    *   **ActionSpec:** Defines what actions the agent can take.
    *   **RewardSpec:** Defines how the agent is rewarded for its actions.
    *   **FilterSpec:** Defines how the agent's observations are filtered.
    *   **TriggerSpec:** Defines when the agent's actions are triggered.

    .. code-block:: python

        from eprllib.Agents.AgentSpec import AgentSpec, ObservationSpec, ActionSpec, RewardSpec, FilterSpec, TriggerSpec
        from eprllib.Agents.Filters.DefaultFilter import DefaultFilter
        from eprllib.Agents.Triggers.SetpointTriggers import DualSetpointTriggerDiscreteAndAvailabilityTrigger

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
            filter=FilterSpec(
                filter_fn=DefaultFilter,
                filter_fn_config={},
            ),
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

*   **ObservationSpec:**

    Defines the agent's observation space. It specifies:

    *   **Variables:** The EnergyPlus variables the agent can observe (e.g., "Zone Air Temperature").
    *   **Meters:** The EnergyPlus meters the agent can observe (e.g., "Electricity:Building").
    *   **Simulation Parameters:** The EnergyPlus simulation parameters the agent can observe.
    *   **Internal Variables:** The internal variables the agent can observe.
    *   **Use Actuator State:** If the agent can observe the state of the actuators.
    *   **Use One Day Weather Prediction:** If the agent can observe the weather prediction.
    *   **Prediction Hours:** The number of hours of weather prediction.
    *   **Prediction Variables:** The variables of the weather prediction.

*   **ActionSpec:**

    Defines the agent's action space. It specifies:

    *   **Actuators:** The EnergyPlus actuators the agent can control (e.g., "Zone Thermostat Heating Setpoint Temperature").

*   **RewardSpec:**

    Defines how the agent is rewarded. It specifies:

    *   **Reward Function:** A function that calculates the reward based on the agent's actions and the environment's state.
    *   **Reward Function Configuration:** The configuration of the reward function.

*   **FilterSpec:**

    Defines how the agent's observations are filtered. It specifies:

    *   **Filter Function:** A function that filters the observations.
    *   **Filter Function Configuration:** The configuration of the filter function.

*   **TriggerSpec:**

    Defines when the agent's actions are triggered. It specifies:

    *   **Trigger Function:** A function that determines when to trigger an action.
    *   **Trigger Function Configuration:** The configuration of the trigger function.

*   **Filters:**

    Filters are modules that can be used to process the agent's observations before they are used by the agent. eprllib provides a ``DefaultFilter``, but you can create custom filters.

*   **Triggers:**

    Triggers are modules that determine when an agent's actions should be executed. eprllib provides a ``DualSetpointTriggerDiscreteAndAvailabilityTrigger``, but you can create custom triggers.

Connectors
----------

Connectors define how agents interact with the environment.

*   **DefaultConnector:**

    The ``DefaultConnector`` is the standard way for agents to interact with the environment. It handles:

    *   Receiving observations from the environment.
    *   Sending actions to the environment.
    *   Receiving rewards from the environment.

*   **Custom Connectors:**

    You can create custom connectors to implement different interaction patterns between agents and the environment. This allows for flexibility in how agents are integrated into the simulation.

Episodes
--------

Episodes define the configuration of the simulation.

*   **Episode:**

    The ``Episode`` class defines the configuration of the simulation. It includes:

    *   **Episode Function:** A function that defines the episode.
    *   **Episode Function Configuration:** The configuration of the episode function.

Integration with RLlib
----------------------

eprllib is designed to work seamlessly with RLlib, a powerful library for reinforcement learning.

*   **Using eprllib Environments with RLlib:**

    eprllib environments (created using ``Environment``) can be directly used with RLlib algorithms. You simply need to register the environment with RLlib and then use it in your RLlib configuration.

    .. code-block:: python

        import ray
        from ray.tune import register_env
        from eprllib.Env.Environment import Environment

        # Register the environment
        register_env(name="EPEnv", env_creator=lambda args: Environment(args))

        # Use the environment in your RLlib configuration
        config = ppo.PPOConfig()
        config = config.environment(env="EPEnv", env_config=env_config)

*   **RLlib Policies and eprllib Agents:**

    RLlib policies are used to control eprllib agents. The policy determines the actions that the agent takes based on its observations.

    .. image:: Images/rllib_integration.png
        :width: 600
        :alt: RLlib and eprllib integration
        :align: center
        :figclass: align-center
        :caption: RLlib and eprllib integration.

By understanding these key concepts, you'll be well-equipped to start developing your own RL agents for building energy optimization and control using eprllib.
