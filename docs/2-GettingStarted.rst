Getting Started with eprllib
=============================

Welcome to eprllib! This guide will help you get started with using eprllib for building
control and energy optimization through Reinforcement Learning (RL).

eprllib leverages RL, a powerful machine learning technique, to develop intelligent
agents that can interact with building simulations. In RL, agents learn to make decisions
by interacting with an environment, taking actions, receiving observations, and obtaining
rewards. This interaction is used to learn an optimal policy, which is a strategy that
maps observations to actions.

The general scheme of RL can be visualized in the following diagram:

.. image:: Images/markov_decision_process.png
    :width: 600
    :alt: Markov Decision Process
    :align: center
    :figclass: align-center
    :caption: The general scheme of Reinforcement Learning (RL) as a Markov Decision Process.

Deep Reinforcement Learning (DRL)
---------------------------------

During the learning process, the RL algorithm attempts to predict the cumulative reward that the
agent will receive if it follows a certain policy. This prediction can be represented by a Value
function, denoted as ``V(obs)``, or an Action-Value function, denoted as ``Q(obs, act)``.

A modern approach to predicting these ``V`` or ``Q`` functions involves using **deep neural networks (DNNs)**
to approximate these values. When DNNs are used, the methodology is referred to as
**Deep Reinforcement Learning (DRL)**. In this context, the DNN model is often referred to as the **policy**.

In essence, the policy is a complex function that, given an observation, outputs the best action to take.

eprllib, EnergyPlus, and RLlib
-------------------------------

eprllib leverages two powerful tools: **EnergyPlus** and **RLlib**.

*   **EnergyPlus** is used to model the building environment. It simulates the building's
    energy performance and provides the environment with which the RL agent interacts.
    `EnergyPlus <https://energyplus.net/>`_
*   **RLlib** is a framework for Deep Reinforcement Learning (DRL). eprllib uses RLlib to train,
    evaluate, save, and restore policies. `RLlib <https://docs.ray.io/en/latest/rllib/index.html>`_

In essence, EnergyPlus provides the simulated world, and RLlib provides the tools to train the
agent within that world.

Running a Simple Experiment with eprllib and RLlib
---------------------------------------------------

Now that you have a basic understanding of the concepts, let's walk through a simple experiment using eprllib and RLlib. This example will demonstrate the core steps involved in setting up and training an agent.

**Steps:**

1.  **Define the Environment:** Use EnergyPlus to create or load a building model that will serve as the environment for the RL agent.
2.  **Define the Agent:** Specify the agent's actions, observations, and reward structure. This is done using eprllib's configuration tools.
3.  **Configure the RL Algorithm:** Choose an appropriate RL algorithm from RLlib and configure its hyperparameters.
4.  **Train the Agent:** Run the training process, allowing the agent to interact with the EnergyPlus environment and learn an optimal policy.
5.  **Evaluate the Agent:** Assess the performance of the trained agent in the EnergyPlus environment.
6.  **Save and Restore the Agent:** Save the trained agent to use it in the future.

**Example:**

The following code provides a basic outline of how to set up and train an agent using eprllib and RLlib. This example uses a simplified environment and agent configuration for clarity.

.. code-block:: python
    :linenos:

    """
    Simple eprllib and RLlib Example
    =================================

    This script demonstrates a basic setup for training an agent using eprllib and RLlib.
    It uses a simplified environment and a basic PPO configuration.
    """

    import os
    from tempfile import TemporaryDirectory

    import ray
    from ray.rllib.algorithms import ppo
    from ray.tune.logger import pretty_print

    from eprllib.Environment.Environment import Environment
    from eprllib.Environment.EnvironmentConfig import EnvironmentConfig
    from eprllib.AgentsConnectors.DefaultConnector import DefaultConnector
    from eprllib.Agents.AgentSpec import (
        AgentSpec,
        ObservationSpec,
        RewardSpec,
        ActionSpec,
        TriggerSpec,
        FilterSpec
    )
    from eprllib.Agents.Filters.DefaultFilter import DefaultFilter
    from eprllib.Agents.Triggers.SetpointTriggers import DualSetpointTriggerDiscreteAndAvailabilityTrigger

    # --- Configuration ---
    # Define the name of the environment
    name = "EPEnv"
    # Define the name of the agent
    agent_name = "HVAC"
    # Define the path of the output
    output_path = TemporaryDirectory("output", "", 'C:/Users/grhen/Documents/Resultados_RLforEP').name
    # --- End Configuration ---

    # --- Environment Configuration ---
    # Create an EnvironmentConfig object to define the environment
    eprllib_config = EnvironmentConfig()
    eprllib_config.generals(
        epjson_path="C:/Users/grhen/OneDrive - docentes.frm.utn.edu.ar/01-Desarrollo del Doctorado/03-Congresos y reuniones/03 - eprllib/Study Cases/Task 1/model-00000000-25772.epJSON",  # Replace with your EPJSON file
        epw_path="C:/Users/grhen/OneDrive - docentes.frm.utn.edu.ar/01-Desarrollo del Doctorado/03-Congresos y reuniones/03 - eprllib/Weather analysis/Chacras_de_Coria_Mendoza_ARG-hour.epw",  # Replace with your EPW file
        output_path=output_path,
        ep_terminal_output=False,
        timeout=10,
        evaluation=False,
    )

    # --- Agent Configuration ---
    # Define the agent's observation, action, reward, filter and trigger
    eprllib_config.agents(
        connector_fn=DefaultConnector,
        connector_fn_config={},
        agents_config={
            agent_name: AgentSpec(
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
                        "agent_name": agent_name,
                        'temperature_range': (18, 28),
                        'actuator_for_cooling': ("Schedule:Compact", "Schedule Value", "cooling_setpoint"),
                        'actuator_for_heating': ("Schedule:Compact", "Schedule Value", "heating_setpoint"),
                        'availability_actuator': ("Schedule:Constant", "Schedule Value", "HVAC_OnOff"),
                    },
                ),
                reward=RewardSpec(
                    reward_fn=lambda agent_name, thermal_zone, beta, people_name, cooling_name, heating_name, cooling_energy_ref, heating_energy_ref, **kwargs: 0,
                    reward_fn_config={
                        "agent_name": agent_name,
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

    # --- Episode Configuration ---
    # Define the episode configuration
    eprllib_config.episodes(
        episode_fn=lambda **kwargs: None,
        episode_fn_config={}
    )

    # --- RLlib Configuration ---
    # Initialize Ray
    ray.init(_temp_dir='C:/Users/grhen/ray_results/tmp')

    # Register the environment
    from ray.tune import register_env
    register_env(name=name, env_creator=lambda args: Environment(args))

    # Build the environment configuration
    env_config = eprllib_config.build()

    # Configure the PPO algorithm
    config = ppo.PPOConfig()
    config = config.environment(env=name, env_config=env_config)  # Use the registered environment name
    config = config.framework("torch")
    config = config.rollouts(num_rollout_workers=0)  # Use 0 workers for simplicity
    config = config.training(model={"fcnet_hiddens": [64, 64]})
    config = config.multi_agent(
        policies={
            'single_policy': None
        },
        policy_mapping_fn=lambda agent_id, episode, worker, **kwargs: 'single_policy',
    )
    algorithm = config.build()

    # --- Training ---
    # Train the agent for a few iterations
    for i in range(5):
        result = algorithm.train()
        print(f"Training iteration {i + 1}:")
        print(pretty_print(result))

    # --- Save and Restore ---
    # Save the trained agent
    checkpoint_path = algorithm.save()
    print(f"Checkpoint saved to {checkpoint_path}")

    # Restore the agent from the checkpoint
    algorithm.restore(checkpoint_path)
    print(f"Checkpoint restored from {checkpoint_path}")

    # --- End ---
    ray.shutdown()

**Explanation:**

1.  **Environment Configuration:**
    *   We start by creating an `EnvironmentConfig` object. This object holds all the information about the EnergyPlus environment, such as the EPJSON file, the EPW file, and the output path.
    *   We use the `eprllib_config.generals()` method to set these general parameters.
    *   You'll need to replace the placeholder paths with your actual file paths.

2.  **Agent Configuration:**
    *   We define the agent's behavior using `eprllib_config.agents()`.
    *   We specify the agent's **observations** (what it can see), **actions** (what it can do), **rewards** (what it's trying to maximize), **filters** and **triggers**.
    *   In this simplified example, the agent observes the outdoor air temperature and the zone mean air temperature.
    *   The agent can control the heating and cooling setpoints and the HVAC on/off.
    *   The reward function is a placeholder in this example.
    * The filter and trigger are defined.

3.  **Episode Configuration:**
    *   We define the episode configuration using `eprllib_config.episodes()`.
    *   In this example, the episode function is a placeholder.

4.  **RLlib Configuration:**
    *   We initialize Ray, which is the framework that RLlib uses for distributed computing.
    *   We register our environment with Ray using `register_env`.
    *   We build the environment configuration using `eprllib_config.build()`.
    *   We configure the PPO algorithm using `ppo.PPOConfig()`.
    *   We specify the environment, the framework (PyTorch), and the number of rollout workers (0 for simplicity).
    *   We define a simple neural network model with two hidden layers of 64 units each.
    *   We define a single policy.
    *   We build the algorithm using `config.build()`.

5.  **Training:**
    *   We train the agent for a few iterations using a `for` loop and `algorithm.train()`.
    *   The `pretty_print()` function is used to display the training results.

6.  **Save and Restore:**
    *   We save the trained agent to a checkpoint using `algorithm.save()`.
    *   We restore the agent from the checkpoint using `algorithm.restore()`.

7.  **End:**
    *   We shutdown the ray.

This example provides a basic framework for training an agent with eprllib and RLlib. You can expand upon this example by adding more complex environment configurations, agent behaviors, and reward functions.

**Next Steps:**

1.  **Replace Placeholders:** Replace the placeholder file paths and reward function with your actual values.
2.  **Run the Code:** Run the code to see the agent training.
3.  **Experiment:** Modify the code to explore different environment configurations, agent behaviors, and hyperparameters.

This simplified example should give you a good starting point for using eprllib and RLlib.
