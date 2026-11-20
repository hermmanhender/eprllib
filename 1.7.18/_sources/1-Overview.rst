Overview
========

``eprllib`` was created to bridge the gap between building energy modeling
with **EnergyPlus** and **Deep Reinforcement Learning (DRL)**. Integrating
these two disciplines has traditionally been complex and time-consuming.
We thought that this library simplifies this process by providing an intuitive 
and flexible way for developing intelligent agents that interact with EnergyPlus
building simulations models and are trained efficiently using RLlib.


Purpose and Scope
-----------------

The primary objective of ``eprllib`` is to facilitate experimentation with
**DRL** algorithms for energy efficiency and building control, allowing not only 
HVAC control but also all the possibilities that EnergyPlus offers like on-site 
renewable energy generation and storage. The library enables researchers and professionals 
to design, train, and evaluate DRL agents that can optimize energy consumption, enhance 
occupant comfort, and improve resource management in buildings.


Key Features
------------

There are many options to train DRL agents today but we consider ``eprllib`` as a tool 
that provides a more accessible and flexible approach to integrating DRL with EnergyPlus. 
The library offers several key features:

*   **Native MultiAgent Environment:** ``eprllib`` is designed to support complex
    multiagent systems, including hierarchical architectures. See ``Connector`` API to see how 
    it enables the modeling of scenarios where multiple agents interact with each other and 
    the building environment. This is not a limitation to setup single-agent applications.
*   **Flexible and Configurable:** The library provides configurable modules: ``Episodes``, 
*   ``Connectors``, ``Filters``, ``ActionMappers``, and ``Rewards``. They allow users to 
*   customize the learning environment and explore diverse control strategies.
*   **Inter-Agent Communication and Coordination:** As mention before, ``eprllib`` includes a 
*   dedicated module for communication and coordination between agents, the ``Connectors`` API. 
*   This is crucial for implementing collaborative strategies and addressing complex problems that 
*   require multiagent interaction.
*   **Deep Integration with EnergyPlus:** Through its close integration with the EnergyPlus API,
    ``eprllib`` provides access to all variables, metrics, parameters, and actuators available in EnergyPlus.
    This enables users to work with detailed and realistic building models, covering a wide range of
    applications, from thermostat control to the management of complete HVAC systems or on-site renewable 
    energy generation and storage.
*   **Ease of Use (RLlib Familiarity):** The library's design was inspired by 
*   `RLlib <https://docs.ray.io/en/latest/rllib/index.html>`_, a popular
    Deep Reinforcement Learning library. This design choice facilitates the learning process for users already
    familiar with RLlib.



Target Audience and Prior Knowledge
-----------------------------------

``eprllib`` is primarily intended for researchers and professionals with **experience in EnergyPlus** and an
interest in **Deep Reinforcement Learning (DRL)**. Users are expected to have prior knowledge in both building 
modeling with EnergyPlus and basic concepts of DRL (agents, environments, rewards, and policies). The intention 
is to provide a tool that allows users to leverage their existing expertise in these areas to develop and 
train DRL agents effectively.


Software requirements
---------------------

The library is operating system agnostic. ``eprllib`` requires **Python >3.10 and <3.13 higher** and **EnergyPlus 9.3 
or higher**, with the last version available recommended. ``ray`` and another dependencies will be automatically 
installed during the ``eprllib`` installation.

Hardware requirements depend on the complexity of the EnergyPlus building models and the RL algorithms employed.


Installation
------------

Installing ``eprllib`` is straightforward using ``pip``:

.. code-block:: bash

    pip install eprllib

This command installs ``eprllib`` and all its dependencies, including ``ray`` and the required libraries for
communication with the EnergyPlus API.

The EnergyPlus software itself must be installed separately, and its API should be accessible to the Python 
environment where ``eprllib`` is installed. For this, the EnergyPlus root must be added to the system's PATH environment variable.

Detailed instructions for installing EnergyPlus can be found on the `EnergyPlus website <https://energyplus.net/downloads>`_.


Workflow Overview
-----------------

The general workflow for using ``eprllib`` involves the following steps:

1.  **Prepare the EnergyPlus Model:** Develop and prepare your EnergyPlus building model as usual. Running a firtst 
    simulation with the EnergyPlus software is recommended to ensure that the model is correctly set up and that all 
    necessary variables and actuators are defined. This step is crucial for ensuring that the DRL agent can interact 
    effectively with the environment during training. Also, the files generated during this first simulation will facilitate
    the configuration of the environment in the next steps.
2.  **Import the modules necessary:** Begin by importing the ``EnvironmentConfig`` class from ``eprllib.Environment.EnvironmentConfig`` 
    into your Python script.
3.  **Configure the Environment:** Create an ``EnvironmentConfig`` object to define the environment
    configuration based on your EnergyPlus model. This object is used to provide the
    EnergyPlus API and RLlib with the necessary parameters.
4.  **Build the Environment:** Use the configured ``EnvironmentConfig`` to build the environment that will be used for training the DRL agent.
5.  **Register the environment into RLlib:** Register the built environment with RLlib to make it available for training.
6.  **Configure the DRL Algorithm:** Set up an `RLlib algorithm <https://docs.ray.io/en/latest/rllib/rllib-algorithms.html>`_
    to train the DRL policy. Here you will provide the EnergyPlus build in the previous steps.
7.  **Execute Training:** Run the training process using `RLlib <https://docs.ray.io/en/latest/rllib/index.html>`_
    or `Tune <https://docs.ray.io/en/latest/tune/index.html>`_.


Example of Use
--------------

A quick example is here to illustrate the workflow described above. This example assumes 
that you have already prepared your EnergyPlus model and that it is ready for interaction 
with the DRL agent.

So we go directly to the second step of the workflow, which is to import the necessary modules and configure the environment. Here 
only the EnvironmentConfig class is imported to illustrate the process, but in a real implementation, you would likely need to 
import additional modules for defining the agents, rewards, and other components of the environment.

.. code-block:: python

    from eprllib.Environment.EnvironmentConfig import EnvironmentConfig


Then, you would proceed to configure the environment by creating an instance of the ``EnvironmentConfig`` class. This configuration 
will include four main parameters build on the classes of ``eprllib`` that includedetails about the EnergyPlus model, the agents, 
the rewards, and any other necessary parameters.

.. code-block:: python

    env_config = EnvironmentConfig(
        generals = ...
        connectors = ...
        agents = ...
        episodes = ...
    )


After configuring the environment, the next step is to build it using the ``build()`` method of the ``EnvironmentConfig`` instance. 
This will create an environment that can be used for training the DRL agent. Is an important step to ensure that the environment is 
correctly configured before building it, as any issues in the configuration may lead to problems during training.

.. code-block:: python

    env_config_built = env_config.build()


Before configure the DRL ``Algorithm`` class, the environment must be registered with RLlib, which allows you to use it for training 
your DRL agent. Follow the `standard RLlib procedure <https://docs.ray.io/en/latest/rllib/rllib-env.html#specifying-by-tune-registered-lambda>`_ 
for registering custom environments as shown in the example below:

.. code-block:: python

    from eprllib.Environment.Environment import Environment
    from ray.tune.registry import register_env

    register_env("my_energyplus_env", lambda config: Environment(config))


Finally, you can configure your DRL algorithm and execute the training process using RLlib or Tune. The specific configuration will depend on the
algorithm you choose and the details of your environment, but the general process involves setting up the algorithm with the registered environment 
and then calling the training function. Here a PPO algorithm is used as an example and Tune is used to execute the training process:

.. code-block:: python

    from ray import tune
    from ray.rllib.algorithms.ppo import PPOConfig

    # Configure the PPO algorithm
    ppo_config = PPOConfig()
    ppo_config.environment(
        # Here you introduce the name of the eprllib environment registered.
        env = "my_energyplus_env",
        # Here you need to pass the built environment config.
        env_config = env_config_built 
        )
    
    # Execute the training process using Tune
    tune.run(
        "PPO",
        config=ppo_config.to_dict(),
        stop={"training_iteration": 100},
        checkpoint_at_end=True,
    )



License and Additional Resources
--------------------------------

``eprllib`` is distributed under the **MIT license**, a permissive open-source license that allows for
its use and modification for both personal and commercial purposes.

For more information, to report issues, or to contribute to the development of ``eprllib``, please
visit the `GitHub repository <https://github.com/hermmanhender/eprllib>`_. The repository also hosts a discussion
forum for each release, where you can interact with other users and developers.
