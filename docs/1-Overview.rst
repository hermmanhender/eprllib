Overview
========

eprllib was created to bridge the gap between building energy modeling
with **EnergyPlus** and **Reinforcement Learning (RL)**. Integrating
these two disciplines has traditionally been complex and time-consuming.
eprllib simplifies this process by providing an intuitive and flexible
interface for developing intelligent agents that interact with EnergyPlus
building simulations.


Purpose and Scope
-----------------

The primary objective of eprllib is to facilitate experimentation with
**Reinforcement Learning (RL)** algorithms for energy efficiency and building
control. The library enables researchers and professionals to design, train,
and evaluate RL agents that can optimize energy consumption, enhance occupant
comfort, and improve resource management in buildings.


Key Features
------------

eprllib offers several key features:

*   **Native Multi-Agent Environment:** eprllib is designed to support complex
    multi-agent systems, including hierarchical architectures. This enables the modeling
    of scenarios where multiple agents interact with each other and the building environment.
    It also supports single-agent setups for simpler applications.
*   **Flexible and Configurable:** The library provides configurable modules
    for **Filters**, **Triggers**, and **Rewards**, allowing users to customize the learning
    environment and explore diverse control strategies.
*   **Inter-Agent Communication and Coordination:** eprllib includes a dedicated module for
    communication and coordination between agents. This is crucial for implementing collaborative
    strategies and addressing complex problems that require multi-agent interaction.
*   **Deep Integration with EnergyPlus:** Through its close integration with the **EnergyPlus API**,
    eprllib provides access to all variables, metrics, parameters, and actuators available in EnergyPlus.
    This enables users to work with detailed and realistic building models, covering a wide range of
    applications, from thermostat control to the management of complete HVAC systems.
*   **Ease of Use (RLlib Familiarity):** eprllib's design is inspired by **RLlib**, a popular
    Reinforcement Learning library. This design choice facilitates the learning process for users already
    familiar with RLlib.



Target Audience and Prior Knowledge
-----------------------------------

eprllib is primarily intended for researchers and professionals with **experience in EnergyPlus** and an
interest in **Reinforcement Learning (RL)**. Users are expected to have prior knowledge in both areas.
eprllib's design, inspired by **RLlib**, facilitates the learning process for those familiar with both tools.
While expertise in RL is not required, a basic understanding of fundamental concepts such as agents,
environments, rewards, and policies is recommended.


Software and Hardware Requirements
----------------------------------

eprllib requires **Python 3.10 or higher** and **EnergyPlus 9.3 or higher**, with **EnergyPlus 24.2** recommended.
**RLlib's** dependencies will be automatically installed during the eprllib installation. Hardware requirements
depend on the complexity of the EnergyPlus building models and the RL algorithms employed. A computer
with ample RAM and processing capacity is recommended for running complex simulations.


Installation
------------

Installing eprllib is straightforward using `pip`:

.. code-block:: bash

    pip install eprllib

This command installs eprllib and all its dependencies, including RLlib and the required libraries for
communication with the EnergyPlus API.


Example of Use
--------------

[A basic example illustrating how to use eprllib to implement an agent within a simulated
EnergyPlus environment will be provided here in a future update.]

The general workflow for using eprllib involves the following steps:

1.  **Import eprllib:** Begin by importing the eprllib library into your Python script.
2.  **Configure the Environment:** Create an `EnvironmentConfig` object to define the environment
    configuration based on your EnergyPlus model. This object is used to provide the
    EnergyPlus API and RLlib with the necessary parameters (see `eprllib.Environment.EnvironmentConfig`).
3.  **Configure the RL Algorithm:** Set up an `RLlib algorithm <https://docs.ray.io/en/latest/rllib/rllib-algorithms.html>`_
    to train the RL policy.
4.  **Execute Training:** Run the training process using `RLlib <https://docs.ray.io/en/latest/rllib/index.html>`_
    or `Tune <https://docs.ray.io/en/latest/tune/index.html>`_.



License and Additional Resources
--------------------------------

eprllib is distributed under the **MIT license**, a permissive open-source license that allows for
its use and modification for both personal and commercial purposes.

For more information, to report issues, or to contribute to the development of eprllib, please
visit the `GitHub repository <https://github.com/hermmanhender/eprllib>`_. The repository also hosts a discussion
forum for each release, where you can interact with other users and developers.
