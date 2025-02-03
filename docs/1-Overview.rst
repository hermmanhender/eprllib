Overview
=========

``eprllib`` was born out of the need to bridge the gap between building modeling with 
**EnergyPlus** and Reinforcement Learning (**RL**). Traditionally, integrating these two 
disciplines has been complex and laborious. eprllib aims to simplify this process, 
offering an intuitive and flexible interface for developing intelligent agents that 
interact with building simulations.

Purpose and Scope
------------------

The main objective of ``eprllib`` is to facilitate experimentation with Reinforcement Learning 
algorithms in the context of energy efficiency and building control. The library allows 
researchers and professionals to design, train, and evaluate agents that can optimize energy 
consumption, improve occupant comfort, and manage resources more efficiently.

Key Features
-------------

``eprllib`` stands out for the following features:

* **Native Multi-Agent Environment**: ``eprllib`` is designed to work with complex multi-agent systems, 
  including hierarchical architectures. This allows for modeling scenarios where multiple agents 
  interact with each other and the building environment. It also supports the use of individual 
  agents for simpler cases.
* **Flexibility and Configurability**: The library offers configurable modules for `Filters <https://hermmanhender.github.io/eprllib/build/html/7-Filters.htlm>`_, 
  `triggers <https://hermmanhender.github.io/eprllib/build/html/6-Triggers.htlm>`_, and `rewards <https://hermmanhender.github.io/eprllib/build/html/8-Rewards.htlm>`_. 
  This allows users to customize the learning environment to suit their specific needs and explore different control strategies.
* **Inter-Agent Communication and Coordination**: ``eprllib`` includes a dedicated module for `communication 
  and coordination between agents <https://hermmanhender.github.io/eprllib/build/html/9-Connectors.htlm>`_. This is 
  essential for implementing collaborative strategies and solving complex problems that require the interaction of multiple agents.
* **Deep Integration with EnergyPlus**: Thanks to its close integration with the EnergyPlus API, ``eprllib`` 
  allows access to all the variables, metrics, parameters, and actuators that EnergyPlus offers. This 
  means that users can work with detailed and realistic building models, covering a wide range of case 
  studies, from thermostat control to the management of complete HVAC systems.
* **Ease of Use**: ``eprllib`` has been developed following a scheme similar to RLlib, a popular Reinforcement 
  Learning library. This facilitates the learning of ``eprllib`` for those users familiar with RLlib.

Target Audience and Prior Knowledge
------------------------------------

``eprllib`` is primarily aimed at researchers with **experience in EnergyPlus** and an interest in Reinforcement 
Learning. It is assumed that users have prior knowledge in both areas, as ``eprllib`` has been developed 
following a scheme similar to RLlib to facilitate the joint learning of both tools. While it is not necessary 
to be an expert in RL, a basic understanding of fundamental concepts such as agents, environments, rewards, 
and policies is recommended.

Software and Hardware Requirements
-----------------------------------
``eprllib`` requires Python 3.10 or higher and EnergyPlus 9.3 or higher, with version 24.2 of the latter being 
recommended. In addition, RLlib's specific dependencies will be required, which will be installed automatically 
during the installation of ``eprllib``. Hardware requirements will depend on the complexity of the building models 
and the RL algorithms used. It is recommended to have a computer with sufficient RAM and processing capacity to 
run complex simulations.

Installation
------------

Installing ``eprllib`` is simple and is done through ``pip``:

.. code-block:: python
    
    pip install eprllib

This command will install ``eprllib`` and all its dependencies, including RLlib and the necessary libraries for 
communication with EnergyPlus.

Example of Use
---------------

[Here, a basic example of use would be included to illustrate how to use ``eprllib`` to implement an agent in a simulated 
EnergyPlus environment. This example will be completed later.]

1. Import ``eprllib``.
2. Configure an EnvConfig object to feed EnergyPlus Python API and RLlib with the environment configuration based on the EnergyPlus model,
    specifying the parameters required (see ``eprllib.Env.EnvConfig``).
3. Configure `RLlib algorithm <https://docs.ray.io/en/latest/rllib/rllib-algorithms.html>`_ to train the policy.
4. Execute the training using `RLlib <https://docs.ray.io/en/latest/rllib/index.html>`_ or `Tune <https://docs.ray.io/en/latest/tune/index.html>`_.


License and Additional Resources
---------------------------------

``eprllib`` is distributed under the MIT license, a permissive open source license that allows its use and 
modification for both personal and commercial purposes.

For more information, to report issues, or to contribute to the development of ``eprllib``, you can visit the 
`GitHub repository <https://github.com/hermmanhender/eprllib>`_. There you will also find a discussion forum for each 
version (Release) where you can interact with other users and developers.
