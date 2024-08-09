Welcome to eprllib's documentation!
===================================

This repository provides a set of methods to establish the computational loop of EnergyPlus within a Markov Decision Process (MDP), treating it as a multi-agent environment compatible with RLlib. The main goal is to offer a simple configuration of EnergyPlus as a standard environment for experimentation with Deep Reinforcement Learning.

Installation
------------

To install EnergyPlusRL, simply use pip:

.. code-block:: python
    
    pip install eprllib

Key Features
------------

* Integration of EnergyPlus and RLlib: This package facilitates setting up a Reinforcement Learning environment using EnergyPlus as the base, allowing for experimentation with energy control policies in buildings.
* Simplified Configuration: To use this environment, you simply need to provide a configuration in the form of a dictionary that includes state variables, metrics, actuators (which will also serve as agents in the environment), and other optional features.
* Flexibility and Compatibility: EnergyPlusRL easily integrates with RLlib, a popular framework for Reinforcement Learning, enabling smooth setup and training of control policies for actionable elements in buildings.

Usage
-----

1. Import eprllib.
2. Configure EnvConfig to provide a EnergyPlus model based configuration, specifying the parameters required (see eprllib.Env.EnvConfig).
3. Configure RLlib algorithm to train the policy.
4. Execute the training using RLlib or Tune.

Example configuration
---------------------

1. Import eprllib (and the libraries that you need).

.. code-block:: python

    import ray
    from ray.tune import register_env
    from ray.rllib.algorithms.ppo.ppo import PPOConfig
    import eprllib
    from eprllib.Env.EnvConfig import EnvConfig, env_config_to_dic
    from eprllib.Env.MultiAgent.EnergyPlusEnv import EnergyPlusEnv_v0

1. Configure EnvConfig to provide a EnergyPlus model based configuration, specifying the parameters required (see eprllib.Env.EnvConfig).

.. code-block:: python

    BuildingModel = EnvConfig()
    BuildingModel.generals(
        epjson_path='path_to_epJSON_file',
        epw_path='path_to_EPW_file',
        output_path='path_to_output_folder',
    )
    BuildingModel.agents(
        agents_config = {
            'Thermal Zone: Room1':{
                'Agent 1 in Room 1': {
                    'ep_actuator_config': (),
                    'thermal_zone': 'Thermal Zone: Room 1',
                    'actuator_type’: 3 ,
                    'agent_id': 1,
                },
            }
        }
    )


3. Configure RLlib algorithm to train the policy.

.. code-block:: python

    # Start a Ray server.
    ray.init()
    # Register the environment.
    register_env(name="EPEnv", env_creator=lambda args: EnergyPlusEnv_v0(args))
    # Configure the algorith and assign the environment registred.
    algo = PPOConfig ( )
    algo.environment(
        env = "EPEnv",
        env_config = env_config_to_dict(BuildingModel)
    )


4. Execute the training using RLlib or Tune.

.. code-block:: python

    # Train the policy with Tune.
    tune.Tuner(
        'PPO',
        tune_config=tune.TuneConfig(
            mode="max",
            metric="episode_reward_mean",
        ),
        run_config=air.RunConfig(
            stop={"episodes_total": 10},
        ),
        param_space=algo.to_dict(),
    ).fit()

.. toctree::
   :maxdepth: 2
   :caption: Contents:


