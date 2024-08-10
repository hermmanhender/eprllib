Overview
========

Why eprllib was developed.

Installation
------------

To install EnergyPlusRL, simply use pip:

.. code-block:: python
    
    pip install eprllib

Organization of the documentation
---------------------------------

The organization of the documentation.

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
