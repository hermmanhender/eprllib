"""
Entrenamiento tarea 1-d
========================

Learning parameters
--------------------
    gamma = 0.8
    lr = 0.0001
    train_batch_size = 10000
    minibatch_size = 1000
    num_epochs = 15

Policy model configuration
---------------------------
    "fcnet_hiddens": [256,256]
    "fcnet_activation": "tanh"

PPO Config
-----------
    use_critic = True
    use_gae = True
    lambda_ = 0.7
    use_kl_loss = True
    kl_coeff = 0.2
    kl_target = 0.7
    shuffle_batch_per_epoch = True
    vf_loss_coeff = 0.9
    entropy_coeff = 0.01
    clip_param = 0.2
    vf_clip_param = 0.2

Exploración
------------
    Curiosity
    
Results
--------
    task_1d_ashrae55sm_PPO_98445_00000
"""
import os
os.environ['RAY_DEDUP_LOGS'] = '0'

import time
import json
from tempfile import TemporaryDirectory

import ray
from ray import air, tune
from ray.tune import register_env
from ray.rllib.algorithms.ppo.ppo import PPOConfig
from ray.rllib.policy.policy import PolicySpec

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

from eprllib.Agents.Rewards.EnergyAndAshrae55SimpleModel import EnergyAndASHRAE55SimpleModel
from eprllib.examples.example_thermostat_files.episode_fn import task_cofiguration
from eprllib.examples.example_thermostat_files.policy_mapping import policy_map_fn

with open("src/eprllib/examples/example_thermostat_files/episode_fn_config.json", "r") as f:
    episode_config = json.load(f)

experiment_name:str = "07-CTD-V"
name:str = "all"
tuning:bool = False
restore:bool = False
checkpoint_path:str = ""
new_rllib_api = False

eprllib_config = EnvironmentConfig()
eprllib_config.generals(
    epjson_path = "src/eprllib/examples/example_central_agent_files/model.epJSON",
    epw_path = "src/eprllib/examples/example_central_agent_files/weathers/ESP_PV_San.Sebastian-Igueldo.080270_TMYx.2004-2018.epw",
    output_path = TemporaryDirectory("output","",'C:/Users/grhen/Documents/Resultados_RLforEP').name,
    ep_terminal_output = False,
    timeout = 10,
    evaluation = False,
)
eprllib_config.connector(
    connector_fn = DefaultConnector,
    connector_fn_config = {},
)
eprllib_config.agents(
    agents_config = {
        "HVAC": AgentSpec(
            observation = ObservationSpec(
                variables = [
                    ("Site Outdoor Air Drybulb Temperature", "Environment"),
                    ("Site Wind Speed", "Environment"),
                    ("Site Outdoor Air Relative Humidity", "Environment"),
                    ("Zone Mean Air Temperature", "Thermal Zone"),
                    ("Zone Air Relative Humidity", "Thermal Zone"),
                    ("Zone People Occupant Count", "Thermal Zone"),
                    ("Zone Thermal Comfort ASHRAE 55 Simple Model Summer or Winter Clothes Not Comfortable Time", "Thermal Zone"),
                ],
                simulation_parameters = {
                    'today_weather_horizontal_ir_at_time': True,
                },
                meters = [
                    "Electricity:Building",
                    "Heating:DistrictHeatingWater",
                    "Cooling:DistrictCooling",
                ],
                use_actuator_state = True,
                use_one_day_weather_prediction = True,
                prediction_hours = 24,
                prediction_variables = {
                    'outdoor_dry_bulb': True,
                },
                internal_variables = [
                    ("Zone Floor Area", "Thermal Zone"),
                ],
                history_len=6,
                user_occupation_funtion = True,
                occupation_schedule = ("Schedule:Constant", "Schedule Value", "occupancy_schedule"),
                user_occupation_forecast = True,
                summer_months = [11, 12, 1, 2],
                
            ),
            action = ActionSpec(
                actuators = [
                    ("Schedule:Compact", "Schedule Value", "heating_setpoint"),
                    ("Schedule:Compact", "Schedule Value", "cooling_setpoint"),
                    ("Schedule:Constant", "Schedule Value", "HVAC_OnOff"),
                ],
            ),
            filter= FilterSpec(
                filter_fn = DefaultFilter,
                filter_fn_config = {},
            ),
            trigger= TriggerSpec(
                trigger_fn = DualSetpointTriggerDiscreteAndAvailabilityTrigger,
                trigger_fn_config = {
                    "agent_name": "HVAC",
                    'temperature_range': (18, 28),
                    'actuator_for_cooling': ("Schedule:Compact", "Schedule Value", "cooling_setpoint"),
                    'actuator_for_heating': ("Schedule:Compact", "Schedule Value", "heating_setpoint"),
                    'availability_actuator': ("Schedule:Constant", "Schedule Value", "HVAC_OnOff"),
                },
            ),
            reward = RewardSpec(
                reward_fn = EnergyAndASHRAE55SimpleModel,
                reward_fn_config = {
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
        ),
    }
)

eprllib_config.episodes(
    episode_fn = task_cofiguration,
    episode_fn_config = episode_config
)

assert eprllib_config.agents_config is not None, "Agents configuration is not defined."

number_of_agents = len([keys for keys in eprllib_config.agents_config.keys()])
ray.init(_temp_dir='C:/Users/grhen/ray_results/tmp')
register_env(name="EPEnv", env_creator=lambda args: Environment(args))
env_config = eprllib_config.to_dict()

if not restore:
    algo = PPOConfig()
    algo.framework(
        framework = 'torch',
    )
    algo.environment(
        env = "EPEnv",
        env_config = env_config,
    )
    algo.fault_tolerance(
        restart_failed_env_runners = True,
    )
    algo.multi_agent(
        policies = {
            'single_policy': PolicySpec()
        },
        policy_mapping_fn = policy_map_fn,
        count_steps_by = "env_steps",
    )
    algo.evaluation(
        # evaluation_interval = 5,
        # evaluation_duration = 10,
        # evaluation_parallel_to_training = True,
        # evaluation_num_env_runners = 1,
        # evaluation_config = {
        #     "explore": False,
        #     "env_config": {
        #         "evaluation": True,
        #     },
        # }
    )
    algo.reporting(
        min_sample_timesteps_per_iteration = 50,
    )
    algo.checkpointing(
        export_native_model_files = True,
    )
    algo.debugging(
        log_level = "ERROR",
        seed = 1,
    )
    algo.api_stack(
        enable_rl_module_and_learner=False,
        enable_env_runner_and_connector_v2=False,
    )
    algo.resources(
        num_gpus = 0,
    )
    algo.training(
    
        # === General Algo Configs ===
        gamma = 0.99,#tune.grid_search([0.7,0.9,0.99]) if tuning else 0.8,#
        lr_schedule = [
            (0, 3e-4),
            (4e6, 1e-4),
            (8e6, 5e-5)
            ],#tune.grid_search([0.0001,0.001,0.01]) if tuning else 0.003,#
        train_batch_size = 100000,#tune.grid_search([100,1000,10000,100000]) if tuning else 12961*7,
        # Each episode has a lenght of 144*3+1=433. To train the model with batch_mode=episodes_complete and using the 8 processes in parallel
        # for each iteration it is possible to set train_batch_size_per_learner=433*8=3464
        minibatch_size = 20000,#tune.grid_search([5000,10000,20000]) if tuning else 12961*7,
        # We can separate the batch into 8 batches of 433 timesteps.
        num_epochs = 30,#tune.grid_search([5,10,15,30]) if tuning else 30,
        vf_share_layers = False,
        
        # === Policy Model configuration ===
        model = {
            # FC Hidden layers
            "fcnet_hiddens": [32,32],#tune.grid_search([[64,64],[128,128],[64,64,64]]) if tuning else [64,64],
            "fcnet_activation": "tanh",
            
            # # LSTM
            # "use_lstm": True,
            # "max_seq_len": 12,
            # "lstm_cell_size": 64,
            
            # # Attention layer
            # "use_attention": True,
            # "attention_num_transformer_units": 2,
            # "attention_dim": 64,
            # "attention_num_heads": 4,
            # "attention_head_dim": 32,
            # "attention_memory_inference": 50,
            # "attention_memory_training": 50,
            # "attention_position_wise_mlp_dim": 32,
            # "attention_init_gru_gate_bias": 2.0,
            # "attention_use_n_prev_actions": 0,
            # "attention_use_n_prev_rewards": 0,
            
            # # Post FC layers
            # "post_fcnet_hiddens": [64,64],
            # "post_fcnet_activation": "tanh",
            },
        
        # === PPO Configs ===
        use_critic = True,
        use_gae = True,
        lambda_ = 0.92,#tune.quniform(0.4, 0.9, 0.1) if tuning else 0.7,#
        use_kl_loss = True,
        kl_coeff = 0.2,#tune.quniform(0.1, 1, 0.1) if tuning else 0.2,#
        kl_target = 0.7,#tune.quniform(0.1, 0.9, 0.1) if tuning else 0.7,#
        shuffle_batch_per_epoch = True,
        vf_loss_coeff = 0.5,#tune.quniform(0.1, 1, 0.1) if tuning else 0.9,#
        entropy_coeff_schedule = [
            (0, 0.02),
            (4e6, 0.005),
            (8e6, 0.001)
            ],#tune.choice([0.,0.1,0.2]) if tuning else 0.01,#
        clip_param = 0.25,#tune.quniform(0.1, 0.3, 0.05) if tuning else 0.25,#
        vf_clip_param = 0.25,#tune.quniform(0.1, 0.3, 0.05) if tuning else 0.3,#
    )
    algo.env_runners(
    
        num_env_runners = 7, # for e-greedy
        # num_env_runners = 0, # for curiosity
        # Number of EnvRunner actors to create for parallel sampling. Setting this to 0 forces sampling 
        # to be done in the local EnvRunner (main process or the Algorithm’s actor when using Tune).
        
        num_envs_per_env_runner = 1, # EnergyPlus don't allow multiple env in the same runner.
        # Number of environments to step through (vector-wise) per EnvRunner. This enables batching when 
        # computing actions through RLModule inference, which can improve performance for inference-bottlenecked 
        # workloads.
        
        sample_timeout_s = 10000, # Default = 60.0
        # The timeout in seconds for calling sample() on remote EnvRunner workers. Results (episode list) from 
        # workers that take longer than this time are discarded. Only used by algorithms that sample synchronously 
        # in turn with their update step (e.g., PPO or DQN). Not relevant for any algos that sample asynchronously, 
        # such as APPO or IMPALA.
        
        create_env_on_local_worker = True,
        # When num_env_runners > 0, the driver (local_worker; worker-idx=0) does not need an environment. This is 
        # because it doesn’t have to sample (done by remote_workers; worker_indices > 0) nor evaluate (done by 
        # evaluation workers; see below).
        
        rollout_fragment_length = 'auto',
        # Divide episodes into fragments of this many steps each during sampling. Trajectories of this size are collected 
        # from EnvRunners and combined into a larger batch of train_batch_size for learning. For example, given 
        # rollout_fragment_length=100 and train_batch_size=1000: 
        # 1. RLlib collects 10 fragments of 100 steps each from rollout workers. 
        # 2. These fragments are concatenated and we perform an epoch of SGD.
        # When using multiple envs per worker, the fragment size is multiplied by num_envs_per_env_runner. This is since we are 
        # collecting steps from multiple envs in parallel. For example, if num_envs_per_env_runner=5, then EnvRunners return 
        # experiences in chunks of 5*100 = 500 steps. The dataflow here can vary per algorithm. For example, PPO further divides 
        # the train batch into minibatches for multi-epoch SGD. Set rollout_fragment_length to “auto” to have RLlib compute an 
        # exact value to match the given batch size.
        
        batch_mode = "complete_episodes", #"complete_episodes", "truncate_episodes"
        # How to build individual batches with the EnvRunner(s). Batches coming from distributed EnvRunners are usually 
        # concat’d to form the train batch. Note that “steps” below can mean different things (either env- or agent-steps) 
        # and depends on the count_steps_by setting, adjustable via AlgorithmConfig.multi_agent(count_steps_by=..): 
        # 1) “truncate_episodes”: Each call to EnvRunner.sample() returns a batch of at most 
        # rollout_fragment_length * num_envs_per_env_runner in size. The batch is exactly rollout_fragment_length * num_envs 
        # in size if postprocessing does not change batch sizes. Episodes may be truncated in order to meet this size requirement. 
        # This mode guarantees evenly sized batches, but increases variance as the future return must now be estimated at truncation 
        # boundaries. 
        # 2) “complete_episodes”: Each call to EnvRunner.sample() returns a batch of at least rollout_fragment_length * num_envs_per_env_runner 
        # in size. Episodes aren’t truncated, but multiple episodes may be packed within one batch to meet the (minimum) batch size. 
        # Note that when num_envs_per_env_runner > 1, episode steps are buffered until the episode completes, and hence batches may 
        # contain significant amounts of off-policy data.
        
        explore = True,
        # exploration_config = {
        #     "type": "Curiosity",  # <- Use the Curiosity module for exploring.
        #     "eta": 1.0,  # Weight for intrinsic rewards before being added to extrinsic ones.
        #     "lr": 0.00001,  # Learning rate of the curiosity (ICM) module.
        #     "feature_dim": 256,  # Dimensionality of the generated feature vectors.
        #     # Setup of the feature net (used to encode observations into feature (latent) vectors).
        #     "inverse_net_hiddens": [256,256],  # Hidden layers of the "inverse" model.
        #     "inverse_net_activation": "relu",  # Activation of the "inverse" model.
        #     "forward_net_hiddens": [256,256],  # Hidden layers of the "forward" model.
        #     "forward_net_activation": "relu",  # Activation of the "forward" model.
        #     "beta": 0.2,  # Weight for the "forward" loss (beta) over the "inverse" loss (1.0 - beta).
        #     # Specify, which exploration sub-type to use (usually, the algo's "default"
        #     # exploration, e.g. EpsilonGreedy for DQN, StochasticSampling for PG/SAC).
        #     "sub_exploration": {
        #         "type": "StochasticSampling",
        #     },
        # },
        # exploration_config = {
        #     "type": "EpsilonGreedy",
        #     "initial_epsilon": 1.,
        #     "final_epsilon": 0.01,
        #     "epsilon_timesteps": 1008 * 25000,
        #     # The timestep counted here are agents timesteps. This means, that the time of exploration 
        #     # is reduced when the number of agents increase.
        # },
        
        observation_filter = "MeanStdFilter",
        # Element-wise observation filter, either "NoFilter" or "MeanStdFilter".
    )
    

    tuner = tune.Tuner(
        "PPO",
        param_space = algo.to_dict(),
        tune_config=tune.TuneConfig(
            mode = "max",
            metric = "env_runners/episode_reward_mean",
            num_samples = 1,
            # This is necesary to iterative execute the search_alg to improve the hyperparameters
            reuse_actors = False,
            trial_name_creator = lambda trial: "{}_{}_{}".format(name, trial.trainable_name, trial.trial_id),
            trial_dirname_creator = lambda trial: "{}_{}_{}".format(name, trial.trainable_name, trial.trial_id),
            
            # == Search algorithm configuration ==
            # search_alg = Repeater(HyperOptSearch(),repeat=10),
            # search_alg = HyperOptSearch(),
            
            # == Scheduler algorithm configuration ==
            # scheduler = ASHAScheduler(
            #     time_attr = 'info/num_env_steps_trained',
            #     max_t= 400000,
            #     grace_period = 200000,
            # ),
        ),
        run_config=air.RunConfig(
            name = "{date}_{name}_{algorithm}".format(
                date = time.strftime("%Y%m%d%H%M%S"),
                name = name,
                algorithm = "PPO",
            ),
            storage_path = f'C:/Users/grhen/ray_results/{experiment_name}',
            stop = {"info/num_env_steps_trained": 1008 * 10000},
            log_to_file = True,
            
            checkpoint_config=air.CheckpointConfig(
                checkpoint_at_end = True,
                checkpoint_frequency = 20,
                num_to_keep = 20,
            ),
            failure_config=air.FailureConfig(
                max_failures = 100,
                # Tries to recover a run up to this many times.
            ),
        ),
    )
    tuner.fit()

else:
    tuner = tune.Tuner.restore(checkpoint_path, 'PPO')
    tuner.fit()
    