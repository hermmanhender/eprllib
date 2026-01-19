"""

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
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
from ray.rllib.models import ModelCatalog
from eprllib.experiments.tesis.configurations.policy_model import CustomTransformerModel
ModelCatalog.register_custom_model("custom_transformer", CustomTransformerModel)

from eprllib.experiments.files.rl_module_transformer import TransformerRLModule

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
# from eprllib.Agents.Triggers.SetpointTriggers import AvailabilityTrigger
from eprllib.Agents.Triggers.AirMassFlowRateTriggers import AirMassFlowRateTrigger
from eprllib.Agents.Triggers.SetpointTriggers import DualSetpointTriggerDiscreteAndAvailabilityTrigger_v2, DualSetpointTriggerContinuosAndAvailabilityTrigger_v2

from eprllib.Agents.Rewards.EnergyAndAshrae55SimpleModel import EnergyAndASHRAE55SimpleModel, EnergyAndASHRAE55SimpleModelEnded
from eprllib.Agents.Rewards.ASHRAE55SimpleModel import ASHRAE55SimpleModelEnded, ASHRAE55SimpleModel
from eprllib.Agents.Rewards.NygardFerguson1990 import NygardFerguson1990
from eprllib.Agents.Rewards.CEN15251 import CEN15251

from eprllib.Agents.Rewards.Coraci2021 import Coraci2021
from eprllib.Agents.Filters.CoraciObsSpace import CoraciObsSpaceFilter
from eprllib.AgentsConnectors.Coraci2021Connector import Coraci2021Connector

from eprllib.experiments.tesis.configurations.curriculum_learning import episode_fn
from eprllib.experiments.tesis.configurations.policy_mapping import policy_map_fn
# from eprllib.experiments.tesis.callbacks import ActionDistributionCallback

with open("src/eprllib/experiments/tesis/configurations/curriculum_task1.json", "r") as f:
    episode_config = json.load(f)

experiment_name:str = "tesis"
name:str = "task1"

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
    connector_fn = Coraci2021Connector,
    connector_fn_config = {},
)
eprllib_config.agents(
    agents_config = {
        "HVAC": AgentSpec(
            observation = ObservationSpec(
                variables = [
                    ("Site Outdoor Air Drybulb Temperature", "Environment"),
                    # ("Site Outdoor Air Relative Humidity", "Environment"),
                    ("Zone Mean Air Temperature", "Thermal Zone"),
                    # ("Zone Air Relative Humidity", "Thermal Zone"),
                    ("Zone People Occupant Count", "Thermal Zone"),
                    # ("Zone Thermal Comfort Fanger Model PMV", "People"),
                    # ("Zone Operative Temperature", "Thermal Zone"),
                    # ("Zone Air Humidity Ratio", "Thermal Zone"),
                    # ("Cooling Coil Total Cooling Energy", "HeatPump Cooling Coil"),
                    # ("Heating Coil Heating Energy", "HeatPump HP Heating Coil"),
                    # ("Heating Coil Heating Energy", "HeatPump Sup Heat Coil"),
                    # ("Zone Thermal Comfort CEN 15251 Adaptive Model Category I Status", "People"),
                    # ("Zone Thermal Comfort CEN 15251 Adaptive Model Category II Status", "People"),
                    # ("Zone Thermal Comfort CEN 15251 Adaptive Model Category III Status", "People"),
                ],
                simulation_parameters = {
                    'hour': True,
                #     'day_of_week': True,
                #     'minutes': True,
                #     'day_of_year': True,
                #     # 'today_weather_horizontal_ir_at_time': True,
                },
                meters = [
                    # "Electricity:Building",
                    "Heating:DistrictHeatingWater",
                    "Cooling:DistrictCooling",
                ],
                use_actuator_state = True,
                use_one_day_weather_prediction = True,
                weather_prediction_hours = 12,
                prediction_variables = {
                    'outdoor_dry_bulb': True,
                },
                internal_variables = [
                    ("Zone Floor Area", "Thermal Zone"),
                ],
                # history_len=4,
                user_occupation_function = False,
                user_type = "Office schedule",
                zone_type = "daytime",
                probability_variation = -1.,
                probability_variation_evening_night_hours = -1.,
                occupation_schedule = ("Schedule:Constant", "Schedule Value", "occupancy_schedule"),
                user_occupation_forecast = False,
                occupation_prediction_hours = 3,
                summer_months = [11, 12, 1, 2],
                # other_obs = {
                #     "war_north": 0,  # Window-to-wall ratio north
                #     "war_south": 0,  # Window-to-wall ratio south
                #     "war_east": 0,   # Window-to-wall ratio east
                #     "war_west": 0,   # Window-to-wall ratio west
                #     "latitude": 0,  # Latitude
                #     "longitude": 0,  # Longitude
                #     # "elevation": 0,  # Elevation
                #     # "time_zone": 0,  # Time zone
                # }
            ),
            # action = ActionSpec(
            #     actuators = [
            #         ("Ideal Loads Air System", "Air Mass Flow Rate", "IdealHVAC"),
            #     ],
            # ),
            action = ActionSpec(
                actuators = [
                    ("Schedule:Compact", "Schedule Value", "heating_setpoint"),
                    ("Schedule:Compact", "Schedule Value", "cooling_setpoint"),
                    ("Schedule:Constant", "Schedule Value", "HVAC_OnOff"),
                    # ("Ideal Loads Air System", "Air Mass Flow Rate", "IdealHVAC"),
                    # ("Ideal Loads Air System", "Air Temperature", "IdealHVAC"),
                ],
            ),
            filter= FilterSpec(
                filter_fn = CoraciObsSpaceFilter,
                filter_fn_config = {},
            ),
            
            # trigger= TriggerSpec(
            #     trigger_fn = DualSetpointTriggerContinuosAndAvailabilityTrigger_v2,
            #     trigger_fn_config = {
            #         'temperature_range': (18, 25),
            #         'deadband': 2,
            #         'actuator_for_cooling': ("Schedule:Compact", "Schedule Value", "cooling_setpoint"),
            #         'actuator_for_heating': ("Schedule:Compact", "Schedule Value", "heating_setpoint"),
            #         'availability_actuator': ("Schedule:Constant", "Schedule Value", "HVAC_OnOff"),
            #         # 'actions_file_path': "C:/Users/grhen/Documents/acciones.csv"
            #     },
            # ),
            
            trigger= TriggerSpec(
                trigger_fn = DualSetpointTriggerDiscreteAndAvailabilityTrigger_v2,
                trigger_fn_config = {
                    'temperature_range': (18, 25),
                    "action_space_dim": 21,
                    'actuator_for_cooling': ("Schedule:Compact", "Schedule Value", "cooling_setpoint"),
                    'actuator_for_heating': ("Schedule:Compact", "Schedule Value", "heating_setpoint"),
                    'availability_actuator': ("Schedule:Constant", "Schedule Value", "HVAC_OnOff"),
                },
            ),
            
            reward = RewardSpec(
                reward_fn = Coraci2021,
                reward_fn_config = {
                    "thermal_zone": "Thermal Zone",
                    # 'people_name': "People",
                    "t_low": 20.0,
                    "t_high": 24.0,
                    "timesteptoreward": 1,
                    'beta': 0.97,
                    'cooling_name': "Cooling:DistrictCooling",
                    'heating_name': "Heating:DistrictHeatingWater"
                },
            ),
        ),
    }
)

eprllib_config.episodes(
    episode_fn = episode_fn,
    episode_fn_config = episode_config,
    # cut_episode_len = 289
)

assert eprllib_config.agents_config is not None, "Agents configuration is not defined."

number_of_agents = len([keys for keys in eprllib_config.agents_config.keys()])
ray.init(_temp_dir='C:/Users/grhen/ray_results/tmp')
register_env(name="EPEnv", env_creator=lambda args: Environment(args))
env_config = eprllib_config.to_dict()


algo = PPOConfig()
algo.framework(
    framework = 'torch',
)
algo.learners(
    num_learners=0
)
algo.environment(
    env = "EPEnv",
    env_config = env_config,
    clip_actions = True,
)
algo.fault_tolerance(
    restart_failed_env_runners = True,
)
algo.multi_agent(
    policies = {
        'single_policy': PolicySpec(),
        # 'always_off_policy': PolicySpec(None,None,None,{deterministic_action:0}),
    },
    policy_mapping_fn = policy_map_fn,
    count_steps_by = "env_steps",
)
# algo.evaluation(
#     evaluation_interval = 2,
#     evaluation_duration = 1,
#     evaluation_parallel_to_training = True,
#     evaluation_num_env_runners = 1,
#     evaluation_config = {
#         "explore": False,
#         "env_config": {
#             "evaluation": True,
#         },
#     }
# )
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
# algo.api_stack(
#     enable_rl_module_and_learner=False,
#     enable_env_runner_and_connector_v2=False,
# )
algo.resources(
    num_gpus = 0,
)
algo.rl_module(
    model_config = DefaultModelConfig(
    #     # FC Hidden layers
        fcnet_hiddens= [256,256,64,64],#tune.grid_search([[64,64],[128,128],[64,64,64]]) if tuning else [64,64],
        fcnet_activation= "relu",
    ),

    # Specify our custom class
    # rl_module_spec=RLModuleSpec(
    #     module_class=TransformerRLModule,
    #     model_config={
    #         "d_model": 64,
    #         "nhead": 2,
    #         "num_encoder_layers": 2,
    #         "dim_feedforward": 128,
    #         "dropout": 0.0, # Disable dropout for this simple test
    #     }
    # )
    
)
algo.training(

    # === General Algo Configs ===
    gamma = 0.90,
    lr = 1e-4,
    # lr_schedule = [
    #     (0, 1e-2),
    #     (2e6, 1e-3),
    #     (4e6, 1e-4)
    #     ],
    # Aumentar el lote de entrenamiento permite disminuir la varianza del gradiente.
    # Esto es especialmente importante cuando se utilizan modelos más complejos, como transformers.
    train_batch_size_per_learner = 289*7, # 289 timesteps per episode (two day plus timestep 0), 7 episodes.
    # Un valor pequeño de minibatch_size puede hacer que el entrenamiento sea más ruidoso, pero también puede ayudar a escapar de mínimos locales.
    # Un valor más grande puede hacer que el entrenamiento sea más estable, pero también puede hacer que el entrenamiento sea más lento.
    minibatch_size = 289*2,
    num_epochs = 7,
    # vf_share_layers = False,
    
    # === Policy Model configuration ===
    # model = {
    #     "custom_model": "custom_transformer",
    #     "custom_model_config": {
    #         "num_encoder_layers": 4,
    #         "nhead": 8,
    #         "d_model": 64,
    #         "dim_feedforward": 4*64,
    #         "dropout": 0.1
    #     },
    # },
    
    
    # === PPO Configs ===
    use_critic = True,
    use_gae = True,
    lambda_ = 0.95, #tune.quniform(0.8, 1.0, 0.05),
    use_kl_loss = True,
    kl_coeff = 0.2,
    kl_target = 0.7,
    shuffle_batch_per_epoch = True,
    vf_loss_coeff = 0.25,
    entropy_coeff = 0.01, #tune.quniform(0.01, 0.1, 0.01),
    # entropy_coeff = [
    #     (0, 0.05),
    #     (5e4, 0.05),
    #     (2e5, 0.01)
    #     ],
    clip_param = 0.2, #tune.quniform(0.1, 0.3, 0.05),
    vf_clip_param = 0.2,#0.25, #tune.quniform(0.1, 0.3, 0.05),
)
algo.env_runners(

    num_env_runners = 7,
    num_envs_per_env_runner = 1,
    sample_timeout_s = 3000000,
    rollout_fragment_length = 'auto',
    batch_mode = "complete_episodes", #"truncate_episodes", "complete_episodes",
    explore = True,
    # observation_filter = "MeanStdFilter",
)
# algo.callbacks(
#     on_train_result=(
#         lambda episode, **kw: print(f"Iteration ended. Action Mean={np.mean(np.array(episode.get_actions()), axis=0)} and Action Std={np.std(np.array(episode.get_actions()), axis=0)}")
#     )
# )
    

tuner = tune.Tuner(
    "PPO",
    param_space = algo.to_dict(),
    tune_config=tune.TuneConfig(
        mode = "max",
        metric = "env_runners/episode_return_mean",
        num_samples = 1,
        # This is necesary to iterative execute the search_alg to improve the hyperparameters
        reuse_actors = False,
        trial_name_creator = lambda trial: "{}-{}".format(trial.trainable_name, trial.trial_id),
        trial_dirname_creator = lambda trial: "{}-{}".format(trial.trainable_name, trial.trial_id),
    ),
    run_config=air.RunConfig(
        name = "{date}-{name}".format(
            date = time.strftime("%Y%m%d%H%M%S"),
            name = name,
        ),
        storage_path = f'C:/Users/grhen/ray_results/{experiment_name}',
        stop = {
            "num_env_steps_sampled_lifetime": 1000000,
            # "learners/single_policy/entropy": 10,
            },
        log_to_file = True,
        
        checkpoint_config=air.CheckpointConfig(
            checkpoint_at_end = True,
            checkpoint_frequency = 5,
            # num_to_keep = 20,
        ),
        failure_config=air.FailureConfig(
            max_failures = 100,
            # Tries to recover a run up to this many times.
        ),
    ),
)
tuner.fit()
