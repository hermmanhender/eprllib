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
from ray.rllib.algorithms.algorithm import Algorithm
from pprint import pprint

from ray.rllib.models import ModelCatalog
from eprllib.examples.example_central_agent_files.policy_model import CustomTransformerModel
ModelCatalog.register_custom_model("custom_transformer", CustomTransformerModel)

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

# from eprllib.Agents.Rewards.EnergyAndAshrae55SimpleModel import EnergyAndASHRAE55SimpleModel
from eprllib.Agents.Rewards.NygardFerguson1990 import NygardFerguson1990
from eprllib.examples.example_thermostat_files.curriculum_learning import episode_fn
from eprllib.examples.example_thermostat_files.policy_mapping import policy_map_fn

with open("src/eprllib/examples/example_thermostat_files/curriculum_task3.json", "r") as f:
    episode_config = json.load(f)

experiment_name: str = "general"
name: str = "task3"
checkpoint_path: str = "C:/Users/grhen/ray_results/general/20250929191319-task2/PPO-98ee5_00000/checkpoint_000025"

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
                    # ("Site Outdoor Air Relative Humidity", "Environment"),
                    ("Zone Mean Air Temperature", "Thermal Zone"),
                    # ("Zone Air Relative Humidity", "Thermal Zone"),
                    ("Zone People Occupant Count", "Thermal Zone"),
                    ("Zone Thermal Comfort Fanger Model PMV", "People")
                ],
                simulation_parameters = {
                    'hour': True,
                    'day_of_week': True,
                    'minutes': True,
                    'day_of_year': True,
                    # 'today_weather_horizontal_ir_at_time': True,
                },
                meters = [
                    "Electricity:Building",
                    "Heating:DistrictHeatingWater",
                    "Cooling:DistrictCooling",
                ],
                use_actuator_state = True,
                use_one_day_weather_prediction = True,
                weather_prediction_hours = 3,
                prediction_variables = {
                    'outdoor_dry_bulb': True,
                },
                internal_variables = [
                    ("Zone Floor Area", "Thermal Zone"),
                ],
                history_len=3,
                user_occupation_funtion = True,
                user_type = "Typical family, office job",
                zone_type = "daytime",
                occupation_schedule = ("Schedule:Constant", "Schedule Value", "occupancy_schedule"),
                user_occupation_forecast = True,
                occupation_prediction_hours = 3,
                summer_months = [11, 12, 1, 2],
                other_obs = {
                    "war_north": 0,  # Window-to-wall ratio north
                    "war_south": 0,  # Window-to-wall ratio south
                    "war_east": 0,   # Window-to-wall ratio east
                    "war_west": 0,   # Window-to-wall ratio west
                    "latitude": 0,  # Latitude
                    "longitude": 0,  # Longitude
                    # "elevation": 0,  # Elevation
                    # "time_zone": 0,  # Time zone
                }
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
                    'temperature_range': (17, 28),
                    "band_gap_range_len": 3,
                    "action_space_dim": 10,
                    'actuator_for_cooling': ("Schedule:Compact", "Schedule Value", "cooling_setpoint"),
                    'actuator_for_heating': ("Schedule:Compact", "Schedule Value", "heating_setpoint"),
                    'availability_actuator': ("Schedule:Constant", "Schedule Value", "HVAC_OnOff"),
                },
            ),
            reward = RewardSpec(
                reward_fn = NygardFerguson1990,
                reward_fn_config = {
                    "thermal_zone": "Thermal Zone",
                    "C2": 6000,
                    'people_name': "People",
                    'cooling_name': "Cooling:DistrictCooling",
                    'heating_name': "Heating:DistrictHeatingWater",
                    'cooling_energy_ref': 0,
                    'heating_energy_ref': 0,
                },
            ),
        ),
    }
)

eprllib_config.episodes(
    episode_fn = episode_fn,
    episode_fn_config = episode_config,
    cut_episode_len = 1
)

assert eprllib_config.agents_config is not None, "Agents configuration is not defined."

number_of_agents = len([keys for keys in eprllib_config.agents_config.keys()])
ray.init(_temp_dir='C:/Users/grhen/ray_results/tmp')
register_env(name="EPEnv", env_creator=lambda args: Environment(args))
env_config = eprllib_config.to_dict()

# Restore the algorithm from checkpoint
print(f"Restoring the old algorithm from checkpoint: {checkpoint_path} ...")
old_algo = Algorithm.from_checkpoint(checkpoint_path)
print("Old algorithm restored.")
# Get the policy from the old algorithm
print("Getting the policy from the old algorithm...")
old_policy = old_algo.get_policy('single_policy')
print("Policy obtained.")
print("Stopping the old algorithm...")
old_algo.stop()
print("Old algorithm stopped.")

old_state = old_policy.get_state()
print("Old policy state obtained.")
serialized_pol_spec = old_state.get("policy_spec")
if serialized_pol_spec is None:
    raise ValueError(
        "No `policy_spec` key was found in given `state`! "
        "Cannot create new Policy."
    )
pol_spec = PolicySpec.deserialize(serialized_pol_spec)

# New algorithm
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
        'single_policy': pol_spec
    },
    policy_mapping_fn = policy_map_fn,
    count_steps_by = "env_steps",
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
    gamma = 0.8,
    lr = 1e-6,
    train_batch_size = 144*7*10, # 144 timesteps per episode (one day), 64 episodes.
    minibatch_size = 144,
    num_epochs = 30,
    vf_share_layers = False,
    
    # === Policy Model configuration ===
    model = {
        "custom_model": "custom_transformer",
        "custom_model_config": {
            "num_encoder_layers": 8,
            "nhead": 10,
            "d_model": 256,
            "dim_feedforward": 4*256,
            "dropout": 0.2
        },
    },
    # model = {
    # #     # FC Hidden layers
    #     "fcnet_hiddens": [256,256,256],#tune.grid_search([[64,64],[128,128],[64,64,64]]) if tuning else [64,64],
    #     "fcnet_activation": "tanh",
    # },
    
    # === PPO Configs ===
    use_critic = True,
    use_gae = True,
    lambda_ = 0.92,
    use_kl_loss = True,
    kl_coeff = 0.2,
    kl_target = 0.7,
    shuffle_batch_per_epoch = True,
    vf_loss_coeff = 0.5,
    entropy_coeff = 0,
    clip_param = 0.05,#tune.quniform(0.1, 0.3, 0.05) if tuning else 0.25,#
    vf_clip_param = 0.05,#tune.quniform(0.1, 0.3, 0.05) if tuning else 0.3,#
)
algo.env_runners(

    num_env_runners = 7,
    num_envs_per_env_runner = 1,
    sample_timeout_s = 10000,
    create_env_on_local_worker = True,
    rollout_fragment_length = 'auto',
    batch_mode = "complete_episodes",
    explore = True,
    observation_filter = "MeanStdFilter",
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
        trial_name_creator = lambda trial: "{}-{}".format(trial.trainable_name, trial.trial_id),
        trial_dirname_creator = lambda trial: "{}-{}".format(trial.trainable_name, trial.trial_id),
    ),
    run_config=air.RunConfig(
        name = "{date}-{name}".format(
            date = time.strftime("%Y%m%d%H%M%S"),
            name = name,
        ),
        storage_path = f'C:/Users/grhen/ray_results/{experiment_name}',
        stop = {"info/num_env_steps_trained": 2500000},
        log_to_file = True,
        
        checkpoint_config=air.CheckpointConfig(
            checkpoint_at_end = True,
            checkpoint_frequency = 10,
            num_to_keep = 20,
        ),
        failure_config=air.FailureConfig(
            max_failures = 100,
            # Tries to recover a run up to this many times.
        ),
    ),
)
tuner.fit()
