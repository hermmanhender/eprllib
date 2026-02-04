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
from eprllib.Agents.Rewards.NygardFerguson1990 import NygardFerguson1990, NygardFerguson1990_comfort
from eprllib.examples.example_thermostat_files.episode_fn import task2_cofiguration
from eprllib.examples.example_thermostat_files.policy_mapping import policy_map_fn

with open("src/eprllib/examples/example_thermostat_files/episode_fn_config_I.json", "r") as f:
    episode_config = json.load(f)

experiment_name:str = "07-CTD-I"
name:str = "NF2"

checkpoint_path:str = "C:/Users/grhen/ray_results/07-CTD-I/20250909235242_Nygard-Ferguson-1990_PPO/Nygard-Ferguson-1990_PPO_4fc1e_00000/checkpoint_000005"

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
                    # "Electricity:Building",
                    "Heating:DistrictHeatingWater",
                    "Cooling:DistrictCooling",
                ],
                use_actuator_state = True,
                use_one_day_weather_prediction = True,
                weather_prediction_hours = 3,
                prediction_variables = {
                    'outdoor_dry_bulb': True,
                },
                # internal_variables = [
                #     ("Zone Floor Area", "Thermal Zone"),
                # ],
                history_len=6,
                # user_occupation_funtion = True,
                # user_type = "Typical family, office job",
                # zone_type = "daytime",
                # occupation_schedule = ("Schedule:Constant", "Schedule Value", "occupancy_schedule"),
                # user_occupation_forecast = False,
                # occupation_prediction_hours = 3,
                # summer_months = [11, 12, 1, 2]
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
    episode_fn = task2_cofiguration,
    episode_fn_config = episode_config,
    cut_episode_len=1
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
        'single_policy': PolicySpec()
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
algo.training(

    # === General Algo Configs ===
    gamma = 0.99,
    lr = 5e-5,
    train_batch_size = 144*31*8,
    minibatch_size = 144,
    num_epochs = 30,
    vf_share_layers = False,
    
    # === Policy Model configuration ===
    model = {
        "custom_model": "custom_transformer",
        "custom_model_config": {
            "num_encoder_layers": 4,
            "nhead": 8,
            "d_model": 128,
            "dim_feedforward": 512,
            "dropout": 0.1
        },
    },
    
    # === PPO Configs ===
    use_critic = True,
    use_gae = True,
    lambda_ = 0.92,
    use_kl_loss = True,
    kl_coeff = 0.2,
    kl_target = 0.7,
    shuffle_batch_per_epoch = True,
    vf_loss_coeff = 0.5,
    entropy_coeff_schedule = [
        (0, 0.02),
        (1.5e6, 0.005),
        (3e6, 0.001)
        ],
    clip_param = 0.15,
    vf_clip_param = 0.15,
)

# Constuct the new algorithm
print("Building the new algorithm...")
new_algo = algo.build()
print("New algorithm built.")

# Set the policy weights to the new algorithm.
print("Setting the policy weights to the new algorithm...")
new_algo.remove_policy('single_policy')
new_algo.add_policy('single_policy', policy=old_policy)
print("Policy weights set.")


# Training iterations
print("Starting training iterations...")
start = time.time()
for iteration in range(1, 100):
    print(f"================ Iteration {iteration} ==================")
    pprint(new_algo.train())
    
    if iteration % 5 == 0:
        checkpoint = new_algo.save()
        print(f"Checkpoint saved at {checkpoint}")
        end = time.time()
        print(f'Training time: {end - start} seconds.')

# Finish training
end = time.time()
print(f'Training time: {end - start} seconds.')
# new_algo.stop()
# print("Algorithm stopped.")
# print("Shutting down Ray...")
ray.shutdown()
print("Ray shut down.")
print("Training complete.")
