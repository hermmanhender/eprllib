"""
Example with Central Agent to control a mix-ventilated building
================================================================

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
    ActionMapperSpec,
    FilterSpec
)

from eprllib.Agents.Filters.DefaultFilter import DefaultFilter

from eprllib.examples.example_central_agent_files.action_mapper import CentralAgentActionMapper
from eprllib.examples.example_central_agent_files.reward_function import IAQThermalComfortEnergyReward
from eprllib.examples.example_central_agent_files.episode_fn import task_cofiguration
from eprllib.examples.example_central_agent_files.policy_mapping import policy_map_fn

# read the json config file as a dict.
with open("src/eprllib/examples/example_central_agent_files/episode_fn_config.json", "r") as f:
    episode_config = json.load(f)

experiment_name = "example_central_agent"
name = "dnn_simple2"
tuning = False
restore = False
checkpoint_path = ""

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
        "Central Agent": AgentSpec(
            observation = ObservationSpec(
                variables = [
                    ("Site Outdoor Air Drybulb Temperature", "Environment"),
                    ("Site Wind Speed", "Environment"),
                    ("Site Outdoor Air Relative Humidity", "Environment"),
                    ("Zone Mean Air Temperature", "Thermal Zone"),
                    ("Zone Air Relative Humidity", "Thermal Zone"),
                    ("Zone People Occupant Count", "Thermal Zone"),
                    ("Zone Air CO2 Concentration", "Thermal Zone"),
                    ("Fan Electricity Energy", "ExhaustFan"),
                ],
                simulation_parameters = {
                    'today_weather_horizontal_ir_at_time': True,
                },
                meters = [
                    "Electricity:Building",
                    "Cooling:DistrictCooling",
                    "Heating:DistrictHeatingWater"
                ],
                use_actuator_state = True,
                use_one_day_weather_prediction = True,
                prediction_hours = 3,
                prediction_variables = {
                    'outdoor_dry_bulb': True,
                },
                # history_len = 6,
            ),
            action = ActionSpec(
                actuators = [
                    ("AirFlow Network Window/Door Opening", "Venting Opening Factor", "window_east"),
                    ("Schedule:Constant", "Schedule Value", "ventilation_factor"),
                    ("Schedule:Constant", "Schedule Value", "HVAC_OnOff"),
                ],
            ),
            filter= FilterSpec(
                filter_fn = DefaultFilter,
                filter_fn_config = {},
            ),
            action_mapper= ActionMapperSpec(
                action_mapper = CentralAgentActionMapper,
                action_mapper_config = {
                    'window_actuator': ("AirFlow Network Window/Door Opening", "Venting Opening Factor", "window_east"),
                    'exhaust_fan_actuator': ("Schedule:Constant", "Schedule Value", "ventilation_factor"),
                    'hvac_availability_actuator': ("Schedule:Constant", "Schedule Value", "HVAC_OnOff"),
                },
            ),
            reward = RewardSpec(
                reward_fn = IAQThermalComfortEnergyReward,
                reward_fn_config = {
                    "thermal_zone": "Thermal Zone",
                    "co2_threshold": 800.0,
                    "exhaust_fan_name": "ExhaustFan",
                    'cooling_name': "Cooling:DistrictCooling",
                    'heating_name': "Heating:DistrictHeatingWater",
                    'cooling_energy_ref': 7100*(10*60), # From W to J in a timestep ot 10 minutes: P*t
                    'heating_energy_ref': 8000*(10*60), # From W to J in a timestep ot 10 minutes: P*t
                    "beta": 0.001,
                    "rho_1": 0.001,  # Weight for thermal comfort
                    "rho_2": 0.999,  # Weight for indoor air quality
                },
            ),
        ),
    }
)

eprllib_config.episodes(
    episode_fn = task_cofiguration,
    # read the json config file as a dict.
    episode_fn_config = episode_config,
)

assert eprllib_config.agents_config is not None, "Agents configuration is not defined."

number_of_agents = len([keys for keys in eprllib_config.agents_config.keys()])
ray.init(_temp_dir='C:/Users/grhen/ray_results/tmp')
register_env(name="EPEnv", env_creator=lambda args: Environment(args))
env_config = eprllib_config.to_dict()

if not restore:
    algo = PPOConfig()
    algo.training(
        
        # === General Algo Configs ===
        gamma = 0.99,
        lr_schedule = [
                (0, 3e-4),
                (2e6, 1e-4),
                (4e6, 5e-5)
                ],
        train_batch_size = 50000,
        minibatch_size = 2000,
        num_epochs = 30,
        
        # === Policy Model configuration ===
        model = {
            # FC Hidden layers
            "fcnet_hiddens": [128,128,128],
            "fcnet_activation": "tanh",
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
                (2e6, 0.005),
                (4e6, 0.001)
                ],
        clip_param = 0.25,
        vf_clip_param = 0.25,
    )
    algo.learners(
        num_learners = 0,
        num_cpus_per_learner = 1,
    )
    algo.environment(
        env = "EPEnv",
        env_config = env_config,
    )
    algo.framework(
        framework = 'torch',
    )
    algo.fault_tolerance(
        restart_failed_env_runners = True,
    )
    algo.env_runners(
        num_env_runners = 7,
        num_envs_per_env_runner = 1,
        sample_timeout_s = 100000,
        create_env_on_local_worker = True,
        rollout_fragment_length = 'auto',
        batch_mode = "complete_episodes",    
        explore = True,        
        observation_filter = "MeanStdFilter",
    )
    algo.multi_agent(
        policies = {
            'single_policy': PolicySpec(),
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
    algo.resources(
        num_gpus = 0,
    )
    algo.api_stack(
    enable_rl_module_and_learner=False,
    enable_env_runner_and_connector_v2=False,
)

    tuning = tune.Tuner(
        "PPO",
        param_space = algo.to_dict(),
        tune_config=tune.TuneConfig(
            num_samples = 1,
            reuse_actors = False,
            trial_name_creator = lambda trial: "{}_{}_{}".format(name, trial.trainable_name, trial.trial_id),
            trial_dirname_creator = lambda trial: "{}_{}_{}".format(name, trial.trainable_name, trial.trial_id),
        ),
        run_config=air.RunConfig(
            name = "{date}_{name}_{algorithm}".format(
                date = time.strftime("%Y%m%d%H%M%S"),
                name = name,
                algorithm = "PPO",
            ),
            storage_path = f'C:/Users/grhen/ray_results/{experiment_name}',
            stop = {"info/num_env_steps_trained": 1008*200000},
            log_to_file = True,
            
            checkpoint_config=air.CheckpointConfig(
                checkpoint_at_end = True,
                checkpoint_frequency = 5,
            ),
            failure_config=air.FailureConfig(
                max_failures = 100,
            ),
        ),
    )
    tuning.fit()

else:
    tuning = tune.Tuner.restore(checkpoint_path, 'PPO')
    tuning.fit()
    