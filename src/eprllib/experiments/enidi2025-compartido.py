"""
Entrenamiento generalizador
============================

Se definen las siguientes propiedades para definir los escenarios de base:

+----------------------+----------------------+-------------------------+-----------------------------+
|Masa térmica interior |Aislación exterior    |Superficies vidriadas    |Proporción de area vidriada  |
+----------------------+----------------------+-------------------------+-----------------------------+
|Alta, Media, Baja     |Alta, Media, Baja     |Norte, Este, Sur, Oeste  |Alta, Media, Baja            |
+----------------------+----------------------+-------------------------+-----------------------------+

Adicionalmente, se establece que cada episodio tenga una longitud de 7 días, donde se pueden apreciar los
fenómenos de inercia térmica y se aumenta la cantidad de episodios considerablemente para la generalización
del problema.

Los climas utilizados corresponden a una misma región climática dentro de la Provincia de Mendoza.

La evaluación de las políticas para los diferentes agentes se realizan con métricas energéticas y de violación
de temperaturas en la vivienda bioclimática del IPV para Mendoza (prototipo 3). Esta vivienda se encuentra
diseñada para diferentes orientaciones, por lo que se comparan las estrategias seguidas en cada caso.

Los resultados son las bases para un manual del usuario.

Arquitectura: [128,128,64,64]

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
from ray.rllib.models import ModelCatalog

from eprllib.Environment.Environment import Environment
from eprllib.Environment.EnvironmentConfig import EnvironmentConfig
from eprllib.AgentsConnectors.FullySharedParametersConnector import FullySharedParametersConnector
from eprllib.Agents.AgentSpec import (
    AgentSpec,
    ObservationSpec,
    RewardSpec,
    ActionSpec,
    TriggerSpec,
    FilterSpec
)
from eprllib.Agents.Filters.FullySharedParametersFilter import FullySharedParametersFilter
from eprllib.Agents.Triggers.WindowsOpeningTriggers import WindowsOpeningTrigger
from eprllib.Agents.Triggers.WindowsShadingTriggers import WindowsShadingTrigger
from eprllib.Agents.Rewards.ASHRAE55SimpleModel import ASHRAE55SimpleModel

from eprllib.experiments.files.episode_function import task_cofiguration
from eprllib.experiments.files.observation_function import task_policy_map_fn
from eprllib.experiments.files.policy_model import CustomTransformerModel

# read the json config file as a dict.
with open("C:/Users/grhen/Documents/GitHub2/eprllib/src/eprllib/experiments/files/episode_function_config.json", "r") as f:
    episode_config = json.load(f)

experiment_name = "training_general"
name = "HVAC_policy"
tuning = False
restore = False
checkpoint_path = "C:/Users/grhen/ray_results/training_general/20250525232639_random_buildings_PPO"

eprllib_config = EnvironmentConfig()
eprllib_config.generals(
    epjson_path = "C:/Users/grhen/OneDrive - docentes.frm.utn.edu.ar/01-Desarrollo del Doctorado/03-Congresos y reuniones/03 - eprllib/Study Cases/Task 1/model-00000000-25772.epJSON",
    epw_path = "C:/Users/grhen/OneDrive - docentes.frm.utn.edu.ar/01-Desarrollo del Doctorado/03-Congresos y reuniones/03 - eprllib/Weather analysis/Chacras_de_Coria_Mendoza_ARG-hour.epw",
    output_path = TemporaryDirectory("output","",'C:/Users/grhen/Documents/Resultados_RLforEP').name,
    ep_terminal_output = False,
    timeout = 10,
    evaluation = False,
)
eprllib_config.connector(
    connector_fn = FullySharedParametersConnector,
    connector_fn_config = {
        'number_of_agents_total': 8,
        "number_of_actuators_total": 8,
    },
)
eprllib_config.agents(
    agents_config = {
        "North Windows": AgentSpec(
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
                ],
                use_actuator_state = True,
                use_one_day_weather_prediction = True,
                prediction_hours = 3,
                prediction_variables = {
                    'outdoor_dry_bulb': True,
                },
                internal_variables = [
                    # ("Zone Air Volume", "Thermal Zone"),
                    ("Zone Floor Area", "Thermal Zone"),
                ],
                other_obs = {
                    "WWR-North": 0.0, # WWR: Window-Wall Ratio
                    "WWR-South": 0.0,
                    "WWR-East": 0.0,
                    "WWR-West": 0.0,
                },
                history_len = 6,
            ),
            action = ActionSpec(
                actuators = [
                    ("AirFlow Network Window/Door Opening", "Venting Opening Factor", "window_north"),
                ],
            ),
            filter= FilterSpec(
                filter_fn = FullySharedParametersFilter,
                filter_fn_config = {},
            ),
            trigger= TriggerSpec(
                trigger_fn = WindowsOpeningTrigger,
                trigger_fn_config = {
                    'window_actuator': ("AirFlow Network Window/Door Opening", "Venting Opening Factor", "window_north"),
                },
            ),
            reward = RewardSpec(
                reward_fn = ASHRAE55SimpleModel,
                reward_fn_config = {
                    "thermal_zone": "Thermal Zone",
                },
            ),
        ),
        "South Windows": AgentSpec(
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
                ],
                use_actuator_state = True,
                use_one_day_weather_prediction = True,
                prediction_hours = 3,
                prediction_variables = {
                    'outdoor_dry_bulb': True,
                },
                internal_variables = [
                    # ("Zone Air Volume", "Thermal Zone"),
                    ("Zone Floor Area", "Thermal Zone"),
                ],
                other_obs = {
                    "WWR-North": 0.0, # WWR: Window-Wall Ratio
                    "WWR-South": 0.0,
                    "WWR-East": 0.0,
                    "WWR-West": 0.0,
                },
                history_len = 6,
            ),
            action = ActionSpec(
                actuators = [
                    ("AirFlow Network Window/Door Opening", "Venting Opening Factor", "window_south"),
                ],
            ),
            filter= FilterSpec(
                filter_fn = FullySharedParametersFilter,
                filter_fn_config = {},
            ),
            trigger= TriggerSpec(
                trigger_fn = WindowsOpeningTrigger,
                trigger_fn_config = {
                    'window_actuator': ("AirFlow Network Window/Door Opening", "Venting Opening Factor", "window_south"),
                },
            ),
            reward = RewardSpec(
                reward_fn = ASHRAE55SimpleModel,
                reward_fn_config = {
                    "thermal_zone": "Thermal Zone",
                },
            ),
        ),
        "East Windows": AgentSpec(
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
                ],
                use_actuator_state = True,
                use_one_day_weather_prediction = True,
                prediction_hours = 3,
                prediction_variables = {
                    'outdoor_dry_bulb': True,
                },
                internal_variables = [
                    # ("Zone Air Volume", "Thermal Zone"),
                    ("Zone Floor Area", "Thermal Zone"),
                ],
                other_obs = {
                    "WWR-North": 0.0, # WWR: Window-Wall Ratio
                    "WWR-South": 0.0,
                    "WWR-East": 0.0,
                    "WWR-West": 0.0,
                },
                history_len = 6,
            ),
            action = ActionSpec(
                actuators = [
                    ("AirFlow Network Window/Door Opening", "Venting Opening Factor", "window_east"),
                ],
            ),
            filter= FilterSpec(
                filter_fn = FullySharedParametersFilter,
                filter_fn_config = {},
            ),
            trigger= TriggerSpec(
                trigger_fn = WindowsOpeningTrigger,
                trigger_fn_config = {
                    'window_actuator': ("AirFlow Network Window/Door Opening", "Venting Opening Factor", "window_east"),
                },
            ),
            reward = RewardSpec(
                reward_fn = ASHRAE55SimpleModel,
                reward_fn_config = {
                    "thermal_zone": "Thermal Zone",
                },
            ),
        ),
        "West Windows": AgentSpec(
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
                ],
                use_actuator_state = True,
                use_one_day_weather_prediction = True,
                prediction_hours = 3,
                prediction_variables = {
                    'outdoor_dry_bulb': True,
                },
                internal_variables = [
                    # ("Zone Air Volume", "Thermal Zone"),
                    ("Zone Floor Area", "Thermal Zone"),
                ],
                other_obs = {
                    "WWR-North": 0.0, # WWR: Window-Wall Ratio
                    "WWR-South": 0.0,
                    "WWR-East": 0.0,
                    "WWR-West": 0.0,
                },
                history_len = 6,
            ),
            action = ActionSpec(
                actuators = [
                    ("AirFlow Network Window/Door Opening", "Venting Opening Factor", "window_west"),
                ],
            ),
            filter= FilterSpec(
                filter_fn = FullySharedParametersFilter,
                filter_fn_config = {},
            ),
            trigger= TriggerSpec(
                trigger_fn = WindowsOpeningTrigger,
                trigger_fn_config = {
                    'window_actuator': ("AirFlow Network Window/Door Opening", "Venting Opening Factor", "window_west"),
                },
            ),
            reward = RewardSpec(
                reward_fn = ASHRAE55SimpleModel,
                reward_fn_config = {
                    "thermal_zone": "Thermal Zone",
                },
            ),
        ),
        
        "North Shades": AgentSpec(
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
                ],
                use_actuator_state = True,
                use_one_day_weather_prediction = True,
                prediction_hours = 3,
                prediction_variables = {
                    'outdoor_dry_bulb': True,
                },
                internal_variables = [
                    # ("Zone Air Volume", "Thermal Zone"),
                    ("Zone Floor Area", "Thermal Zone"),
                ],
                other_obs = {
                    "WWR-North": 0.0, # WWR: Window-Wall Ratio
                    "WWR-South": 0.0,
                    "WWR-East": 0.0,
                    "WWR-West": 0.0,
                },
                history_len = 6,
            ),
            action = ActionSpec(
                actuators = [
                    ("Window Shading Control", "Control Status", "window_north"),
                ],
            ),
            filter= FilterSpec(
                filter_fn = FullySharedParametersFilter,
                filter_fn_config = {},
            ),
            trigger= TriggerSpec(
                trigger_fn = WindowsShadingTrigger,
                trigger_fn_config = {
                    'shading_actuator': ("Window Shading Control", "Control Status", "window_north"),
                },
            ),
            reward = RewardSpec(
                reward_fn = ASHRAE55SimpleModel,
                reward_fn_config = {
                    "thermal_zone": "Thermal Zone",
                },
            ),
        ),
        "South Shades": AgentSpec(
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
                ],
                use_actuator_state = True,
                use_one_day_weather_prediction = True,
                prediction_hours = 3,
                prediction_variables = {
                    'outdoor_dry_bulb': True,
                },
                internal_variables = [
                    # ("Zone Air Volume", "Thermal Zone"),
                    ("Zone Floor Area", "Thermal Zone"),
                ],
                other_obs = {
                    "WWR-North": 0.0, # WWR: Window-Wall Ratio
                    "WWR-South": 0.0,
                    "WWR-East": 0.0,
                    "WWR-West": 0.0,
                },
                history_len = 6,
            ),
            action = ActionSpec(
                actuators = [
                    ("Window Shading Control", "Control Status", "window_south"),
                ],
            ),
            filter= FilterSpec(
                filter_fn = FullySharedParametersFilter,
                filter_fn_config = {},
            ),
            trigger= TriggerSpec(
                trigger_fn = WindowsShadingTrigger,
                trigger_fn_config = {
                    'shading_actuator': ("Window Shading Control", "Control Status", "window_south"),
                },
            ),
            reward = RewardSpec(
                reward_fn = ASHRAE55SimpleModel,
                reward_fn_config = {
                    "thermal_zone": "Thermal Zone",
                },
            ),
        ),
        "East Shades": AgentSpec(
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
                ],
                use_actuator_state = True,
                use_one_day_weather_prediction = True,
                prediction_hours = 3,
                prediction_variables = {
                    'outdoor_dry_bulb': True,
                },
                internal_variables = [
                    # ("Zone Air Volume", "Thermal Zone"),
                    ("Zone Floor Area", "Thermal Zone"),
                ],
                other_obs = {
                    "WWR-North": 0.0, # WWR: Window-Wall Ratio
                    "WWR-South": 0.0,
                    "WWR-East": 0.0,
                    "WWR-West": 0.0,
                },
                history_len = 6,
            ),
            action = ActionSpec(
                actuators = [
                    ("Window Shading Control", "Control Status", "window_east"),
                ],
            ),
            filter= FilterSpec(
                filter_fn = FullySharedParametersFilter
                filter_fn_config = {},
            ),
            trigger= TriggerSpec(
                trigger_fn = WindowsShadingTrigger,
                trigger_fn_config = {
                    'shading_actuator': ("Window Shading Control", "Control Status", "window_east"),
                },
            ),
            reward = RewardSpec(
                reward_fn = ASHRAE55SimpleModel,
                reward_fn_config = {
                    "thermal_zone": "Thermal Zone",
                },
            ),
        ),
        "West Shades": AgentSpec(
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
                ],
                use_actuator_state = True,
                use_one_day_weather_prediction = True,
                prediction_hours = 3,
                prediction_variables = {
                    'outdoor_dry_bulb': True,
                },
                internal_variables = [
                    # ("Zone Air Volume", "Thermal Zone"),
                    ("Zone Floor Area", "Thermal Zone"),
                ],
                other_obs = {
                    "WWR-North": 0.0, # WWR: Window-Wall Ratio
                    "WWR-South": 0.0,
                    "WWR-East": 0.0,
                    "WWR-West": 0.0,
                },
                history_len = 6,
            ),
            action = ActionSpec(
                actuators = [
                    ("Window Shading Control", "Control Status", "window_west"),
                ],
            ),
            filter= FilterSpec(
                filter_fn = FullySharedParametersFilter,
                filter_fn_config = {},
            ),
            trigger= TriggerSpec(
                trigger_fn = WindowsShadingTrigger,
                trigger_fn_config = {
                    'shading_actuator': ("Window Shading Control", "Control Status", "window_west"),
                },
            ),
            reward = RewardSpec(
                reward_fn = ASHRAE55SimpleModel,
                reward_fn_config = {
                    "thermal_zone": "Thermal Zone",
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

assert eprllib_config.agents_config is not None, "Agents configuration is not set."

number_of_agents = len([keys for keys in eprllib_config.agents_config.keys()])
ray.init(_temp_dir='C:/Users/grhen/ray_results/tmp')
register_env(name="EPEnv", env_creator=lambda args: Environment(args))
ModelCatalog.register_custom_model("custom_transformer", CustomTransformerModel)
env_config = eprllib_config.to_dict()
# print(env_config)
# file_path = to_json(eprllib_config, f"C:/Users/grhen/Documents/GitHub/SimpleCases/configurations/env_config_{experiment_name}_{name}.json")


if not restore:
    algo = PPOConfig()
    algo.training(
        
        # === General Algo Configs ===
        gamma = 0.8,#tune.grid_search([0.7,0.9,0.99]) if tuning else 0.8,#
        lr_schedule = [
                (0, 3e-4),
                (20e6, 1e-4),
                (30e6, 5e-5)
                ],#tune.grid_search([0.0001,0.001,0.01]) if tuning else 0.003,#
        train_batch_size = 50000,#tune.grid_search([100,1000,10000,100000]) if tuning else 12961*7,
        # Each episode has a lenght of 144*3+1=433. To train the model with batch_mode=episodes_complete and using the 8 processes in parallel
        # for each iteration it is possible to set train_batch_size_per_learner=433*8=3464
        minibatch_size = 1000,#tune.grid_search([5000,10000,20000]) if tuning else 12961*7,
        # We can separate the batch into 8 batches of 433 timesteps.
        num_epochs = 30,#tune.grid_search([5,10,15,30]) if tuning else 30,
        
        # === Policy Model configuration ===
        model = {
            # FC Hidden layers
            "fcnet_hiddens": [256,256],
            "fcnet_activation": "tanh",
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
                (20e6, 0.005),
                (30e6, 0.001)
                ],#tune.choice([0.,0.1,0.2]) if tuning else 0.01,#
        clip_param = 0.25,#tune.quniform(0.1, 0.3, 0.05) if tuning else 0.25,#
        vf_clip_param = 0.25,#tune.quniform(0.1, 0.3, 0.05) if tuning else 0.3,#
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
        sample_timeout_s = 10000, # Default = 60.0
        create_env_on_local_worker = True,
        rollout_fragment_length = 'auto',
        batch_mode = "truncate_episodes", #"complete_episodes", "truncate_episodes"
        explore = True,
        observation_filter = "MeanStdFilter",
    )
    algo.multi_agent(
        policies = {
            'HVAC_policy': PolicySpec()
        },
        policy_mapping_fn = task_policy_map_fn,
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
            stop = {"info/num_env_steps_trained": 1008*28000},
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
    tuning.fit()

else:
    tuning = tune.Tuner.restore(checkpoint_path, 'PPO')
    tuning.fit()
    