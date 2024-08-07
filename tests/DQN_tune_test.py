
import time
import sys
sys.path.append('C:/Users/grhen/Documents/GitHub/natural_ventilation_EP_RLlib/src')
from tempfile import TemporaryDirectory
import gymnasium as gym
import ray
from ray import air, tune
from ray.tune import register_env
from ray.rllib.algorithms.dqn.dqn import DQNConfig
from ray.rllib.policy.policy import PolicySpec
from eprllib.env.multiagent.EnergyPlusEnvironment import EnergyPlusEnv_v0
from eprllib.tools import rewards, utils
from numpy.random import choice

env_config={
    # === ENERGYPLUS OPTIONS === #
    'epjson': "tests/files/prot_3_ceiling.epJSON",
    "epw_training": choice(["tests/files/GEF_Lujan_de_cuyo-hour-H1.epw",
                            "tests/files/GEF_Lujan_de_cuyo-hour-H2.epw",
                            "tests/files/GEF_Lujan_de_cuyo-hour-H3.epw"]),
    "epw": "tests/files/GEF_Lujan_de_cuyo-hour-H4.epw",
    'output': TemporaryDirectory("output","DQN_",'C:/Users/grhen/Documents/Resultados_RLforEP').name,
    'ep_terminal_output': False,
    
    # === EXPERIMENT OPTIONS === #
    'is_test': False,
    'episode_len_fn': utils.len_episode,
    
    # === ENVIRONMENT OPTIONS === #
    'action_space': gym.spaces.Discrete(2),
    'reward_function': rewards.reward_function_T3,
    "ep_variables":{
        "To": ("Site Outdoor Air Drybulb Temperature", "Environment"),
        "Ti": ("Zone Mean Air Temperature", "Thermal Zone: Living"),
        "v": ("Site Wind Speed", "Environment"),
        "d": ("Site Wind Direction", "Environment"),
        "RHo": ("Site Outdoor Air Relative Humidity", "Environment"),
        "RHi": ("Zone Air Relative Humidity", "Thermal Zone: Living"),
        "pres": ("Site Outdoor Air Barometric Pressure", "Environment"),
        "occupancy": ("Zone People Occupant Count", "Thermal Zone: Living"),
        "ppd": ("Zone Thermal Comfort Fanger Model PPD", "Living Occupancy")
    },
    "ep_meters": {
        "electricity": "Electricity:Zone:THERMAL ZONE: LIVING",
        "gas": "NaturalGas:Zone:THERMAL ZONE: LIVING",
    },
    "ep_actuators": {
        "opening_window_1": ("AirFlow Network Window/Door Opening", "Venting Opening Factor", "living_NW_window"),
        "opening_window_2": ("AirFlow Network Window/Door Opening", "Venting Opening Factor", "living_E_window"),
    },
    # Set the type of actuator that correspond to each agent.
    # 1: Cooling set point
    # 2: Heating set point
    # 3: Acondicionated Air Flow Rate
    # 4: North Window Opening
    # 5: East Window Opening
    # 6: South Window Opening
    # 7: West Window Opening
    # 8: North Window Shading
    # 9: East Window Shading
    # 10: South Window Shading
    # 11: West Window Shading
    # 12: Fan Flow Rate
    "ep_actuators_type": {
        "opening_window_1": 4,
        "opening_window_2": 5,
    },
    'time_variables': [
        'hour',
        'day_of_year',
        'day_of_the_week',
        ],
    'weather_variables': [
        'is_raining',
        'sun_is_up',
        "today_weather_beam_solar_at_time",
        ],
    "infos_variables": ["ppd", "occupancy", "Ti"],
    "no_observable_variables": ["ppd"],
    
    # === OPTIONAL === #
    "timeout": 10,
    "T_confort": 23.5,
    "weather_prob_days": 2,
    "cut_episode_len": 7,
}


ray.init()
register_env(name="EPEnv", env_creator=lambda args: EnergyPlusEnv_v0(args))

def policy_mapping_fn(agent_id, episode, worker, **kwargs):
    return "shared_policy"


algo = DQNConfig().training(
    # === General Algo Configs === #
    gamma = 0.99,
    lr = 0.01,
    grad_clip = 40,
    grad_clip_by = 'global_norm',
    train_batch_size = 256,
    model = {
        "fcnet_hiddens": [256,256,256],
        "fcnet_activation": "relu",
        },
    optimizer = {},
    # === DQN Configs === #
    num_atoms = 100,
    v_min = -343,
    v_max = 0,
    noisy = True,
    sigma0 = 0.7,
    dueling = True,
    hiddens = [256],
    double_q = True,
    n_step = 12,
    replay_buffer_config = {
        '_enable_replay_buffer_api': True,
        'type': 'MultiAgentPrioritizedReplayBuffer',
        'capacity': 5000000,
        'prioritized_replay_alpha': 0.7,
        'prioritized_replay_beta': 0.6,
        'prioritized_replay_eps': 1e-6,
        'replay_sequence_length': 1,
        },
    categorical_distribution_temperature = 0.5,
).environment(
    env="EPEnv",
    env_config=env_config,
).framework(
    framework = 'torch',
).fault_tolerance(
    recreate_failed_workers = True,
    restart_failed_sub_environments=False,
).rollouts(
    num_rollout_workers = 7,
    create_env_on_local_worker=True,
    rollout_fragment_length = 'auto',
    enable_connectors = True,
    num_envs_per_worker=1,
).experimental(
    _enable_new_api_stack = False,
).multi_agent(
    policies = {
        'shared_policy': PolicySpec(),
    },
    policy_mapping_fn = policy_mapping_fn,
).reporting(
    min_sample_timesteps_per_iteration = 1000,
).checkpointing(
    export_native_model_files = True,
).debugging(
    log_level = "ERROR",
).resources(
    num_gpus = 0,
)
algo.exploration(
    exploration_config={
        "type": "EpsilonGreedy",
        "initial_epsilon": 1.,
        "final_epsilon": 0.,
        "epsilon_timesteps": 6*24*365*100,
    }
)

tune.Tuner(
    "DQN",
    tune_config=tune.TuneConfig(
        mode="max",
        metric="episode_reward_mean",
        reuse_actors=False,
        trial_name_creator=utils.trial_str_creator,
        trial_dirname_creator=utils.trial_str_creator,
    ),
    run_config=air.RunConfig(
        name="{date}_VN_marl".format(
            date=str(time.time())
        ),
        stop={"episodes_total": 200},
        log_to_file=True,
        
        checkpoint_config=air.CheckpointConfig(
            checkpoint_at_end = True,
            checkpoint_frequency = 50,
        ),
        failure_config=air.FailureConfig(
            max_failures=100
        ),
    ),
    param_space=algo.to_dict(),
).fit()

ray.shutdown()