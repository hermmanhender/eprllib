"""
# Example: Computed Schedule

## Problem Statement

Many models have schedule inputs that could be used to control the object, 
but creating the schedules is often cumbersome and may not always result in 
optimal behavior due to dynamic environmental factors. Leveraging schedules 
as inputs for Deep Reinforcement Learning (DRL) offers a promising approach 
to learning optimal control policies. In this example, we demonstrate how 
DRL can be applied to learn the optimal policy for setting heating and 
cooling zone temperature schedules.

## DRL Design Discussion

As an example, we will utilize the model of the Small Office Reference 
Building (RefBldgSmallOfficeNew2004_Chicago.idf) and utilize the 
EnergyPlus API Python to optimize heating and cooling zone temperature 
setpoint schedules. The input object `Schedule:Constant` has been configured 
to serve as an actuator (or agent within the scope of eprllib).

To facilitate the DRL process, we must explicitly define the action space, 
detailing the possible actions the agent can take in adjusting the temperature 
setpoints. Additionally, we need to specify the variables comprising the 
observation space, providing clarity on the information available to the 
agent at each time step. For the reward function, we define a range of 
temperatures within which we aim to maintain the environment to ensure 
comfort for the building occupants.

Furthermore, it is essential to elucidate how the DRL algorithm interacts 
with the EnergyPlus simulation. By detailing the learning process over 
time, readers gain a deeper understanding of the practical implementation 
and the iterative nature of DRL in optimizing building energy management.


built-in variables:
    Hour
    DayOfWeek
"""

# import the necessary libraries
import time
from tempfile import TemporaryDirectory
import gymnasium as gym
import ray
from ray import air, tune
from ray.tune import register_env
from ray.rllib.algorithms.dqn.dqn import DQNConfig
from ray.rllib.policy.policy import PolicySpec
from eprllib.env.multiagent.marl_ep_gym_env import EnergyPlusEnv_v0
from eprllib.tools import rewards, utils, action_transformers
from numpy.random import choice

# define the eprllib configuration
env_config={
    # === ENERGYPLUS OPTIONS === #
    'epjson': "energyplus/testfiles/RefBldgSmallOfficeNew2004_Chicago.idf",
    "epw_training": "path_to/Chicago.epw",
    "epw": "path_to/Chicago.epw",
    'output': TemporaryDirectory("output","eprllib",'path_to_outputs_folder'),
    'ep_terminal_output': False,
    
    # === EXPERIMENT OPTIONS === #
    'is_test': False,
    
    # === ENVIRONMENT OPTIONS === #
    'action_space': gym.spaces.Discrete(4),
    'action_transformer': action_transformers.thermostat_dual,
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
        "cooling_setpoint": ("Schedule:Constant", "Schedule Value", "CLDSP_SC"),
        "heating_serpoint": ("Schedule:Constant", "Schedule Value", "HTDSP_SC"),
    },
    "infos_variables": ["ppd", "occupancy", "Ti"],
    "no_observable_variables": ["ppd"],
    
    # === OPTIONAL === #
    "timeout": 10,
    "T_confort": 22,
    "weather_prob_days": 2
}

# inicialize ray server and after that register the environment
ray.init()
register_env(name="EPEnv", env_creator=lambda args: EnergyPlusEnv_v0(args))

# configurate the algorithm
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

# init the training loop
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

# close the ray server
ray.shutdown()