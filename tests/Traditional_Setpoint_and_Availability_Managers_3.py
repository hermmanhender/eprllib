"""
# Example 2.2. Traditional Setpoint and Availability Managers

## Problem Statement

The traditional way of modeling supervisory control of HVAC systems in 
EnergyPlus is to use SetpointManagers and AvailabilityManagers. To gain 
experience with eprllib we model a Large Oﬀice Reference Building 
(RefBldgLargeOﬀiceNew2004_Chicago.idf) with a SetpointManager:NightCycle 
actuator.

## EMS Design Discussion

A review of the example file shows that three types of traditional HVAC 
managers are being used: scheduled setpoints, mixed air setpoints, and 
night cycle availability. We will discuss these separately. In this example, 
we will use the SetpointManager:NightCycle actuator.

The input object AvailabilityManager:NightCycle functions by monitoring 
zone temperature and starting up the air system (if needed) to keep the 
building within the thermostat range. The sensors here are the zone air 
temperatures, which are set up by using EnergyManagementSystem:Sensor 
objects in the same way as for Example 1. We will need one zone temperature 
sensor for each zone that is served by the air system so we can emulate the 
“CycleOnAny” model being used. The other sensors we need are the desired 
zone temperatures used by the thermostat. We access these temperatures 
directly from the schedules (HTGSETP_SCH and CLGSETP_SCH in the example) 
by using EnergyManagementSystem:Sensor objects. To control the air system’s 
operation status, we use an EnergyManagementSystem:Actuator object that is 
assigned to an “AirLoopHVAC” component type using the control variable 
called “Availability Status.” EnergyPlus recognizes four availability 
states that control the behavior of the air system. Inside EnergyPlus 
these are integers, but EMS has only real-valued variables, so we will use 
the following whole numbers:

* NoAction = 0.0
* ForceOff = 1.0
* CycleOn = 2.0
* CycleOnZoneFansOnly = 3.0.

The traditional AvailabilityManager:NightCycle object operates by turning 
on the system for a prescribed amount of time (1800 seconds in the example 
file), and then turning it off for the same amount of time. You should be 
able to model this starting and stopping in EMS by using Trend variables to 
record the history of the actions. However, this cycling is not necessarily 
how real buildings are operated, and for this example we do not try to 
precisely emulate the traditional EnergyPlus night cycle manager. Rather, we 
use a simpler temperature-based control to start and stop the air system for 
the night cycle. The algorithm first assumes an offset tolerance of 0.83°C 
and calculates limits for when heating should turn on and off and when 
cooling should turn on and off. It then finds the maximum and minimum zone 
temperatures for all the zones attached to the air system. These use the 
@Max and @Min built-in functions, which take on two operators at a time. 
Then a series of logic statements is used to compare temperatures and decide 
what the availability status of the air system should be.
"""

# import the necessary libraries
import time
from tempfile import TemporaryDirectory
from gymnasium.spaces import Discrete
import ray
from ray import air, tune
from ray.tune import register_env
from ray.rllib.algorithms.dqn.dqn import DQNConfig
from ray.rllib.policy.policy import PolicySpec
from eprllib.env.multiagent.marl_ep_gym_env import EnergyPlusEnv_v0
from eprllib.tools import rewards, utils, action_transformers

# define the eprllib configuration
env_config={
    # === ENERGYPLUS OPTIONS === #
    'epjson': "C:/EnergyPlusV23-2-0/ExampleFiles/RefBldgSmallOfficeNew2004_Chicago.idf",
    "epw_training": "C:/EnergyPlusV23-2-0/WeatherData/USA_IL_Chicago-OHare.Intl.AP.725300_TMY3.epw",
    "epw": "C:/EnergyPlusV23-2-0/WeatherData/USA_IL_Chicago-OHare.Intl.AP.725300_TMY3.epw",
    'output': TemporaryDirectory("output","eprllib",'path_to_outputs_folder'),
    'ep_terminal_output': False,
    
    # === EXPERIMENT OPTIONS === #
    'is_test': False,
    
    # === ENVIRONMENT OPTIONS === #
    'action_space': Discrete(4),
    'action_transformer': action_transformers.thermostat_dual,
    'reward_function': rewards.PPD_Energy_reward,
    "ep_variables":{
        "To": ("Site Outdoor Air Drybulb Temperature", "Environment"),
        "Ti": ("Zone Mean Air Temperature", "Core_ZN"),
        "v": ("Site Wind Speed", "Environment"),
        "d": ("Site Wind Direction", "Environment"),
        "RHo": ("Site Outdoor Air Relative Humidity", "Environment"),
        "RHi": ("Zone Air Relative Humidity", "Core_ZN"),
        "pres": ("Site Outdoor Air Barometric Pressure", "Environment"),
        "occupancy": ("Zone People Occupant Count", "Core_ZN"),
        "ppd": ("Zone Thermal Comfort Fanger Model PPD", "Core_ZN People")
    },
    "ep_meters": {
        "heating_meter": "Heating:Electricity: Core_ZN",
        "cooling_meter": "Cooling:Electricity: Core_ZN",
    },
    "ep_actuators": {
        "cooling_setpoint": ("Zone Temperature Control", "Cooling Setpoint", "Core_ZN"),
        "heating_serpoint": ("Zone Temperature Control", "Heating Setpoint", "Core_ZN"),
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
    "infos_variables": ["ppd", 'heating_meter', 'cooling_meter'],
    "no_observable_variables": ["ppd"],
    
    # === OPTIONAL === #
    "timeout": 10,
    'beta_reward': 0.5,
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
    ),
    run_config=air.RunConfig(
        stop={"episodes_total": 200},
        checkpoint_config=air.CheckpointConfig(
            checkpoint_at_end = True,
            checkpoint_frequency = 10,
        ),
    ),
    param_space=algo.to_dict(),
).fit()

# close the ray server
ray.shutdown()
