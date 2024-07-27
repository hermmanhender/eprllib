"""
# Example 3. Hygro-thermal Window Opening Control for Airflow Network

## Problem Statement

A user of EnergyPlus version 3.1 posted the following question on the Yahoo! 
list (circa spring 2009):

    I am currently trying to model a simple ventilation system based on an 
    exhaust fan and outdoor air variable aperture paths that open according 
    to the indoor relative humidity.

    As I didn’t find any object to directly do this, I am trying to use an 
    AirflowNetwork: MultiZone: Component: DetailedOpening object and its 
    AirflowNetwork: multizone: Surface object to model the variable aperture. 
    But the Ventilation Control Mode of the surface object can only be done 
    via Temperature or Enthalpy controls (or other not interesting for my 
    purpose), and not via humidity.

    So my questions are:

    1. is it possible to make the surface object variable according to the 
    relative humidity? (maybe adapting the program?)
    2. or is there an other way to make it?

Because the traditional EnergyPlus controls for window openings do not support 
humidity-based controls (or did not as of Version 3.1), the correct 
response to Question #1 was “No.” But with the EMS, we can now answer 
Question #2 as “Yes.” How can we take the example file called 
HybridVentilationControl.idf and implement humidity-based control for a 
detailed opening in the airflow network model?

## EMS Design Discussion

The main EMS sensor will be the zone air humidity, so we use an 
EnergyManagementSystem:Sensor object that maps to the output variable 
called System Node Relative Humidity for the zone’s air node. This zone 
has the detailed opening.

The EMS will actuate the opening in an airflow network that is defined by 
the input object AirflowNetwork:MultiZone:Component:DetailedOpening. The 
program will setup the actuator for this internally, but we need to use 
an EnergyManagementSystem:Actuator object to declare that we want to use 
the actuator and provide the variable name we want for the Erl programs.

Because we do not know the exactly what the user had in mind, for this example 
we assume that the desired behavior for the opening area is that the opening 
should vary linearly with room air relative humidity. When the humidity 
increases, we want the opening to be larger. When the humidity decreases, 
we want the opening to be smaller. For relative humidity below 25% we close 
the opening. At 60% or higher relative humidity, the opening should be 
completely open. We formulate a model equation for opening factor as:

    ```
    if RH < 0.25:
        F_open = 0.0
    elif RH > 0.6:
        F_open = 1.0
    else:
        F_open = (RH - 0.25)/(0.6 - 0.25)
    ```
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
from eprllib.env.multiagent.EnergyPlusEnvironment import EnergyPlusEnv_v0
from eprllib.tools import ActionTransformers, rewards, utils

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
    'action_transformer': ActionTransformers.thermostat_dual,
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
