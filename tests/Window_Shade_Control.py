"""
# Example 6. Window Shade Control

## Problem Statement

EnergyPlus offers a wide range of control options in the WindowShadingControl 
object, but it is easy to imagine custom schemes for controlling shades or 
blinds that are not available. We need to ask, Can we use the EMS to override 
the shading controls?

## EMS Design Discussion

We will take the example file PurchAirWindowBlind.idf and use EMS to add a 
new control scheme. This file has an interior blind that can be either “on” 
or “off.” The control scheme has three parts:

* Deploy the blind whenever too much direct sun would enter the zone and cause 
discomfort for the occupants.
* Deploy the blind whenever there is a significant cooling load.
* Leave the blind open whenever the first two constraints have not triggered.

We assume that a model for the direct sun control is based on incidence angle, 
where the angle is defined as zero for normal incidence relative to the 
plane of the window. When the direct solar incidence angle is less than 45 
degrees, we want to draw the blind. EnergyPlus has a report variable called 
“Surface Ext Solar Beam Cosine Of Incidence Angle,” for which we will use a 
sensor in our EnergyManagementSystem:Sensor input object. This sensor is a 
cosine value that we turn into an angle value with the built-in function 
@ArcCos. Then we will use the built-in function @RadToDeg to convert from 
radians to degrees. This new window/solar incidence angle in degree may be 
an interesting report variable, so we use an 
EnergyManagementSystem:OutputVariable input object to create custom output.

Because the transmitted solar is a problem only when there is a cooling load, 
we also trigger the blind based on the current data for cooling. The report 
variable called “Zone/Sys Sensible Cooling Rate” is used in an EMS sensor 
to obtain an Erl variable with the most recent data about zone cooling load 
required to meet setpoint. When this value is positive, we know the zone 
cannot make good use of passive solar heating, so we close the blind.

The EMS actuator will override the operation of a WindowShadingControl input 
object. Related to this, the EDD file shows

! <EnergyManagementSystem:Actuator Available>, Component Unique Name, 
Component Type, Control Type

EnergyManagementSystem:Actuator Available,ZN001:WALL001:WIN001,Window 
Shading Control,Control Status

Although the user-defined name for the WindowShadingControl is “INCIDENT 
SOLAR ON BLIND,” the component unique name of the actuator that is available 
is called “ZN001:WALL001:WIN001.” There could be multiple windows, all with 
shades, and each is governed by a single WindowShadingControl input object. 
The EMS actuator could override each window separately. The Control Type is 
called “Control Status,” and requires you to set the status to one of a set 
of possible control flags. For this case, with only an interior shade, there 
are two states for the actuator to take. The first shows the shade is “off,” 
and corresponds to a value of 0.0. The second shows the interior shade is 
“on,” and corresponds to a value of 6.0.
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
