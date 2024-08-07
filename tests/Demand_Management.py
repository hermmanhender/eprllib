"""
# Example 9. Demand Management

## Problem Statement

Demand management refers to controlling a building to reduce the peak 
electrical power draws or otherwise improve the load profile from the 
perspective of the electric utility. Managing electricity demand is an 
important application for EMS. We should ask, Can we take the model from 
example 2 and use the EMS to add demand management?

## EMS Design Discussion

Example 2 is a model of a large oﬀice building, but unfortunately the utility 
tariff is not a demandbased rate. Therefore, we change to a different set of 
utility rate input objects so the model has demand charges.

For this example, we assume that demand is managed by turning down the lights 
and increasing the cooling setpoint. The EMS calling point chosen is 
“BeginTimestepBeforePredictor” because it allows you to change the lighting 
power levels and temperature setpoints before you predict the zone loads.

To manage the demand, we first need to develop some targets based on some a 
priori idea of what level of demand should be considered “high.” Therefore, 
we first run the model without demand management and note the simulation 
results for demand. There are many ways to obtain the demand results, but one 
method is to obtain them from the tabular report for Tariffs called “Native 
Variables.” In that report, the row called PeakDemand is the demand used to 
calculate demand charges and is listed in kW. We will use these values to 
construct a target level of demand for each month by taking these results 
and multiplying by 0.85 in an effort to reduce demand by 15%. For example, 
the demand for January was 1,154.01 kW, so we make our target level to be
0.85 * 1154.01 = 980.91 kW and the demand for August was 1,555.20 kW, so 
the target is 0.85 * 1555.20 = 1,321.92 kW.

To develop our Erl program, we separate the overall task into two parts:

1. Determine the current state of demand management control.
2. Set the controls based on that control state.

We then divide the Erl programs into two main programs and give them 
descriptive names: “Determine_Current_Demand_Manage_State”; 
“Dispatch_Demand_Changes_By_State.”

The Erl program to determine the control state determines the current status 
for the demand management controls. You can record and manage the control 
state by setting the value of a global variable called “argDmndMngrState.” 
For this example, we develop four control states that represent four levels 
of demand management:

* Level 1 is assigned a value of 0.0 and represents no override to implement 
changes to demandrelated controls.
* Level 2 is assigned a value of 1.0 and represents moderately aggressive 
overrides for demandrelated controls.
* Level 3 is assigned a value of 2.0 and represents more aggressive override.
* Level 4 is assigned a value of 3.0 and represents the most aggressive 
overrides.

We develop an algorithm for choosing the control state by assuming it should 
be a function of how close the current power level is to the target power 
level, the current direction for changes in power use, and the recent history 
of control state. The current demand is obtained by using a sensor that is 
based on the “Total Electric Demand” output variable. This current demand 
is compared to the target demand levels discussed as well as a “level 1” 
demand level that is set to be 90% of the target. If the current demand is 
higher than the level 1 demand but lower than the target, the state will 
tend to be at level 1. If the current demand is higher than the target, the 
current state will be either level 2 or level 3 depending on the trend 
direction. However, we do not want the response to be too quick because it 
leads to too much bouncing between control states. Therefore, we also 
introduce some numerical damping with the behavior that once a control 
state is selected it should be selected for at least two timesteps before 
dropping down to a lower level. This damping is modeled with the help of a 
trend variable that records the control state over time so we can retrieve 
what the control state was during the past two timesteps.

Once the control state is determined, the Erl programs will use EMS actuators 
to override controls based on the state. The following table summarizes the 
control adjustments used in our example for each control state.

TABLE

For control state level 0, the actuators are all set to Null so they stop 
overriding controls and return the model to normal operation.

To alter the lighting power density with EMS, you could use either a direct 
method that employs a Lights actuator or an indirect method that modifies the 
lighting schedule. For this example we use the direct method with 
EnergyManagementSystem:Actuator input objects that enable you to override 
the “Electricity Rate” for each zone’s lights. We also set up internal 
variables to obtain the Lighting Power Design Level for each Lights object. 
Finally, we set up an EMS sensor to obtain the lighting schedule value to 
use in Erl programs. If the demand management control state is 1, 2, or 3, 
we use the following model to determine a new lighting power level:

Power = (Adjustment Factor) × (Lighting Power Design Level) × (Schedule Value)

There are also two ways to alter the cooling setpoint with EMS. To dynamically 
alter the cooling setpoints, we modify the schedule rather than directly 
actuating Zone Temperature Control actuators. Changing the schedule allows 
one actuator to override all the zones; the more direct approach would 
require actuators for each zone. (This can be used to advantage if different 
zones are to be managed differently.) The algorithm applies an offset 
depending on the control state. In the input file, the schedule for cooling 
setpoints is called CLGSETP_SCH, so we set up an actuator for this Schedule 
Value. Because the algorithm is a simple offset from the original schedule, 
we need to keep track of the values in the original schedule. We cannot use 
the same schedule as an input to the algorithm because once an actuator 
overrides the schedule value it will corrupt the original schedule. This 
would be an example of a circular reference problem. Therefore, we make a 
copy of the cooling setpoint schedule, naming it CLGSETP_SCH_Copy, and use 
the copy in a EnergyManagementSystem:Sensor object to obtain the current 
scheduled value for the setpoint. When we override the CLGSETP_SCH schedule, 
it will not corrupt the values from the CLGSTEP_SCH_Copy schedule used as 
input.
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
