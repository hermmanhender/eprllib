"""
# Example 10. Plant Loop Override Control

## Problem Statement

A common occurrence in EnergyPlus central plant simulations is for a component 
to be designed well, but during the course of an annual simulation, it is 
operated outside of its allowable region. This is due to the governing 
control strategies (operating schemes). These operation schemes may not have 
the intelligence to say, turn off a cooling tower when the outdoor 
temperature is too low.

Within the EnergyPlus example files, the cooling tower offers warnings stating 
that the tower temperature is going below a low temperature limit. We should 
ask, can we use a simple EMS addition to an input file to override the loop 
and turn off the cooling tower to avoid these situations and therefore the 
warnings?

## EMS Design Discussion

For this example, we will start with the example file that is packaged with 
EnergyPlus called EcoRoofOrlando.idf. This is one example of an input file 
where a cooling tower throws warnings due to operating at too low of a 
temperature. Although the input file has many aspects related to zone and 
HVAC, we will only be interested in the loop containing the tower, which is a 
CondenserLoop named Chiller Plant Condenser Loop. The loop has a minimum loop 
temperature of 5 degrees Celsius, as specified by the CondenserLoop object.

In order to avoid these warnings and properly shut off the tower, EMS will be 
used to check the outdoor temperature and shut down the whole loop. Special 
care must be taken when manipulating plant and condenser loops with EMS. The 
most robust way found is to both disable the loop equipment and also override 
(turn off) the loop. Skipping either of these can cause mismatches where 
either components are still expecting flow but the pump is not running, or the 
pump is trying to force flow through components which are disabled. Either 
of these cases can cause unstable conditions and possibly fatal flow errors.

The outdoor air temperature must be checked in order to determine what the 
EMS needs to do at a particular point in the simulation. This is handled by 
use of an EMS sensor monitoring the Outdoor Dry Bulb standard E+ output 
variable.

To manage the loop and pump, actuators are employed on both. The pump 
actuator is a mass flow rate override. This can be used to set the flow to 
zero, effectively shutting off the pump. The loop actuator is an on/off 
supervisory control, which allows you to “shut the entire loop down.” This 
actuator will not actually shut the loop down, it effectively empties the 
equipment availability list, so that there is no equipment available to 
reject/absorb load on the supply side. If you use this actuator alone 
to “shut down the loop,” you may find that the pump is still flowing fluid 
around the loop, but the equipment will not be running.

The EMS calling point chosen is “InsideHVACSystemIterationLoop,” so that the 
operation will be updated every time the plant loops are simulated.

The Erl program is quite simple for this case. If the outdoor dry bulb 
temperature goes below a certain value, the loop and pump actuators are 
set to zero. If the outdoor temperature is equal to or above this value, the 
actuators are set to Null, relinquishing control back to the regular 
operation schemes. In modifying this specific input file it was found that 
the outdoor dry bulb temperature which removed these warnings was six 
degrees Celsius. We also create a custom output variable called “EMS 
Condenser Flow Override On” to easily record when the overrides have occurred.
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
