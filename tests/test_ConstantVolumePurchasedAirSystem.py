"""
# Example 7. Constant Volume Purchased Air System

## Problem Statement

The simplest way to add HVAC control to an EnergyPlus thermal zone is to 
use the ZoneHVAC:IdealLoadsAirSystem. This was called purchased air in 
older versions. The ideal loads air system is intended for load calculations. 
You provide input for the supply air conditions of drybulb and humidity 
ratio, but the flow rate cannot be controlled. The model operates by 
varying the flow rate to exactly meet the desired setpoints. However, you 
may want to experiment with various designs in a slightly different way in 
which, given a prescribed supply air situation, then adjust the design to 
maximize the thermal comfort. It would be interesting to use the 
simple-toinput purchased air model to examine how a zone responds to a 
system, rather than how the system responds to a zone. We should ask, 
Can we use the EMS to prescribe the supply air flow rates for a purchased 
air model?

## EMS Design Discussion

For this example we begin with the input file from Example 6 (primarily 
because it already has purchased air). We examine the typical mass flow 
rates the air system provides to have some data to judge what an appropriate 
constant flow rate might be. A cursory review of the data indicates that 
cooling flow rates of 0.3 kg/s are chosen for two zones and 0.4 kg/s is 
chosen for the third. Heating flow rates of 0.1 and 0.15 kg/s are also chosen.

We want the model to respond differently for heating and cooling. We define 
two operating states and create global variables to hold that state for 
each zone. The first state is when the zone calls for heating; we will 
assign a value of 1.0. The second is when the zone calls for cooling; we 
assign 2.0.

To sense the state we will use EMS sensors associated with the output 
variable called “Zone/Sys Sensible Load Predicted.” We will set up one of 
these for each zone and use it as input data. If this value is less than 
zero, the zone is in the cooling state. If it is greater than zero, the zone 
is in the heating state. This predicted load is calculated during the 
predictor part of the model, so we choose the EMS calling point called 
“AfterPredictorAfterHVACManagers.”

An EMS actuator is available for the ideal loads air system that overrides 
the air mass flow rate (kg/s) delivered by the system when it is on. The 
override is not absolute in that the model will still apply the limits 
defined in the input object and overrides only if the system is “on.” 
The internal logic will turn off the air system if the zone is in the 
thermostat dead band or scheduled “off” by availability managers. This “off” 
state is modeled inside the ideal loads air system so it does not need to 
be calculated in Erl. This control leads to a constant volume system that 
cycles in an attempt to control the zone conditions. In practice, it can 
achieve relatively good control when loads do not exceed the available capacity.
"""
import sys
import os

# Add the scr directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))


# import the necessary libraries
from typing import Dict, Any
from tempfile import TemporaryDirectory
import ray
from ray import air, tune
from ray.tune import register_env
from ray.rllib.algorithms.ppo.ppo import PPOConfig
from ray.rllib.policy.policy import PolicySpec
from eprllib.Env.MultiAgent.EnergyPlusEnvironment import EnergyPlusEnv_v0
from eprllib.Env.EnvConfig import EnvConfig, env_config_to_dict
from eprllib.ActionFunctions.DualSetPointThermostat import DualSetPointThermostat
from eprllib.RewardFunctions.EnergyTemperature import EnergyTemperatureReward

# Define a custom reward function
reward_fn = EnergyTemperatureReward(
    reward_fn_config = {
        "Heating Thermostat": {
            'beta': 0.5,
            'T_interior_name': "Zone Mean Air Temperature",
            'cooling_name': "Cooling:Electricity",
            'heating_name': "Heating:Electricity",
        },
        "Cooling Thermostat": {
            'beta': 0.5,
            'T_interior_name': "Zone Mean Air Temperature",
            'cooling_name': "Cooling:Electricity",
            'heating_name': "Heating:Electricity",
        },
    }
)

# Use a build-in action function
action_fn = DualSetPointThermostat(
    action_fn_config = {
        "agents_type": {
            "Heating Thermostat": 2,
            "Cooling Thermostat": 1
        },
    }
)

# Define the environment configuration
RefBldgSallOfficeNew2004_Chicago = EnvConfig()
RefBldgSallOfficeNew2004_Chicago.generals(
    epjson_path = "C:/Users/grhen/Documents/GitHub/natural_ventilation_EP_RLlib/tests/ExampleFiles/RefBldgSmallOfficeNew2004_Chicago.idf", 
    epw_path = "C:/Users/grhen/Documents/GitHub/natural_ventilation_EP_RLlib/tests/WeatherData/USA_IL_Chicago-OHare.Intl.AP.725300_TMY3.epw",
    output_path = TemporaryDirectory("_output","eprllib_test_").name,
    ep_terminal_output = True
)
RefBldgSallOfficeNew2004_Chicago.agents(
    agents_config = {
        "Heating Thermostat": {
            "ep_actuator_config": ("Zone Temperature Control", "Heating Setpoint", "Core_ZN"),
            "thermal_zone": "Core_ZN",
            "thermal_zone_indicator": 1,
            "actuator_type": 2,
            "agent_indicator": 1
        },
        "Cooling Thermostat": {
            "ep_actuator_config": ("Zone Temperature Control", "Cooling Setpoint", "Core_ZN"),
            "thermal_zone": "Core_ZN",
            "thermal_zone_indicator": 1,
            "actuator_type": 1,
            "agent_indicator": 2
        }
    }
)
RefBldgSallOfficeNew2004_Chicago.observations(
    ep_environment_variables = [
        "Site Outdoor Air Drybulb Temperature",
        "Site Outdoor Air Relative Humidity",
    ],
    ep_thermal_zones_variables = [
        "Zone Mean Air Temperature",
    ],
    ep_meters = [
        "Cooling:Electricity",
        "Heating:Electricity",
    ],
    infos_variables = {
        "Core_ZN": [
            "Zone Mean Air Temperature",
            "Cooling:Electricity", 
            "Heating:Electricity",
        ],
    },
    use_agent_indicator = True,
)
RefBldgSallOfficeNew2004_Chicago.actions(
    action_fn = action_fn
)
RefBldgSallOfficeNew2004_Chicago.rewards(
    reward_fn = reward_fn
)
# transfor the config to a dictionary
env_config = env_config_to_dict(RefBldgSallOfficeNew2004_Chicago)

# inicialize ray server and after that register the environment
ray.init()
register_env(name="EPEnv", env_creator=lambda args: EnergyPlusEnv_v0(args))

algo = PPOConfig().training(
    # General Algo Configs
    gamma = 0.8,
    # Float specifying the discount factor of the Markov Decision process.
    lr = 0.0001,
    # The learning rate (float) or learning rate schedule
    model = {
        "fcnet_hiddens": [64,64],
        "fcnet_activation": "relu",
        },
    # Arguments passed into the policy model. See models/catalog.py for a full list of the 
    # available model options.
    train_batch_size = 8000,
    # PPO Configs
    lr_schedule = None, # List[List[int | float]] | None = NotProvided,
    # Learning rate schedule. In the format of [[timestep, lr-value], [timestep, lr-value], …] 
    # Intermediary timesteps will be assigned to interpolated learning rate values. A schedule 
    # should normally start from timestep 0.
    use_critic = True, # bool | None = NotProvided,
    # Should use a critic as a baseline (otherwise don’t use value baseline; required for using GAE).
    use_gae = True, # bool | None = NotProvided,
    # If true, use the Generalized Advantage Estimator (GAE) with a value function, 
    # see https://arxiv.org/pdf/1506.02438.pdf.
    lambda_ = 0.2,
    # The GAE (lambda) parameter.  The generalized advantage estimator for 0 < λ < 1 makes a 
    # compromise between bias and variance, controlled by parameter λ.
    use_kl_loss = True, # bool | None = NotProvided,
    # Whether to use the KL-term in the loss function.
    kl_coeff = 9,
    # Initial coefficient for KL divergence.
    kl_target = 0.05,
    # Target value for KL divergence.
    sgd_minibatch_size = 800, # if not tune_runner else tune.choice([48, 128]), # int | None = NotProvided,
    # Total SGD batch size across all devices for SGD. This defines the minibatch size 
    # within each epoch.
    num_sgd_iter = 40, # if not tune_runner else tune.randint(30, 60), # int | None = NotProvided,
    # Number of SGD iterations in each outer loop (i.e., number of epochs to execute per train batch).
    shuffle_sequences = True, # bool | None = NotProvided,
    # Whether to shuffle sequences in the batch when training (recommended).
    vf_loss_coeff = 0.4,
    # Coefficient of the value function loss. IMPORTANT: you must tune this if you set 
    # vf_share_layers=True inside your model’s config.
    entropy_coeff = 10,
    # Coefficient of the entropy regularizer.
    entropy_coeff_schedule = None, # List[List[int | float]] | None = NotProvided,
    # Decay schedule for the entropy regularizer.
    clip_param = 0.1, # if not tune_runner else tune.uniform(0.1, 0.4), # float | None = NotProvided,
    # The PPO clip parameter.
    vf_clip_param = 10, # if not tune_runner else tune.uniform(0, 50), # float | None = NotProvided,
    # Clip param for the value function. Note that this is sensitive to the scale of the 
    # rewards. If your expected V is large, increase this.
    grad_clip = None, # float | None = NotProvided,
    # If specified, clip the global norm of gradients by this amount.
).environment(
    env = "EPEnv",
    env_config = env_config,
).framework(
    framework = 'torch',
).fault_tolerance(
    recreate_failed_env_runners = True,
).env_runners(
    num_env_runners = 0,
    create_env_on_local_worker = True,
    rollout_fragment_length = 'auto',
    enable_connectors = True,
    num_envs_per_env_runner = 1,
    explore = True,
    exploration_config = {
        "type": "EpsilonGreedy",
        "initial_epsilon": 1.,
        "final_epsilon": 0.,
        "epsilon_timesteps": 6*24*365*8,
    },
).multi_agent(
    policies = {
        'shared_policy': PolicySpec(),
    },
    policy_mapping_fn = lambda agent_id, episode, worker, **kwargs: "shared_policy",
).reporting( # multi_agent config va aquí
    min_sample_timesteps_per_iteration = 1000,
).checkpointing(
    export_native_model_files = True,
).debugging(
    log_level = "ERROR",
).resources(
    num_gpus = 0,
)

my_new_ppo = algo.build()
results = my_new_ppo.train()
ray.shutdown()

"""# init the training loop
tune.Tuner(
    "PPO",
    tune_config=tune.TuneConfig(
        mode="max",
        metric="episode_reward_mean",
    ),
    run_config=air.RunConfig(
        stop={"episodes_total": 16},
        checkpoint_config=air.CheckpointConfig(
            checkpoint_at_end = True,
            checkpoint_frequency = 10,
        ),
    ),
    param_space=algo.to_dict(),
).fit()

# close the ray server
ray.shutdown()"""
