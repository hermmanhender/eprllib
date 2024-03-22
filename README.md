# eprllib: EnergyPlus as a Markov Decission Process (MDP) environment for Deep Reinforcement Learning (DRL) in RLlib 

This repository provides a set of methods to establish the computational loop of EnergyPlus within a Markov Decision Process (MDP), treating it as a multi-agent environment compatible with RLlib. The main goal is to offer a simple configuration of EnergyPlus as a standard environment for experimentation with Deep Reinforcement Learning.

## Installation

To install EnergyPlusRL, simply use pip:

```
pip install eprllib
```

## Key Features

* Integration of EnergyPlus and RLlib: This package facilitates setting up a Reinforcement Learning environment using EnergyPlus as the base, allowing for experimentation with energy control policies in buildings.
* Simplified Configuration: To use this environment, you simply need to provide a configuration in the form of a dictionary that includes state variables, metrics, actuators (which will also serve as agents in the environment), and other optional features.
* Flexibility and Compatibility: EnergyPlusRL easily integrates with RLlib, a popular framework for Reinforcement Learning, enabling smooth setup and training of control policies for actionable elements in buildings.

## Usage

1. Import the package into your Python script.
2. Define your environment configuration in a dictionary, specifying state variables, metrics, actuators, and other relevant features as needed. (See Documentation section to know all the parameters).
3. Configure RLlib for training control policies using the EnergyPlusRL environment.
4. Execute the training of your Reinforcement Learning model and evaluate the results obtained.

## Example configuration

```
import ray
from ray.tune import register_env
from ray import tune, air
from ray.rllib.algorithms.dqn.dqn import DQNConfig
from ray.rllib.policy.policy import PolicySpec
import gymnasium as gym
from eprllib.env.multiagent.marl_ep_gym_env import EnergyPlusEnv_v0

env_config={
    # === ENERGYPLUS OPTIONS === #
    'epjson': '/content/prot_3_ceiling.epJSON',
    "epw_training": np.random.choice(["/content/GEF_Lujan_de_cuyo-hour-H1.epw","/content/GEF_Lujan_de_cuyo-hour-H2.epw","/content/GEF_Lujan_de_cuyo-hour-H3.epw"]),
    "epw": "/content/GEF_Lujan_de_cuyo-hour-H4.epw",
    # Configure the output directory for the EnergyPlus simulation.
    'output': '/content/output',
    # For dubugging is better to print in the terminal the outputs of the EnergyPlus simulation process.
    'ep_terminal_output': False,

    # === EXPERIMENT OPTIONS === #
    # For evaluation process 'is_test=True' and for trainig False.
    'is_test': False,

    # === ENVIRONMENT OPTIONS === #
    # action space for simple agent case
    'action_space': gym.spaces.Discrete(2),
    # observation space for simple agent case
    # This is equal to the the sume of:
    #   + ep_variables
    #   + ep_meters
    #   + ep_actuators
    #   + weather_prob_days*144
    #   - no_observable_variables
    #   + 1 (agent_indicator)
    #   + 6 ('day_of_the_week','is_raining','sun_is_up','hora','simulation_day','rad')
    'observation_space': gym.spaces.Box(float("-inf"), float("inf"), (307,)),
    'reward_function': eprllib.tools.rewards.reward_function_T3,
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
    "infos_variables": ["ppd", "occupancy", "Ti"],
    "no_observable_variables": ["ppd"],

    # === OPTIONAL === #
    "timeout": 10,
    "T_confort": 23.5,
    "weather_prob_days": 2
}

def policy_mapping_fn(agent_id, episode, worker, **kwargs):
    return "shared_policy"

# To register the custom environment.
ray.init()
register_env(name="EPEnv", env_creator=lambda args: EnergyPlusEnv_v0(args))

algo = DQNConfig().training(
    gamma = 0.99,
    lr = 0.01,
).environment(
    env="EPEnv",
    env_config=env_config,
).framework(
    framework = 'torch',
).rollouts(
    num_rollout_workers = 0,
).experimental(
    _enable_new_api_stack = False,
).multi_agent(
    policies = {
        'shared_policy': PolicySpec(),
    },
    policy_mapping_fn = policy_mapping_fn,
)

tune.Tuner(
    algorithm,
    tune_config=tune.TuneConfig(
        mode="max",
        metric="episode_reward_mean",
    ),
    run_config=air.RunConfig(
        stop={"episodes_total": 800},
    ),
    param_space=algo.to_dict(),
).fit()
```

## Contribution

Contributions are welcome! If you wish to improve this project or add new features, feel free to submit a pull request.

## Licency

MIT License

Copyright (c) 2024 hermmanhender

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

