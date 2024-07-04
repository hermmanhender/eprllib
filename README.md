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
    'epjson': "C:/Users/grhen/Documents/GitHub/eprllib_experiments/active_climatization/files/prot_3_ceiling_SetPointHVAC_PowerLimit.epJSON",
    "epw_training": choice(["C:/Users/grhen/Documents/GitHub/eprllib_experiments/active_climatization/files/GEF_Lujan_de_cuyo-hour-H1.epw",
                            "C:/Users/grhen/Documents/GitHub/eprllib_experiments/active_climatization/files/GEF_Lujan_de_cuyo-hour-H2.epw",
                            "C:/Users/grhen/Documents/GitHub/eprllib_experiments/active_climatization/files/GEF_Lujan_de_cuyo-hour-H3.epw"]),
    "epw": "C:/Users/grhen/Documents/GitHub/eprllib_experiments/active_climatization/files/GEF_Lujan_de_cuyo-hour-H4.epw",
    # Configure the output directory for the EnergyPlus simulation.
    'output': TemporaryDirectory("output","",'C:/Users/grhen/Documents/Resultados_RLforEP').name,
    # For dubugging is better to print in the terminal the outputs of the EnergyPlus simulation process.
    'ep_terminal_output': False,
    
    # === EXPERIMENT OPTIONS === #
    # For evaluation process 'is_test=True' and for trainig False.
    'is_test': False,
    
    # === ENVIRONMENT OPTIONS === #
    # action space for simple agent case
    'action_space': gym.spaces.Discrete(6),
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
        "heating": "Heating:DistrictHeatingWater",
        "cooling": "Cooling:DistrictCooling",
    },
    "ep_actuators": {
        "heating_setpoint": ("Schedule:Compact", "Schedule Value", "HVACTemplate-Always 19"),
        "cooling_setpoint": ("Schedule:Compact", "Schedule Value", "HVACTemplate-Always 25"),
        "AirMassFlowRate": ("Ideal Loads Air System", "Air Mass Flow Rate", "Thermal Zone: Living Ideal Loads Air System"),
    },
    'ep_actuators_type': {
        "heating_setpoint": 2,
        "cooling_setpoint": 1,
        "AirMassFlowRate": 3,
    },
    'time_variables': [
        'hour',
        'day_of_year',
        'day_of_week',
        ],
    'weather_variables': [
        'is_raining',
        'sun_is_up',
        "today_weather_beam_solar_at_time",
        ],
    "infos_variables": ["ppd", 'heating', 'cooling', 'occupancy','Ti'],
    "no_observable_variables": ["ppd"],
    
    # === OPTIONAL === #
    "timeout": 10,
    'cut_episode_len': None, # longitud en días del periodo cada el cual se trunca un episodio
    "weather_prob_days": 2,
    # Action transformer
    'action_transformer': thermostat_dual_mass_flow_rate,
    # Reward function config
    'reward_function': normalize_reward_function,
    'reward_function_config': {
        # cut_reward_len_timesteps: Este parámetro permite que el agente no reciba una recompensa 
        # en cada paso de tiempo, en cambio las variables para el cálculo de la recompensa son 
        # almacenadas en una lista para luego utilizar una recompensa promedio cuando se alcanza 
        # la cantidad de pasos de tiempo indicados por 'cut_reward_len_timesteps'.
        'cut_reward_len_timesteps': 144,
        # Parámetros para la exclusión de términos de la recompensa
        'comfort_reward': True,
        'energy_reward': True,              
        # beta_reward: Parámetros de ponderación para la energía y el confort.
        'beta_reward': 0.5,               
        # energy_ref: El valor de referencia depende del entorno. Este puede corresponder a la energía máxima que puede demandar el entorno en un paso de tiempo, un valor de energía promedio u otro.
        'cooling_energy_ref': 1500000,
        'heating_energy_ref': 1500000,
        # Nombres de las variables utilizadas en su configuración del entorno.
        'occupancy_name': 'occupancy',
        'ppd_name': 'ppd',
        'T_interior_name': 'Ti',
        'cooling_name': 'cooling',
        'heating_name': 'heating',
    },
    'episode_config': {
        # Net Conditioned Building Area [m2]
        'building_area': 20.75,
        'aspect_ratio': 1.35,
        # Conditioned Window-Wall Ratio, Gross Window-Wall Ratio [%]
        'window_area_relation_north': 56.67,
        'window_area_relation_east': 18.19,
        'window_area_relation_south': 2.59,
        'window_area_relation_west': 0,
        'inercial_mass': "auto",
        'construction_u_factor': "auto",
        # User-Specified Maximum Total Cooling Capacity [W]
        'E_cool_ref': 2500,
        # User-Specified Maximum Sensible Heating Capacity [W]
        'E_heat_ref': 2500,
    },
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

Copyright (c) 2024 Germán Rodolfo Henderson

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

-------------------------------------------------------------------------------------------------
Copyright 2023 Ray Authors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

-------------------------------------------------------------------------------------------------
EnergyPlus, Copyright (c) 1996-2024, The Board of Trustees of the University of Illinois, The Regents of the University of California, through Lawrence Berkeley National Laboratory (subject to receipt of any required approvals from the U.S. Dept. of Energy), Oak Ridge National Laboratory, managed by UT-Battelle, Alliance for Sustainable Energy, LLC, and other contributors. All rights reserved.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

-------------------------------------------------------------------------------------------------