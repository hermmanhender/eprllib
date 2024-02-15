# DRL-based natural ventilation management model with RLlib and EnergyPlus

## Description

Buildings have proven to be one of the energy sinks in recent decades. Both the industrial and commercial sectors and the residential sector are consumers of a large part of the energy and responsible for greenhouse gas emissions due to these consumptions. In general, these consumptions are associated with the use of energy inside buildings. In the residential sector, in particular, most of the primary energy is consumed for heating, ventilation or air conditioning (HVAC) and for cooking food. Currently, HVAC systems are automated to achieve high performance and high comfort features. However, they have many problems still unresolved. Its automation is based on instantaneous variables and does not consider the user's activity or the use of passive air conditioning strategies in the home, such as natural ventilation, taking advantage of the sun through windows. On the other hand, the accelerated development of artificial intelligence in recent years provides novel tools to address complex problems such as the energy operation of a building. Within these tools, DRL (Deep Reinforcement Learning) adapts well to the type of problem to be solved. It is for all this that the development of a control model that manages energy use is proposed, considering the user's activity and the bioclimatic strategies that a home has, particularly for this work, natural ventilation.

## Repository organization

The repository is ordned to be easy to read and to develope.

The **`agents`** folder is where the building models are allocated.
```
-> agents
    -> conventional.py
    -> user.py
```

In the **`env`** folder there are alocated two files, the RLlib implementation for EnergyPlus and the
EnergyPlus Runner implementation of EnergyPlus Python API. This two scripts are executed in two threads 
to allow the simulation of the NREL software in a reinforcement learning way.
```
-> env
    ->nv_rllib_env.py
    ->nv_ep_runner.py
```
The **`epjson`** folder is where the building models are allocated.
```
-> epjson
    -> file_to_simulate.epJSON
```
The **`epw`** folder is where the weather file and the statistical of the respective wheather are allocated. You will find in the preprocess folder a notebook to calculate the statisticals pkl file from the epw.
```
-> epw
    -> wheather_file.epw
    -> wheather_stats.pkl
```
The **`postprocess`** folder...
```
-> postprocess
    -> postprocess_file.ipynb
```
The **`tools`** folder...
```
-> tools
    -> devices_space_action.py
    -> ep_episode_config.py
    -> weather_utils.py
```
Finally we have three scripts in the main repository that are used to configurate the 
experiment to be running or execute the evaluation of the policy trained and compare with the conventional policy.

-> init_training.py
-> init_conventional.py
-> init_evaluation.py
-> centralized_action_space.csv

### Conventional controls

Control mechanisms for different elements are established:

* Blnds or shades
* Windows
* On-Off of heating and cooling systems
* Thermostats dual comfort temperature

These are established according to intuitive conventional rules that are currently used in residential buildings.

### Clima stadistics

### Aditional tools

## How to use and how to work

![Implementación del entorno de EnergyPlus en RLlib.](execution_flow.png)

## Contribution

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


