"""This module contains the reward functions that different authors have proposed as well as some 
developed within the framework of the development of this library. It is expected over time to be 
able to establish reward functions that optimize the operation of buildings.

For greater flexibility of the library, all reward functions must have the following format:

    ```
    def reward_function_name(EnvObject, infos: dict) -> float:
        ...
        return reward
    ```
    
The arguments correspond to the following:
    EnvObject: It is the set of properties contained in `self` of the environment used.
    infos (Dict): This is the dictionary that is shared with the `reset()` and `step()` methods 
    of the Gymnasium library along with the observation and, if applicable, the `terminated` and 
    `truncated` values.

It has been preferred to use the `infos` dictionary and not the observation, since the latter is 
a numpy array and cannot be called by key values, which is prone to errors when developing the program 
and indexing a arrangement may change.

# Property `cut_reward_len_timesteps`
This property is used to define the number of timesteps that the reward is calculated for. Not all
the rewards function have this property, so it is not mandatory to define it.

# Property `beta_reward`
This property is used to define the value of the beta parameter of the reward function. Not all
the rewards function have this property, so it is not mandatory to define it.
"""
# Importing the neccesary libraries
from typing import Dict
from math import exp

# Defining the reward functions
def dalamagkidis_2007(EnvObject, infos: Dict) -> float:
    """El autor plantea una función de recompensa con tres térmicos ponderados. Cada uno de estos 
    términos corresponde a una penalidad por: disconfort, uso de energía, elevada consentración de CO2.
    Cada uno de los términos es normalizado. En el caso del disconfort, el valor máximo que puede tomar es
    100%. En el caso de la energía, se debe indicar un valor de referencia como 'energy_ref'; su valor por 
    default es 1. Para la normalización de la concentración de CO2 se toma el valor de referencia indicado
    por el autor en su trabajo; este se puede modificar con la variables 'co2_ref' y su valor de default
    es 870.
    
    En la implementación se permite optar por utilizar o no cada término de la penalización.

    Args:
        EnvObject: It is the set of properties contained in `self` of the environment used.
        infos (Dict): This is the dictionary that is shared with the `reset()` and `step()` methods 
        of the Gymnasium library along with the observation and, if applicable, the `terminated` and 
        `truncated` values.

    Returns:
        float: reward value.
        
    Configuration Example:
        ```
        from eprllib.tools.rewards import dalamagkidis_2007
        
        env_config = {
            # Your config here
            ...
            "ep_variables":{
                ...
                "occupancy": ("Zone People Occupant Count", "Thermal Zone: Living"),
                "ppd": ("Zone Thermal Comfort Fanger Model PPD", "Living Occupancy"),
            },
            "ep_meters": {
                ...
                "heating": "Heating:DistrictHeatingWater",
                "cooling": "Cooling:DistrictCooling",
            },
            "ep_actuators": {
                ...
            },
            "infos_variables": [
                ...
                "ppd", 
                'heating', 
                'cooling',
                'occupancy'
            ],
            ...
            # Reward config
            'reward_function': dalamagkidis_2007,
            'reward_function_config': {
                # cut_reward_len_timesteps: Este parámetro permite que el agente no reciba una recompensa en cada paso de tiempo, en cambio las variables para el cálculo de la recompensa son almacenadas en una lista para luego utilizar una recompensa promedio cuando se alcanza la cantidad de pasos de tiempo indicados por 'cut_reward_len_timesteps'.
                'cut_reward_len_timesteps': 1,
                # Parámetros para la exclusión de términos de la recompensa
                'comfort_reward': True,
                'energy_reward': True,
                'co2_reward': True,                
                # w1: Parámetros de ponderación para el confort.
                'w1': 0.80,
                # w2: Parámetros de ponderación para el uso de energía.
                'w2': 0.01,
                # w3: Parámetros de ponderación para la concentración de CO2.
                'w3': 0.20,                 
                # energy_ref: El valor de referencia depende del entorno. Este puede corresponder a la energía máxima que puede demandar el entorno en un paso de tiempo, un valor de energía promedio u otro.
                'energy_ref': 6805274,
                # co2_ref: Este parámtero indica el valor de referencia de consentración de CO2 que se espera tener en un ambiente con una calidad de aire óptima.
                'co2_ref': 870,
                # Nombres de las variables utilizadas en su configuración del entorno.
                'occupancy_name': 'occupancy',
                'ppd_name': 'ppd',
                T_interior_name: 'Ti',
                'cooling_name': 'cooling',
                'heating_name': 'heating',
                'co2_name': 'co2'
            }
        }
        ```
    """
    if not EnvObject.env_config.get('reward_function_config', False):
        raise Exception('The reward function configuration is not defined')
    
    # define which rewards will be considered
    comfort_reward = EnvObject.env_config['reward_function_config'].get('comfort_reward', True)
    energy_reward = EnvObject.env_config['reward_function_config'].get('energy_reward', True)
    co2_reward = EnvObject.env_config['reward_function_config'].get('co2_reward', True)
    
    # if the EnvObject don't have the list to append the values here obtained, one list is created as a property of the EnvObject
    if not hasattr(EnvObject, 'ppd_list') and comfort_reward:
        EnvObject.ppd_list = []
    if not hasattr(EnvObject, 'energy_list') and energy_reward:
        EnvObject.energy_list = []
    if not hasattr(EnvObject, 'co2_list') and co2_reward:
        EnvObject.co2_list = []
    
    # define the number of timesteps per episode
    cut_reward_len_timesteps = EnvObject.env_config['reward_function_config'].get('cut_reward_len_timesteps', 1)
    
    # define the ponderation parameters
    w1 = EnvObject.env_config['reward_function_config'].get('w1', 0.80)
    w2 = EnvObject.env_config['reward_function_config'].get('w2', 0.01)
    w3 = EnvObject.env_config['reward_function_config'].get('w3', 0.20)
    
    # get the values of the energy, PPD, and CO2 from the infos dict
    agent_ids = EnvObject.env_config['agent_ids']
    if comfort_reward:
        ppd_name = EnvObject.env_config['reward_function_config'].get('ppd_name', False)
        T_interior_name = EnvObject.env_config['reward_function_config'].get('T_interior_name', False)
        occupancy_name = EnvObject.env_config['reward_function_config'].get('occupancy_name', False)
        if not ppd_name or not occupancy_name or not T_interior_name:
            raise Exception('The names of the variables are not defined')
        ppd = infos[agent_ids[0]][ppd_name]
        T_interior = infos[agent_ids[0]][T_interior_name]
        occupancy = infos[agent_ids[0]][occupancy_name]
        if occupancy == 0:
            ppd = 0
        EnvObject.ppd_list.append(ppd)
    if energy_reward:
        energy_ref = EnvObject.env_config['reward_function_config'].get('energy_ref',False)
        cooling_name = EnvObject.env_config['reward_function_config'].get('cooling_name', False)
        heating_name = EnvObject.env_config['reward_function_config'].get('heating_name', False)
        if not energy_ref or not cooling_name or not heating_name:
            raise Exception('The names of the variables are not defined')
        cooling_meter = infos[agent_ids[0]][cooling_name]
        heating_meter = infos[agent_ids[0]][heating_name]
        EnvObject.energy_list.append(cooling_meter+heating_meter)
    if co2_reward:
        co2_ref = EnvObject.env_config['reward_function_config'].get('co2_ref',False)
        co2_name = EnvObject.env_config['reward_function_config'].get('co2_name', False)
        occupancy_name = EnvObject.env_config['reward_function_config'].get('occupancy_name', False)
        if not co2_ref or not co2_name or not occupancy_name:
            raise Exception('The names of the variables are not defined')
        co2 = infos[agent_ids[0]][co2_name]
        occupancy = infos[agent_ids[0]][occupancy_name]
        if occupancy == 0:
            co2 = 0
        EnvObject.co2_list.append(co2)
    
    # calculate the reward if the timestep is divisible by the cut_reward_len_timesteps.
    # if don't return 0.
    if EnvObject.timestep % cut_reward_len_timesteps == 0:
        if comfort_reward:
            rew1 = -w1*(sum(EnvObject.ppd_list)/cut_reward_len_timesteps/100)
            # If there are not people, only the reward is calculated when the environment is far away
            # from the comfort temperature ranges. This limits are recommended in EnergyPlus documentation:
            # InputOutput Reference p.522
            if T_interior > 29.4:
                rew1 += -10
            elif T_interior < 16.7:
                rew1 += -10
        else:
            rew1 = 0
        if energy_reward:
            rew2 = -w2*(sum(EnvObject.energy_list)/cut_reward_len_timesteps/energy_ref)
        else:
            rew2 = 0
        if co2_reward:
            rew3 = -w3*(sum(1/(1+exp(-0.06(co2-co2_ref))))/cut_reward_len_timesteps)
        else:
            rew3 = 0
        reward = rew1 + rew2 + rew3
        
        # emptly the lists
        if comfort_reward:
            EnvObject.ppd_list = []
        if energy_reward:
            EnvObject.energy_list = []
        if co2_reward:
            EnvObject.co2_list = []
            
        return reward
    else:
        return 0