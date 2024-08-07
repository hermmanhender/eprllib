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
import numpy as np

class RewardFunction:
    def __init__(
        self,
        EnvObject
    ):
        self.EnvObject = EnvObject
    
    def calculate_reward(
        self,
        infos: Dict[str,Dict[str,Any]] = None
        ) -> Dict[str,float]:
        return NotImplemented

# Defining the reward functions
def dalamagkidis_2007(EnvObject, infos: Dict) -> Dict[str,float]:
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
                'T_interior_name': 'Ti',
                'cooling_name': 'cooling',
                'heating_name': 'heating',
                'co2_name': 'co2'
            }
        }
        ```
    """
    reward_dict = {key: 0. for key in EnvObject._thermal_zone_ids}
    if not EnvObject.env_config.get('reward_function_config', False):
        ValueError('The reward function configuration is not defined. The default function will be use.')
    # thermal zone variables dicts
    cut_reward_len_timesteps_tz = {thermal_zone: 0 for thermal_zone in EnvObject._thermal_zone_ids}
    w1_tz = {thermal_zone: 0 for thermal_zone in EnvObject._thermal_zone_ids}
    w2_tz = {thermal_zone: 0 for thermal_zone in EnvObject._thermal_zone_ids}
    w3_tz = {thermal_zone: 0 for thermal_zone in EnvObject._thermal_zone_ids}
    ppd_name_tz = {thermal_zone: '' for thermal_zone in EnvObject._thermal_zone_ids}
    T_interior_name_tz = {thermal_zone: '' for thermal_zone in EnvObject._thermal_zone_ids}
    occupancy_name_tz = {thermal_zone: '' for thermal_zone in EnvObject._thermal_zone_ids}
    energy_ref_tz = {thermal_zone: 0 for thermal_zone in EnvObject._thermal_zone_ids}
    cooling_name_tz = {thermal_zone: '' for thermal_zone in EnvObject._thermal_zone_ids}
    heating_name_tz = {thermal_zone: '' for thermal_zone in EnvObject._thermal_zone_ids}
    co2_ref_tz = {thermal_zone: 0 for thermal_zone in EnvObject._thermal_zone_ids}
    co2_name_tz = {thermal_zone: '' for thermal_zone in EnvObject._thermal_zone_ids}
    
    for ThermalZone in EnvObject._thermal_zone_ids:
        # define which rewards will be considered
        comfort_reward = EnvObject.env_config['reward_function_config'][ThermalZone].get('comfort_reward', True)
        energy_reward = EnvObject.env_config['reward_function_config'][ThermalZone].get('energy_reward', True)
        co2_reward = EnvObject.env_config['reward_function_config'][ThermalZone].get('co2_reward', True)
        
        # if the EnvObject don't have the list to append the values here obtained, one list is created as a property of the EnvObject
        if not hasattr(EnvObject, 'ppd_dict') and comfort_reward:
            EnvObject.ppd_dict = {key: [] for key in EnvObject._agent_ids}
        if not hasattr(EnvObject, 'energy_dict') and energy_reward:
            EnvObject.energy_dict = {key: [] for key in EnvObject._agent_ids}
        if not hasattr(EnvObject, 'co2_dict') and co2_reward:
            EnvObject.co2_dict = {key: [] for key in EnvObject._agent_ids}
        
        # define the number of timesteps per episode
        cut_reward_len_timesteps_tz[ThermalZone] = EnvObject.env_config['reward_function_config'][ThermalZone].get('cut_reward_len_timesteps', 1)
        
        # define the ponderation parameters for thermal zone
        w1_tz[ThermalZone] = EnvObject.env_config['reward_function_config'][ThermalZone].get('w1', 0.80)
        w2_tz[ThermalZone] = EnvObject.env_config['reward_function_config'][ThermalZone].get('w2', 0.01)
        w3_tz[ThermalZone] = EnvObject.env_config['reward_function_config'][ThermalZone].get('w3', 0.20)
        
        if comfort_reward:
            ppd_name_tz[ThermalZone] = EnvObject.env_config['reward_function_config'][ThermalZone].get('ppd_name', False)
            T_interior_name_tz[ThermalZone] = EnvObject.env_config['reward_function_config'][ThermalZone].get('T_interior_name', False)
            occupancy_name_tz[ThermalZone] = EnvObject.env_config['reward_function_config'][ThermalZone].get('occupancy_name', False)
            if not ppd_name_tz[ThermalZone] or not occupancy_name_tz[ThermalZone] or not T_interior_name_tz[ThermalZone]:
                ValueError('The names of the variables are not defined')
        
        if energy_reward:
            energy_ref_tz[ThermalZone] = EnvObject.env_config['reward_function_config'][ThermalZone].get('energy_ref',False)
            cooling_name_tz[ThermalZone] = EnvObject.env_config['reward_function_config'][ThermalZone].get('cooling_name', False)
            heating_name_tz[ThermalZone] = EnvObject.env_config['reward_function_config'][ThermalZone].get('heating_name', False)
            if not energy_ref_tz[ThermalZone] or not cooling_name_tz[ThermalZone] or not heating_name_tz[ThermalZone]:
                ValueError('The names of the variables are not defined')
        
        if co2_reward:
            co2_ref_tz[ThermalZone] = EnvObject.env_config['reward_function_config'][ThermalZone].get('co2_ref',False)
            co2_name_tz[ThermalZone] = EnvObject.env_config['reward_function_config'][ThermalZone].get('co2_name', False)
            occupancy_name_tz[ThermalZone] = EnvObject.env_config['reward_function_config'][ThermalZone].get('occupancy_name', False)
            if not co2_ref_tz[ThermalZone] or not co2_name_tz[ThermalZone] or not occupancy_name_tz[ThermalZone]:
                ValueError('The names of the variables are not defined')
    
    # agent variables dicts
    cut_reward_len_timesteps_a = {agent: 0 for agent in EnvObject._agent_ids}
    w1_a = {agent: 0 for agent in EnvObject._agent_ids}
    w2_a = {agent: 0 for thermal_zone in EnvObject._agent_ids}
    w3_a = {agent: 0 for agent in EnvObject._agent_ids}
    ppd_name_a = {agent: '' for agent in EnvObject._agent_ids}
    T_interior_name_a = {agent: '' for agent in EnvObject._agent_ids}
    occupancy_name_a = {agent: '' for agent in EnvObject._agent_ids}
    energy_ref_a = {agent: 0 for agent in EnvObject._agent_ids}
    cooling_name_a = {agent: '' for agent in EnvObject._agent_ids}
    heating_name_a = {agent: '' for agent in EnvObject._agent_ids}
    co2_ref_a = {agent: 0 for agent in EnvObject._agent_ids}
    co2_name_a = {agent: '' for agent in EnvObject._agent_ids}
    
    # asign the ThermalZone values above defined to each agent
    for agent in EnvObject._agent_ids:
        agent_thermal_zone = EnvObject.env_config['agents_config'][agent]['thermal_zone']
        
        comfort_reward = EnvObject.env_config['reward_function_config'][agent_thermal_zone].get('comfort_reward', True)
        energy_reward = EnvObject.env_config['reward_function_config'][agent_thermal_zone].get('energy_reward', True)
        co2_reward = EnvObject.env_config['reward_function_config'][agent_thermal_zone].get('co2_reward', True)
        
        cut_reward_len_timesteps_a[agent] = cut_reward_len_timesteps_tz[agent_thermal_zone]
        w1_a[agent] = w1_tz[agent_thermal_zone]
        w2_a[agent] = w2_tz[agent_thermal_zone]
        w3_a[agent] = w3_tz[agent_thermal_zone]
        
        if comfort_reward:
            ppd_name_a[agent] = ppd_name_tz[ThermalZone]
            T_interior_name_a[agent] = T_interior_name_tz[ThermalZone]
            occupancy_name_a[agent] = occupancy_name_tz[ThermalZone]
            
        if energy_reward:
            energy_ref_a[agent] = energy_ref_tz[ThermalZone]
            cooling_name_a[agent] = cooling_name_tz[ThermalZone]
            heating_name_a[agent] = heating_name_tz[ThermalZone]
        
        if co2_reward:
            co2_ref_a[agent] = co2_ref_tz[ThermalZone]
            co2_name_a[agent] = co2_name_tz[ThermalZone]
            occupancy_name_a[agent] = occupancy_name_tz[ThermalZone]
    
    # get the values of the energy, PPD, and CO2 from the infos dict
    for agent in EnvObject._agent_ids:
        agent_thermal_zone = EnvObject.env_config['agents_config'][agent]['thermal_zone']
        
        comfort_reward = EnvObject.env_config['reward_function_config'][agent_thermal_zone].get('comfort_reward', True)
        energy_reward = EnvObject.env_config['reward_function_config'][agent_thermal_zone].get('energy_reward', True)
        co2_reward = EnvObject.env_config['reward_function_config'][agent_thermal_zone].get('co2_reward', True)
        
        if comfort_reward:
            _ppd_name = ppd_name_a[agent]
            _occupancy_name = occupancy_name_a[agent]
            ppd = infos[agent][_ppd_name]
            occupancy = infos[agent][_occupancy_name]
            if occupancy == 0:
                ppd = 0
            EnvObject.ppd_dict[agent].append(ppd)
        if energy_reward:
            _cooling_name = cooling_name_a[agent]
            _heating_name = heating_name_a[agent]
            cooling_meter = infos[agent][_cooling_name]
            heating_meter = infos[agent][_heating_name]
            EnvObject.energy_dict[agent].append(cooling_meter+heating_meter)
        if co2_reward:
            co2 = infos[agent][co2_name]
            occupancy = infos[agent][_occupancy_name]
            if occupancy == 0:
                co2 = 0
            EnvObject.co2_dict[agent].append(co2)
    
    # calculate the reward if the timestep is divisible by the cut_reward_len_timesteps.
    # if don't return 0.
    for agent in EnvObject._agent_ids:
        if EnvObject.timestep % cut_reward_len_timesteps_a[agent] == 0:
            
            agent_thermal_zone = EnvObject.env_config['agents_config'][agent]['thermal_zone']
            comfort_reward = EnvObject.env_config['reward_function_config'][agent_thermal_zone].get('comfort_reward', True)
            energy_reward = EnvObject.env_config['reward_function_config'][agent_thermal_zone].get('energy_reward', True)
            co2_reward = EnvObject.env_config['reward_function_config'][agent_thermal_zone].get('co2_reward', True)
            
            if comfort_reward:
                _T_interior_name = T_interior_name_a[agent]
                _T_interior = infos[agent][_T_interior_name]
                rew1 = -w1_a[agent]*(sum(EnvObject.ppd_dict[agent])/cut_reward_len_timesteps_a[agent]/100)
                # If there are not people, only the reward is calculated when the environment is far away
                # from the comfort temperature ranges. This limits are recommended in EnergyPlus documentation:
                # InputOutput Reference p.522
                if _T_interior > 29.4:
                    rew1 += -10
                elif _T_interior < 16.7:
                    rew1 += -10
            else:
                rew1 = 0
            if energy_reward:
                rew2 = -w2_a[agent]*(sum(EnvObject.energy_dict[agent])/cut_reward_len_timesteps_a[agent]/energy_ref_a[agent])
            else:
                rew2 = 0
            if co2_reward:
                rew3 = 0
                for co2 in range(len([key for key in EnvObject.co2_dict[agent].keys()])):
                    rew3 += -w3_a[agent]*(1/(1+exp(-0.06*(EnvObject.co2_dict[agent][co2]-co2_ref_a[agent]))))
                rew3 = rew3/cut_reward_len_timesteps_a[agent]
            else:
                rew3 = 0
            
            reward_dict[agent] = rew1 + rew2 + rew3
            
            # emptly the lists
            if comfort_reward:
                EnvObject.ppd_dict = {key: [] for key in EnvObject._agent_ids}
            if energy_reward:
                EnvObject.energy_dict = {key: [] for key in EnvObject._agent_ids}
            if co2_reward:
                EnvObject.co2_dict = {key: [] for key in EnvObject._agent_ids}
            
    return reward_dict
    
def normalize_reward_function(EnvObject, infos: Dict) -> Dict[str,float]:
    """This function returns the normalize reward calcualted as the sum of the penalty of the energy 
    amount of one week divide per the maximun reference energy demand and the average PPD comfort metric
    divide per the maximal PPF value that can be take (100). Also, each term is divide per the longitude
    of the episode and multiply for a ponderation factor of beta for the energy and (1-beta) for the comfort.
    Both terms are negatives, representing a penalti for demand energy and for generate discomfort.

    Args:
        self (Environment): RLlib environment.
        obs (dict): Zone Mean Air Temperature for the Thermal Zone in °C.
        infos (dict): infos dict must to provide the occupancy level and the Zone Mean Temperature.

    Returns:
        float: reward normalize value
    """
    reward_dict = {key: 0. for key in EnvObject._agent_ids}
    if not EnvObject.env_config.get('reward_function_config', False):
        ValueError('The reward function configuration is not defined. The default function will be use.')
    
    # if the EnvObject don't have the list to append the values here obtained, one list is created as a property of the EnvObject
    if not hasattr(EnvObject, 'ppd_dict'):
        EnvObject.ppd_dict = {key: [] for key in EnvObject._agent_ids}
    if not hasattr(EnvObject, 'energy_dict'):
        EnvObject.energy_dict = {key: [] for key in EnvObject._agent_ids}
    
    # variables dicts
    beta_reward_tz = {thermal_zone: 0 for thermal_zone in EnvObject._thermal_zone_ids}
    cut_reward_len_timesteps_tz = {thermal_zone: 0 for thermal_zone in EnvObject._thermal_zone_ids}
    ppd_name_tz = {thermal_zone: '' for thermal_zone in EnvObject._thermal_zone_ids}
    T_interior_name_tz = {thermal_zone: '' for thermal_zone in EnvObject._thermal_zone_ids}
    occupancy_name_tz = {thermal_zone: '' for thermal_zone in EnvObject._thermal_zone_ids}
    cooling_energy_ref_tz = {thermal_zone: 0 for thermal_zone in EnvObject._thermal_zone_ids}
    heating_energy_ref_tz = {thermal_zone: 0 for thermal_zone in EnvObject._thermal_zone_ids}
    cooling_name_tz = {thermal_zone: '' for thermal_zone in EnvObject._thermal_zone_ids}
    heating_name_tz = {thermal_zone: '' for thermal_zone in EnvObject._thermal_zone_ids}
    
    for ThermalZone in EnvObject._thermal_zone_ids:
        # define which rewards will be considered
        comfort_reward = EnvObject.env_config['reward_function_config'][ThermalZone].get('comfort_reward', True)
        energy_reward = EnvObject.env_config['reward_function_config'][ThermalZone].get('energy_reward', True)
        
        # define the number of timesteps per episode
        cut_reward_len_timesteps_tz[ThermalZone] = EnvObject.env_config['reward_function_config'][ThermalZone].get('cut_reward_len_timesteps', 1)
        
        # define the beta reward
        beta_reward_tz[ThermalZone] = EnvObject.env_config['reward_function_config'][ThermalZone].get('beta_reward', 0.5)
    
        if comfort_reward:
            ppd_name_tz[ThermalZone] = EnvObject.env_config['reward_function_config'][ThermalZone].get('ppd_name', False)
            T_interior_name_tz[ThermalZone] = EnvObject.env_config['reward_function_config'][ThermalZone].get('T_interior_name', False)
            occupancy_name_tz[ThermalZone] = EnvObject.env_config['reward_function_config'][ThermalZone].get('occupancy_name', False)
            if not ppd_name_tz[ThermalZone] or not occupancy_name_tz[ThermalZone] or not T_interior_name_tz[ThermalZone]:
                ValueError('The names of the variables are not defined')
        
        if energy_reward:
            cooling_energy_ref_tz[ThermalZone] = EnvObject.env_config['reward_function_config'][ThermalZone].get('cooling_energy_ref', False)
            heating_energy_ref_tz[ThermalZone] = EnvObject.env_config['reward_function_config'][ThermalZone].get('heating_energy_ref', False)
            cooling_name_tz[ThermalZone] = EnvObject.env_config['reward_function_config'][ThermalZone].get('cooling_name', False)
            heating_name_tz[ThermalZone] = EnvObject.env_config['reward_function_config'][ThermalZone].get('heating_name', False)
            if not cooling_energy_ref_tz[ThermalZone] or not heating_energy_ref_tz[ThermalZone] or not cooling_name_tz[ThermalZone] or not heating_name_tz[ThermalZone]:
                ValueError('The names of the variables are not defined')
    
    
    beta_reward_a = {agent: 0 for agent in EnvObject._agent_ids}
    cut_reward_len_timesteps_a = {agent: 0 for agent in EnvObject._agent_ids}
    ppd_name_a = {agent: '' for agent in EnvObject._agent_ids}
    T_interior_name_a = {agent: '' for agent in EnvObject._agent_ids}
    occupancy_name_a = {agent: '' for agent in EnvObject._agent_ids}
    cooling_energy_ref_a = {agent: 0 for agent in EnvObject._agent_ids}
    heating_energy_ref_a = {agent: 0 for agent in EnvObject._agent_ids}
    cooling_name_a = {agent: '' for agent in EnvObject._agent_ids}
    heating_name_a = {agent: '' for agent in EnvObject._agent_ids}
    
    # asign the ThermalZone values above defined to each agent
    for agent in EnvObject._agent_ids:
        agent_thermal_zone = EnvObject.env_config['agents_config'][agent]['thermal_zone']
        comfort_reward = EnvObject.env_config['reward_function_config'][agent_thermal_zone].get('comfort_reward', True)
        energy_reward = EnvObject.env_config['reward_function_config'][agent_thermal_zone].get('energy_reward', True)
        
        cut_reward_len_timesteps_a[agent] = cut_reward_len_timesteps_tz[agent_thermal_zone]
        beta_reward_a[agent] = beta_reward_tz[agent_thermal_zone]
        
        if comfort_reward:
            ppd_name_a[agent] = ppd_name_tz[agent_thermal_zone]
            T_interior_name_a[agent] = T_interior_name_tz[agent_thermal_zone]
            occupancy_name_a[agent] = occupancy_name_tz[agent_thermal_zone]
            
        if energy_reward:
            cooling_energy_ref_a[agent] = cooling_energy_ref_tz[agent_thermal_zone]
            heating_energy_ref_a[agent] = heating_energy_ref_tz[agent_thermal_zone]
            cooling_name_a[agent] = cooling_name_tz[agent_thermal_zone]
            heating_name_a[agent] = heating_name_tz[agent_thermal_zone]
        
    # get the values of the energy, PPD, and CO2 from the infos dict
    for agent in EnvObject._agent_ids:
        agent_thermal_zone = EnvObject.env_config['agents_config'][agent]['thermal_zone']
        comfort_reward = EnvObject.env_config['reward_function_config'][agent_thermal_zone].get('comfort_reward', True)
        energy_reward = EnvObject.env_config['reward_function_config'][agent_thermal_zone].get('energy_reward', True)
        
        if comfort_reward:
            _ppd_name = ppd_name_a[agent]
            _occupancy_name = occupancy_name_a[agent]
            _T_interior_name = T_interior_name_a[agent]
            ppd = infos[agent][_ppd_name]
            occupancy = infos[agent][_occupancy_name]
            T_interior = infos[agent][_T_interior_name]
            if occupancy > 0:
                if infos[agent][_ppd_name] < 5:
                    ppd = infos[agent][_ppd_name] = 5
                else:
                    ppd = infos[agent][_ppd_name]
            else:
                ppd = infos[agent][_ppd_name] = 0
            # If there are not people, only the reward is calculated when the environment is far away
            # from the comfort temperature ranges. This limits are recommended in EnergyPlus documentation:
            # InputOutput Reference p.522
            if T_interior > 29.4:
                ppd = infos[agent][_ppd_name] = 100
            elif T_interior < 16.7:
                ppd = infos[agent][_ppd_name] = 100
                
            EnvObject.ppd_dict[agent].append(ppd)
        
        if energy_reward:
            _cooling_name = cooling_name_a[agent]
            _heating_name = heating_name_a[agent]
            cooling_meter = infos[agent][_cooling_name]
            heating_meter = infos[agent][_heating_name]
            
            EnvObject.energy_dict[agent].append(cooling_meter/cooling_energy_ref_a[agent]+heating_meter/heating_energy_ref_a[agent])
    
    # calculate the reward if the timestep is divisible by the cut_reward_len_timesteps.
    # if don't return 0.
    for agent in EnvObject._agent_ids:
        if EnvObject.timestep % cut_reward_len_timesteps_a[agent] == 0:
            agent_thermal_zone = EnvObject.env_config['agents_config'][agent]['thermal_zone']
            comfort_reward = EnvObject.env_config['reward_function_config'][agent_thermal_zone].get('comfort_reward', True)
            energy_reward = EnvObject.env_config['reward_function_config'][agent_thermal_zone].get('energy_reward', True)
            
            if comfort_reward:
                ppd_avg = sum(EnvObject.ppd_dict[agent])/len(EnvObject.ppd_dict[agent])
                rew1 = -(1-beta_reward_a[agent])*(1/(1+np.exp(-0.1*(ppd_avg-45))))
            else:
                rew1 = 0
            
            if energy_reward:
                rew2 = -beta_reward_a[agent]*(sum(EnvObject.energy_dict[agent])/len(EnvObject.energy_dict[agent]))
            else:
                rew2 = 0
            
            reward_dict[agent] = rew1 + rew2
            
    # emptly the lists
    for agent in EnvObject._agent_ids:
        if EnvObject.timestep % cut_reward_len_timesteps_a[agent] == 0:
            EnvObject.ppd_dict = {key: [] for key in EnvObject._agent_ids}
            EnvObject.energy_dict = {key: [] for key in EnvObject._agent_ids}
            
    return reward_dict
