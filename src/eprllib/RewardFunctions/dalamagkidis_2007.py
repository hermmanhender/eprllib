"""
Dalamagkidis et al. (2007)
==========================

This module contains the implementation of the reward function proposed by Dalamagkidis et al. (2007).
The reward function is based on three components: discomfort, energy usage, and CO2 concentration.

The discomfort component is calculated based on the temperature difference between the indoor air temperature
and the outdoor temperature. The energy usage component is calculated based on the energy consumption of the
building. The CO2 concentration component is calculated based on the concentration of CO2 in the building.

The reward function is weighted by three factors: discomfort, energy usage, and CO2 concentration. Each factor
is normalized to a maximum value of 100%.

The reward function is implemented as a class that inherits from the `RewardFunction` base class. The `calculate_reward`
method takes the environment information as input and returns the reward value as a dictionary, where the keys
are the agent names and the values are the corresponding reward values.
"""

from typing import Dict, Any
from math import exp
from eprllib.Env.MultiAgent.EnergyPlusEnvironment import EnergyPlusEnv_v0
from eprllib.RewardFunctions.RewardFunctions import RewardFunction

class dalamagkidis_2007(RewardFunction):
    def __init__(
        self,
        EnvObject: EnergyPlusEnv_v0
        ):
        super().__init__(EnvObject)
        
    def calculate_reward(
    self,
    infos: Dict[str,Dict[str,Any]] = None
    ) -> Dict[str,float]:
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
        reward_dict = {key: 0. for key in self.EnvObject._thermal_zone_ids}
        if not self.EnvObject.env_config.get('reward_fn_config', False):
            ValueError('The reward function configuration is not defined. The default function will be use.')
        # thermal zone variables dicts
        cut_reward_len_timesteps_tz = {thermal_zone: 0 for thermal_zone in self.EnvObject._thermal_zone_ids}
        w1_tz = {thermal_zone: 0 for thermal_zone in self.EnvObject._thermal_zone_ids}
        w2_tz = {thermal_zone: 0 for thermal_zone in self.EnvObject._thermal_zone_ids}
        w3_tz = {thermal_zone: 0 for thermal_zone in self.EnvObject._thermal_zone_ids}
        ppd_name_tz = {thermal_zone: '' for thermal_zone in self.EnvObject._thermal_zone_ids}
        T_interior_name_tz = {thermal_zone: '' for thermal_zone in self.EnvObject._thermal_zone_ids}
        occupancy_name_tz = {thermal_zone: '' for thermal_zone in self.EnvObject._thermal_zone_ids}
        energy_ref_tz = {thermal_zone: 0 for thermal_zone in self.EnvObject._thermal_zone_ids}
        cooling_name_tz = {thermal_zone: '' for thermal_zone in self.EnvObject._thermal_zone_ids}
        heating_name_tz = {thermal_zone: '' for thermal_zone in self.EnvObject._thermal_zone_ids}
        co2_ref_tz = {thermal_zone: 0 for thermal_zone in self.EnvObject._thermal_zone_ids}
        co2_name_tz = {thermal_zone: '' for thermal_zone in self.EnvObject._thermal_zone_ids}
        
        for ThermalZone in self.EnvObject._thermal_zone_ids:
            # define which rewards will be considered
            comfort_reward = self.EnvObject.env_config['reward_fn_config'][ThermalZone].get('comfort_reward', True)
            energy_reward = self.EnvObject.env_config['reward_fn_config'][ThermalZone].get('energy_reward', True)
            co2_reward = self.EnvObject.env_config['reward_fn_config'][ThermalZone].get('co2_reward', True)
            
            # if the EnvObject don't have the list to append the values here obtained, one list is created as a property of the EnvObject
            if not hasattr(self.EnvObject, 'ppd_dict') and comfort_reward:
                self.EnvObject.ppd_dict = {key: [] for key in self.EnvObject._agent_ids}
            if not hasattr(self.EnvObject, 'energy_dict') and energy_reward:
                self.EnvObject.energy_dict = {key: [] for key in self.EnvObject._agent_ids}
            if not hasattr(self.EnvObject, 'co2_dict') and co2_reward:
                self.EnvObject.co2_dict = {key: [] for key in self.EnvObject._agent_ids}
            
            # define the number of timesteps per episode
            cut_reward_len_timesteps_tz[ThermalZone] = self.EnvObject.env_config['reward_fn_config'][ThermalZone].get('cut_reward_len_timesteps', 1)
            
            # define the ponderation parameters for thermal zone
            w1_tz[ThermalZone] = self.EnvObject.env_config['reward_fn_config'][ThermalZone].get('w1', 0.80)
            w2_tz[ThermalZone] = self.EnvObject.env_config['reward_fn_config'][ThermalZone].get('w2', 0.01)
            w3_tz[ThermalZone] = self.EnvObject.env_config['reward_fn_config'][ThermalZone].get('w3', 0.20)
            
            if comfort_reward:
                ppd_name_tz[ThermalZone] = self.EnvObject.env_config['reward_fn_config'][ThermalZone].get('ppd_name', False)
                T_interior_name_tz[ThermalZone] = self.EnvObject.env_config['reward_fn_config'][ThermalZone].get('T_interior_name', False)
                occupancy_name_tz[ThermalZone] = self.EnvObject.env_config['reward_fn_config'][ThermalZone].get('occupancy_name', False)
                if not ppd_name_tz[ThermalZone] or not occupancy_name_tz[ThermalZone] or not T_interior_name_tz[ThermalZone]:
                    ValueError('The names of the variables are not defined')
            
            if energy_reward:
                energy_ref_tz[ThermalZone] = self.EnvObject.env_config['reward_fn_config'][ThermalZone].get('energy_ref',False)
                cooling_name_tz[ThermalZone] = self.EnvObject.env_config['reward_fn_config'][ThermalZone].get('cooling_name', False)
                heating_name_tz[ThermalZone] = self.EnvObject.env_config['reward_fn_config'][ThermalZone].get('heating_name', False)
                if not energy_ref_tz[ThermalZone] or not cooling_name_tz[ThermalZone] or not heating_name_tz[ThermalZone]:
                    ValueError('The names of the variables are not defined')
            
            if co2_reward:
                co2_ref_tz[ThermalZone] = self.EnvObject.env_config['reward_fn_config'][ThermalZone].get('co2_ref',False)
                co2_name_tz[ThermalZone] = self.EnvObject.env_config['reward_fn_config'][ThermalZone].get('co2_name', False)
                occupancy_name_tz[ThermalZone] = self.EnvObject.env_config['reward_fn_config'][ThermalZone].get('occupancy_name', False)
                if not co2_ref_tz[ThermalZone] or not co2_name_tz[ThermalZone] or not occupancy_name_tz[ThermalZone]:
                    ValueError('The names of the variables are not defined')
        
        # agent variables dicts
        cut_reward_len_timesteps_a = {agent: 0 for agent in self.EnvObject._agent_ids}
        w1_a = {agent: 0 for agent in self.EnvObject._agent_ids}
        w2_a = {agent: 0 for agent in self.EnvObject._agent_ids}
        w3_a = {agent: 0 for agent in self.EnvObject._agent_ids}
        ppd_name_a = {agent: '' for agent in self.EnvObject._agent_ids}
        T_interior_name_a = {agent: '' for agent in self.EnvObject._agent_ids}
        occupancy_name_a = {agent: '' for agent in self.EnvObject._agent_ids}
        energy_ref_a = {agent: 0 for agent in self.EnvObject._agent_ids}
        cooling_name_a = {agent: '' for agent in self.EnvObject._agent_ids}
        heating_name_a = {agent: '' for agent in self.EnvObject._agent_ids}
        co2_ref_a = {agent: 0 for agent in self.EnvObject._agent_ids}
        co2_name_a = {agent: '' for agent in self.EnvObject._agent_ids}
        
        # asign the ThermalZone values above defined to each agent
        for agent in self.EnvObject._agent_ids:
            agent_thermal_zone = self.EnvObject.env_config['agents_config'][agent]['thermal_zone']
            
            comfort_reward = self.EnvObject.env_config['reward_fn_config'][agent_thermal_zone].get('comfort_reward', True)
            energy_reward = self.EnvObject.env_config['reward_fn_config'][agent_thermal_zone].get('energy_reward', True)
            co2_reward = self.EnvObject.env_config['reward_fn_config'][agent_thermal_zone].get('co2_reward', True)
            
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
        for agent in self.EnvObject._agent_ids:
            agent_thermal_zone = self.EnvObject.env_config['agents_config'][agent]['thermal_zone']
            
            comfort_reward = self.EnvObject.env_config['reward_fn_config'][agent_thermal_zone].get('comfort_reward', True)
            energy_reward = self.EnvObject.env_config['reward_fn_config'][agent_thermal_zone].get('energy_reward', True)
            co2_reward = self.EnvObject.env_config['reward_fn_config'][agent_thermal_zone].get('co2_reward', True)
            
            if comfort_reward:
                _ppd_name = ppd_name_a[agent]
                _occupancy_name = occupancy_name_a[agent]
                _co2_name = co2_name_a[agent]
                
                ppd = infos[agent][_ppd_name]
                occupancy = infos[agent][_occupancy_name]
                
                if occupancy == 0:
                    ppd = 0
                self.EnvObject.ppd_dict[agent].append(ppd)
            if energy_reward:
                _cooling_name = cooling_name_a[agent]
                _heating_name = heating_name_a[agent]
                cooling_meter = infos[agent][_cooling_name]
                heating_meter = infos[agent][_heating_name]
                self.EnvObject.energy_dict[agent].append(cooling_meter+heating_meter)
            if co2_reward:
                co2 = infos[agent][_co2_name]
                occupancy = infos[agent][_occupancy_name]
                if occupancy == 0:
                    co2 = 0
                self.EnvObject.co2_dict[agent].append(co2)
        
        # calculate the reward if the timestep is divisible by the cut_reward_len_timesteps.
        # if don't return 0.
        for agent in self.EnvObject._agent_ids:
            if self.EnvObject.timestep % cut_reward_len_timesteps_a[agent] == 0:
                
                agent_thermal_zone = self.EnvObject.env_config['agents_config'][agent]['thermal_zone']
                comfort_reward = self.EnvObject.env_config['reward_fn_config'][agent_thermal_zone].get('comfort_reward', True)
                energy_reward = self.EnvObject.env_config['reward_fn_config'][agent_thermal_zone].get('energy_reward', True)
                co2_reward = self.EnvObject.env_config['reward_fn_config'][agent_thermal_zone].get('co2_reward', True)
                
                if comfort_reward:
                    _T_interior_name = T_interior_name_a[agent]
                    _T_interior = infos[agent][_T_interior_name]
                    rew1 = -w1_a[agent]*(sum(self.EnvObject.ppd_dict[agent])/cut_reward_len_timesteps_a[agent]/100)
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
                    rew2 = -w2_a[agent]*(sum(self.EnvObject.energy_dict[agent])/cut_reward_len_timesteps_a[agent]/energy_ref_a[agent])
                else:
                    rew2 = 0
                if co2_reward:
                    rew3 = 0
                    for co2 in range(len([key for key in self.EnvObject.co2_dict[agent].keys()])):
                        rew3 += -w3_a[agent]*(1/(1+exp(-0.06*(self.EnvObject.co2_dict[agent][co2]-co2_ref_a[agent]))))
                    rew3 = rew3/cut_reward_len_timesteps_a[agent]
                else:
                    rew3 = 0
                
                reward_dict[agent] = rew1 + rew2 + rew3
                
                # emptly the lists
                if comfort_reward:
                    self.EnvObject.ppd_dict = {key: [] for key in self.EnvObject._agent_ids}
                if energy_reward:
                    self.EnvObject.energy_dict = {key: [] for key in self.EnvObject._agent_ids}
                if co2_reward:
                    self.EnvObject.co2_dict = {key: [] for key in self.EnvObject._agent_ids}
                
        return reward_dict
    