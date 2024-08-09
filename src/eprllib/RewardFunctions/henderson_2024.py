"""
Henderson et al. (2024)
=======================

This module contains the reward function class for the EnergyPlusEnv_v0 environment.
The reward function is based on the Henderson et al. (2024) paper, which proposes a reward function
for optimizing the energy consumption and comfort of a building. The reward function is divided into
two components: energy demand penalty and comfort penalty. The energy demand penalty is based on the
maximum energy demand for the entire episode, and the comfort penalty is based on the average PPD comfort
metric for the entire episode. The reward function is normalized by dividing each term by the maximum
value for the entire episode, and multiplying by a ponderation factor for the energy and (1-beta) for
the comfort. Both terms are negatives, representing a penalti for demand energy and for generate discomfort.

The reward function is designed to be used in conjunction with the EnergyPlusEnv_v0 environment, which
provides the necessary information for calculating the reward.
"""

import numpy as np
from typing import Dict, Any
from eprllib.Env.MultiAgent.EnergyPlusEnvironment import EnergyPlusEnv_v0
from eprllib.RewardFunctions.RewardFunctions import RewardFunction

class henderson_2024(RewardFunction):
    def __init__(
        self,
        EnvObject: EnergyPlusEnv_v0
        ):
        super().__init__(EnvObject)
        
    def calculate_reward(
    self,
    infos: Dict[str,Dict[str,Any]] = None
    ) -> Dict[str,float]:
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
        reward_dict = {key: 0. for key in self.EnvObject._agent_ids}
        if not self.EnvObject.env_config.get('reward_fn_config', False):
            ValueError('The reward function configuration is not defined. The default function will be use.')
        
        # if the EnvObject don't have the list to append the values here obtained, one list is created as a property of the EnvObject
        if not hasattr(self.EnvObject, 'ppd_dict'):
            self.EnvObject.ppd_dict = {key: [] for key in self.EnvObject._agent_ids}
        if not hasattr(self.EnvObject, 'energy_dict'):
            self.EnvObject.energy_dict = {key: [] for key in self.EnvObject._agent_ids}
        
        # variables dicts
        beta_reward_tz = {thermal_zone: 0 for thermal_zone in self.EnvObject._thermal_zone_ids}
        cut_reward_len_timesteps_tz = {thermal_zone: 0 for thermal_zone in self.EnvObject._thermal_zone_ids}
        ppd_name_tz = {thermal_zone: '' for thermal_zone in self.EnvObject._thermal_zone_ids}
        T_interior_name_tz = {thermal_zone: '' for thermal_zone in self.EnvObject._thermal_zone_ids}
        occupancy_name_tz = {thermal_zone: '' for thermal_zone in self.EnvObject._thermal_zone_ids}
        cooling_energy_ref_tz = {thermal_zone: 0 for thermal_zone in self.EnvObject._thermal_zone_ids}
        heating_energy_ref_tz = {thermal_zone: 0 for thermal_zone in self.EnvObject._thermal_zone_ids}
        cooling_name_tz = {thermal_zone: '' for thermal_zone in self.EnvObject._thermal_zone_ids}
        heating_name_tz = {thermal_zone: '' for thermal_zone in self.EnvObject._thermal_zone_ids}
        
        for ThermalZone in self.EnvObject._thermal_zone_ids:
            # define which rewards will be considered
            comfort_reward = self.EnvObject.env_config['reward_fn_config'][ThermalZone].get('comfort_reward', True)
            energy_reward = self.EnvObject.env_config['reward_fn_config'][ThermalZone].get('energy_reward', True)
            
            # define the number of timesteps per episode
            cut_reward_len_timesteps_tz[ThermalZone] = self.EnvObject.env_config['reward_fn_config'][ThermalZone].get('cut_reward_len_timesteps', 1)
            
            # define the beta reward
            beta_reward_tz[ThermalZone] = self.EnvObject.env_config['reward_fn_config'][ThermalZone].get('beta_reward', 0.5)
        
            if comfort_reward:
                ppd_name_tz[ThermalZone] = self.EnvObject.env_config['reward_fn_config'][ThermalZone].get('ppd_name', False)
                T_interior_name_tz[ThermalZone] = self.EnvObject.env_config['reward_fn_config'][ThermalZone].get('T_interior_name', False)
                occupancy_name_tz[ThermalZone] = self.EnvObject.env_config['reward_fn_config'][ThermalZone].get('occupancy_name', False)
                if not ppd_name_tz[ThermalZone] or not occupancy_name_tz[ThermalZone] or not T_interior_name_tz[ThermalZone]:
                    ValueError('The names of the variables are not defined')
            
            if energy_reward:
                cooling_energy_ref_tz[ThermalZone] = self.EnvObject.env_config['reward_fn_config'][ThermalZone].get('cooling_energy_ref', False)
                heating_energy_ref_tz[ThermalZone] = self.EnvObject.env_config['reward_fn_config'][ThermalZone].get('heating_energy_ref', False)
                cooling_name_tz[ThermalZone] = self.EnvObject.env_config['reward_fn_config'][ThermalZone].get('cooling_name', False)
                heating_name_tz[ThermalZone] = self.EnvObject.env_config['reward_fn_config'][ThermalZone].get('heating_name', False)
                if not cooling_energy_ref_tz[ThermalZone] or not heating_energy_ref_tz[ThermalZone] or not cooling_name_tz[ThermalZone] or not heating_name_tz[ThermalZone]:
                    ValueError('The names of the variables are not defined')
        
        
        beta_reward_a = {agent: 0 for agent in self.EnvObject._agent_ids}
        cut_reward_len_timesteps_a = {agent: 0 for agent in self.EnvObject._agent_ids}
        ppd_name_a = {agent: '' for agent in self.EnvObject._agent_ids}
        T_interior_name_a = {agent: '' for agent in self.EnvObject._agent_ids}
        occupancy_name_a = {agent: '' for agent in self.EnvObject._agent_ids}
        cooling_energy_ref_a = {agent: 0 for agent in self.EnvObject._agent_ids}
        heating_energy_ref_a = {agent: 0 for agent in self.EnvObject._agent_ids}
        cooling_name_a = {agent: '' for agent in self.EnvObject._agent_ids}
        heating_name_a = {agent: '' for agent in self.EnvObject._agent_ids}
        
        # asign the ThermalZone values above defined to each agent
        for agent in self.EnvObject._agent_ids:
            agent_thermal_zone = self.EnvObject.env_config['agents_config'][agent]['thermal_zone']
            comfort_reward = self.EnvObject.env_config['reward_fn_config'][agent_thermal_zone].get('comfort_reward', True)
            energy_reward = self.EnvObject.env_config['reward_fn_config'][agent_thermal_zone].get('energy_reward', True)
            
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
        for agent in self.EnvObject._agent_ids:
            agent_thermal_zone = self.EnvObject.env_config['agents_config'][agent]['thermal_zone']
            comfort_reward = self.EnvObject.env_config['reward_fn_config'][agent_thermal_zone].get('comfort_reward', True)
            energy_reward = self.EnvObject.env_config['reward_fn_config'][agent_thermal_zone].get('energy_reward', True)
            
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
                    
                self.EnvObject.ppd_dict[agent].append(ppd)
            
            if energy_reward:
                _cooling_name = cooling_name_a[agent]
                _heating_name = heating_name_a[agent]
                cooling_meter = infos[agent][_cooling_name]
                heating_meter = infos[agent][_heating_name]
                
                self.EnvObject.energy_dict[agent].append(cooling_meter/cooling_energy_ref_a[agent]+heating_meter/heating_energy_ref_a[agent])
        
        # calculate the reward if the timestep is divisible by the cut_reward_len_timesteps.
        # if don't return 0.
        for agent in self.EnvObject._agent_ids:
            if self.EnvObject.timestep % cut_reward_len_timesteps_a[agent] == 0:
                agent_thermal_zone = self.EnvObject.env_config['agents_config'][agent]['thermal_zone']
                comfort_reward = self.EnvObject.env_config['reward_fn_config'][agent_thermal_zone].get('comfort_reward', True)
                energy_reward = self.EnvObject.env_config['reward_fn_config'][agent_thermal_zone].get('energy_reward', True)
                
                if comfort_reward:
                    ppd_avg = sum(self.EnvObject.ppd_dict[agent])/len(self.EnvObject.ppd_dict[agent])
                    rew1 = -(1-beta_reward_a[agent])*(1/(1+np.exp(-0.1*(ppd_avg-45))))
                else:
                    rew1 = 0
                
                if energy_reward:
                    rew2 = -beta_reward_a[agent]*(sum(self.EnvObject.energy_dict[agent])/len(self.EnvObject.energy_dict[agent]))
                else:
                    rew2 = 0
                
                reward_dict[agent] = rew1 + rew2
                
        # emptly the lists
        for agent in self.EnvObject._agent_ids:
            if self.EnvObject.timestep % cut_reward_len_timesteps_a[agent] == 0:
                self.EnvObject.ppd_dict = {key: [] for key in self.EnvObject._agent_ids}
                self.EnvObject.energy_dict = {key: [] for key in self.EnvObject._agent_ids}
                
        return reward_dict
