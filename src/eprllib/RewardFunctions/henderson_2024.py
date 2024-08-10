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
from eprllib.RewardFunctions.RewardFunctions import RewardFunction

class henderson_2024(RewardFunction):
    def __init__(
        self,
        reward_fn_config: Dict[str,Any]
        ) -> Dict[str,float]:
        """
        This class implements the Henderson et al. (2024) reward function for the EnergyPlusEnv_v0 environment.
        The reward function is divided into two components: energy demand penalty and comfort penalty. The energy 
        demand penalty is based on the maximum energy demand for the entire episode, and the comfort penalty is 
        based on the average PPD comfort metric for the entire episode. The reward function is normalized by 
        dividing each term by the maximum value for the entire episode, and multiplying by a ponderation factor 
        for the energy and (1-beta) for the comfort. Both terms are negatives, representing a penalti for demand 
        energy and for generate discomfort.

        Args:
            reward_fn_config (Dict[str,Any]): The dictionary is to configurate the variables that use each agent
            to calculate the reward. The dictionary must to have the following keys: agent_thermal_zone_ids, comfort_reward,
            energy_reward, cut_reward_len_timesteps, beta, ppd_name, T_interior_name, occupancy_name, cooling_energy_ref,
            heating_energy_ref, cooling_name, heating_name. All this variables start with the name of the agent and then
            the valuo of the reference name.

        Returns:
            Dict[str,float]: The reward value for each agent in the timestep.
        """
        super().__init__(reward_fn_config)
        self.agents = {agent for agent in reward_fn_config.keys()}
        
        self.thermal_zone: Dict[str,str] = {agent: None for agent in self.agents}
        self.comfort_reward: Dict[str,bool] = {agent: None for agent in self.agents}
        self.energy_reward: Dict[str,bool] = {agent: None for agent in self.agents}
        self.cut_reward_len_timesteps: Dict[str,float] = {agent: None for agent in self.agents}
        self.beta: Dict[str,float] = {agent: None for agent in self.agents}
        self.ppd_name: Dict[str,str] = {agent: None for agent in self.agents}
        self.T_interior_name: Dict[str,str] = {agent: None for agent in self.agents}
        self.occupancy_name: Dict[str,str] = {agent: None for agent in self.agents}
        self.cooling_energy_ref: Dict[str,str] = {agent: None for agent in self.agents}
        self.heating_energy_ref: Dict[str,str] = {agent: None for agent in self.agents}
        self.cooling_name: Dict[str,str] = {agent: None for agent in self.agents}
        self.heating_name: Dict[str,str] = {agent: None for agent in self.agents}
        
        for agent in self.agents:
            self.thermal_zone[agent] = reward_fn_config[agent]["thermal_zone"]
            self.comfort_reward[agent] = reward_fn_config[agent]['comfort_reward']
            self.energy_reward[agent] = reward_fn_config[agent]['energy_reward']
            self.cut_reward_len_timesteps[agent] = reward_fn_config[agent]['cut_reward_len_timesteps']
            self.beta[agent] = reward_fn_config[agent]['beta']
            self.ppd_name[agent] = reward_fn_config[agent]['ppd_name']
            self.T_interior_name[agent] = reward_fn_config[agent]['T_interior_name']
            self.occupancy_name[agent] = reward_fn_config[agent]['occupancy_name']
            self.cooling_energy_ref[agent] = reward_fn_config[agent]['cooling_energy_ref']
            self.heating_energy_ref[agent] = reward_fn_config[agent]['heating_energy_ref']
            self.cooling_name[agent] = reward_fn_config[agent]['cooling_name']
            self.heating_name[agent] = reward_fn_config[agent]['heating_name']
        
        # dictionary to save the values of each agent when the reward is dalyed.
        self.ppd_dict = {key: [] for key in self.agents}
        self.energy_dict = {key: [] for key in self.agents}
        
        self.timestep = 0
        
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
        self.timestep += 1
        # dictionary to save the reward values of each agent.
        reward_dict = {key: 0. for key in self.agents}
        
        # asign the ThermalZone values above defined to each agent
        for agent in self.agents:
            if self.comfort_reward[agent]:
                ppd = infos[agent][self.ppd_name[agent]]
                occupancy = infos[agent][self.occupancy_name[agent]]
                T_interior = infos[agent][self.T_interior_name[agent]]
                if occupancy > 0:
                    if infos[agent][self.ppd_name[agent]] < 5:
                        ppd = infos[agent][self.ppd_name[agent]] = 5
                    else:
                        ppd = infos[agent][self.ppd_name[agent]]
                else:
                    ppd = infos[agent][self.ppd_name[agent]] = 0
                # If there are not people, only the reward is calculated when the environment is far away
                # from the comfort temperature ranges. This limits are recommended in EnergyPlus documentation:
                # InputOutput Reference p.522
                if T_interior > 29.4:
                    ppd = infos[agent][self.ppd_name[agent]] = 100
                elif T_interior < 16.7:
                    ppd = infos[agent][self.ppd_name[agent]] = 100
                    
                self.ppd_dict[agent].append(ppd)
            
            if self.energy_reward[agent]:
                cooling_meter = infos[agent][self.cooling_name[agent]]
                heating_meter = infos[agent][self.heating_name[agent]]
                
                self.energy_dict[agent].append(cooling_meter/self.cooling_energy_ref[agent]+heating_meter/self.heating_energy_ref[agent])
        
        # calculate the reward if the timestep is divisible by the cut_reward_len_timesteps.
        # if don't return 0.
        for agent in self.agents:
            if self.timestep % self.cut_reward_len_timesteps[agent] == 0:
                if self.comfort_reward[agent]:
                    ppd_avg = sum(self.ppd_dict[agent])/len(self.ppd_dict[agent])
                    rew1 = -(1-self.beta[agent])*(1/(1+np.exp(-0.1*(ppd_avg-45))))
                else:
                    rew1 = 0
                
                if self.energy_reward[agent]:
                    rew2 = -self.beta[agent]*(sum(self.energy_dict[agent])/len(self.energy_dict[agent]))
                else:
                    rew2 = 0
                
                reward_dict[agent] = rew1 + rew2
                
        # emptly the lists
        for agent in self.agents:
            if self.timestep % self.cut_reward_len_timesteps[agent] == 0:
                self.ppd_dict = {key: [] for key in self.agents}
                self.energy_dict = {key: [] for key in self.agents}
                
        return reward_dict
