"""
Nygard (1990)
=======================

"""

import numpy as np
from typing import Dict, Any
from eprllib.RewardFunctions.RewardFunctions import RewardFunction

class Nygard1990(RewardFunction):
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
            to calculate the reward. The dictionary must to have the following keys:
                1. agent_thermal_zone_ids,
                2. cut_reward_len_timesteps,
                3. Cdyn,
                4. pmv_name,
                5. T_interior_name,
                6. occupancy_name,
                7. cooling_name,
                8. heating_name.
            All this variables start with the name of the agent and then
            the valuo of the reference name.

        Returns:
            Dict[str,float]: The reward value for each agent in the timestep.
        """
        super().__init__(reward_fn_config)
        self.agents = {agent for agent in reward_fn_config.keys()}
        
        self.thermal_zone: Dict[str,str] = {agent: None for agent in self.agents}
        self.cut_reward_len_timesteps: Dict[str,float] = {agent: None for agent in self.agents}
        self.Cdyn: Dict[str,float] = {agent: None for agent in self.agents}
        self.pmv_name: Dict[str,str] = {agent: None for agent in self.agents}
        self.T_interior_name: Dict[str,str] = {agent: None for agent in self.agents}
        self.occupancy_name: Dict[str,str] = {agent: None for agent in self.agents}
        self.cooling_name: Dict[str,str] = {agent: None for agent in self.agents}
        self.heating_name: Dict[str,str] = {agent: None for agent in self.agents}
        
        for agent in self.agents:
            self.thermal_zone[agent] = reward_fn_config[agent]["thermal_zone"]
            self.cut_reward_len_timesteps[agent] = reward_fn_config[agent]['cut_reward_len_timesteps']
            self.Cdyn[agent] = reward_fn_config[agent]['Cdyn']
            self.pmv_name[agent] = reward_fn_config[agent]['pmv_name']
            self.T_interior_name[agent] = reward_fn_config[agent]['T_interior_name']
            self.occupancy_name[agent] = reward_fn_config[agent]['occupancy_name']
            self.cooling_name[agent] = reward_fn_config[agent]['cooling_name']
            self.heating_name[agent] = reward_fn_config[agent]['heating_name']
        
        # dictionary to save the values of each agent when the reward is dalyed.
        self.pmv_dict = {key: [] for key in self.agents}
        self.energy_dict = {key: [] for key in self.agents}
        
        self.timestep = 0
        
        self.Cp = 1
        self.Cu = self.Cu_calculation
        
    def Cu_calculation(
        self,
        Ti:float, # Interior temperature, °C
        Cdyn:float, # effective (dynamuc) thermal capacity of the room, J/K
        pmv_avg:float, # Fanger Predicted Mean Vote (PMV) factor, adim
        Dpmv:float=0.2, # PMV variation, adim. Default: 0.2
        Tset_:float=19, # Thermostat for heating, °C. Default: 19 °C
        Tset:float=24 # Thermostat for cooling, °C. Default: 24°C
        ) -> float:
        """
        """
        Cu = pmv_avg/(Tset-Ti)*(np.exp(Dpmv**2)-1)/(Cdyn*Dpmv)
        Cu_ = pmv_avg/(Ti-Tset_)*(np.exp(Dpmv**2)-1)/(Cdyn*Dpmv)
        
        return max(Cu, Cu_)
        
    def calculate_reward(
    self,
    infos: Dict[str,Dict[str,Any]] = None,
    truncated_flag: bool = False
    ) -> Dict[str,float]:
        """This function returns the normalize reward calcualted as the sum of the penalty of the energy 
        amount of one week divide per the maximun reference energy demand and the average PPD comfort metric
        divide per the maximal PPF value that can be take (100). Also, each term is divide per the longitude
        of the episode and multiply for a ponderation factor of beta for the energy and (1-beta) for the comfort.
        Both terms are negatives, representing a penalti for demand energy and for generate discomfort.

        Args:
            self (Environment): RLlib environment.
            infos (dict): infos dict must to provide the occupancy level and the Zone Mean Temperature.

        Returns:
            float: reward normalize value
        """
        self.timestep += 1
        # dictionary to save the reward values of each agent.
        reward_dict = {key: 0. for key in self.agents}
        
        # asign the ThermalZone values above defined to each agent
        for agent in self.agents:
            pmv = infos[agent][self.pmv_name[agent]]
            occupancy = infos[agent][self.occupancy_name[agent]]
            T_interior = infos[agent][self.T_interior_name[agent]]
            cooling_meter = infos[agent][self.cooling_name[agent]]
            heating_meter = infos[agent][self.heating_name[agent]]
            
            if occupancy > 0:
                if infos[agent][self.pmv_name[agent]] < 0:
                    pmv = infos[agent][self.pmv_name[agent]] = 0
                else:
                    pmv = infos[agent][self.pmv_name[agent]]
            else:
                pmv = infos[agent][self.pmv_name[agent]] = 0
            if T_interior > 29.4:
                pmv = infos[agent][self.pmv_name[agent]] = 3
            elif T_interior < 16.7:
                pmv = infos[agent][self.pmv_name[agent]] = 3
                
            self.pmv_dict[agent].append(pmv)
            self.energy_dict[agent].append(cooling_meter+heating_meter)
            
            
        
        # calculate the reward if the timestep is divisible by the cut_reward_len_timesteps.
        # if don't return 0.
        for agent in self.agents:
            if self.timestep % self.cut_reward_len_timesteps[agent] == 0 or truncated_flag:
                
                if len(self.ppd_dict[agent]) > 0:
                    pmv_avg = sum(self.pmv_dict[agent])/len(self.pmv_dict[agent])
                    total_energy = sum(self.energy_dict[agent])
                    Cu = self.Cu(T_interior, self.Cdyn[agent], pmv_avg)
                    rew = -(Cu*total_energy + self.Cp*(np.exp(pmv_avg**2)-1))
                else:
                    rew = 0
                
                reward_dict[agent] = rew
                
        # emptly the lists
        for agent in self.agents:
            if self.timestep % self.cut_reward_len_timesteps[agent] == 0 or truncated_flag:
                self.ppd_dict = {agent: []}
                self.energy_dict = {agent: []}
                
        # Print the reward_dict if one of the values are NaN or Inf
        for agent in self.agents:
            if np.isnan(reward_dict[agent]) or np.isinf(reward_dict[agent]):
                print(f"The reward value is NaN or Inf: {reward_dict}")
        return reward_dict
