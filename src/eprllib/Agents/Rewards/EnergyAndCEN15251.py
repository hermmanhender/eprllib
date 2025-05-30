"""
Energy and CEN 15251 reward function
=====================================

"""
from typing import Any, Dict
from eprllib.Agents.Rewards.BaseReward import BaseReward
from eprllib.Agents.Rewards.EnergyRewards import EnergyWithMeters, HierarchicalEnergyWithMeters
from eprllib.Agents.Rewards.CEN15251 import CEN15251, HierarchicalCEN15251
from eprllib.Utils.annotations import override
from eprllib import logger

class EnergyAndCEN15251(BaseReward):
    def __init__(
        self,
        reward_fn_config: Dict[str,Any],
        ):
        """
        This reward funtion takes the energy demand in the time step by the heating and cooling system and 
        calculate the energy reward as the sum of both divide by the maximal energy consumption of the 
        heating and cooling active system, respectibly.
        
        Also, take the Zone Thermal Comfort ASHRAE 55 Simple Model Summer or Winter Clothes Not Comfortable Time[hr]
        variable to determine the comfort reward.
        
        Thogother, they constitute the total reward that, ponderated with the beta factor, gives to the agents
        a signal to optimize the policy.

        Args:
            reward_fn_config (Dict[str,Any]): The dictionary is to configurate the variables that use each agent
            to calculate the reward. The dictionary must to have the following keys:
            
                1. thermal_zone,
                2. beta,
                3. cooling_name,
                4. heating_name,
                5. cooling_energy_ref,
                6. heating_energy_ref.
            
            All this variables start with the name of the agent and then the value of the reference name.
        """
        super().__init__(reward_fn_config)
        self.comfort_reward = CEN15251({
            "thermal_zone": reward_fn_config['thermal_zone'],
            "people_name": reward_fn_config['people_name']
        })
        self.energy_reward = EnergyWithMeters({
            "cooling_name": reward_fn_config['cooling_name'],
            "heating_name": reward_fn_config['heating_name'],
            "cooling_energy_ref": reward_fn_config['cooling_energy_ref'],
            "heating_energy_ref": reward_fn_config['heating_energy_ref']
        })
        self.beta = reward_fn_config['beta']
    
    @override(BaseReward)
    def get_reward(
    self,
    infos: Dict[str,Any] = None,
    terminated_flag: bool = False,
    truncated_flag: bool = False
    ) -> float:
        """
        This function returns the normalize reward calcualted as the sum of the penalty of the energy 
        amount of one week divide per the maximun reference energy demand and the average PPD comfort metric
        divide per the maximal PPF value that can be take (100). Also, each term is divide per the longitude
        of the episode and multiply for a ponderation factor of beta for the energy and (1-beta) for the comfort.
        Both terms are negatives, representing a penalti for demand energy and for generate discomfort.

        Args:
            infos (dict): infos dict must to provide the occupancy level and the Zone Mean Temperature.

        Returns:
            float: reward normalize value
        """
        if infos.get("goal"):
            self.beta = infos["goal"]
        
        reward = 0.
        reward += (1-self.beta) * self.comfort_reward.get_reward(infos, terminated_flag, truncated_flag)
        reward += self.beta * self.energy_reward.get_reward(infos, terminated_flag, truncated_flag)
        return reward

# === Hierarchical version ===

class HierarchicalEnergyAndCEN15251(BaseReward):
    def __init__(
        self,
        reward_fn_config: Dict[str,Any],
        ):
        """
        This reward funtion takes the energy demand in the time step by the heating and cooling system and 
        calculate the energy reward as the sum of both divide by the maximal energy consumption of the 
        heating and cooling active system, respectibly.
        
        Also, take the Zone Thermal Comfort ASHRAE 55 Simple Model Summer or Winter Clothes Not Comfortable Time[hr]
        variable to determine the comfort reward.
        
        Thogother, they constitute the total reward that, ponderated with the beta factor, gives to the agents
        a signal to optimize the policy.

        Args:
            reward_fn_config (Dict[str,Any]): The dictionary is to configurate the variables that use each agent
            to calculate the reward. The dictionary must to have the following keys:
            
                1. thermal_zone,
                2. beta,
                3. cooling_name,
                4. heating_name,
                5. cooling_energy_ref,
                6. heating_energy_ref.
            
            All this variables start with the name of the agent and then the value of the reference name.
        """
        super().__init__(reward_fn_config)
        self.beta = reward_fn_config['beta']
        self.comfort_reward = HierarchicalCEN15251({
            "thermal_zone": reward_fn_config['thermal_zone'],
            "people_name": reward_fn_config['people_name']
        })
        self.energy_reward = HierarchicalEnergyWithMeters({
            "cooling_name": reward_fn_config['cooling_name'],
            "heating_name": reward_fn_config['heating_name'],
            "cooling_energy_ref": reward_fn_config['cooling_energy_ref'],
            "heating_energy_ref": reward_fn_config['heating_energy_ref']
        })
        
    @override(BaseReward)
    def get_reward(
    self,
    infos: Dict[str,Any] = None,
    terminated_flag: bool = False,
    truncated_flag: bool = False
    ) -> float:
        """
        This function returns the normalize reward calcualted as the sum of the penalty of the energy 
        amount of one week divide per the maximun reference energy demand and the average PPD comfort metric
        divide per the maximal PPF value that can be take (100). Also, each term is divide per the longitude
        of the episode and multiply for a ponderation factor of beta for the energy and (1-beta) for the comfort.
        Both terms are negatives, representing a penalti for demand energy and for generate discomfort.

        Args:
            infos (dict): infos dict must to provide the occupancy level and the Zone Mean Temperature.

        Returns:
            float: reward normalize value
        """
        reward = 0.
        reward += (1-self.beta) * self.comfort_reward.get_reward(infos, terminated_flag, truncated_flag)
        reward += self.beta * self.energy_reward.get_reward(infos, terminated_flag, truncated_flag)
        
        return reward
    