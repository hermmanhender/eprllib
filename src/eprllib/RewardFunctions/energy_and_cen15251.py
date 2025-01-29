"""
Energy and CEN 15251 reward function
=====================================

The standard CEN 15251 stablish three categories of acceptability of the thermal condition
as is show in table below.

.. list-table:: Title
   :widths: 25 75
   :header-rows: 1

   * - Category
     - Explanation
   * - Category I (90%) Acceptability 
     - High level of expectation and is recommended for spaces occupied by very sensitive and fragile 
       persons with special requirements like handicapped, sick, very joung children and elderly persons.
   * - Category II (80%) Acceptability 
     - Normal level of expectation and should be used for new buildings and renovations.
   * - Category III (65%) Acceptability
     - An acceptable, moderate level of expectation and may be used for existing buildings.
   * - Cat. IV
     - Values outside the criteria for the above categories. This category should only be accepted for a 
       limited part of the year.

See Engineering Reference of EnergyPlus documentation 19.1.6 Adaptive Comfort Model Based on European Standard
EN15251-2007 for more information.

"""
from typing import Any, Dict
from eprllib.RewardFunctions.RewardFunctions import RewardFunction
from eprllib.RewardFunctions.energy_rewards import energy_with_meters, herarchical_energy_with_meters
from eprllib.RewardFunctions.comfort_rewards import cen15251, herarchical_cen15251
from eprllib.Utils.annotations import override

class reward_fn(RewardFunction):
    def __init__(
        self,
        reward_fn_config: Dict[str,Any],
        ) -> Dict[str,float]:
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
            
                1. agent_name,
                2. thermal_zone,
                3. beta,
                4. cooling_name,
                5. heating_name,
                6. cooling_energy_ref,
                7. heating_energy_ref.
            
            All this variables start with the name of the agent and then
            the value of the reference name.

        Returns:
            Dict[str,float]: The reward value for each agent in the timestep.
        """
        super().__init__(reward_fn_config)
        self.comfort_reward = cen15251({
            "agent_name": reward_fn_config["agent_name"],
            "thermal_zone": reward_fn_config['thermal_zone'],
            "people_name": reward_fn_config['people_name']
        })
        self.energy_reward = energy_with_meters({
            "agent_name": reward_fn_config["agent_name"],
            "cooling_name": reward_fn_config['cooling_name'],
            "heating_name": reward_fn_config['heating_name'],
            "cooling_energy_ref": reward_fn_config['cooling_energy_ref'],
            "heating_energy_ref": reward_fn_config['heating_energy_ref']
        })
        self.beta = reward_fn_config['beta']
    
    @override(RewardFunction)
    def get_reward(
    self,
    infos: Dict[str,Any] = None,
    terminated_flag: bool = False,
    truncated_flag: bool = False
    ) -> float:
        """This function returns the normalize reward calcualted as the sum of the penalty of the energy 
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

# === Herarchical version ===

class herarchical_reward_fn(RewardFunction):
    def __init__(
        self,
        reward_fn_config: Dict[str,Any],
        ) -> Dict[str,float]:
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
            
                1. agent_name,
                2. thermal_zone,
                3. beta,
                4. cooling_name,
                5. heating_name,
                6. cooling_energy_ref,
                7. heating_energy_ref.
            
            All this variables start with the name of the agent and then
            the value of the reference name.

        Returns:
            Dict[str,float]: The reward value for each agent in the timestep.
        """
        super().__init__(reward_fn_config)
        self.comfort_reward = herarchical_cen15251({
            "agent_name": reward_fn_config["agent_name"],
            "thermal_zone": reward_fn_config['thermal_zone'],
            "people_name": reward_fn_config['people_name']
        })
        self.energy_reward = herarchical_energy_with_meters({
            "agent_name": reward_fn_config["agent_name"],
            "cooling_name": reward_fn_config['cooling_name'],
            "heating_name": reward_fn_config['heating_name'],
            "cooling_energy_ref": reward_fn_config['cooling_energy_ref'],
            "heating_energy_ref": reward_fn_config['heating_energy_ref']
        })
        self.beta = reward_fn_config['beta']
    
    @override(RewardFunction)
    def get_reward(
    self,
    infos: Dict[str,Any] = None,
    terminated_flag: bool = False,
    truncated_flag: bool = False
    ) -> float:
        """This function returns the normalize reward calcualted as the sum of the penalty of the energy 
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
    