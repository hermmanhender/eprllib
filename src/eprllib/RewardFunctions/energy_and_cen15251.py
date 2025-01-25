"""
Energy and CEN 15251 reward function
=====================================

The standard CEN 15251 stablish three categories of acceptability of the thermal condition
as is show in table below.

+==================================+===============================================================+
|Category I (90%) Acceptability    | High level of expectation and is recommended for spaces       |
|                                  | occupied by very sensitive and fragile persons with special   |
|                                  | requirements like handicapped, sick, very joung children and  |
|                                  | elderly persons.                                              |
+----------------------------------+---------------------------------------------------------------+
|Category II (80%) Acceptability   | Normal level of expectation and should be used for new        |
|                                  | buildings and renovations.                                    |
+----------------------------------+---------------------------------------------------------------+
|Category III (65%) Acceptability  | An acceptable, moderate level of expectation and may be used  |
|                                  | for existing buildings.                                       |
+----------------------------------+---------------------------------------------------------------+
|Cat. IV                           | Values outside the criteria for the above categories. This    |
|                                  | category should only be accepted for a limited part of the    |
|                                  | year.                                                         |
+==================================+===============================================================+

See Engineering Reference of EnergyPlus documentation 19.1.6 Adaptive Comfort Model Based on European Standard
EN15251-2007 for more information.

"""
import numpy as np
from typing import Any, Dict
from eprllib.RewardFunctions.RewardFunctions import RewardFunction
from eprllib.Utils.observation_utils import (
    get_variable_name,
    get_meter_name
)

class reward_fn(RewardFunction):
    def __init__(
        self,
        reward_fn_config: Dict[str,Any],
        ) -> Dict[str,float]:
        """
        This reward funtion takes the energy demand in the time step by the heating and cooling system and 
        calculate the energy reward as the sum of both divide by the maximal energy consumption of the 
        heating and cooling active system, respectibly.
        
        Also, take the PMV Fanger's factor and the occupancy in the thermal zone a calculate a reward using
        both elements.
        
        Thogother, they constitute the total reward that, ponderated with the beta factor, gives to the agents
        a signal to optimize the policy.

        Args:
            reward_fn_config (Dict[str,Any]): The dictionary is to configurate the variables that use each agent
            to calculate the reward. The dictionary must to have the following keys:
            
                1. agent_name,
                2. people_name,
                3. thermal_zone,
                4. beta,
                5. cooling_name,
                6. heating_name,
                7. cooling_energy_ref,
                8. heating_energy_ref.
            
            All this variables start with the name of the agent and then
            the value of the reference name.
        """
        super().__init__(reward_fn_config)
        
        self.agent_name = reward_fn_config["agent_name"]
        people = reward_fn_config['people_name']
        self.thermal_zone = reward_fn_config['thermal_zone']
        self.beta = reward_fn_config["beta"]
        self.cat1_name = get_variable_name(
            self.agent_name,
            "Zone Thermal Comfort CEN 15251 Adaptive Model Category I Status",
            people
        )
        self.cat2_name = get_variable_name(
            self.agent_name,
            "Zone Thermal Comfort CEN 15251 Adaptive Model Category II Status",
            people
        )
        self.cat3_name = get_variable_name(
            self.agent_name,
            "Zone Thermal Comfort CEN 15251 Adaptive Model Category III Status",
            people
        )
        self.cooling_name = get_meter_name(
            self.agent_name,
            reward_fn_config['cooling_name']
        )
        self.heating_name = get_meter_name(
            self.agent_name,
            reward_fn_config['heating_name']
        )
        self.cooling_energy_ref = self.reward_fn_config['cooling_energy_ref']
        self.heating_energy_ref = self.reward_fn_config['heating_energy_ref']

    
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
            terminated (bool): terminated condition of the episode.
            truncated (bool): truncated condition of the episode.
            
        Returns:
            float: reward normalize value.
        """
        reward = 0. 
        
        temp_int = infos.get(get_variable_name(
                self.agent_name,
                "Zone Mean Air Temperature",
                self.thermal_zone
            ),
            KeyError(f"The parameter {get_variable_name(self.agent_name,'Zone Mean Air Temperature',self.thermal_zone)} not found")
        )
        cat1 = infos[self.cat1_name]
        cat2 = infos[self.cat2_name]
        cat3 = infos[self.cat3_name]
        cooling_meter = infos[self.cooling_name]
        heating_meter = infos[self.heating_name]
        
        if cat1 == 1:
            reward += -(1-self.beta) * 0.0
        elif cat1 == 0:
            if cat2 == 1:
                reward += -(1-self.beta) * 0.05
            elif cat2 == 0:
                if cat3 == 1:
                    reward += -(1-self.beta) * 0.50
                elif cat3 == 0:
                    reward += -(1-self.beta) * 1
        else:
            if temp_int > 29.4 or temp_int < 16.7:
                reward += -(1-self.beta) * 1
            else:
                reward += -(1-self.beta) * 0.0
        # reward based on energy
        reward += -self.beta * np.clip((cooling_meter/self.cooling_energy_ref + heating_meter/self.heating_energy_ref),0,1)
            
        return reward
