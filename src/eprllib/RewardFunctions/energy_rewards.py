"""
Energy rewards
===============

The rewards based on energy variables/meters.

Herarchical versions
---------------------

Note that herarchical versions are the same function but insteade of using only the inmediate
value of the variable/meter, the reward is calculated using a List of the last timesteps and
integrating or averaging them.
"""
import numpy as np
from typing import Any, Dict, List, Tuple, Union
from eprllib.RewardFunctions.RewardFunctions import RewardFunction
from eprllib.Utils.observation_utils import (
    get_meter_name
)

class energy_with_meters(RewardFunction):
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
                2. cooling_name,
                3. heating_name,
                4. cooling_energy_ref,
                5. heating_energy_ref.
            
            All this variables start with the name of the agent and then
            the value of the reference name.

        Returns:
            Dict[str,float]: The reward value for each agent in the timestep.
        """
        super().__init__(reward_fn_config)
        
        agent_name = reward_fn_config["agent_name"]
        self.cooling = get_meter_name(
            agent_name,
            reward_fn_config['cooling_name']
        )
        self.heating = get_meter_name(
            agent_name,
            reward_fn_config['heating_name']
        )
    
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
        return - np.clip(
            (infos[self.cooling] / self.reward_fn_config['cooling_energy_ref'] \
                + infos[self.heating] / self.reward_fn_config['heating_energy_ref']),
            0,
            1
        )


# === Herarchical versions ===

class herarchical_energy_with_meters:
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
        
        agent_name = reward_fn_config["agent_name"]
        self.cooling = get_meter_name(
            agent_name,
            reward_fn_config['cooling_name']
        )
        self.heating = get_meter_name(
            agent_name,
            reward_fn_config['heating_name']
        )
        
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
            return - np.clip(
                (sum(infos[self.cooling]) / (self.reward_fn_config['cooling_energy_ref'] * len(infos[self.cooling])) \
                    + sum(infos[self.heating]) / (self.reward_fn_config['heating_energy_ref'] * len(infos[self.heating]))),
                0,
                1
            )
