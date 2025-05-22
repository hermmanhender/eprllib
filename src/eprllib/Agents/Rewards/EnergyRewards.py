"""
Energy rewards
===============

The rewards based on energy variables/meters.

Hierarchical versions
----------------------

Note that hierarchical versions are the same function but insteade of using only the inmediate
value of the variable/meter, the reward is calculated using a List of the last timesteps and
integrating or averaging them.
"""
import numpy as np
from typing import Any, Dict
from eprllib.Agents.Rewards.BaseReward import BaseReward
from eprllib.Utils.observation_utils import get_meter_name
from eprllib.Utils.annotations import override
from eprllib.Utils.agent_utils import get_agent_name, config_validation

class EnergyWithMeters(BaseReward):
    REQUIRED_KEYS = {
        "cooling_name": str,
        "heating_name": str,
        "cooling_energy_ref": float|int,
        "heating_energy_ref": float|int
    }
    
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
            
                1. cooling_name,
                2. heating_name,
                3. cooling_energy_ref,
                4. heating_energy_ref.
            
            All this variables start with the name of the agent and then the value of the reference name.
        """
        # Validate the config.
        config_validation(reward_fn_config, self.REQUIRED_KEYS)
        
        super().__init__(reward_fn_config)
        
        self.agent_name = None
        self.cooling = None
        self.heating = None
    
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
        if self.agent_name is None:
            self.agent_name = get_agent_name(infos)
            self.cooling = get_meter_name(
                self.agent_name,
                self.reward_fn_config['cooling_name']
            )
            self.heating = get_meter_name(
                self.agent_name,
                self.reward_fn_config['heating_name']
            )
            
        return - np.clip(
            (infos[self.cooling] / self.reward_fn_config['cooling_energy_ref'] \
                + infos[self.heating] / self.reward_fn_config['heating_energy_ref']),
            0,
            1
        )


# === Hierarchical versions ===

class HierarchicalEnergyWithMeters(BaseReward):
    REQUIRED_KEYS = {
        "cooling_name": str,
        "heating_name": str,
        "cooling_energy_ref": float|int,
        "heating_energy_ref": float|int
    }
    
    def __init__(
        self,
        reward_fn_config: Dict[str,Any],
        ):
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
            
                1. cooling_name,
                2. heating_name,
                3. cooling_energy_ref,
                4. heating_energy_ref.
            
            All this variables start with the name of the agent and then the value of the reference name.
        """
        # Validate the config.
        config_validation(reward_fn_config, self.REQUIRED_KEYS)
        
        super().__init__(reward_fn_config)
        
        self.agent_name = None
        self.cooling = None
        self.heating = None
    
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
            if self.agent_name is None:
                self.agent_name = get_agent_name(infos)
                self.cooling = get_meter_name(
                    self.agent_name,
                    self.reward_fn_config['cooling_name']
                )
                self.heating = get_meter_name(
                    self.agent_name,
                    self.reward_fn_config['heating_name']
                )
                
            return - np.clip(
                (sum(infos[self.cooling]) / (self.reward_fn_config['cooling_energy_ref'] * len(infos[self.cooling])) \
                    + sum(infos[self.heating]) / (self.reward_fn_config['heating_energy_ref'] * len(infos[self.heating]))),
                0,
                1
            )
