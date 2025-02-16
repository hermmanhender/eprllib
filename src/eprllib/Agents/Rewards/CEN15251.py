"""
CEN15251
=========

The rewards based on comfort variables/meters.

Hierarchical versions
----------------------

Note that hierarchical versions are the same function but instead of using only the immediate
value of the variable/meter, the reward is calculated using a list of the last timesteps and
integrating or averaging them.
"""
from typing import Any, Dict
from eprllib.Agents.Rewards.BaseReward import BaseReward
from eprllib.Utils.observation_utils import get_variable_name
from eprllib.Utils.annotations import override
from eprllib.Utils.agent_utils import get_agent_name

class CEN15251(BaseReward):
    """
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
            
                1. people_name,
                2. thermal_zone,
                3. beta,
                4. cooling_name,
                5. heating_name,
                6. cooling_energy_ref,
                7. heating_energy_ref.
            
            All this variables start with the name of the agent and then the value of the reference name.
        """
        super().__init__(reward_fn_config)
        
        self.agent_name = None
        self.cat1_name = None
        self.cat2_name = None
        self.cat3_name = None
        self.temp_int = None
    
    @override(BaseReward)
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
            self (Environment): RLlib environment.
            infos (dict): infos dict must to provide the occupancy level and the Zone Mean Temperature.

        Returns:
            float: reward normalize value
        """
        if self.agent_name is None:
            self.agent_name = get_agent_name(infos)
            self.cat1_name = get_variable_name(
                self.agent_name,
                "Zone Thermal Comfort CEN 15251 Adaptive Model Category I Status",
                self.reward_fn_config['people_name']
            )
            self.cat2_name = get_variable_name(
                self.agent_name,
                "Zone Thermal Comfort CEN 15251 Adaptive Model Category II Status",
                self.reward_fn_config['people_name']
            )
            self.cat3_name = get_variable_name(
                self.agent_name,
                "Zone Thermal Comfort CEN 15251 Adaptive Model Category III Status",
                self.reward_fn_config['people_name']
            )
            self.temp_int = get_variable_name(
                self.agent_name,
                "Zone Mean Air Temperature",
                self.reward_fn_config['thermal_zone']
            )
        
        temp_int = infos.get(
            self.temp_int,
            KeyError(f"The parameter {self.temp_int} not found. The agent name auto-detected was {self.agent_name} and the infos provided is: {infos}")
        )
        
        if infos[self.cat1_name] == 1:
            return 0
        elif infos[self.cat1_name] == 0:
            if infos[self.cat2_name] == 1:
                return -0.05
            elif infos[self.cat2_name] == 0:
                if infos[self.cat3_name] == 1:
                    return -0.50
                elif infos[self.cat3_name] == 0:
                    return -1
        else:
            if temp_int > 29.4 or temp_int < 16.7:
                return -1
            else:
                return 0


# === Hierarchical versions ===
   
class HierarchicalCEN15251(BaseReward):
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
            
                1. people_name,
                2. thermal_zone,
                3. beta,
                4. cooling_name,
                5. heating_name,
                6. cooling_energy_ref,
                7. heating_energy_ref.
            
            All this variables start with the name of the agent and then the value of the reference name.
        """
        super().__init__(reward_fn_config)
        
        self.agent_name = None
        self.cat1_name = None
        self.cat2_name = None
        self.cat3_name = None
        self.temp_int = None
        
    @override(BaseReward)
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
            self (Environment): RLlib environment.
            infos (dict): infos dict must to provide the occupancy level and the Zone Mean Temperature.

        Returns:
            float: reward normalize value
        """
        if self.agent_name is None:
            self.agent_name = get_agent_name(infos)
            self.cat1_name = get_variable_name(
                self.agent_name,
                "Zone Thermal Comfort CEN 15251 Adaptive Model Category I Status",
                self.reward_fn_config['people_name']
            )
            self.cat2_name = get_variable_name(
                self.agent_name,
                "Zone Thermal Comfort CEN 15251 Adaptive Model Category II Status",
                self.reward_fn_config['people_name']
            )
            self.cat3_name = get_variable_name(
                self.agent_name,
                "Zone Thermal Comfort CEN 15251 Adaptive Model Category III Status",
                self.reward_fn_config['people_name']
            )
            self.temp_int = get_variable_name(
                self.agent_name,
                "Zone Mean Air Temperature",
                self.reward_fn_config['thermal_zone']
            )
        
        reward = 0. 
        
        temp_int = infos.get(
            self.temp_int,
            KeyError(f"The parameter {self.temp_int} not found. The agent name auto-detected was {self.agent_name} and the infos provided is: {infos}")
        )
        for ix in range(len(infos[self.cat1_name])):
            
            if infos[self.cat1_name][ix] == 1:
                reward += 0
            elif infos[self.cat1_name][ix] == 0:
                if infos[self.cat2_name][ix] == 1:
                    reward -= 0.05
                elif infos[self.cat2_name][ix] == 0:
                    if infos[self.cat3_name][ix] == 1:
                        reward -= 0.50
                    elif infos[self.cat3_name][ix] == 0:
                        reward -= 1
            else:
                if temp_int[ix] > 29.4 or temp_int[ix] < 16.7:
                    reward -= 1
                else:
                    reward += 0
        return reward
    