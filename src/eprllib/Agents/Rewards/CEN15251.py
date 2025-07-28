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
from typing import Any, Dict, List # type: ignore
from numpy.typing import NDArray
from numpy import float32
from eprllib.Agents.Rewards.BaseReward import BaseReward
from eprllib.Utils.observation_utils import get_variable_name
from eprllib.Utils.annotations import override
from eprllib.Utils.agent_utils import config_validation
from eprllib import logger

class CEN15251(BaseReward):
    """
    The standard CEN 15251 stablish three categories of acceptability of the thermal condition
    as is show in table below.

    .. list-table:: Comfort categories based on CEN 15251 standard.
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
    REQUIRED_KEYS = {
        "thermal_zone": str,
        "people_name": str
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
            
                1. people_name,
                2. thermal_zone
            
            All this variables start with the name of the agent and then the value of the reference name.
        """
        # Validate the config.
        config_validation(reward_fn_config, self.REQUIRED_KEYS)
        
        super().__init__(reward_fn_config)
        
        self.agent_name: str = "None"
        self.cat1_name:int = 0
        self.cat2_name:int = 0
        self.cat3_name:int = 0
        self.temp_int:int = 0
        
        logger.info(f"Reward function config: {reward_fn_config}")
    
    @override(BaseReward)
    def set_initial_parameters(
        self,
        agent_name: str,
        obs_indexed: Dict[str, int]
    ) -> None:
        if self.agent_name is "None":
            self.agent_name = agent_name
            self.cat1_name = obs_indexed[get_variable_name(
                self.agent_name,
                "Zone Thermal Comfort CEN 15251 Adaptive Model Category I Status",
                self.reward_fn_config['people_name']
            )]
            self.cat2_name = obs_indexed[get_variable_name(
                self.agent_name,
                "Zone Thermal Comfort CEN 15251 Adaptive Model Category II Status",
                self.reward_fn_config['people_name']
            )]
            self.cat3_name = obs_indexed[get_variable_name(
                self.agent_name,
                "Zone Thermal Comfort CEN 15251 Adaptive Model Category III Status",
                self.reward_fn_config['people_name']
            )]
            self.temp_int = obs_indexed[get_variable_name(
                self.agent_name,
                "Zone Mean Air Temperature",
                self.reward_fn_config['thermal_zone']
            )]
        
        logger.info(f"Agent {self.agent_name} reward function initialized with variables.")
            
    @override(BaseReward)
    def get_reward(
        self,
        obs: NDArray[float32],
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
            float: reward normalize value.
        """
        temp_int:float = obs[self.temp_int]
        
        if obs[self.cat1_name] == 1:
            return 0.
        elif obs[self.cat1_name] == 0:
            if obs[self.cat2_name] == 1:
                return -0.05
            elif obs[self.cat2_name] == 0:
                if obs[self.cat3_name] == 1:
                    return -0.50
                elif obs[self.cat3_name] == 0:
                    return -1.
                else:
                    if temp_int > 29.4 or temp_int < 16.7:
                        return -1.
                    else:
                        return 0.
            else:
                if temp_int > 29.4 or temp_int < 16.7:
                    return -1.
                else:
                    return 0.
        else:
            if temp_int > 29.4 or temp_int < 16.7:
                return -1.
            else:
                return 0.


# === Hierarchical versions ===
   
class HierarchicalCEN15251(BaseReward):
    REQUIRED_KEYS = {
        "thermal_zone": str,
        "people_name": str
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
            
                1. people_name,
                2. thermal_zone
            
            All this variables start with the name of the agent and then the value of the reference name.
        """
        # Validate the config.
        config_validation(reward_fn_config, self.REQUIRED_KEYS)
        
        super().__init__(reward_fn_config)
        
        self.agent_name:str = "None"
        self.cat1_name:int = 0
        self.cat2_name:int = 0
        self.cat3_name:int = 0
        self.temp_int:int = 0
    
    @override(BaseReward)
    def set_initial_parameters(
        self,
        agent_name: str,
        obs_indexed: Dict[str, int]
    ) -> None:
        if self.agent_name is "None":
            self.agent_name = agent_name
            self.cat1_name = obs_indexed[get_variable_name(
                self.agent_name,
                "Zone Thermal Comfort CEN 15251 Adaptive Model Category I Status",
                self.reward_fn_config['people_name']
            )]
            self.cat2_name = obs_indexed[get_variable_name(
                self.agent_name,
                "Zone Thermal Comfort CEN 15251 Adaptive Model Category II Status",
                self.reward_fn_config['people_name']
            )]
            self.cat3_name = obs_indexed[get_variable_name(
                self.agent_name,
                "Zone Thermal Comfort CEN 15251 Adaptive Model Category III Status",
                self.reward_fn_config['people_name']
            )]
            self.temp_int = obs_indexed[get_variable_name(
                self.agent_name,
                "Zone Mean Air Temperature",
                self.reward_fn_config['thermal_zone']
            )]
            
    @override(BaseReward)
    def get_reward(
    self,
    obs: NDArray[float32],
    terminated: bool = False,
    truncated: bool = False
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
        reward = 0. 
        
        temp_int:List[float] = obs[self.temp_int]
        
        for ix in range(len(obs[self.cat1_name])):
            
            if obs[self.cat1_name][ix] == 1:
                reward += 0
            elif obs[self.cat1_name][ix] == 0:
                if obs[self.cat2_name][ix] == 1:
                    reward -= 0.05
                elif obs[self.cat2_name][ix] == 0:
                    if obs[self.cat3_name][ix] == 1:
                        reward -= 0.50
                    elif obs[self.cat3_name][ix] == 0:
                        reward -= 1
                    else:
                        if temp_int[ix] > 29.4 or temp_int[ix] < 16.7:
                            reward -= 1
                        else:
                            reward += 0
            else:
                if temp_int[ix] > 29.4 or temp_int[ix] < 16.7:
                    reward -= 1
                else:
                    reward += 0
        return reward
    