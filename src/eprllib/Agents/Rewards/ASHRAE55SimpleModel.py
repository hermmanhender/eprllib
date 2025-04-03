"""
ASHRAE55SimpleModel
====================

The rewards based on whether the humidity ratio and the operative temperature
is within the region shown in ASHRAE Standard 55-2004 in the figure. For the calculation the operative
temperature is simplified to be the average of the air temperature and the mean radiant temperature. For
summer, the 0.5 Clo level is used and, for winter, the 1.0 Clo level is used. The graphs below are based on
the following tables which extend the ASHRAE values to zero humidity ratio.

.. list-table:: Table 1: Winter Clothes (1.0 Clo)
    :widths: 50 50
    :header-rows: 1

    * - Operative temperature, °C
        - Humidity Ratio (kgWater/kgDryAir)
    * - 19.6
        - 0.012
    * - 23.9 
        - 0.012
    * - 26.3
        - 0.000
    * - 21.7
        - 0.000

.. list-table:: Table 2: Summer Clothes (0.5 Clo)
    :widths: 50 50
    :header-rows: 1

    * - Operative temperature, °C
        - Humidity Ratio (kgWater/kgDryAir)
    * - 23.6
        - 0.012
    * - 26.8 
        - 0.012
    * - 28.3
        - 0.000
    * - 25.1
        - 0.000

We use in this implementation the EnergyPlus variable:

* Zone Thermal Comfort ASHRAE 55 Simple Model Summer or Winter Clothes Not Comfortable Time[hr]: The time when the 
zone is occupied that the combination of humidity ratio and operative temperature is not in the ASHRAE 55-2004 summer 
or winter clothes region.

If you use this implementation for an agent, the observation space must to contain the following variable:

.. code-block:: python
    
    eprllib_config = EnvConfig()
    eprllib_config.agents(
        agents_config = {
            "agent_name": AgentSpec(
                observation = ObservationSpec(
                    variables = [
                        ...
                        ("Zone Thermal Comfort ASHRAE 55 Simple Model Summer or Winter Clothes Not Comfortable Time", "Thermal Zone Name"),
                        ...
                    ],
                    ...
                ),
                ...
            ),
            ...
        },
        ...
    )
    

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
from eprllib.Utils.agent_utils import get_agent_name, config_validation

class ASHRAE55SimpleModel(BaseReward):
    REQUIRED_KEYS = {
        "thermal_zone": str,
    }
    
    def __init__(
        self,
        reward_fn_config: Dict[str, Any],
    ):
        """
        This reward function takes the energy demand in the time step by the heating and cooling system and 
        calculates the energy reward as the sum of both divided by the maximal energy consumption of the 
        heating and cooling active system, respectively.
        
        Also, it takes the Zone Thermal Comfort ASHRAE 55 Simple Model Summer or Winter Clothes Not Comfortable Time[hr]
        variable to determine the comfort reward.
        
        Together, they constitute the total reward that, weighted with the beta factor, gives the agents
        a signal to optimize the policy.

        Args:
            reward_fn_config (Dict[str, Any]): The dictionary to configure the variables that each agent uses
            to calculate the reward. The dictionary must have the following keys:
            
                1. thermal_zone,
            
            All these variables start with the name of the agent and then the value of the reference name.
        """
        # Validate the config.
        config_validation(reward_fn_config, self.REQUIRED_KEYS)
        
        super().__init__(reward_fn_config)
        
        self.agent_name = None
        self.comfort = None
    
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
            infos (dict): infos dict must to provide the Zone Thermal Comfort ASHRAE 55 Simple Model Summer or Winter Clothes Not Comfortable Time.

        Returns:
            float: reward value.
        """
        if self.agent_name is None:
            self.agent_name = get_agent_name(infos)
            self.comfort = get_variable_name(
                self.agent_name, 
                "Zone Thermal Comfort ASHRAE 55 Simple Model Summer or Winter Clothes Not Comfortable Time", 
                self.reward_fn_config['thermal_zone']
            )
        if infos[self.comfort] > 0:
            return -1
        else:
            return 0

# === Hierarchical versions ===

class HierarchicalASHRAE55SimpleModel(BaseReward):
    REQUIRED_KEYS = {
        "thermal_zone": str,
    }
    
    def __init__(
        self,
        reward_fn_config: Dict[str,Any],
        ):
        """
        """
        # Validate the config.
        config_validation(reward_fn_config, self.REQUIRED_KEYS)
        
        super().__init__(reward_fn_config)
        
        self.agent_name = None
        self.comfort = None
    
    @override(BaseReward)
    def get_reward(
    self,
    infos: Dict[str,Any] = None,
    terminated_flag: bool = False,
    truncated_flag: bool = False
    ) -> float:
        """
        """
        if self.agent_name is None:
            self.agent_name = get_agent_name(infos)
            self.comfort = get_variable_name(
                self.agent_name,
                "Zone Thermal Comfort ASHRAE 55 Simple Model Summer or Winter Clothes Not Comfortable Time",
                self.reward_fn_config['thermal_zone']
            )
            
        reward = 0
        for ix in range(len(infos[self.comfort])):
            if infos[self.comfort][ix] > 0:
                reward -= 1
        
        return reward/len(infos[self.comfort])
      