"""
Coraci2021
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
import numpy as np
from typing import Any, Dict, List
from numpy.typing import NDArray
from numpy import float32
from eprllib.Agents.Rewards.BaseReward import BaseReward
from eprllib.Agents.Rewards.EnergyRewards import EnergyWithMetersEnded
from eprllib.Utils.observation_utils import get_variable_name
from eprllib.Utils.annotations import override
from eprllib.Utils.agent_utils import config_validation

from eprllib import logger

class Coraci2021(BaseReward):
    REQUIRED_KEYS = {
        "thermal_zone": str
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
        
        self.agent_name: str = "None"
        self.dt_index: int = 0
        self.zone_people_occupant_count_index: int = 0
        
        self.t_low: float = reward_fn_config.get("t_low", 20.0)
        self.t_high: float = reward_fn_config.get("t_high", 22.0)
        self.beta: float = reward_fn_config.get("beta", .8)
        
        self.energy_reward = EnergyWithMetersEnded({
            "cooling_name": reward_fn_config['cooling_name'],
            "heating_name": reward_fn_config['heating_name'],
            "comfort_temperature": reward_fn_config.get("comfort_temperature", 21.0),
            "timesteptoreward": reward_fn_config.get("timesteptoreward", 1),
            "thermal_zone": reward_fn_config['thermal_zone']
        })
        
        self.cumulated_reward: List[float] = []
        self.reward_timestep: int = 0
        self.timesteptoreward: int = reward_fn_config.get("timesteptoreward", 1)
        self.clip: float = reward_fn_config.get("clip", 0.15)
    
    @override(BaseReward)
    def set_initial_parameters(
        self,
        agent_name: str,
        obs_indexed: Dict[str, int]
    ) -> None:
        """
        This method can be overridden in subclasses to set initial parameters based on the provided infos.

        Args:
            infos (Dict[str, Any]): The infos dictionary containing necessary information for initialization.
        """
        self.energy_reward.set_initial_parameters(agent_name, obs_indexed)
        
        if self.agent_name == "None":
            self.agent_name = agent_name
            
            self.dt_index = obs_indexed[f"{self.agent_name}: dt: 0"]
            logger.info(f"'{self.agent_name}: dt: 0': {self.dt_index}")
            
            self.zone_people_occupant_count_index = obs_indexed[get_variable_name(
                self.agent_name, 
                "Zone People Occupant Count", 
                self.reward_fn_config['thermal_zone']
            )]
            logger.info(f"Agent name: {self.agent_name}, Zone People Occupant Count Index: {self.zone_people_occupant_count_index}")
            
            # Reset the state
            self.cumulated_reward = []
            self.reward_timestep = 0
    
    
    @override(BaseReward)
    def get_reward(
        self,
        prev_obs: NDArray[float32],
        prev_action: Any,
        obs: NDArray[float32],
        terminated: bool,
        truncated: bool
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
        # Create a variable to cumulate the energy and comfort reward for the timestep
        reward: float = 0.0
        
        # === ENERGY ===
        reward += (1 - self.beta) * self.energy_reward.get_reward(prev_obs, prev_action, obs, terminated, truncated)
        
        # If the place is occupied.
        if obs[self.zone_people_occupant_count_index] != 0:
            
            # === COMFORT ===
            # Get the internal temperature
            dt = np.clip(abs(obs[self.dt_index]),self.clip,1) # El clip a 0.15 hace que la recompensa por confort no aumente con temperaturas más elevadas de 20.5°C
            # Calculate the reward for the actual state.
            reward += - self.beta * np.clip(((dt)**3),0,1)
        
        # Save the time step reward to use it with timesteptoreward. 
        self.cumulated_reward.append(reward)
        
        # Increase the time step in 1.
        self.reward_timestep += 1

        # If is the end of the episode or the time steps to reward is complete, delivery the mean reward.
        if terminated or truncated or (self.reward_timestep % self.timesteptoreward == 0):
            reward2return = float(sum(self.cumulated_reward))
            self.cumulated_reward = []
            self.reward_timestep = 0
            return reward2return
        
        # Return 0 when the conditions are not met.
        return 0.0