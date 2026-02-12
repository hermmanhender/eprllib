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
from typing import Any, Dict, Tuple, List
import numpy as np
from numpy.typing import NDArray
from numpy import float32
from shapely.geometry import Point, Polygon
from eprllib.Agents.Rewards.BaseReward import BaseReward
from eprllib.Utils.observation_utils import get_variable_name
from eprllib.Utils.annotations import override
from eprllib.Utils.agent_utils import config_validation

from eprllib import logger

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
        
        self.agent_name: str = "None"
        self.zone_operative_temperature_index: int = 0
        self.zone_air_humidity_ratio_index: int = 0
        self.zone_people_occupant_count_index: int = 0
        
        self.max_temperature: float = reward_fn_config.get("max_temperature", 50.0)
        self.max_humidity: float = reward_fn_config.get("max_humidity", 0.03)
        
        # Se disminuyó 1 °C y 0.001 kgH2O/kgAire en los puntos que definen la zona de confort para hacer tender al control a estar más cerca del centro.
        self.thermal_zone_poligon: List[Tuple[float, float]] = reward_fn_config.get("thermal_zone_poligon", [
            (20.7/self.max_temperature, 0.0/self.max_humidity), 
            (27.3/self.max_temperature, 0.0/self.max_humidity), 
            (25.8/self.max_temperature, 0.011/self.max_humidity), 
            (18.6/self.max_temperature, 0.011/self.max_humidity)
            ])
        self.thermal_zone = Polygon(self.thermal_zone_poligon)
    
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
        if self.agent_name == "None":
            self.agent_name = agent_name
            self.zone_operative_temperature_index = obs_indexed[get_variable_name(
                self.agent_name, 
                "Zone Operative Temperature", 
                self.reward_fn_config['thermal_zone']
            )]
            logger.info(f"Agent name: {self.agent_name}, Zone Operative Temperature Index: {self.zone_operative_temperature_index}")
            
            self.zone_air_humidity_ratio_index = obs_indexed[get_variable_name(
                self.agent_name, 
                "Zone Air Humidity Ratio", 
                self.reward_fn_config['thermal_zone']
            )]
            logger.info(f"Agent name: {self.agent_name}, Zone Air Humidity Ratio Index: {self.zone_air_humidity_ratio_index}")
    
            self.zone_people_occupant_count_index = obs_indexed[get_variable_name(
                self.agent_name, 
                "Zone People Occupant Count", 
                self.reward_fn_config['thermal_zone']
            )]
            logger.info(f"Agent name: {self.agent_name}, Zone People Occupant Count Index: {self.zone_people_occupant_count_index}")
    
    
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
        actual_state = Point(obs[self.zone_operative_temperature_index]/self.max_temperature, obs[self.zone_air_humidity_ratio_index]/self.max_humidity)
        
        if obs[self.zone_people_occupant_count_index] == 0:
            return 0.0
        
        # If the point touch the limits of the polygon, the penalty is 0.
        elif self.thermal_zone.touches(actual_state):
            return 0.0
        
        # If the point is inside the polygon, the penalty is +1.
        elif self.thermal_zone.contains(actual_state):
            return 0.0
        
        else:
            return -np.clip(self.thermal_zone.exterior.distance(actual_state), 0.0, 1.0)
 
class ASHRAE55SimpleModelEnded(BaseReward):
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
        
        self.agent_name: str = "None"
        self.zone_operative_temperature_index: int = 0
        self.zone_air_humidity_ratio_index: int = 0
        self.zone_people_occupant_count_index: int = 0
        
        self.max_temperature: float = reward_fn_config.get("max_temperature", 50.0)
        self.max_humidity: float = reward_fn_config.get("max_humidity", 0.03)
        self.thermal_zone_poligon: List[Tuple[float, float]] = reward_fn_config.get("thermal_zone_poligon", [
            (21.7/self.max_temperature, 0.0/self.max_humidity), 
            (28.3/self.max_temperature, 0.0/self.max_humidity), 
            (26.8/self.max_temperature, 0.012/self.max_humidity), 
            (19.6/self.max_temperature, 0.012/self.max_humidity)
            ])
        self.thermal_zone = Polygon(self.thermal_zone_poligon)
        
        self.cumulated_reward: List[float] = []
        self.reward_timestep: int = 0
    
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
        if self.agent_name == "None":
            self.agent_name = agent_name
            self.zone_operative_temperature_index = obs_indexed[get_variable_name(
                self.agent_name, 
                "Zone Operative Temperature", 
                self.reward_fn_config['thermal_zone']
            )]
            logger.info(f"Agent name: {self.agent_name}, Zone Operative Temperature Index: {self.zone_operative_temperature_index}")
            
            self.zone_air_humidity_ratio_index = obs_indexed[get_variable_name(
                self.agent_name, 
                "Zone Air Humidity Ratio", 
                self.reward_fn_config['thermal_zone']
            )]
            logger.info(f"Agent name: {self.agent_name}, Zone Air Humidity Ratio Index: {self.zone_air_humidity_ratio_index}")
    
            self.zone_people_occupant_count_index = obs_indexed[get_variable_name(
                self.agent_name, 
                "Zone People Occupant Count", 
                self.reward_fn_config['thermal_zone']
            )]
            logger.info(f"Agent name: {self.agent_name}, Zone People Occupant Count Index: {self.zone_people_occupant_count_index}")
    
    
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
        reward: float = 0.0
        
        
        actual_state = Point(obs[self.zone_operative_temperature_index]/self.max_temperature, obs[self.zone_air_humidity_ratio_index]/self.max_humidity)
        
        if obs[self.zone_people_occupant_count_index] == 0:
            reward += 0.0
        
        # If the point touch the limits of the polygon, the penalty is 0.
        elif self.thermal_zone.touches(actual_state):
            reward += 0.0
        
        # If the point is inside the polygon, the penalty is +1.
        elif self.thermal_zone.contains(actual_state):
            reward += 0.0
        
        else:
            reward += -np.clip(self.thermal_zone.exterior.distance(actual_state), 0.0, 1.0)
 
        self.cumulated_reward.append(reward)
        
        if terminated or truncated or (self.reward_timestep % 6 == 0):
            # Calculate the average reward over the cumulated rewards.
            reward_to_return = np.clip(sum(self.cumulated_reward) / len(self.cumulated_reward), -1.0, 0.0)
            
            self.cumulated_reward = []
            self.reward_timestep = 0
            return reward_to_return
        
        self.reward_timestep += 1
        
        return 0.0
    
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
        
        self.agent_name = "None"
        self.comfort: str = "None"
    
    @override(BaseReward)
    def set_initial_parameters(
        self,
        obs: NDArray[float32],
        agent_name: str,
        obs_indexed: Dict[str, int]
    ) -> None:
        if self.agent_name == "None":
            self.agent_name = agent_name
            self.comfort_index = obs_indexed[get_variable_name(
                self.agent_name, 
                "Zone Thermal Comfort ASHRAE 55 Simple Model Summer or Winter Clothes Not Comfortable Time", 
                self.reward_fn_config['thermal_zone']
            )]
            logger.info(f"Agent name: {self.agent_name}, Comfort variable: {self.comfort_index}")
            
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
        """
        reward = 0
        for ix in range(len(obs[self.comfort_index])):
            if obs[self.comfort_index][ix] > 0:
                reward -= 1
        
        return reward/len(obs[self.comfort_index])
      