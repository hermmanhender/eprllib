"""
Predictive Thermal Control of Building Systems - Anna Marie Nygard-Ferguson (1990)
===================================================================================

This reward function implement the cost function proposed by Nygard-Ferguson (1990) in the These No 876.
The cost function takes into account both the energy consumption and the thermal comfort of the building occupants.

Reference:
    - Nygard-Ferguson, A. M. (1990). Predictive Control of Building Systems. PhD thesis, Ecole Polytechnique Federale de Lausanne.

"""
from typing import Any, Dict, List
from numpy.typing import NDArray
from numpy import float32
from eprllib.Agents.Rewards.BaseReward import BaseReward
from eprllib.Utils.observation_utils import get_meter_name, get_parameter_name
from eprllib.Utils.annotations import override
from eprllib.Utils.agent_utils import config_validation

class LiuAndHenze2006(BaseReward):
    REQUIRED_KEYS: Dict[str, Any] = {
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
        
        Also, it takes the Zone Thermal Comfort Fanger Model PMV variable to determine the comfort reward.
        
        Together, they constitute the total reward that, weighted with the beta factor C2, gives the agents
        a signal to optimize the policy.

        Args:
            reward_fn_config (Dict[str, Any]): The dictionary to configure the variables that each agent uses
            to calculate the reward. The dictionary must have the following keys:
            
                1. thermal_zone (str): The name of the thermal zone where the agent is located.
                2. cooling_energy_ref (float|int): The reference cooling energy demand for normalization.
                3. heating_energy_ref (float|int): The reference heating energy demand for normalization.
                4. cooling_name (str): The name of the cooling energy meter.
                5. heating_name (str): The name of the heating energy meter.
                6. timestep_per_hour (int, optional): The number of timesteps per hour. Default is 6.
                7. C2 (float|int, optional): The ponderation factor for the comfort reward. Default is 1500.
            
            All these variables start with the name of the agent and then the value of the reference name.
        """
        # Validate the config.
        config_validation(reward_fn_config, self.REQUIRED_KEYS)
        
        super().__init__(reward_fn_config)
        
        self.agent_name: str = "None"
        self.cooling_index: int = 0
        self.heating_index: int = 0
        self.hour_index: int = 0
        
        self.number_of_timesteps_per_hour: int = reward_fn_config.get("number_of_timesteps_per_hour", 6)
        self.prices: List[float|int]|float|int = reward_fn_config.get("prices", 0.2)
    
    @override(BaseReward)
    def set_initial_parameters(
        self,
        agent_name: str,
        obs_indexed: Dict[str, int]
    ) -> None:
        """
        This method can be overridden in subclasses to set initial parameters based on the provided obs_indexed.

        Args:
            agent_name (str): The name of the agent.
            obs_indexed (Dict[str, Any]): The obs_indexed dictionary containing necessary information for initialization.
        """
        if self.agent_name == "None":
            self.agent_name = agent_name
            
            if type(self.prices) is list:
                # check that the length of the prices list is 24
                if len(self.prices) != 24:
                    raise ValueError("The length of the prices list must be 24.")
                self.hour_index = obs_indexed[get_parameter_name(
                    self.agent_name,
                    "hour"
                )]
            
            self.cooling_index = obs_indexed[get_meter_name(
                self.agent_name,
                self.reward_fn_config['cooling_name']
            )]
            self.heating_index = obs_indexed[get_meter_name(
                self.agent_name,
                self.reward_fn_config['heating_name']
            )]
    
    
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
            obs (NDArray[float32]): The observation array containing the necessary information for calculating the reward.
            terminated (bool): Indicates if the episode has terminated.
            truncated (bool): Indicates if the episode has been truncated.

        Returns:
            float: reward value.
        """
        reward: float = 0.0
        
        # === Energy reward ===
        if type(self.prices) is list:
            assert type(self.prices) is list
            r_k: float|int = self.prices[obs[self.hour_index]]
            assert type(r_k) is float|int
        else:
            assert type(self.prices) is float|int
            r_k: float|int = self.prices

        
        reward += -r_k*(obs[self.cooling_index] + obs[self.heating_index])*60/self.number_of_timesteps_per_hour
        
        return reward
