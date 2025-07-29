"""
CO2 Concentration for IAQ reward function
==========================================


"""
import logging
import numpy as np
from typing import Any, Dict
from numpy.typing import NDArray
from numpy import float32
from eprllib.Agents.Rewards.BaseReward import BaseReward
from eprllib.Utils.observation_utils import get_variable_name, get_meter_name
from eprllib.Utils.annotations import override
from eprllib.Utils.agent_utils import config_validation

logger = logging.getLogger("ray.rllib")

class IAQThermalComfortEnergyReward(BaseReward):
    REQUIRED_KEYS: Dict[str, Any] = {
        "thermal_zone": str,
        "co2_threshold": float | int,
        "exhaust_fan_name": str,
        "cooling_name": str,
        "heating_name": str,
        "cooling_energy_ref": float | int,
        "heating_energy_ref": float | int,
        "beta": float,
        "rho_1": float,
        "rho_2": float,
    }
    def __init__(
        self,
        reward_fn_config: Dict[str, Any],
    ) -> None:
        """
        
        """
        config_validation(reward_fn_config, self.REQUIRED_KEYS)
        
        super().__init__(reward_fn_config)
        
        self.agent_name: str = "None"
        self.concentration: int = 0
        self.fan_energy: int = 0
        self.occupancy: int = 0
        
        self.max_energy = max(reward_fn_config['cooling_energy_ref'],reward_fn_config['heating_energy_ref']) #1.111*(250 * 0.1111)/0.6 * (60*60/6) # air density*(Pressure*Flow)/efficiency * ((seconds in one minute)*(minutes in one hour)/(timestep per hour)

        self.beta = reward_fn_config['beta']
        self.rho_1 = reward_fn_config['rho_1']
        self.rho_2 = reward_fn_config['rho_2']
        # Check that the sum of rho_1 and rho_2 is equal to 1.
        if not np.isclose(self.rho_1 + self.rho_2, 1.0):
            raise ValueError("The sum of rho_1 and rho_2 must be equal to 1.")
        
    @override(BaseReward)
    def set_initial_parameters(
        self,
        agent_name: str,
        obs_indexed: Dict[str, int]
    ) -> None:
        # === Asignation of variables ===
        if self.agent_name is "None":
            self.agent_name = agent_name
            self.concentration = obs_indexed[get_variable_name(
                self.agent_name, 
                "Zone Air CO2 Concentration", 
                self.reward_fn_config['thermal_zone']
            )]
            self.fan_energy = obs_indexed[get_variable_name(
                self.agent_name, 
                "Fan Electricity Energy", 
                self.reward_fn_config['exhaust_fan_name']
            )]
            self.occupancy = obs_indexed[get_variable_name(
                self.agent_name, 
                "Zone People Occupant Count",
                self.reward_fn_config['thermal_zone']
            )]
            self.comfort = obs_indexed[get_variable_name(
                self.agent_name, 
                # "Zone Thermal Comfort ASHRAE 55 Simple Model Summer or Winter Clothes Not Comfortable Time",
                "Zone Mean Air Temperature",
                self.reward_fn_config['thermal_zone']
            )]
            self.cooling = obs_indexed[get_meter_name(
                self.agent_name,
                self.reward_fn_config['cooling_name']
            )]
            self.heating = obs_indexed[get_meter_name(
                self.agent_name,
                self.reward_fn_config['heating_name']
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
            infos (dict): infos dict must to provide the occupancy level and the Zone Mean Temperature.

        Returns:
            float: reward normalize value
        """
        
        
        # === Reward calculation ===
        # Energy
        e_rew = -1 * np.clip(
            ((obs[self.cooling] + obs[self.heating] + obs[self.fan_energy]) \
                / ( max(self.reward_fn_config['cooling_energy_ref'], self.reward_fn_config['heating_energy_ref']) + self.max_energy)),
            0,
            1
        )
        
        # CO2 Concentration
        iaq_rew = 0
        if obs[self.occupancy] > 0:
            if obs[self.concentration] > self.reward_fn_config['co2_threshold']:
                iaq_rew = -1
        
        # Thermal comfort
        tc_rew = 0
        if obs[self.occupancy] > 0:
            if obs[self.comfort] < 19 or obs[self.comfort] > 26:
                tc_rew = -1
        
        # Total reward
        reward = 0.
        reward += (1-self.beta) * (self.rho_1*tc_rew + self.rho_2*iaq_rew)
        reward += self.beta * e_rew
        
        return reward
        