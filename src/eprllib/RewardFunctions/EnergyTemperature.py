"""
Energy and Temperature Reward function
======================================

This reward function calculates the energy and temperature reward for each agent in the environment.
The energy reward is calculated using the values of energy and temperature from the environment.
"""
from typing import Dict, Any
from eprllib.RewardFunctions.RewardFunctions import RewardFunction

class EnergyTemperatureReward(RewardFunction):
    def __init__(
        self,
        reward_fn_config: Dict[str,Any]
    ):
        super().__init__(reward_fn_config)
        self.agents = {agent for agent in reward_fn_config.keys()}
        
        self.beta: Dict[str,float] = {agent: None for agent in self.agents}
        self.T_interior_name: Dict[str,str] = {agent: None for agent in self.agents}
        self.cooling_name: Dict[str,str] = {agent: None for agent in self.agents}
        self.heating_name: Dict[str,str] = {agent: None for agent in self.agents}
        
        for agent in self.agents:
            self.beta[agent] = reward_fn_config[agent]['beta']
            self.T_interior_name[agent] = reward_fn_config[agent]['T_interior_name']
            self.cooling_name[agent] = reward_fn_config[agent]['cooling_name']
            self.heating_name[agent] = reward_fn_config[agent]['heating_name']

    def calculate_reward(
    self,
    infos: Dict[str,Dict[str,Any]] = None
    ) -> Dict[str,float]:
        """
        This function returns reward of each timestep. Also, each term is multiply for a ponderation 
        factor of beta for the energy and (1-beta) for the comfort.
        Both terms are negatives, representing a penalti for demand energy and for generate discomfort.

        Args:
            self (Environment): RLlib environment.
            obs (dict): Zone Mean Air Temperature for the Thermal Zone in °C.
            infos (dict): infos dict must to provide the Zone Mean Temperature and the energy metrics.

        Returns:
            float: reward value.
        """
        # dictionary to save the reward values of each agent.
        reward_dict = {agent: 0. for agent in self.agents}
        
        # asign the ThermalZone values above defined to each agent
        for agent in self.agents:
            T_interior = infos[agent][self.T_interior_name[agent]]
            cooling_meter = infos[agent][self.cooling_name[agent]]
            heating_meter = infos[agent][self.heating_name[agent]]
                
            rew1 = -(1-self.beta[agent])*(T_interior-22)**2
            rew2 = -self.beta[agent]*(cooling_meter + heating_meter)
            
            reward_dict[agent] = rew1 + rew2
                
        return reward_dict
