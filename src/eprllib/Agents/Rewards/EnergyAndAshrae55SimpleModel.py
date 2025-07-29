"""
Energy and ASHRAE 55 Simple Model reward function
==================================================


"""
from typing import Any, Dict # type: ignore
from numpy.typing import NDArray
from numpy import float32
from eprllib.Agents.Rewards.BaseReward import BaseReward
from eprllib.Agents.Rewards.EnergyRewards import EnergyWithMeters, HierarchicalEnergyWithMeters
from eprllib.Agents.Rewards.ASHRAE55SimpleModel import ASHRAE55SimpleModel, HierarchicalASHRAE55SimpleModel
from eprllib.Utils.annotations import override

class EnergyAndASHRAE55SimpleModel(BaseReward):
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
            
                1. thermal_zone,
                2. beta,
                3. cooling_name,
                4. heating_name,
                5. cooling_energy_ref,
                6. heating_energy_ref.
            
            All this variables start with the name of the agent and then the value of the reference name.
        """
        super().__init__(reward_fn_config)
        self.comfort_reward = ASHRAE55SimpleModel({
            "thermal_zone": reward_fn_config['thermal_zone']
        })
        self.energy_reward = EnergyWithMeters({
            "cooling_name": reward_fn_config['cooling_name'],
            "heating_name": reward_fn_config['heating_name'],
            "cooling_energy_ref": reward_fn_config['cooling_energy_ref'],
            "heating_energy_ref": reward_fn_config['heating_energy_ref']
        })
        self.beta = reward_fn_config['beta']
    
    @override(BaseReward)
    def set_initial_parameters(
        self,
        agent_name: str,
        obs_indexed: Dict[str, int]
    ) -> None:
        self.comfort_reward.set_initial_parameters(agent_name, obs_indexed)
        self.energy_reward.set_initial_parameters(agent_name, obs_indexed)
        
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
        reward = 0.
        reward += (1-self.beta) * self.comfort_reward.get_reward(obs, terminated, truncated)
        reward += self.beta * self.energy_reward.get_reward(obs, terminated, truncated)
        return reward

# === Hierarchical version ===

class HierarchicalEnergyAndASHRAE55SimpleModel(BaseReward):
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
            
                1. thermal_zone,
                2. beta,
                3. cooling_name,
                4. heating_name,
                5. cooling_energy_ref,
                6. heating_energy_ref.
            
            All this variables start with the name of the agent and then the value of the reference name.
        """
        super().__init__(reward_fn_config)
        self.comfort_reward = HierarchicalASHRAE55SimpleModel({
            "thermal_zone": reward_fn_config['thermal_zone']
        })
        self.energy_reward = HierarchicalEnergyWithMeters({
            "cooling_name": reward_fn_config['cooling_name'],
            "heating_name": reward_fn_config['heating_name'],
            "cooling_energy_ref": reward_fn_config['cooling_energy_ref'],
            "heating_energy_ref": reward_fn_config['heating_energy_ref']
        })
        self.beta = reward_fn_config['beta']
    
    @override(BaseReward)
    def set_initial_parameters(
        self,
        agent_name: str,
        obs_indexed: Dict[str, int]
    ) -> None:
        self.comfort_reward.set_initial_parameters(agent_name, obs_indexed)
        self.energy_reward.set_initial_parameters(agent_name, obs_indexed)
        
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
        reward = 0.
        reward += (1-self.beta) * self.comfort_reward.get_reward(obs, terminated, truncated)
        reward += self.beta * self.energy_reward.get_reward(obs, terminated, truncated)
        return reward
    
class LowLevelEnergyAndASHRAE55SimpleModel(BaseReward):
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
            
                1. thermal_zone,
                2. beta,
                3. cooling_name,
                4. heating_name,
                5. cooling_energy_ref,
                6. heating_energy_ref.
            
            All this variables start with the name of the agent and then the value of the reference name.
        """
        super().__init__(reward_fn_config)
        self.comfort_reward = ASHRAE55SimpleModel({
            "thermal_zone": reward_fn_config['thermal_zone']
        })
        self.energy_reward = EnergyWithMeters({
            "cooling_name": reward_fn_config['cooling_name'],
            "heating_name": reward_fn_config['heating_name'],
            "cooling_energy_ref": reward_fn_config['cooling_energy_ref'],
            "heating_energy_ref": reward_fn_config['heating_energy_ref']
        })
        self.beta = reward_fn_config['beta']
    
    @override(BaseReward)
    def set_initial_parameters(
        self,
        agent_name: str,
        obs_indexed: Dict[str, int]
    ) -> None:
        self.comfort_reward.set_initial_parameters(agent_name, obs_indexed)
        self.energy_reward.set_initial_parameters(agent_name, obs_indexed)
        
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
        reward = 0.
        reward += (1-self.beta) * self.comfort_reward.get_reward(obs, terminated, truncated)
        reward += self.beta * self.energy_reward.get_reward(obs, terminated, truncated)
        return reward