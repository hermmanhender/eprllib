"""This module contains the reward functions that different authors have proposed as well as some 
developed within the framework of the development of this library. It is expected over time to be 
able to establish reward functions that optimize the operation of buildings.

For greater flexibility of the library, all reward functions must have the following format:

    ```
    def reward_function_name(EnvObject, infos: dict) -> float:
        ...
        return reward
    ```
    
The arguments correspond to the following:
    EnvObject: It is the set of properties contained in `self` of the environment used.
    infos (Dict): This is the dictionary that is shared with the `reset()` and `step()` methods 
    of the Gymnasium library along with the observation and, if applicable, the `terminated` and 
    `truncated` values.

It has been preferred to use the `infos` dictionary and not the observation, since the latter is 
a numpy array and cannot be called by key values, which is prone to errors when developing the program 
and indexing a arrangement may change.

# Property `cut_reward_len_timesteps`
This property is used to define the number of timesteps that the reward is calculated for. Not all
the rewards function have this property, so it is not mandatory to define it.

# Property `beta_reward`
This property is used to define the value of the beta parameter of the reward function. Not all
the rewards function have this property, so it is not mandatory to define it.
"""
# Importing the neccesary libraries
from typing import Dict
from math import exp

# Defining the reward functions

def yang_2015(EnvObject, infos: Dict) -> float:
    """This reward function is based on the work of Yang et al. 2015. It balance between exergy and
    comfort. The reward is calculated for the operation of a heat pump and the internal temperatures
    in the divece must to be known. The ecuation modeled here is:
    
    ```
    r = 1/2*[beta*g(Delta T) + (1-beta)*f(Delta P)]
    g(Delta T) = -C_T*|(Delta T) - (Delta T_optimal)|
    f(Delta P) = 0 if (Delta P) >= 0 else (Delta P)/C_P
    ```

    The values of the constants $C_T$ and $C_P$ are obtained for the specific case (see the paper). The 
    defauts are 0.5/°C and 2.5 kW, for $C_T$ and $C_P$ respectively. The beta value default is 0.5.
    
    All this parameters are configurable in the EnvObject config. If don't, the defaults values are used.

    Args:
        EnvObject: _description_
        infos (Dict): _description_

    Returns:
        float: reward value.
    """
    return 9999

def dalamagkidis_2007(EnvObject, infos: Dict) -> float:
    """_description_

    Args:
        EnvObject: _description_
        infos (Dict): _description_

    Returns:
        float: reward value.
    """
    # define which rewards will be considered
    comfort_reward = EnvObject.env_config.get('comfort_reward', True)
    energy_reward = EnvObject.env_config.get('energy_reward', True)
    co2_reward = EnvObject.env_config.get('co2_reward', True)
    
    # if the EnvObject don't have the list to append the values here obtained, one list is created as a property of the EnvObject
    if not hasattr(EnvObject, 'ppd_list') and comfort_reward:
        EnvObject.ppd_list = []
    if not hasattr(EnvObject, 'energy_list') and energy_reward:
        EnvObject.energy_list = []
    if not hasattr(EnvObject, 'co2_list') and co2_reward:
        EnvObject.co2_list = []
    
    # define the number of timesteps per episode
    cut_reward_len_timesteps = EnvObject.env_config.get('cut_reward_len_timesteps', 144)
    
    # define the ponderation parameters
    w1 = EnvObject.env_config.get('w1', 0.80)
    w2 = EnvObject.env_config.get('w2', 0.01)
    w3 = EnvObject.env_config.get('w3', 0.20)
    
    # get the values of the energy, PPD, and CO2 from the infos dict
    agent_ids = EnvObject.env_config['agent_ids']
    if comfort_reward:
        ppd = infos[agent_ids[0]]['ppd']
        occupancy = infos[agent_ids[0]]['occupancy']
        if occupancy == 0:
            ppd = 0
        EnvObject.ppd_list.append(ppd)
    if energy_reward:
        energy_ref = EnvObject.env_config.get('energy_ref',1)
        cooling_meter = infos[agent_ids[0]]['cooling']
        heating_meter = infos[agent_ids[0]]['heating']
        EnvObject.energy_list.append(cooling_meter+heating_meter)
    if co2_reward:
        co2 = infos[agent_ids[0]]['co2']
        occupancy = infos[agent_ids[0]]['occupancy']
        if occupancy == 0:
            co2 = 0
        EnvObject.co2_list.append(co2)
    
    # calculate the reward if the timestep is divisible by the cut_reward_len_timesteps.
    # if don't return 0.
    if EnvObject.timestep % cut_reward_len_timesteps == 0:
        if comfort_reward:
            rew1 = (-w1*(sum(EnvObject.ppd_list)/100))/cut_reward_len_timesteps
        else:
            rew1 = 0
        if energy_reward:
            rew2 = (-w2*(sum(EnvObject.energy_list)/energy_ref))/cut_reward_len_timesteps
        else:
            rew2 = 0
        if co2_reward:
            rew3 = (-w3*(sum(1/(1+exp(-0.06(co2-870))))))/cut_reward_len_timesteps
        else:
            rew3 = 0
        reward = rew1 + rew2 + rew3
        
        # emptly the lists
        if comfort_reward:
            EnvObject.ppd_list = []
        if energy_reward:
            EnvObject.energy_list = []
        if co2_reward:
            EnvObject.co2_list = []
            
        return reward
    else:
        return 0






def reward_function_T3(EnvObject, infos: Dict) -> float:
    """This function returns the reward calcualted as the absolute value of the cube in the 
    difference between set point temperatur for comfort and the temperature measured in the 
    thermal zone when there are people in the zone but zero when is not.

    Args:
        config (Dict[str, Any]): env_config dictionary. Optionaly you can configurate the 'T_confot' variable.
        obs (dict): Zone Mean Air Temperature for the Thermal Zone in °C.
        infos (dict): infos dict must to provide the occupancy level and the Zone Mean Temperature.
        agent_ids (list): list of agent acting in the environment.

    Returns:
        float: reward value
    """
    agent_ids = EnvObject.config['agent_ids']
    T_confort = EnvObject.config.get('T_confort', 22)
    occupancy = infos[agent_ids[0]]['occupancy']
    T_zone = infos[agent_ids[0]]['Ti']
    if occupancy > 0: # When there are people in the thermal zone, a reward is calculated.
        reward = -(min(abs((T_confort - T_zone)**3),343.))
    else:
        # If there are not people, only the reward is calculated when the environment is far away
        # from the comfort temperature ranges. This limits are recommended in EnergyPlus documentation:
        # InputOutput Reference p.522
        if T_zone > 29.4:
            reward = -343.
        elif T_zone < 16.7:
            reward = -343.
        else:
            reward = 0.
    return reward

def reward_function_T2(EnvObject, infos: Dict) -> float:
    """This function returns the reward calcualted as the absolute value of the square in the 
    difference between set point temperatur for comfort and the temperature measured in the 
    thermal zone when there are people in the zone but zero when is not.

    Args:
        config (Dict[str, Any]): env_config dictionary. Optionaly you can configurate the 'T_confot' variable.
        obs (dict): Zone Mean Air Temperature for the Thermal Zone in °C.
        infos (dict): infos dict must to provide the occupancy level and the Zone Mean Temperature.
        agent_ids (list): list of agent acting in the environment.

    Returns:
        float: reward value
    """
    agent_ids = EnvObject.config['agent_ids']
    T_confort = EnvObject.config.get('T_confort', 22)
    occupancy = infos[agent_ids[0]]['occupancy']
    T_zone = infos[agent_ids[0]]['Ti']
    if occupancy > 0: # When there are people in the thermal zone, a reward is calculated.
        reward = -(min(abs((T_confort - T_zone)**2),49.))
    else:
        # If there are not people, only the reward is calculated when the environment is far away
        # from the comfort temperature ranges. This limits are recommended in EnergyPlus documentation:
        # InputOutput Reference p.522
        if T_zone > 29.4:
            reward = -49.
        elif T_zone < 16.7:
            reward = -49.
        else:
            reward = 0.
    return reward

def reward_function_T3_Energy(EnvObject, infos: Dict) -> float:
    """This function returns the reward calcualted as the absolute value of the cube in the 
    difference between set point temperatur for comfort and the temperature measured in the 
    thermal zone when there are people in the zone but zero when is not.

    Args:
        config (Dict[str, Any]): env_config dictionary. Optionaly you can configurate the 'T_confot' variable.
        obs (dict): Zone Mean Air Temperature for the Thermal Zone in °C.
        infos (dict): infos dict must to provide the occupancy level and the Zone Mean Temperature.
        agent_ids (list): list of agent acting in the environment.

    Returns:
        float: reward value
    """
    agent_ids = EnvObject.config['agent_ids']
    
    beta_reward = EnvObject.config.get('beta_reward', 0.5)
    T_confort = EnvObject.config.get('T_confort', 22)
    occupancy = infos[agent_ids[0]]['occupancy']
    T_zone = infos[agent_ids[0]]['Ti']
    cooling_meter = infos[agent_ids[0]]['cooling_meter']
    heating_meter = infos[agent_ids[0]]['heating_meter']
    
    if occupancy > 0: # When there are people in the thermal zone, a reward is calculated.
        reward = -beta_reward*(cooling_meter+heating_meter) -(1-beta_reward)*(min(abs((T_confort - T_zone)**3),343.))
    else:
        # If there are not people, only the reward is calculated when the environment is far away
        # from the comfort temperature ranges. This limits are recommended in EnergyPlus documentation:
        # InputOutput Reference p.522
        if T_zone > 29.4:
            reward = -beta_reward*(cooling_meter+heating_meter) -(1-beta_reward)*343.
        elif T_zone < 16.7:
            reward = -beta_reward*(cooling_meter+heating_meter) -(1-beta_reward)*343.
        else:
            reward = -beta_reward*(cooling_meter+heating_meter)
    return reward
    
def PPD_Energy_reward(EnvObject, infos: Dict) -> float:
    """This function returns the reward calcualted as the absolute value of the cube in the 
    difference between set point temperatur for comfort and the temperature measured in the 
    thermal zone when there are people in the zone but zero when is not.

    Args:
        config (Dict[str, Any]): env_config dictionary. Optionaly you can configurate the 'T_confot' variable.
        obs (dict): Zone Mean Air Temperature for the Thermal Zone in °C.
        infos (dict): infos dict must to provide the occupancy level and the Zone Mean Temperature.
        agent_ids (list): list of agent acting in the environment.

    Returns:
        float: reward value
    """
    agent_ids = EnvObject.config['agent_ids']
    
    beta_reward = EnvObject.config.get('beta_reward', 0.5)
    cooling_meter = infos[agent_ids[0]]['cooling_meter']
    heating_meter = infos[agent_ids[0]]['heating_meter']
    
    occupancy = infos[agent_ids[0]]['occupancy']
    if occupancy == 0:
        ppd = 0
    else:
        ppd = infos[agent_ids[0]]['ppd']
    
    return -beta_reward*(cooling_meter + heating_meter) -(1-beta_reward)*ppd
    
def reward_function_ppd(EnvObject, infos: Dict) -> float:
    """This function returns the reward calcualted as the absolute value of the cube in the 
    difference between set point temperatur for comfort and the temperature measured in the 
    thermal zone when there are people in the zone but zero when is not.

    Args:
        config (Dict[str, Any]): env_config dictionary. Optionaly you can configurate the 'T_confot' variable.
        obs (dict): Zone Mean Air Temperature for the Thermal Zone in °C.
        infos (dict): infos dict must to provide the occupancy level and the Zone Mean Temperature.
        agent_ids (list): list of agent acting in the environment.

    Returns:
        float: reward value
    """
    agent_ids = EnvObject.config['agent_ids']
    occupancy = infos[agent_ids[0]]['occupancy']
    T_zone = infos[agent_ids[0]]['Ti']
    occupancy = infos[agent_ids[0]]['occupancy']
    if occupancy == 0:
        ppd = 0
    else:
        ppd = infos[agent_ids[0]]['ppd']
    if occupancy > 0: # When there are people in the thermal zone, a reward is calculated.
        reward = -ppd
    else:
        # If there are not people, only the reward is calculated when the environment is far away
        # from the comfort temperature ranges. This limits are recommended in EnergyPlus documentation:
        # InputOutput Reference p.522
        if T_zone > 29.4:
            reward = -150
        elif T_zone < 16.7:
            reward = -150
        else:
            reward = 0.
    return reward

def normalize_reward_function(EnvObject, infos: Dict) -> float:
    """This function returns the normalize reward calcualted as the sum of the penalty of the energy 
    amount of one week divide per the maximun reference energy demand and the average PPD comfort metric
    divide per the maximal PPF value that can be take (100). Also, each term is divide per the longitude
    of the episode and multiply for a ponderation factor of beta for the energy and (1-beta) for the comfort.
    Both terms are negatives, representing a penalti for demand energy and for generate discomfort.

    Args:
        EnvObject: RLlib environment.
        infos (dict): infos dict must to provide the occupancy level and the Zone Mean Temperature.

    Returns:
        float: reward normalize value
    """
    # define the number of timesteps per episode
    cut_reward_len = EnvObject.env_config.get('cut_reward_len', 1)
    cut_reward_len_timesteps = cut_reward_len * 144
    # define the beta reward
    beta_reward = EnvObject.env_config.get('beta_reward', 0.5)
    # get the values of the energy and PPD
    agent_ids = EnvObject.env_config['agent_ids']
    cooling_meter = infos[agent_ids[0]]['cooling']
    heating_meter = infos[agent_ids[0]]['heating']
    occupancy = infos[agent_ids[0]]['occupancy']
    if occupancy == 0:
        ppd = 0
    else:
        ppd = infos[agent_ids[0]]['ppd']
    EnvObject.energy_list.append(cooling_meter+heating_meter)
    EnvObject.ppd_list.append(ppd)
    
    if EnvObject.timestep % cut_reward_len_timesteps == 0:
        reward = (-beta_reward*(sum(EnvObject.energy_list)/EnvObject.env_config['energy_ref']) \
            -(1-beta_reward)*(sum(EnvObject.ppd_list)/100)) \
                / cut_reward_len_timesteps
        # emptly the lists
        EnvObject.energy_list = []
        EnvObject.ppd_list = []
        return reward
    else:
        return 0