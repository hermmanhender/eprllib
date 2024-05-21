from typing import Any, Dict

def reward_function_T3(config: Dict[str, Any], obs: dict, infos: dict) -> float:
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
    agent_ids = config['agent_ids']
    T_confort = config.get('T_confort', 22)
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

def reward_function_T2(config: Dict[str, Any], obs: dict, infos: dict) -> float:
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
    agent_ids = config['agent_ids']
    T_confort = config.get('T_confort', 22)
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

def reward_function_T3_Energy(config: Dict[str, Any], obs: dict, infos: dict) -> float:
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
    agent_ids = config['agent_ids']
    
    beta_reward = config['beta_reward'] if config.get('beta_reward', False) else 0.5
    T_confort = config.get('T_confort', 22)
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
    
def PPD_Energy_reward(config: Dict[str, Any], obs: dict, infos: dict) -> float:
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
    agent_ids = config['agent_ids']
    
    beta_reward = config['beta_reward'] if config.get('beta_reward', False) else 0.5
    cooling_meter = infos[agent_ids[0]]['cooling_meter']
    heating_meter = infos[agent_ids[0]]['heating_meter']
    PPD = infos[agent_ids[0]]['ppd']
    
    return -beta_reward*(cooling_meter + heating_meter) -(1-beta_reward)*PPD
    
def reward_function_ppd(config: Dict[str, Any], obs: dict, infos: dict) -> float:
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
    agent_ids = config['agent_ids']
    occupancy = infos[agent_ids[0]]['occupancy']
    T_zone = infos[agent_ids[0]]['Ti']
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

def normalize_reward_function(self, obs: dict, infos: dict) -> float:
    """This function returns the normalize reward calcualted as the sum of the penalty of the energy 
    amount of one week divide per the maximun reference energy demand and the average PPD comfort metric
    divide per the maximal PPF value that can be take (100). Also, each term is divide per the longitude
    of the episode and multiply for a ponderation factor of beta for the energy and (1-beta) for the comfort.
    Both terms are negatives, representing a penalti for demand energy and for generate discomfort.

    Args:
        self (Environment): RLlib environment.
        obs (dict): Zone Mean Air Temperature for the Thermal Zone in °C.
        infos (dict): infos dict must to provide the occupancy level and the Zone Mean Temperature.

    Returns:
        float: reward normalize value
    """
    # define the number of timesteps per episode
    cut_reward_len = self.env_config.get('cut_reward_len', 1)
    cut_reward_len_timesteps = cut_reward_len * 144
    # define the beta reward
    beta_reward = self.env_config.get('beta_reward', 0.5)
    # get the values of the energy and PPD
    agent_ids = self.env_config['agent_ids']
    cooling_meter = infos[agent_ids[0]]['cooling']
    heating_meter = infos[agent_ids[0]]['heating']
    ppd = infos[agent_ids[0]]['ppd']
    self.energy_list.append(cooling_meter+heating_meter)
    self.ppd_list.append(ppd)
    
    if self.timestep % cut_reward_len_timesteps == 0:
        reward = (-beta_reward*(sum(self.energy_list)/self.env_config['energy_ref']) \
            -(1-beta_reward)*(sum(self.ppd_list)/100)) \
                / cut_reward_len_timesteps
        # emptly the lists
        self.energy_list = []
        self.ppd_list = []
        return reward
    else:
        return 0