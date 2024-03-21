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
    T_confort = config.get('T_confort', 23.5)
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
    T_confort = config.get('T_confort', 23.5)
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