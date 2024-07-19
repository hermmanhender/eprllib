"""This module contain the utils used in the Multi-agent definition files.
"""

from typing import Tuple, Dict, Any, List, Set
from gymnasium.spaces import Box

def env_value_inspection(env_config:Dict):
    """Examine that all the neccesary arguments to use in the MDP env definition exist.

    Args:
        env_config (Dict): env_config main dict.
    """
    if not env_config.get('ep_actuators', False):
        ValueError("No actuators defined in the environment configuration. Set the dictionary in"\
            "env_config['ep_actuators'] with the corresponded actuators.")
    if not env_config.get('action_space', False):
            ValueError("No action space defined in the environment configuration. Set the dictionary in"\
                "env_config['action_space'] with the corresponded action space.")

def runner_value_inspection(env_config:Dict):
    if not env_config.get('ep_actuators', False):
        ValueError("No actuators defined in the environment configuration. Set the dictionary in"\
            "env_config['ep_actuators'] with the corresponded actuators.")
    
    if env_config.get('use_building_properties', True):
            # a loop control the existency of the building_properties
            for key in env_config['episode_config'].keys():
                if key in [
                    'building_area', 'aspect_ratio', 'window_area_relation_north', 
                    'window_area_relation_east', 'window_area_relation_south', 
                    'window_area_relation_west', 'inercial_mass', 
                    'construction_u_factor', 'E_cool_ref', 'E_heat_ref',
                ]:
                    pass
                else:
                    ValueError(f'Building property {key} not found.')
    
def actuators_to_agents(agent_config:Dict) -> Tuple[List,Set,Dict,Dict]:
    """Take the ep_actuator dict and transform it to the agent, thermal zone, and actuator type dict.

    Args:
        agent_config (Dict): ep_actuator dict in the env_config.

    Returns:
        Tuple[Dict,Dict,Dict,Dict]: agent, thermal zone, and actuator type.
    """
    keys = agent_ids = list(agent_config.keys())
    values = list(agent_config.values())

    assert len(keys) == len(values)
    for i in range(len(values)):
        assert len(values[0])==len(values[i])
        
    act = []
    for i in range(len(keys)):
        act.append(values[i][0])
    actuators = {}
    for i in range(len(keys)):
        actuators[keys[i]] = act[i]

    lth = []
    for i in range(len(keys)):
        lth.append(values[i][1])
    thermal_zones = {}
    for i in range(len(keys)):
        thermal_zones[keys[i]] = lth[i]
    thermal_zone_ids = set(lth)
    typ = []
    for i in range(len(keys)):
        typ.append(values[i][2])
    actuator_type = {}
    for i in range(len(keys)):
        actuator_type[keys[i]] = typ[i]
    
    agents_str = ", ".join(agent_ids)
    print(f"The environment is defined with {len(agent_ids)} agents: {agents_str}")
    
    return agent_ids, thermal_zone_ids, actuators, thermal_zones, actuator_type

def obs_space(env_config:Dict):
    """This method construct the observation space of the environment.

    Args:
        env_config (Dict): The environment configuration dictionary.

    Returns:
        space.Box: The observation space of the environment.
    """
    obs_space_len = 0
    # actuator state.
    if env_config.get('use_actuator_state', True):
        obs_space_len += 1
    # agent_indicator.
    if env_config.get('use_agent_indicator', True):
        obs_space_len += 1
    # agent type.
    if env_config.get('use_agent_type', True):
        obs_space_len += 1
    # building properties.
    if env_config.get('use_building_properties', True):
        obs_space_len += len(list(env_config['building_properties'].keys())[0].values())
    # weather prediction.
    if env_config.get('use_one_day_weather_prediction', True):
        obs_space_len += 24*6
    # variables and meters.
    if env_config.get('ep_variables', False):
        obs_space_len += len(env_config['ep_variables'])
    if env_config.get('ep_meters', False):
        obs_space_len += len(env_config['ep_meters'])
    if env_config.get('time_variables', False):
        obs_space_len += len(env_config['time_variables'])
    if env_config.get('weather_variables', False):
        obs_space_len += len(env_config['weather_variables'])
    # discount the not observable variables.
    if env_config.get('no_observable_variables', False):
        obs_space_len -= len(env_config['no_observable_variables'])
    # construct the observation space.
    return Box(float("-inf"), float("inf"), (obs_space_len,))