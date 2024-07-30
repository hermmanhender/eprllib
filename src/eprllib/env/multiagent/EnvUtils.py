"""This module contain the utils used in the Multi-agent definition files.
"""

from typing import Tuple, Dict, Any, List, Set
from gymnasium.spaces import Box, Discrete
import numpy as np

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
    if isinstance(env_config, str):
        ValueError("env_config is a string.")
    else:
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
    
def actuators_to_agents(agent_config:Dict[str, List]) -> Tuple[List,List,Dict,Dict]:
    """Take the ep_actuator dict and transform it to the agent, thermal zone, and actuator type dict.

    Args:
        agent_config (Dict): ep_actuator dict in the env_config.

    Returns:
        Tuple[Dict,Dict,Dict,Dict]: agent, thermal zone, and actuator type.
    """
    # get the keys and values from the ep_actuator definitions
    keys = agent_ids = list(agent_config.keys())
    values = list(agent_config.values())
    
    # Verify that all the agents have the same quantity of attributes
    for i in range(len(values)):
        assert len(values[0])==len(values[i])
    
    # compose the actuators for EnergyPlus handle
    actuator = []
    for i in range(len(keys)):
        actuator_tuple = (values[i][0], values[i][1], values[i][2])
        actuator.append(actuator_tuple)
    # asign the actuator handle config to the agent
    agents_actuators = {}
    for i in range(len(keys)):
        agents_actuators[keys[i]] = actuator[i]

    # identify the thermal zones where there are agents
    agent_thermal_zone_names = []
    for i in range(len(keys)):
        agent_thermal_zone_names.append(values[i][3])
    # asign the thermal zone to each agent
    agents_thermal_zones = {}
    for i in range(len(keys)):
        agents_thermal_zones[keys[i]] = agent_thermal_zone_names[i]
    # a list with the thermal zones with agents (without repeat)
    thermal_zone_ids = []
    for zone in agent_thermal_zone_names:
        if not zone in  thermal_zone_ids:
            thermal_zone_ids.append(zone)
    
    # agent type
    typ = []
    for i in range(len(keys)):
        typ.append(values[i][4])
    agents_types = {}
    for i in range(len(keys)):
        agents_types[keys[i]] = typ[i]
    
    agents_str = ", ".join(agent_ids)
    print(f"The environment is defined with {len(agent_ids)} agents: {agents_str}")
    
    return agent_ids, thermal_zone_ids, agents_actuators, agents_thermal_zones, agents_types

def obs_space(env_config:Dict, _thermal_none_ids:Set):
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
        for thermal_zone in _thermal_none_ids:
            thermal_zone_name = thermal_zone
            break
        obs_space_len += len([key for key in env_config['building_properties'][thermal_zone_name].keys()])
        
    # weather prediction.
    if env_config.get('use_one_day_weather_prediction', True):
        obs_space_len += 24*6
        
    # variables and meters.
    if env_config.get('ep_environment_variables', False):
        obs_space_len += len(env_config['ep_environment_variables'])
    
    if env_config.get('ep_thermal_zones_variables', False):
        obs_space_len += len(env_config['ep_thermal_zones_variables'])
    
    if env_config.get('ep_object_variables', False):
        for thermal_zone in _thermal_none_ids:
            thermal_zone_name = thermal_zone
            break
        obs_space_len += len([key for key in env_config['ep_object_variables'][thermal_zone_name].keys()])
        
    if env_config.get('ep_meters', False):
        obs_space_len += len(env_config['ep_meters'])
        
    if env_config.get('time_variables', False):
        obs_space_len += len(env_config['time_variables'])
        
    if env_config.get('weather_variables', False):
        obs_space_len += len(env_config['weather_variables'])
        
    # discount the not observable variables.
    if env_config.get('no_observable_variables', False):
        for thermal_zone in _thermal_none_ids:
            thermal_zone_name = thermal_zone
            break
        obs_space_len -= len(env_config['no_observable_variables'][thermal_zone_name])
        
    # construct the observation space.
    return Box(float("-inf"), float("inf"), (obs_space_len,))

def continuous_action_space():
    """This method construct the action space of the environment.
    
    Returns:
        gym.Box: Continuous action space with limits between [0,1].
    """
    return Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)

def discrete_action_space():
    """This method construct the action space of the environment.
    
    Returns:
        gym.Discrete: Discrete action space with limits between [0,10] and a step of 1.
    """
    return Discrete(11)

def environment_variables(env_config: Dict[str, Any]) -> Tuple[Dict[str, Tuple [str, str]], Dict[str, int]]:
    
    variables:Dict[str, Tuple [str, str]] = {}
    if env_config.get('ep_environment_variables', False):
        variables = {variable: (variable, 'Environment') for variable in env_config['ep_environment_variables']}
    
    var_handles: Dict[str, int] = {}
    
    return variables, var_handles

def thermal_zone_variables(env_config: Dict[str, Any], _thermal_zone_ids:Set) -> Tuple[Dict[str, Dict[str, Tuple[str,str]]],Dict[str,int]]:
    thermal_zone_variables: Dict[str, Tuple [str, str]] = {thermal_zone: {} for thermal_zone in _thermal_zone_ids}
    if env_config.get('ep_thermal_zones_variables', False):
        for thermal_zone in _thermal_zone_ids:
            thermal_zone_variables[thermal_zone].update({variable: (variable, thermal_zone) for variable in env_config['ep_thermal_zones_variables']})
    thermal_zone_var_handles: Dict[str, int] = {thermal_zone: {} for thermal_zone in _thermal_zone_ids}
    
    return thermal_zone_variables, thermal_zone_var_handles

def object_variables(env_config: Dict[str, Any], _thermal_zone_ids: Set) -> Tuple[Dict[str, Dict[str, Tuple[str,str]]],Dict[str,int]]:
    object_variables: Dict[str, Dict[str, Tuple [str, str]]] = {}
    if env_config.get('ep_object_variables', False):
        object_variables = env_config['ep_object_variables']
    object_var_handles = {thermal_zone: {} for thermal_zone in _thermal_zone_ids}
    
    return object_variables, object_var_handles

def meters(env_config: Dict[str, Any]) -> Tuple[Dict[str, str], Dict[str,int]]:
    meters: Dict[str,str] = {}
    if env_config.get('ep_meters', False):
        meters = {key: key for key in env_config['ep_meters']}
    meter_handles: Dict[str, int] = {}
    
    return meters, meter_handles

def actuators(env_config: Dict[str, Any], _agent_ids:Set) -> Tuple[Dict[str,Tuple[str,str,str]], Dict[str,int]]:
    
    actuators: Dict[str,Tuple[str,str,str]] = {agent: env_config['agents_config'][agent]['ep_actuator_config'] for agent in _agent_ids}
    actuator_handles: Dict[str, int] = {}
    
    return actuators, actuator_handles