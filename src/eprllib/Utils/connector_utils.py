

import eprllib.Utils.observation_utils as observation_utils
from typing import Dict, Any, Tuple # type: ignore

def set_variables_in_obs(
    env_config: Dict[str, Any],
    agent: str,
    obs_indexed: Dict[str, int],
    obs_space_len: int = 0
) -> Tuple[Dict[str,int], int]:
    """
    Set the variables in the observation indexed dictionary.
    :param env_config: environment configuration
    :type env_config: Dict[str, Any]
    :param agent: agent name
    :type agent: str
    :param obs_indexed: indexed observation dictionary
    :type obs_indexed: Dict[str, int]
    :return: indexed observation dictionary and observation space length
    :rtype: Tuple[Dict[str, int], int]
    """
    if env_config["agents_config"][agent]["observation"]['variables'] is not None:
        for var in range(len(env_config["agents_config"][agent]["observation"]['variables'])):
            obs_indexed.update({observation_utils.get_variable_name(
                agent,
                env_config["agents_config"][agent]["observation"]['variables'][var][0],
                env_config["agents_config"][agent]["observation"]['variables'][var][1]
                ): obs_space_len})
            obs_space_len += 1
    return obs_indexed, obs_space_len

def set_internal_variables_in_obs(
    env_config: Dict[str, Any],
    agent: str,
    obs_indexed: Dict[str, int],
    obs_space_len: int = 0
) -> Tuple[Dict[str,int], int]:
    """
    Set the internal variables in the observation indexed dictionary.
    :param env_config: environment configuration
    :type env_config: Dict[str, Any]
    :param agent: agent name
    :type agent: str
    :param obs_indexed: indexed observation dictionary
    :type obs_indexed: Dict[str, int]
    :return: indexed observation dictionary and observation space length
    :rtype: Tuple[Dict[str, int], int]
    """
    if env_config["agents_config"][agent]["observation"]['internal_variables'] is not None:
        for var in range(len(env_config["agents_config"][agent]["observation"]['internal_variables'])):
            obs_indexed.update({observation_utils.get_internal_variable_name(
                agent,
                env_config["agents_config"][agent]["observation"]['internal_variables'][var][0],
                env_config["agents_config"][agent]["observation"]['internal_variables'][var][1]
                ): obs_space_len})
            obs_space_len += 1
    return obs_indexed, obs_space_len

def set_meters_in_obs(
    env_config: Dict[str, Any],
    agent: str,
    obs_indexed: Dict[str, int],
    obs_space_len: int = 0
) -> Tuple[Dict[str,int], int]:
    """
    Set the meters in the observation indexed dictionary.
    :param env_config: environment configuration
    :type env_config: Dict[str, Any]
    :param agent: agent name
    :type agent: str
    :param obs_indexed: indexed observation dictionary
    :type obs_indexed: Dict[str, int]
    :return: indexed observation dictionary and observation space length
    :rtype: Tuple[Dict[str, int], int]
    """
    if env_config["agents_config"][agent]["observation"]['meters'] is not None:
        for meter in range(len(env_config["agents_config"][agent]["observation"]['meters'])):
            obs_indexed.update({observation_utils.get_meter_name(
                agent,
                env_config["agents_config"][agent]["observation"]['meters'][meter]
                ): obs_space_len})
            obs_space_len += 1
    return obs_indexed, obs_space_len

def set_zone_simulation_parameters_in_obs(
    env_config: Dict[str, Any],
    agent: str,
    obs_indexed: Dict[str, int],
    obs_space_len: int = 0
) -> Tuple[Dict[str,int], int]:
    """
    Set zone simulation parameters in the observation indexed dictionary.
    :param env_config: environment configuration
    :type env_config: Dict[str, Any]
    :param agent: agent name
    :type agent: str
    :param obs_indexed: indexed observation dictionary
    :type obs_indexed: Dict[str, int]
    :return: indexed observation dictionary and observation space length
    :rtype: Tuple[Dict[str, int], int]
    """
    if env_config["agents_config"][agent]["observation"]['zone_simulation_parameters'] is not None:
        for key, value in env_config["agents_config"][agent]["observation"]['zone_simulation_parameters'].items():
            if value:
                obs_indexed.update({observation_utils.get_parameter_name(
                    agent,
                    key
                    ): obs_space_len})
                obs_space_len += 1
    return obs_indexed, obs_space_len

def set_prediction_variables_in_obs(
    env_config: Dict[str, Any],
    agent: str,
    obs_indexed: Dict[str, int],
    obs_space_len: int = 0
) -> Tuple[Dict[str,int], int]:
    """
    Set prediction variables in the observation indexed dictionary.
    :param env_config: environment configuration
    :type env_config: Dict[str, Any]
    :param agent: agent name
    :type agent: str
    :param obs_indexed: indexed observation dictionary
    :type obs_indexed: Dict[str, int]
    :return: indexed observation dictionary and observation space length
    :rtype: Tuple[Dict[str, int], int]
    """
    if env_config["agents_config"][agent]["observation"]['use_one_day_weather_prediction']:
        for key, value in env_config["agents_config"][agent]["observation"]['prediction_variables'].items():
            if value:
                for hour in range(env_config["agents_config"][agent]["observation"]['prediction_hours']):
                    obs_indexed.update({observation_utils.get_parameter_prediction_name(
                        agent,
                        key,
                        hour + 1
                    ): obs_space_len})
                    obs_space_len += 1
    return obs_indexed, obs_space_len

def set_other_obs_in_obs(
    env_config: Dict[str, Any],
    agent: str,
    obs_indexed: Dict[str, int],
    obs_space_len: int = 0
) -> Tuple[Dict[str,int], int]:
    """
    Set other obs in the observation indexed dictionary.
    :param env_config: environment configuration
    :type env_config: Dict[str, Any]
    :param agent: agent name
    :type agent: str
    :param obs_indexed: indexed observation dictionary
    :type obs_indexed: Dict[str, int]
    :return: indexed observation dictionary and observation space length
    :rtype: Tuple[Dict[str, int], int]
    """
    if env_config["agents_config"][agent]["observation"]['other_obs'] is not None:
        for other_obs in range(len(env_config["agents_config"][agent]["observation"]['other_obs'])):
            obs_indexed.update({observation_utils.get_other_obs_name(
                agent,
                env_config["agents_config"][agent]["observation"]['other_obs'][other_obs]
                ): obs_space_len})
            obs_space_len += 1
    return obs_indexed, obs_space_len

def set_actuators_in_obs(
    env_config: Dict[str, Any],
    agent: str,
    obs_indexed: Dict[str, int],
    obs_space_len: int = 0
) -> Tuple[Dict[str,int], int]:
    """
    Set actuators in the observation indexed dictionary.
    :param env_config: environment configuration
    :type env_config: Dict[str, Any]
    :param agent: agent name
    :type agent: str
    :param obs_indexed: indexed observation dictionary
    :type obs_indexed: Dict[str, int]
    :return: indexed observation dictionary and observation space length
    :rtype: Tuple[Dict[str, int], int]
    """
    if env_config["agents_config"][agent]["observation"]['use_actuator_state']:
        for actuator in range(len(env_config["agents_config"][agent]["action"]['actuators'])):
            actuator_component_type = env_config["agents_config"][agent]["action"]['actuators'][actuator][0]
            actuator_control_type = env_config["agents_config"][agent]["action"]['actuators'][actuator][1]
            actuator_key = env_config["agents_config"][agent]["action"]['actuators'][actuator][2]
            obs_indexed.update({observation_utils.get_actuator_name(
                agent,
                actuator_component_type,
                actuator_control_type,
                actuator_key
                ): obs_space_len})
            obs_space_len += 1
    return obs_indexed, obs_space_len