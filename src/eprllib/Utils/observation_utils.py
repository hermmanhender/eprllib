"""
Observation utilities
======================

This module contain some utilities used within the observation space, function, and other parts
of the programm related with the observations.

"""

def get_variable_name(agent: str, variable_name:str, variable_key:str) -> str:
    """
    This function is used to get the variable name in the observation space.

    Args:
        agent (str): The agent name.
        variable_name (str): The variable name.
        variable_key (str): The variable key.

    Returns:
        str: The variable name in the observation space.
    """
    return f"{agent}: {variable_name}: {variable_key}"

def get_internal_variable_name(agent: str, variable_type:str, variable_key:str) -> str:
    """
    This function is used to get the internal variable name in the observation space.

    Args:
        agent (str): The agent name.
        variable_type (str): The variable type.
        variable_key (str): The variable key.

    Returns:
        str: The internal variable name in the observation space.
    """
    return get_variable_name(agent, variable_type, variable_key)

def get_meter_name(agent: str, meter_name:str) -> str:
    """
    This function is used to get the meter name in the observation space.

    Args:
        agent (str): The agent name.
        meter_name (str): The meter name.

    Returns:
        str: The meter name in the observation space.
    """
    return f"{agent}: {meter_name}"

def get_actuator_name(agent: str, actuator_component_type:str, actuator_control_type:str, actuator_key:str) -> str:
    """
    This function is used to get the actuator name in the observation space.

    Args:
        agent (str): The agent name.
        actuator_component_type (str): The actuator component type.
        actuator_control_type (str): The actuator control type.
        actuator_key (str): The actuator key.

    Returns:
        str: The actuator name in the observation space.
    """
    return f"{agent}: {actuator_component_type}: {actuator_control_type}: {actuator_key}"

def get_parameter_name(agent: str, parameter_name:str) -> str:
    """
    This function is used to get the parameter name in the observation space.

    Args:
        agent (str): The agent name.
        parameter_name (str): The parameter name.

    Returns:
        str: The parameter name in the observation space.
    """
    return f"{agent}: {parameter_name}"

def get_parameter_prediction_name(agent: str, parameter_name:str, hour:int) -> str:
    """
    This function is used to get the parameter prediction name in the observation space.

    Args:
        agent (str): The agent name.
        parameter_name (str): The parameter name.
        hour (int): The hour.

    Returns:
        str: The parameter prediction name in the observation space.
    """
    return f"{agent}: {parameter_name}: {hour}"

def get_other_obs_name(agent: str, other_obs_name:str) -> str:
    """
    This function is used to get the other observation name in the observation space.

    Args:
        agent (str): The agent name.
        other_obs_name (str): The other observation name.

    Returns:
        str: The other observation name in the observation space.
    """
    return f"{agent}: {other_obs_name}"
