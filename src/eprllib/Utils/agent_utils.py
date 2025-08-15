"""
Agent utilities
================

"""
from typing import Any, Dict, List, get_args, get_origin # type: ignore
from eprllib import logger

def get_agent_name(state: Dict[str, Any]|List[str]) -> str:
    """
    Get the agent name from the state dictionary. The state dictionay can be the agent_states dictionary or the infos 
    dictionary. Both contain the agent name in all the variables keys as: "agent_name: variable[0]: variable[1]: ...".
    This method is util for autodetect the agent name in the implementation of filters, triggers, or rewards classes for
    agents.

    Args:
        state (Dict[str, Any] | List): The state dictionary.

    Returns:
        str: The agent name.
    
    Raises:
        TypeError: If the state is neither a dictionary nor a list.
        ValueError: If the state is empty.
    """
    # As we don't know the agent that belong this filter, we auto-dectect his name form the name of the variables names
    # inside the agent_states_copy dictionary. The agent_states dict has keys with the format of "agent_name: ...".
    if isinstance(state, List):
        # Check if the list is empty
        if not state:
            msg = "The state list is empty"
            logger.error(msg)
            raise ValueError(msg)
        return state[0].split(':')[0]
    elif isinstance(state, Dict):
        # Check if the dictionary is empty
        if not state:
            msg = "The state dictionary is empty"
            logger.error(msg)
            raise ValueError(msg)
        return list(state.keys())[0].split(':')[0]
    else:
        msg = f"The state must be a dictionary or a list, but got a {type(state).__name__}"
        logger.error(msg)
        raise TypeError(msg)




def _validate_type(value: Any, expected_type: Any, path: str = "") -> None:
    """Recursively validates a value against an expected type."""
    origin = get_origin(expected_type)
    
    if origin is tuple:
        # if the origin is a list, try to convert in a tuple
        if isinstance(value, list):
            value = tuple(value)
        
        elif not isinstance(value, tuple):
            msg = f"Expected tuple at {path}, got {type(value).__name__}"
            logger.error(msg)
            raise TypeError(msg)
        
        expected_types = get_args(expected_type)
        if len(value) != len(expected_types):
            msg = f"Expected tuple with {len(expected_types)} elements at {path}, got {len(value)}"
            logger.error(msg)
            raise ValueError(msg)
        
        for i, (item, item_type) in enumerate(zip(value, expected_types)):
            _validate_type(item, item_type, f"{path}[{i}]")
    
    elif origin is dict:
        if not isinstance(value, dict):
            msg = f"Expected dict at {path}, got {type(value).__name__}"
            logger.error(msg)
            raise TypeError(msg)
        
        key_type, value_type = get_args(expected_type)
        for k, v in value.items():
            _validate_type(k, key_type, f"{path}.key")
            _validate_type(v, value_type, f"{path}[{k}]")
    
    elif origin is list:
        if not isinstance(value, list):
            msg = f"Expected list at {path}, got {type(value).__name__}"
            logger.error(msg)
            raise TypeError(msg)
        
        item_type = get_args(expected_type)[0]
        for i, item in enumerate(value):
            _validate_type(item, item_type, f"{path}[{i}]")
    
    elif expected_type is Any:
        pass  # Any type is always valid
    
    elif not isinstance(value, expected_type):
        msg = f"Expected {expected_type.__name__} at {path}, got {type(value).__name__}"
        logger.error(msg)
        raise TypeError(msg)


def config_validation(config: Dict[str, Any], required_keys: Dict[str, Any]) -> None:
    """
    Validates a configuration dictionary against required keys and nested types.

    Args:
        config: The configuration dictionary to validate.
        required_keys: A dictionary where keys are required key names and values are
                       expected types (supports nested types like Dict[str, Tuple[int, float, Any]]).

    Raises:
        ValueError: If a required key is missing.
        TypeError: If a value has an incorrect type.
    """
    for key, expected_type in required_keys.items():
        if key not in config:
            msg = f"Missing required key: '{key}'"
            logger.error(msg)
            raise ValueError(msg)
        
        _validate_type(config[key], expected_type, f"config['{key}']")
