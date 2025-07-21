"""
Agent utilities
================

"""
from typing import Any, Dict, List, get_args, get_origin
from eprllib import logger

def get_agent_name(state: Dict[str, Any] | List) -> str:
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
    if isinstance(state, list):
        # Check if the list is empty
        if not state:
            msg = "The state list is empty"
            logger.error(msg)
            raise ValueError(msg)
        return state[0].split(':')[0]
    elif isinstance(state, dict):
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


def config_validation(config: Dict[str, Any], required_keys: Dict[str, Any]) -> None:
    """
    Validates a configuration dictionary against a set of required keys and types.

    Args:
        config: The configuration dictionary to validate.
        required_keys: A dictionary where keys are required key names and values are
                       expected types (or Tuple type hints).

    Raises:
        ValueError: If a required key is missing.
        TypeError: If a value has an incorrect type.
    """
    for key, expected_type in required_keys.items():
        if key not in config:
            msg = f"Missing required key: '{key}'"
            logger.error(msg)
            raise ValueError(msg)

        value = config[key]
        origin = get_origin(expected_type)

        if origin is tuple:  # Check if it's a Tuple type hint
            if not isinstance(value, tuple):
                msg = f"The key '{key}' must be a tuple, but got a {type(value).__name__}"
                logger.error(msg)
                raise TypeError(msg)

            expected_types_in_tuple = get_args(expected_type)
            if len(value) != len(expected_types_in_tuple):
                msg = f"The key '{key}' must be a tuple with {len(expected_types_in_tuple)} elements, but has {len(value)}"
                logger.error(msg)
                raise ValueError(msg)

            for i, (item, item_expected_type) in enumerate(zip(value, expected_types_in_tuple)):
                if not isinstance(item, item_expected_type):
                    msg = f"The element {i} in key '{key}' must be a {item_expected_type.__name__}, but got a {type(item).__name__}"
                    logger.error(msg)
                    raise TypeError(msg)

        elif not isinstance(value, expected_type):
            msg = (f"The key '{key}' expects the type {expected_type.__name__}, "
                            f"but got {type(value).__name__}")
            logger.error(msg)
            raise TypeError(msg)

