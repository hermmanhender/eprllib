"""
Agent utilities
================

"""
from typing import Any, Dict, List, get_args, get_origin

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
    """
    # As we don't know the agent that belong this filter, we auto-dectect his name form the name of the variables names
    # inside the agent_states_copy dictionary. The agent_states dict has keys with the format of "agent_name: ...".
    if isinstance(state, list):
        return state[0].split(':')[0]
    elif isinstance(state, dict):
        return list(state.keys())[0].split(':')[0]


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
            raise ValueError(f"The following key is missing: '{key}'")

        value = config[key]
        origin = get_origin(expected_type)

        if origin is tuple:  # Check if it's a Tuple type hint
            if not isinstance(value, tuple):
                raise TypeError(f"The key '{key}' must be a tuple, but got a {type(value).__name__}")

            expected_types_in_tuple = get_args(expected_type)
            if len(value) != len(expected_types_in_tuple):
                raise ValueError(f"The key '{key}' must be a tuple with {len(expected_types_in_tuple)} elements, but has {len(value)}")

            for i, (item, item_expected_type) in enumerate(zip(value, expected_types_in_tuple)):
                if not isinstance(item, item_expected_type):
                    raise TypeError(f"The element {i} in key '{key}' must be a {item_expected_type.__name__}, but got a {type(item).__name__}")

        elif not isinstance(value, expected_type):
            raise TypeError(f"The key '{key}' expects the type {expected_type.__name__}, "
                            f"but got {type(value).__name__}")

