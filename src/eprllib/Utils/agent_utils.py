"""
Agent utilities
================

"""
from typing import Any, Dict, List, Tuple

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
    
def config_validation(config, required_keys):
    for key, expected_type in required_keys.items():
        if key not in config:
            raise ValueError(f"The following key is missing: '{key}'")
        
        # Special case for Tuple[str, str, str]
        if expected_type == Tuple[str, str, str]:
            if not isinstance(expected_type, tuple):
                raise TypeError(f"The key '{key}' must be a tuple, but goot a {type(expected_type).__name__}")
            if len(expected_type) != 3:
                raise ValueError(f"The key '{key}' must be a tuple with 3 elements, but has {len(expected_type)}")
            if not all(isinstance(item, str) for item in expected_type):
                raise TypeError(f"All the elements in the key '{key}' must be strings, but was gotten: {[type(item).__name__ for item in expected_type]}")
        
        elif expected_type == Tuple[str, str]:
            if not isinstance(expected_type, tuple):
                raise TypeError(f"The key '{key}' must be a tuple, but goot a {type(expected_type).__name__}")
            if len(expected_type) != 2:
                raise ValueError(f"The key '{key}' must be a tuple with 2 elements, but has {len(expected_type)}")
            if not all(isinstance(item, str) for item in expected_type):
                raise TypeError(f"All the elements in the key '{key}' must be strings, but was gotten: {[type(item).__name__ for item in expected_type]}")
            
        elif expected_type == Tuple[int, int]:
            if not isinstance(expected_type, tuple):
                raise TypeError(f"The key '{key}' must be a tuple, but goot a {type(expected_type).__name__}")
            if len(expected_type) != 2:
                raise ValueError(f"The key '{key}' must be a tuple with 2 elements, but has {len(expected_type)}")
            if not all(isinstance(item, int) for item in expected_type):
                raise TypeError(f"All the elements in the key '{key}' must be integers, but was gotten: {[type(item).__name__ for item in expected_type]}")
        
        elif not isinstance(config[key], expected_type):
            raise TypeError(f"The key '{key}' expect the type {expected_type.__name__}, "
                            f"but got {type(config[key]).__name__}")
            