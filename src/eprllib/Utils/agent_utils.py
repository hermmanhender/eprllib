"""
Agent utilities
================

"""
from typing import Any, Dict, List

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
    