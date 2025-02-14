"""
General utilities
==================

Work in progress...
"""
from typing import Set, Dict, List, get_origin, get_args, Union, _SpecialGenericAlias


def variable_checking(
    epJSON_file:str,
) -> Set:
    """
    This function check if the epJSON file has the required variables.

    Args:
        epJSON_file(str): path to the epJSON file.

    Return:
        set: list of missing variables.
    """
    pass

from typing import get_origin, get_args, Union, _SpecialGenericAlias

# Importar las clases que deseas validar
from eprllib.AgentsConnectors.BaseConnector import BaseConnector
from eprllib.Episodes.BaseEpisode import BaseEpisode
from eprllib.Agents.Rewards.BaseReward import BaseReward
from eprllib.Agents.Triggers.BaseTrigger import BaseTrigger
from eprllib.Agents.Filters.BaseFilter import BaseFilter

def validate_properties(obj, expected_types):
    """
    Enhanced version that supports Union types and optional properties
    
    Args:
        obj: The object to validate
        expected_types: A dictionary mapping property names to either:
                       - A type
                       - A tuple of types (Union)
                       - (type, bool) tuple where bool indicates if property is optional
    """
    errors = []
    is_valid = True
    
    for prop_name, type_spec in expected_types.items():
        # Handle optional properties
        is_optional = False
        expected_type = type_spec
        
        if isinstance(type_spec, tuple) and len(type_spec) == 2 and isinstance(type_spec[1], bool):
            expected_type, is_optional = type_spec
            
        # Check if property exists
        if not hasattr(obj, prop_name):
            if not is_optional:
                errors.append(f"Missing required property: {prop_name}")
                is_valid = False
            continue
            
        actual_value = getattr(obj, prop_name)
        
        # Handle None for optional properties
        if actual_value is None and is_optional:
            continue
            
        # Handle union types
        valid_types = (expected_type,) if isinstance(expected_type, type) else expected_type
        
        def is_instance_of_type(value, types):
            if isinstance(types, _SpecialGenericAlias):
                types = (types,)
            for typ in types:
                origin = get_origin(typ)
                if origin:
                    if origin is Union:
                        if any(is_instance_of_type(value, get_args(typ))):
                            return True
                    elif isinstance(value, origin):
                        return True
                elif isinstance(value, typ):
                    return True
            return False

        if not is_instance_of_type(actual_value, valid_types):
            errors.append(
                f"Property '{prop_name}' has incorrect type. "
                f"Expected {', '.join(t.__name__ if isinstance(t, type) else str(t) for t in valid_types)}, "
                f"got {type(actual_value).__name__}"
            )
            is_valid = False
    
    return is_valid, errors
