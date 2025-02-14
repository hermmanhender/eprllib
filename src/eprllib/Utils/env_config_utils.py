"""
Utilities for the environment configuration
============================================

Work in progress...
"""

from typing import Set, Dict, Optional
import inspect
from eprllib.Environment.EnvironmentConfig import EnvironmentConfig
from gymnasium.spaces import Box, Discrete
import sys
import numpy as np
from typing import get_origin, get_args, Union, _SpecialGenericAlias

# Importar las clases que deseas validar
from eprllib.AgentsConnectors.BaseConnector import BaseConnector
from eprllib.Episodes.BaseEpisode import BaseEpisode
from eprllib.Agents.Rewards.BaseReward import BaseReward
from eprllib.Agents.Triggers.BaseTrigger import BaseTrigger
from eprllib.Agents.Filters.BaseFilter import BaseFilter

def EP_API_add_path(version:Optional[str]="24-2-0", path:Optional[str]=None):
    """
    This method add the EnergyPlus Python API to the system path. This allow to use the 
    EnergyPlus program in Python. The minimal version of EnergyPlus is 9.3.0 and the default
    version (and the stable one) for eprllib is 23-2-0.

    Args:
        version (str, optional): Numeric version of EnergyPlus. Defaults to "23-2-0".
        path (Optional[str], optional): Complete path to the EnergyPlus installation directory 
        if this is different that the default installation. Defaults to None.
    """
    new_path = None
    if path is not None:
        new_path = path
    else:
        os_platform = sys.platform
        if os_platform == "linux":
            new_path = f"/usr/local/EnergyPlus-{version}"
        else:
            new_path = f"C:/EnergyPlusV{version}"

    # Check if the new path is already in sys.path
    if new_path not in sys.path:
        sys.path.insert(0, new_path)
        print(f"EnergyPlus API path added: {new_path}")

def env_config_validation(MyEnvConfig: EnvironmentConfig) -> bool:
    """
    Validate the EnvConfig object before to be used in the env_config parameter of RLlib environment config.
    """
    # Check that the variables defined in EnvConfig are the allowed in the EnvConfig base
    # class.
    allowed_vars = inspect.get_annotations(EnvironmentConfig).keys()
    for var in vars(MyEnvConfig):
        if var not in allowed_vars:
            raise ValueError(f"The variable {var} is not allowed in EnvConfig. Allowed variables are {allowed_vars}")
    return True

def to_json(
    MyEnvConfig: EnvironmentConfig,
    output_path: str = None
    ) -> str:
    """Convert an EnvConfig object into a json string before to be used in the env_config parameter of RLlib environment config.

    Args:
        MyEnvConfig (EnvConfig): _description_
        output_path (str, optional): _description_. Defaults to None.

    Returns:
        str: _description_
    """
    import json
    import time
    
    env_config_json = json.dumps(MyEnvConfig.build())

    # generate a unique number based on time
    time_id = str(int(time.time()))
    # check the implementation of output_path
    if output_path is None:
        output_path = './'
    path = output_path+f'/{time_id}_env_config.json'
    # save the json string to a file
    with open(path, 'x') as f:
        f.write(env_config_json)
    
    print(f"EnvConfig saved to {path}")
    
    return path

def from_json(
    path: str
    ) -> EnvironmentConfig:
    """Convert a json file into an EnvConfig object before to be used in the env_config parameter of RLlib environment config.

    Args:
        path (str): _description_

    Returns:
        EnvConfig: _description_
    """
    import json
    with open(path, 'r') as f:
        env_config_json = f.read()
    env_config_dict = json.loads(env_config_json)
    env_config = EnvironmentConfig(**env_config_dict)
    return env_config

def continuous_action_space():
    """
    This method construct the action space of the environment.
    
    Returns:
        gym.Box: Continuous action space with limits between [0,1].
    """
    return Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)

def discrete_action_space(n:int=2):
    """
    This method construct the action space of the environment.
    
    Returns:
        gym.Discrete: Discrete action space with limits between [0,10] and a step of 1.
    """
    return Discrete(n)


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
