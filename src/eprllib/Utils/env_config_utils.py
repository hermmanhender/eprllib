"""
EnvironmentConfig Utilities
==============================

Work in progress...
"""

from typing import Set, Dict, Optional, List, Any
import inspect
from eprllib.Environment.EnvironmentConfig import EnvironmentConfig
from gymnasium.spaces import Box, Discrete
import sys
import numpy as np
from typing import get_origin, get_args, Union, Tuple

from eprllib import logger

def EP_API_add_path(path: Optional[str] = None) -> str:
    """
    Manages EnergyPlus paths for eprllib.

    If a 'path' argument is provided, it's used directly as the EnergyPlus installation path
    after validation.
    Otherwise, this method auto-detects installed EnergyPlus versions based on `LIST_OF_VERSIONS`.
    - If multiple versions are found, the latest one (as per `LIST_OF_VERSIONS`) is chosen,
      and a message is printed.
    - If no version is found, an error message is printed, and the program exits.

    The selected EnergyPlus installation (either user-provided or auto-detected) is then
    copied to a unique temporary directory. This temporary copy's path is added to `sys.path`,
    allowing isolated EnergyPlus environments for parallel execution.
    Temporary directories are registered for cleanup on program exit using `atexit`.

    Args:
        path (Optional[str], optional): Full path to an EnergyPlus installation directory.
            If None, auto-detection is performed. Defaults to None.

    Returns:
        str: The path to the temporary EnergyPlus copy that was added to `sys.path`.

    Raises:
        FileNotFoundError: If 'path' is provided but does not exist or is not a directory.
        RuntimeError: If auto-detection fails to find any suitable EnergyPlus installation,
                      or if copying the installation to a temporary directory fails.
    """
    logger.debug("EnvConfigUtils: Attempting to auto-detect EnergyPlus installation...")
    os_platform = sys.platform
    original_ep_path: Optional[str] = None
    if os_platform.startswith("linux"):  # Covers "linux" and "linux2"
        original_ep_path = f"/usr/local/EnergyPlus-{path}"
    elif os_platform == "win32":
        original_ep_path = f"C:/EnergyPlusV{path}"
    elif os_platform == "darwin":
        original_ep_path = f"/Applications/EnergyPlus-{path}"

    if original_ep_path is not None:
        if original_ep_path not in sys.path:
            sys.path.insert(0, original_ep_path)
            logger.debug(f"EnvConfigUtils: EnergyPlus API path added to sys.path: {original_ep_path}")
        else:
            logger.debug(f"EnvConfigUtils: EnergyPlus API path already in sys.path: {original_ep_path}")
        
        return original_ep_path
    
    else:
        logger.error(f"EnvConfigUtils: Warning: EnergyPlus auto-detection is not configured for this OS: {os_platform}. "
                "Please provide the path manually if detection fails.")
        raise RuntimeError(f"EnergyPlus auto-detection failed for OS: {os_platform}. "
                           "Please provide the path manually or ensure EnergyPlus is installed correctly.")
        

def env_config_validation(MyEnvConfig: EnvironmentConfig) -> bool:
    """
    Validate the EnvConfig object before to be used in the env_config parameter of RLlib environment config.
    """
    # Check that the variables defined in EnvConfig are the allowed in the EnvConfig base
    # class.
    allowed_vars = inspect.get_annotations(EnvironmentConfig).keys()
    for var in vars(MyEnvConfig):
        if var not in allowed_vars:
            msg = f"EnvConfigUtils: The variable '{var}' is not allowed in EnvConfig. Allowed variables are: {allowed_vars}"
            logger.error(msg)
            raise ValueError(msg)
    return True

def to_json(
    MyEnvConfig: EnvironmentConfig,
    output_path: Optional[str] = None
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
    
    env_config_json = json.dumps(MyEnvConfig.to_dict())

    # generate a unique number based on time
    time_id = str(int(time.time()))
    # check the implementation of output_path
    if output_path is None:
        output_path = './'
    path = output_path+f'/{time_id}_env_config.json'
    # save the json string to a file
    with open(path, 'x') as f:
        f.write(env_config_json)
    
    logger.info(f"EnvConfigUtils: EnvConfig saved to {path}")
    
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
    return Box(low=0.0, high=1.0, shape=(1,), dtype=np.float64)

def discrete_action_space(n:int=2):
    """
    This method construct the action space of the environment.
    
    Returns:
        gym.Discrete: Discrete action space with limits between [0,10] and a step of 1.
    """
    return Discrete(n)


def variable_checking(
    epJSON_file:str,
) -> Set[Any]|None:
    """
    This function check if the epJSON file has the required variables.

    Args:
        epJSON_file(str): path to the epJSON file.

    Return:
        set: list of missing variables.
    """
    pass

def validate_properties(obj: Any, expected_types: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Enhanced version that supports Union types and optional properties
    
    Args:
        obj: The object to validate
        expected_types: A dictionary mapping property names to either:
                       - A type
                       - A tuple of types (Union)
                       - (type, bool) tuple where bool indicates if property is optional
    """
    errors: List[str] = []
    is_valid = True
    
    for prop_name, type_spec in expected_types.items():
        # Handle optional properties
        is_optional = False
        expected_type = type_spec
        
        if isinstance(type_spec, tuple):
            if len(type_spec) == 2:
                if isinstance(type_spec[1], bool):
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
