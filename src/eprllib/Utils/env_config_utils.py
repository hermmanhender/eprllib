"""
Utilities for the environment configuration
============================================

Work in progress...
"""

from typing import Set, Dict, Optional, List
import inspect
from eprllib.Environment.EnvironmentConfig import EnvironmentConfig
from gymnasium.spaces import Box, Discrete
import sys
import os
import shutil
import tempfile
import atexit
import numpy as np
from typing import get_origin, get_args, Union, _SpecialGenericAlias

# Importar las clases que deseas validar
from eprllib.AgentsConnectors.BaseConnector import BaseConnector
from eprllib.Episodes.BaseEpisode import BaseEpisode
from eprllib.Agents.Rewards.BaseReward import BaseReward
from eprllib.Agents.Triggers.BaseTrigger import BaseTrigger
from eprllib.Agents.Filters.BaseFilter import BaseFilter

LIST_OF_VERSIONS = [
        "9-3-0",
        "9-4-0",
        "9-5-0",
        "9-6-0",
        "22-1-0",
        "22-2-0",
        "23-1-0",
        "23-2-0",
        "24-1-0",
        "24-2-0",
        "25-1-0"
    ]


_temp_dirs_to_clean: List[str] = []
_cleanup_registered_for_ep_api = False

def _cleanup_all_temp_ep_dirs():
    """Cleans up all temporary EnergyPlus directories created by EP_API_add_path."""
    global _temp_dirs_to_clean
    if not _temp_dirs_to_clean:
        return
    print("Cleaning up temporary EnergyPlus directories created by eprllib...")
    for d_path in _temp_dirs_to_clean:
        print(f"Attempting to remove temporary directory: {d_path}")
        shutil.rmtree(d_path, ignore_errors=True)
    _temp_dirs_to_clean = [] # Clear the list

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
    global _cleanup_registered_for_ep_api, _temp_dirs_to_clean

    original_ep_path: Optional[str] = None

    if path is not None:
        if not os.path.isdir(path):
            raise FileNotFoundError(
                f"Provided EnergyPlus path does not exist or is not a directory: {path}"
            )
        print(f"Using user-provided EnergyPlus path: {path}")
        original_ep_path = path
    else:
        print("Attempting to auto-detect EnergyPlus installation...")
        os_platform = sys.platform
        detected_installations: Dict[str, str] = {} # version_str -> path

        path_template: Optional[str] = None
        if os_platform.startswith("linux"):  # Covers "linux" and "linux2"
            path_template = "/usr/local/EnergyPlus-{}"
        elif os_platform == "win32":
            path_template = "C:/EnergyPlusV{}"
        elif os_platform == "darwin":
            path_template = "/Applications/EnergyPlus-{}"

        if path_template:
            for v_str in LIST_OF_VERSIONS:
                potential_path = path_template.format(v_str)
                if os.path.isdir(potential_path):
                    detected_installations[v_str] = potential_path
        else:
            print(f"Warning: EnergyPlus auto-detection is not configured for this OS: {os_platform}. "
                  "Please provide the path manually if detection fails.")

        if not detected_installations:
            error_msg = (
                "Auto-detection failed: No EnergyPlus installation found in standard locations "
                f"for versions: {', '.join(LIST_OF_VERSIONS)}. "
                "Please provide the path manually or ensure EnergyPlus is installed correctly."
            )
            print(error_msg)
            sys.exit(error_msg)

        latest_version_found: Optional[str] = None
        for v_str in reversed(LIST_OF_VERSIONS): # Check from newest to oldest
            if v_str in detected_installations:
                latest_version_found = v_str
                break
        
        original_ep_path = detected_installations[latest_version_found]

        if len(detected_installations) > 1:
            print(f"Multiple EnergyPlus versions detected: {list(detected_installations.values())}.")
            print(f"Using the latest detected version ({latest_version_found}): {original_ep_path}")
        else:
            print(f"Detected EnergyPlus version ({latest_version_found}): {original_ep_path}")

    # Create a temporary copy of the EnergyPlus installation
    # ep_install_dir_name = os.path.basename(original_ep_path)
    # temp_base_dir = tempfile.mkdtemp(prefix="eprllib_ep_")
    # temp_ep_path_for_env = os.path.join(temp_base_dir, ep_install_dir_name)

    # print(f"Copying EnergyPlus from '{original_ep_path}' to temporary location '{temp_ep_path_for_env}'...")
    # try:
    #     shutil.copytree(original_ep_path, temp_ep_path_for_env)
    #     print("Copy successful.")
    # except Exception as e:
    #     shutil.rmtree(temp_base_dir, ignore_errors=True) # Clean up base temp dir on copy failure
    #     error_msg = f"Error copying EnergyPlus installation: {e}"
    #     print(error_msg)
    #     sys.exit(f"Failed to create temporary EnergyPlus environment: {error_msg}")

    # if not _cleanup_registered_for_ep_api:
    #     atexit.register(_cleanup_all_temp_ep_dirs)
    #     _cleanup_registered_for_ep_api = True
    # _temp_dirs_to_clean.append(temp_base_dir)

    # if temp_ep_path_for_env not in sys.path:
    #     sys.path.insert(0, temp_ep_path_for_env)
    #     print(f"EnergyPlus API path from temporary copy added to sys.path: {temp_ep_path_for_env}")
    # else:
    #     print(f"EnergyPlus API path from temporary copy already in sys.path: {temp_ep_path_for_env}")

    # return temp_ep_path_for_env
    
    if original_ep_path not in sys.path:
        sys.path.insert(0, original_ep_path)
        print(f"EnergyPlus API path from temporary copy added to sys.path: {original_ep_path}")
    else:
        print(f"EnergyPlus API path from temporary copy already in sys.path: {original_ep_path}")
    
    return original_ep_path



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
