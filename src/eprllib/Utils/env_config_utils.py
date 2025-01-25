"""
Utilities for the environment configuration
============================================

Work in progress...
"""

from typing import Dict, Optional
import inspect
from eprllib.Env.EnvConfig import EnvConfig
from gymnasium.spaces import Box, Discrete
import sys

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

def env_config_validation(MyEnvConfig: EnvConfig) -> bool:
    """
    Validate the EnvConfig object before to be used in the env_config parameter of RLlib environment config.
    """
    # Check that the variables defined in EnvConfig are the allowed in the EnvConfig base
    # class.
    allowed_vars = inspect.get_annotations(EnvConfig).keys()
    for var in vars(MyEnvConfig):
        if var not in allowed_vars:
            raise ValueError(f"The variable {var} is not allowed in EnvConfig. Allowed variables are {allowed_vars}")
    return True

def env_config_to_dict(MyEnvConfig: EnvConfig) -> Dict:
    """
    Convert an EnvConfig object into a dict before to be used in the env_config parameter of RLlib environment config.
    """
    # Check that the variables defined in EnvConfig are the allowed in the EnvConfig base
    # class.
    # if env_config_validation(MyEnvConfig):
    return vars(MyEnvConfig)


def to_json(
    MyEnvConfig: EnvConfig,
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
    if env_config_validation(MyEnvConfig):
        env_config_json = json.dumps(env_config_to_dict(MyEnvConfig))
    else:
        return
    
    # generate a unique number based on time
    time_id = str(int(time.time()))
    # check the implementation of output_path
    if output_path is None:
        output_path = './'
    path = output_path+f'/{time_id}_env_config.json'
    # save the json string to a file
    with open(path, 'x') as f:
        f.write(env_config_json)
    return path

def from_json(
    path: str
    ) -> EnvConfig:
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
    env_config = EnvConfig(**env_config_dict)
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
