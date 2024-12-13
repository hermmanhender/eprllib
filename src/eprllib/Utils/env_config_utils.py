

from typing import Dict
import inspect
from eprllib.Env.EnvConfig import EnvConfig

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
    if env_config_validation(MyEnvConfig):
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
    if output_path == None:
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