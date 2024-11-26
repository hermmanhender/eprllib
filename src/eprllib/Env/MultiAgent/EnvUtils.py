"""
Environment Utilities
=====================

This module contain the methods used in the environment process.
"""

from typing import Tuple, Dict, List, Optional
from gymnasium.spaces import Box, Discrete
import numpy as np
import sys

def EP_API_add_path(version:Optional[str]="23-2-0", path:Optional[str]=None):
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
    
def actuators_to_agents(agent_config:Dict[str, List]) -> Tuple[List,List,Dict,Dict]:
    """
    Take the ep_actuator dict and transform it to the agent, thermal zone, and actuator type dict.

    Args:
        agent_config (Dict): ep_actuator dict in the env_config.

    Returns:
        Tuple[Dict,Dict,Dict,Dict]: agent, thermal zone, and actuator type.
    """
    # get the keys and values from the ep_actuator definitions
    keys = agent_ids = list(agent_config.keys())
    values = list(agent_config.values())
    
    # Verify that all the agents have the same quantity of attributes
    for i in range(len(values)):
        assert len(values[0])==len(values[i])
    
    # compose the actuators for EnergyPlus handle
    actuator = []
    for i in range(len(keys)):
        actuator_tuple = (values[i][0], values[i][1], values[i][2])
        actuator.append(actuator_tuple)
    # asign the actuator handle config to the agent
    agents_actuators = {}
    for i in range(len(keys)):
        agents_actuators[keys[i]] = actuator[i]

    # identify the thermal zones where there are agents
    agent_thermal_zone_names = []
    for i in range(len(keys)):
        agent_thermal_zone_names.append(values[i][3])
    # asign the thermal zone to each agent
    agents_thermal_zones = {}
    for i in range(len(keys)):
        agents_thermal_zones[keys[i]] = agent_thermal_zone_names[i]
    # a list with the thermal zones with agents (without repeat)
    thermal_zone_ids = []
    for zone in agent_thermal_zone_names:
        if not zone in  thermal_zone_ids:
            thermal_zone_ids.append(zone)
    
    # agent type
    typ = []
    for i in range(len(keys)):
        typ.append(values[i][4])
    agents_types = {}
    for i in range(len(keys)):
        agents_types[keys[i]] = typ[i]
    
    agents_str = ", ".join(agent_ids)
    print(f"The environment is defined with {len(agent_ids)} agents: {agents_str}")
    
    return agent_ids, thermal_zone_ids, agents_actuators, agents_thermal_zones, agents_types

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
