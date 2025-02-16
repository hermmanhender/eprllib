"""
Fully-shared-parameters policy (Observation Function)
=====================================================

This module contains the filter class for the fully-shared-parameters policy. This filter builds the observation
vector for the agent using this policy. The main reason to use this filter is to remove the actuator state from
the agent state vector, as this will be added in the vector of actuator_states in the ``FullySharedParametersConnector``
class for ``AgentsConnectors``.
"""

from typing import Any, Dict
import numpy as np
from eprllib.Agents.Filters.BaseFilter import BaseFilter
from eprllib.Utils.observation_utils import get_actuator_name
from eprllib.Utils.annotations import override

class FullySharedParametersFilter(BaseFilter):
    """
    Filter class for the fully-shared-parameters policy.

    This filter builds the observation vector for the agent using this policy. The main reason to use this 
    filter is to remove the actuator state from the agent state vector, as this will be added in the vector 
    of actuator_states in the ``FullySharedParametersConnector`` class for ``AgentsConnectors``.
    """
    def __init__(
        self,
        filter_fn_config: Dict[str, Any]
    ):
        """
        Initializes the FullySharedParametersFilter class.

        Args:
            filter_fn_config (Dict[str, Any]): Configuration dictionary for the filter function.
        """
        super().__init__(filter_fn_config)
    
    @override(BaseFilter)
    def get_filtered_obs(
        self,
        env_config: Dict[str, Any],
        agent_states: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Sets the observation for the agent by removing the actuator state from the agent state vector.

        Args:
            env_config (Dict[str, Any]): Configuration dictionary for the environment.
            agent_states (Dict[str, Any], optional): Dictionary containing the states of the agent.

        Returns:
            Dict[str, Any]: Dictionary containing the observations for the agent.
        """
        agent_states_copy = agent_states.copy()
        # Remove from agent_states and save the actuator items.
        for agent in env_config["agents_config"].keys():
            for actuator_config in env_config["agents_config"][agent]["action"]["actuators"]:
                _ = agent_states_copy.pop(get_actuator_name(agent, actuator_config[0], actuator_config[1], actuator_config[2]), None)
        
        return np.array(list(agent_states_copy.values()), dtype='float32')