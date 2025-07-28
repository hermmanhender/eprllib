"""
Fully-shared-parameters filter
===============================

This module contains the filter class for the fully-shared-parameters case. This filter remove the actuator state from
the agent state dictionary and return the observation as a plain vector (a numpy array) without the actuator information.

The actuator state could be added after as a augmented observation vector in the ``FullySharedParametersConnector``
class for ``AgentsConnectors``. The use of both methods together avoid the duplication of information in the observation
space.
"""
import numpy as np
from numpy.typing import NDArray
from typing import Any, Dict # type: ignore
from eprllib.Agents.Filters.BaseFilter import BaseFilter
from eprllib.Utils.observation_utils import get_actuator_name
from eprllib.Utils.annotations import override
from eprllib.Utils.agent_utils import get_agent_name
from eprllib import logger

class FullySharedParametersFilter(BaseFilter):
    """
    Filter class for the fully-shared-parameters policy.
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
        
        self.agent_name = None
    
    @override(BaseFilter)
    def get_filtered_obs(
        self,
        env_config: Dict[str, Any],
        agent_states: Dict[str, Any],
    ) -> NDArray[np.float32]:
        """
        Filter the observation for the agent by removing the actuator state from the agent state vector.

        Args:
            env_config (Dict[str, Any]): Configuration dictionary for the environment.
            agent_states (Dict[str, Any], optional): Dictionary containing the states of the agent.

        Returns:
            NDarray: Filtered observations as a numpy array of float32 values.
        """
        # Generate a copy of the agent_states to avoid conflicts with global variables.
        agent_states_copy = agent_states.copy()
        
        # As we don't know the agent that belong this filter, we auto-dectect his name form the name of the variables names
        # inside the agent_states_copy dictionary. The agent_states dict has keys with the format of "agent_name: ...".
        if self.agent_name is None:
            self.agent_name = get_agent_name(agent_states_copy)
        
        # Remove from agent_states_copy the actuators state that the agent manage, if any.
        for actuator_config in env_config["agents_config"][self.agent_name]["action"]["actuators"]:
            _ = agent_states_copy.pop(get_actuator_name(self.agent_name, actuator_config[0], actuator_config[1], actuator_config[2]), None)
        
        logger.debug(f"Filtered observation for agent {self.agent_name}: {agent_states_copy}")
        # Return a flat array with the values of the agent_states_copy without actuators state.
        return np.array(list(agent_states_copy.values()), dtype='float32')
    