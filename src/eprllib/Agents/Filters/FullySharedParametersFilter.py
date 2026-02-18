"""
Fully-shared-parameters filter
===============================

This module contains the filter class for the fully-shared-parameters case. This filter remove the actuator state from
the agent state dictionary and return the observation as a plain vector (a numpy array) without the actuator information.

The actuator state could be added after as a augmented observation vector in the ``FullySharedParametersConnector``
class for ``Connectors``. The use of both methods together avoid the duplication of information in the observation
space.
"""
import numpy as np
from numpy.typing import NDArray
from typing import Any, Dict

from eprllib.Agents.Filters.BaseFilter import BaseFilter
from eprllib.Utils.observation_utils import get_actuator_name
from eprllib.Utils.annotations import override
from eprllib import logger

class FullySharedParametersFilter(BaseFilter):
    """
    Filter class for the fully-shared-parameters policy.
    """
    @override(BaseFilter)
    def setup(self) -> None:
        """
        Initializes the FullySharedParametersFilter class.

        Args:
            filter_fn_config (Dict[str, Any]): Configuration dictionary for the filter function.
        """
        pass
    
    @override(BaseFilter)
    def _get_filtered_obs(
        self,
        env_config: Dict[str, Any],
        agent_states: Dict[str, Any],
    ) -> NDArray[np.float64]:
        """
        Filter the observation for the agent by removing the actuator state from the agent state vector.

        Args:
            env_config (Dict[str, Any]): Configuration dictionary for the environment.
            agent_states (Dict[str, Any], optional): Dictionary containing the states of the agent.

        Returns:
            NDarray: Filtered observations as a numpy array of float64 values.
        """
        # Remove from agent_states_copy the actuators state that the agent manage, if any.
        for actuator_config in env_config["agents_config"][self.agent_name]["action"]["actuators"]:
            _ = agent_states.pop(get_actuator_name(self.agent_name, actuator_config[0], actuator_config[1], actuator_config[2]), None)
        
        logger.debug(f"FullySharedParametersFilter: Filtered observation for agent {self.agent_name}: {agent_states}")
        # Return a flat array with the values of the agent_states_copy without actuators state.
        return np.array(list(agent_states.values()), dtype='float64')
    