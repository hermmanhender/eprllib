"""
Base Filter
============

This module contains the base class for defining ``Filter`` functions used in agent specifications.
``Filter``s are used to preprocess observations before they are fed to the agent. The ``BaseFilter``
class provides the basic structure and methods that can be extended to create custom ``Filter``s.

This class can not be used directly in ``eprllib``, but as a base to create new ```Filter``s. All the ``Filter``s
must be based in this class.

The methods provided here are used during inizialization and execution of the environment.
You have to overwrite the following methods:

    - ``setup(self)``
    - ``_get_filtered_obs``
    
"""
from typing import Any, Dict
from numpy import floating
from numpy.typing import NDArray

from eprllib import logger
from eprllib.Utils.annotations import OverrideToImplementCustomLogic

class BaseFilter:
    """
    Base class for defining filter functions used in agent specifications.
    Filters are used to preprocess observations before they are fed to the agent.
    """
    filter_fn_config: Dict[str, Any] = {}
    agent_name: str = ""
    
    def __init__(
        self,
        agent_name: str,
        filter_fn_config: Dict[str, Any] = {}
    ):
        """
        Initializes the BaseFilter class.

        Args:
            filter_fn_config (Dict[str, Any]): Configuration dictionary for the filter function.
        """
        self.filter_fn_config = filter_fn_config
        self.agent_name = agent_name
        
        logger.info(f"BaseFilter: The BaseFilter was correctly inicializated with {self.filter_fn_config} config.")
        
        # Make sure, `setup()` is only called once, no matter what.
        if hasattr(self, "_is_setup") and self._is_setup:
            raise RuntimeError(
                "``BaseActionMapper.setup()`` called twice within your ActionMapper implementation "
                f"{self}! Make sure you are using the proper inheritance order "
                " and that you are NOT overriding the constructor, but "
                "only the ``setup()`` method of your subclass."
            )
        try:
            self.setup()
        except AttributeError as e:
            raise e

        self._is_setup:bool = True
    
    def get_filtered_obs(
        self,
        env_config: Dict[str, Any],
        agent_states: Dict[str, Any],
    ) -> NDArray[floating[Any]]:
        # Check if the agent_states dictionary is empty
        if not agent_states:
            msg = "agent_states dictionary is empty"
            logger.error(msg)
            raise ValueError(msg)
        
        # Check if all values in the agent_states dictionary are numeric
        if not all(isinstance(value, (int, float)) for value in agent_states.values()):
            msg = "All values in agent_states must be numeric"
            logger.error(msg)
            raise ValueError(msg)
        
        # Generate a copy of the agent_states to avoid conflicts with global variables.
        agent_states_copy = agent_states.copy()
        
        return self._get_filtered_obs(env_config, agent_states_copy)
    
    
    # ===========================
    # === OVERRIDABLE METHODS ===
    # ===========================
    
    @OverrideToImplementCustomLogic
    def setup(self):
        """
        Sets up the components of the module.

        This is called automatically during the __init__ method of this class.
        """
        pass
    
    
    @OverrideToImplementCustomLogic
    def _get_filtered_obs(
        self,
        env_config: Dict[str, Any],
        agent_states: Dict[str, Any],
    ) -> NDArray[floating[Any]]:
        """
        Returns the filtered observations for the agent based on the environment configuration
        and agent states. This method processes the raw observations according to the filter
        configuration specified in filter_fn_config.

        Args:
            env_config (Dict[str, Any]): Configuration dictionary for the environment. Can include settings 
            that affect how the observations are filtered.
            
            agent_states (Dict[str, Any], optional): Dictionary containing the states of the agent.

        Returns:
            NDarray: Filtered observations as a numpy array of float64 values.
        """
        msg = "BaseFilter: This method should be implemented in a subclass."
        logger.error(msg)
        raise NotImplementedError(msg)