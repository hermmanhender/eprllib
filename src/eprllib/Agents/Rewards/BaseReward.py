"""
Reward Function
================

This module contains the base class for defining reward functions.

It is preferred to use the ``infos`` dictionary and not the observation, since the latter is 
a numpy array and cannot be called by key values, which is prone to errors when developing the program 
and indexing an array may change.

The terminated and truncated flags are arguments in the reward function ``get_reward`` method to allow
implementations with dispersed reward. This flags allow return the final reward when the episode ends.
"""
from typing import Dict, Any, List
from numpy.typing import NDArray
from numpy import floating

from eprllib import logger
from eprllib.Utils.annotations import OverrideToImplementCustomLogic

class BaseReward:
    """
    This class is the base class for defining reward functions.
    """
    reward_fn_config: Dict[str, Any] = {}
    agent_name: str = ""
    cumulated_reward: List[float] = []
    reward_timestep: int = 0
    
    def __init__(
        self,
        agent_name: str,
        reward_fn_config: Dict[str, Any] = {}
    ) -> None:
        """
        Initializes the base reward function with the given configuration.

        Args:
            reward_fn_config (Dict[str, Any]): Configuration dictionary for the reward function.
        """
        self.reward_fn_config = reward_fn_config
        self.agent_name = agent_name
        
        self.cumulated_reward = []
        self.reward_timestep = 0
        
        logger.info(f"BaseReward: The BaseReward was correctly inicializated with {self.reward_fn_config} config.")
        
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
        self._set_initial_parameters_once: bool = False
    
    
    def reset(self) -> None:
        """
        Resets the reward function to its initial state.
        """
        self.cumulated_reward = []
        self.reward_timestep = 0
    
    def set_initial_parameters(
        self,
        obs_indexed: Dict[str, int]
    ) -> None:
        if not self._set_initial_parameters_once:
            self._set_initial_parameters(obs_indexed)
            self._set_initial_parameters_once = True
            logger.info(f"BaseReward: The initial parameters were correctly set.")
        else:
            logger.info(f"BaseReward: The initial parameters were already set.")
    
    
    # ===========================
    # === OVERRIDABLE METHODS ===
    # ===========================
    
    @OverrideToImplementCustomLogic
    def setup(self) -> None:
        """
        This method can be overridden in subclasses to perform setup tasks.
        """
        pass
    
    @OverrideToImplementCustomLogic
    def _set_initial_parameters(
        self,
        obs_indexed: Dict[str, int]
    ) -> None:
        """
        This method can be overridden in subclasses to set initial parameters based on the provided infos.
        
        Example:
            ```
            self.zone_people_occupant_count_index = obs_indexed[get_variable_name(
                self.agent_name, 
                "Zone People Occupant Count", 
                self.reward_fn_config['thermal_zone']
            )]
            ```

        Args:
            infos (Dict[str, Any]): The infos dictionary containing necessary information for initialization.
        """
        pass
    
    @OverrideToImplementCustomLogic
    def get_reward(
        self,
        prev_obs: NDArray[floating[Any]],
        prev_action: Any,
        obs: NDArray[floating[Any]],
        terminated: bool,
        truncated: bool
        ) -> float:
        """
        This method must be implemented in the subclass to calculate the reward.

        Args:
            infos (Dict[str, Any]): The infos dictionary containing the necessary information for calculating the reward.
            terminated (bool): Indicates if the episode has terminated.
            truncated (bool): Indicates if the episode has been truncated.

        Returns:
            float: The calculated reward.
        """
        msg = "BaseReward: This method must be implemented in the subclass."
        logger.error(msg)
        raise NotImplementedError(msg)
