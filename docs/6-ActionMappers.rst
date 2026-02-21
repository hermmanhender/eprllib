ActionMapper API
================

Introduction
------------

In ``eprllib``, ``ActionMappers`` determine how the policy actions are executed into EnergyPlus actuators. They 
provide a mechanism to control the timing of actions, allowing for more complex and nuanced agent behavior. This 
document provides a detailed explanation of the ``ActionMapper`` API in ``eprllib``.

.. image:: Images/triggers.png
    :width: 600
    :alt: Triggers diagram
    :align: center


Creating custom ``ActionMapper`` functions
------------------------------------------

``ActionMapper`` functions are responsible for determining how the policy actions are executed into EnergyPlus actuators.
To define a custom ``ActionMapper`` function, you need to follow these steps:

1. Override the ``setup(self)`` method.
2. Override the ``get_action_space_dim(self)`` method.
3. Override the ``actuator_names(self, actuators_config: Dict[str, Tuple[str,str,str]])`` method.
4. Override the ``_agent_to_actuator_action(self, action: Any, actuators: List[str])`` method.
5. Optionally, you can override the methods ``get_actuator_action(self, action: float | int, actuator: str)`` and ``action_to_goal(self, action: int | float)``.


    NOTE: It is recommended use the decorator ``override`` in each method. See: ``eprllib.Utils.annotations.override``

To see a practical example, we will configurate an ``ActionMapper`` function that convert Discrete actions to opening windows signal actuator actions.
Here is the full example:

.. code-block:: python

    import gymnasium as gym
    from typing import Any, Dict, List, Tuple, Optional
    from eprllib.Agents.ActionMappers.BaseActionMapper import BaseActionMapper
    from eprllib.Utils.observation_utils import get_actuator_name
    from eprllib.Utils.annotations import override

    class WindowsOpeningDiscreteActionMapper(BaseActionMapper):
    
        @override(BaseActionMapper)
        def setup(self):
            # Here you have access to the self.action_mapper_config and self.agent_name
            self.window_actuator: Optional[str] = None
            self.action_space_dim: int = self.action_mapper_config.get("action_space_dim", 11)
        
        @override(BaseActionMapper)    
        def get_action_space_dim(self) -> gym.Space[Any]:
            """
            Get the action space of the environment.

            Returns:
                gym.Space: Action space of the environment.
            """
            return gym.spaces.Discrete(self.action_space_dim)
        
        @override(BaseActionMapper)
        def actuator_names(self, actuators_config: Dict[str, Tuple[str,str,str]]) -> None:
            self.window_actuator = get_actuator_name(
                self.agent_name,
                actuators_config["window_actuator"][0],
                actuators_config["window_actuator"][1],
                actuators_config["window_actuator"][2]
            )
            
        @override(BaseActionMapper)
        def _agent_to_actuator_action(self, action: Any, actuators: List[str]) -> Dict[str, Any]:
            """
            Transform the agent action to actuator action.

            Args:
                action (Any): The action to be transformed.
                actuators (List[str]): List of actuators controlled by the agent.

            Returns:
                Dict[str, Any]: Transformed actions for the actuators.
            """
            actuator_dict_actions = {actuator: None for actuator in actuators}
            
            if self.window_actuator in actuator_dict_actions.keys() and self.window_actuator is not None:
                actuator_dict_actions.update({self.window_actuator: action / 10})
                
            return actuator_dict_actions
            