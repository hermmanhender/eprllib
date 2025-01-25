Actions
========

Actions are taken by the agents in every timestep. The final action is implemented in 
an EnergyPlus actuator that must be defined inside the agent configuration. An agent
can control more than one actuator.

action argument in agent definitions
-------------------------------------

The action argument inside the AgentSpec is defined with help of the ActionSpec class:

.. code-block:: python
    from eprllib.Agent.AgentSpec import ActionSpec
    from eprllib.ActionFuntions.ActionFuntions import ActionFunction
    from eprllib.ActionFuntions.setpoin_control import discrete_dual_setpoint_and_availability

    action = ActionSpec(
        actuators: List[Tuple[str,str,str]] = [ # (component_type, control_type, actuator_key)
            ("Schedule:Compact", "Schedule Value", "heating_setpoint"),
            ("Schedule:Compact", "Schedule Value", "cooling_setpoint"),
            ("Schedule:Constant", "Schedule Value", "HVAC_OnOff"),
        ],
        action_fn: ActionFunction = discrete_dual_setpoint_and_availability,
        action_fn_config: Dict[str, Any] = {
            'agent_name': "Setpoint_agent",
            'temperature_range': (18, 28),
            'actuator_for_cooling': ("Schedule:Compact", "Schedule Value", "cooling_setpoint"),
            'actuator_for_heating': ("Schedule:Compact", "Schedule Value", "heating_setpoint"),
            'availability_actuator': ("Schedule:Constant", "Schedule Value", "HVAC_OnOff"),
    )

Action functions has a main rolle in eprllib. They give flexibility to the environment configuration 
throug three methods called in the simulation process:

.. code-block:: python
    from typing import Dict, Any, List
    import gymnasium as gym

    class ActionFunction:
        def __init__(self, action_fn_config: Dict[str,Any] = {}):
            self.action_fn_config = action_fn_config
        
        def get_action_space_dim(self) -> gym.Space:
            return NotImplementedError("This method should be implemented in the child class.")
        
        def agent_to_actuator_action(self, action: Any, actuators: List[str]) -> Dict[str,Any]:
            raise NotImplementedError("This method should be implemented in the child class.")

        def get_actuator_action(self, action:float|int, actuator: str) -> Any:            
            return action

Work in progres...