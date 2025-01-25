Actions
========

Actions are taken by the agents in every timestep. The final action is implemented in 
an EnergyPlus actuator that must be defined inside the agent configuration. An agent
can control more than one actuator.

action argument in agent definitions
-------------------------------------
An agent is defined as:

.. code-block:: python
    from eprllib.Agent.AgentSpec import AgentSpec

    agent = AgentSpec( # define the action specification here.
        observation = ...,
        action = ...,
        reward = ...
    )

The action argument inside the AgentSpec is defined with help of the ActionSpec class:

.. code-block:: python
    from eprllib.Agent.AgentSpec import ActionSpec

    action = ActionSpec(
        actuators: List[Tuple[str,str,str]] = ...,
        action_fn: ActionFunction = ...,
        action_fn_config: Dict[str, Any] = ...
    )

Action functions has a main rolle in eprllib. They give flexibility to the environment configuration 
throug three methods called in the simulation process:

.. code-block:: python
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