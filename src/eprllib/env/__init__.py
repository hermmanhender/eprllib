"""The 'env' module contain the base environments to wrap an EnergyPlus model
into a Markov Desition  Process (MDP) compatible with RLlib.

There is a single agent environment and a multi-agent environment.
Each of them are compose from two parts: the `EnergyPlusRunner` and the `GymEnv`.
The first one is a wrapper for the EnergyPlus simulation, while the second one
is a wrapper for the RLlib interface (based on the OpenAI Gym interface).

The `EnergyPlusRunner` is responsible for setting up the EnergyPlus simulation.
It is a non-blocking process that runs in a separate thread. It is also
responsible for handling the output of the simulation and updating the state
of the simulation. It is also responsible for providing the observations and
actions to the RLlib environment.

The `GymEnv` is responsible for providing the observations and actions to the
RLlib agent. It is also responsible for handling the input from the RLlib agent
and updating the state of the simulation.

The `EnergyPlusRunner` and `GymEnv` are designed to be used in a non-blocking
way to allow the RLlib agent to interact with the simulation in real-time.
This is done by using a separate thread for the `EnergyPlusRunner` and a
separate process for the `GymEnv`. This allows the RLlib agent to interact
with the simulation in real-time while the simulation is running in the
background.
"""
