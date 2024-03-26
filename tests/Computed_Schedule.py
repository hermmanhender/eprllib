"""
# Example: Computed Schedule

## Problem Statement

Many models have schedule inputs that could be used to control the object, 
but creating the schedules is often cumbersome and may not always result in 
optimal behavior due to dynamic environmental factors. Leveraging schedules 
as inputs for Deep Reinforcement Learning (DRL) offers a promising approach 
to learning optimal control policies. In this example, we demonstrate how 
DRL can be applied to learn the optimal policy for setting heating and 
cooling zone temperature schedules.

## DRL Design Discussion

As an example, we will utilize the model of the Small Office Reference 
Building (RefBldgSmallOfficeNew2004_Chicago.idf) and utilize the 
EnergyPlus API Python to optimize heating and cooling zone temperature 
setpoint schedules. The input object `Schedule:Constant` has been configured 
to serve as an actuator (or agent within the scope of eprllib).

To facilitate the DRL process, we must explicitly define the action space, 
detailing the possible actions the agent can take in adjusting the temperature 
setpoints. Additionally, we need to specify the variables comprising the 
observation space, providing clarity on the information available to the 
agent at each time step. For the reward function, we define a range of 
temperatures within which we aim to maintain the environment to ensure 
comfort for the building occupants.

Furthermore, it is essential to elucidate how the DRL algorithm interacts 
with the EnergyPlus simulation. By detailing the learning process over 
time, readers gain a deeper understanding of the practical implementation 
and the iterative nature of DRL in optimizing building energy management.


built-in variables:
    Hour
    DayOfWeek
"""

# import the necessary libraries

# define the eprllib configuration

# inicialize ray server and after that register the environment

# configurate the algorithm

# init the training loop

# close the ray server