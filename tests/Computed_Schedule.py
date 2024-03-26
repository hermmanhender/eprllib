"""
# Example 5. Computed Schedule

## Problem Statement

Many models have schedule inputs that could be used to control the object, 
but creating the schedules it is too cumbersome. We need to ask, Can we use 
the EMS to dynamically calculate a schedule?

## EMS Design Discussion

As an example, we will take the model from example 1 and use the EMS to 
replicate the heating and cooling zone temperature setpoint schedules. 
The input object Schedule:Constant has been set up to be available as an 
actuator. We then add EnergyManagementSystem:Actuator objects that set these 
actuators up as Erl variables.

To devise an Erl program to compute the schedule, we need to use the 
built-in variables that describe time and day. The built-in variable Hour 
will provide information about the time of day; DayOfWeek will provide 
information about whether it is the weekend or a weekday.
"""