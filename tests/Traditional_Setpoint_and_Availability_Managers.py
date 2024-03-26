"""
# Example 2.1. Traditional Setpoint and Availability Managers

## Problem Statement

The traditional way of modeling supervisory control of HVAC systems in 
EnergyPlus is to use SetpointManagers and AvailabilityManagers. To gain 
experience with eprllib we model a Large Oﬀice Reference Building 
(RefBldgLargeOﬀiceNew2004_Chicago.idf) with a SetpointManager:Scheduled actuator.

## EMS Design Discussion

A review of the example file shows that three types of traditional HVAC 
managers are being used: scheduled setpoints, mixed air setpoints, and 
night cycle availability. We will discuss these separately. In this example, 
we will use the SetpointManager:Scheduled actuator.

The input object SetpointManager:Scheduled functions by placing a setpoint 
value on a specified node based on the value in a schedule. Therefore, our 
EMS program will do the same. First we will need to access the schedule. 
In this example, a schedule called Seasonal-Reset-Supply-AirTemp-Sch contains 
the temperature values desired for the air system’s supply deck. We use an 
EnergyManagementSystem:Sensor object based on the output variable called 
“Schedule Value” to fill schedule values into an Erl variable called 
Seasonal_Reset_SAT_Sched. Once we have the sensor and actuator setup, 
putting the setpoint on the node involves a single line of Erl code, 
“SET VAV_1_SAT_setpoint = Seasonal_Reset_SAT_Sched.”
"""