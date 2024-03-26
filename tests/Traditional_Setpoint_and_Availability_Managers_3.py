"""
# Example 2.2. Traditional Setpoint and Availability Managers

## Problem Statement

The traditional way of modeling supervisory control of HVAC systems in 
EnergyPlus is to use SetpointManagers and AvailabilityManagers. To gain 
experience with eprllib we model a Large Oﬀice Reference Building 
(RefBldgLargeOﬀiceNew2004_Chicago.idf) with a SetpointManager:NightCycle 
actuator.

## EMS Design Discussion

A review of the example file shows that three types of traditional HVAC 
managers are being used: scheduled setpoints, mixed air setpoints, and 
night cycle availability. We will discuss these separately. In this example, 
we will use the SetpointManager:NightCycle actuator.

The input object AvailabilityManager:NightCycle functions by monitoring 
zone temperature and starting up the air system (if needed) to keep the 
building within the thermostat range. The sensors here are the zone air 
temperatures, which are set up by using EnergyManagementSystem:Sensor 
objects in the same way as for Example 1. We will need one zone temperature 
sensor for each zone that is served by the air system so we can emulate the 
“CycleOnAny” model being used. The other sensors we need are the desired 
zone temperatures used by the thermostat. We access these temperatures 
directly from the schedules (HTGSETP_SCH and CLGSETP_SCH in the example) 
by using EnergyManagementSystem:Sensor objects. To control the air system’s 
operation status, we use an EnergyManagementSystem:Actuator object that is 
assigned to an “AirLoopHVAC” component type using the control variable 
called “Availability Status.” EnergyPlus recognizes four availability 
states that control the behavior of the air system. Inside EnergyPlus 
these are integers, but EMS has only real-valued variables, so we will use 
the following whole numbers:

* NoAction = 0.0
* ForceOff = 1.0
* CycleOn = 2.0
* CycleOnZoneFansOnly = 3.0.

The traditional AvailabilityManager:NightCycle object operates by turning 
on the system for a prescribed amount of time (1800 seconds in the example 
file), and then turning it off for the same amount of time. You should be 
able to model this starting and stopping in EMS by using Trend variables to 
record the history of the actions. However, this cycling is not necessarily 
how real buildings are operated, and for this example we do not try to 
precisely emulate the traditional EnergyPlus night cycle manager. Rather, we 
use a simpler temperature-based control to start and stop the air system for 
the night cycle. The algorithm first assumes an offset tolerance of 0.83°C 
and calculates limits for when heating should turn on and off and when 
cooling should turn on and off. It then finds the maximum and minimum zone 
temperatures for all the zones attached to the air system. These use the 
@Max and @Min built-in functions, which take on two operators at a time. 
Then a series of logic statements is used to compare temperatures and decide 
what the availability status of the air system should be.
"""