"""
# Example 2.2. Traditional Setpoint and Availability Managers

## Problem Statement

The traditional way of modeling supervisory control of HVAC systems in 
EnergyPlus is to use SetpointManagers and AvailabilityManagers. To gain 
experience with eprllib we model a Large Oﬀice Reference Building 
(RefBldgLargeOﬀiceNew2004_Chicago.idf) with a SetpointManager:Mixed actuator.

## EMS Design Discussion

A review of the example file shows that three types of traditional HVAC 
managers are being used: scheduled setpoints, mixed air setpoints, and 
night cycle availability. We will discuss these separately. In this example, 
we will use the SetpointManager:Mixed actuator.

The input object SetpointManager:Mixed air functions by placing a setpoint 
value on a specified node based on the value of the setpoint at another node 
and the temperature rise across the fan. The temperature rise is found by 
taking the temperature at the fan outlet node and subtracting the temperature 
at the fan inlet node. The EMS needs two additional sensors to obtain these 
temperatures, which are set up by using a pair EnergyManagementSystem:Sensor 
objects. The example file has three mixed air setpoint managers that place 
setpoints on the outlet of the outdoor air mixer, the outlet of the cooling 
coil, and the outlet of the heating coil. Therefore, we need three actuators 
to place setpoints at these three nodes, which are set up using three 
EnergyManagementSystem:Actuator objects. Each mixed air setpoint calculation 
is a simple single-line of program code such as 
“SET VAV_1_CoolC_Setpoint = Seasonal_Reset_SAT_Sched - 
(T_VAV1FanOut - T_VAV1FanIn).”
"""