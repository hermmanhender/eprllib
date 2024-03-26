"""
# Example 7. Constant Volume Purchased Air System

## Problem Statement

The simplest way to add HVAC control to an EnergyPlus thermal zone is to 
use the ZoneHVAC:IdealLoadsAirSystem. This was called purchased air in 
older versions. The ideal loads air system is intended for load calculations. 
You provide input for the supply air conditions of drybulb and humidity 
ratio, but the flow rate cannot be controlled. The model operates by 
varying the flow rate to exactly meet the desired setpoints. However, you 
may want to experiment with various designs in a slightly different way in 
which, given a prescribed supply air situation, then adjust the design to 
maximize the thermal comfort. It would be interesting to use the 
simple-toinput purchased air model to examine how a zone responds to a 
system, rather than how the system responds to a zone. We should ask, 
Can we use the EMS to prescribe the supply air flow rates for a purchased 
air model?

## EMS Design Discussion

For this example we begin with the input file from Example 6 (primarily 
because it already has purchased air). We examine the typical mass flow 
rates the air system provides to have some data to judge what an appropriate 
constant flow rate might be. A cursory review of the data indicates that 
cooling flow rates of 0.3 kg/s are chosen for two zones and 0.4 kg/s is 
chosen for the third. Heating flow rates of 0.1 and 0.15 kg/s are also chosen.

We want the model to respond differently for heating and cooling. We define 
two operating states and create global variables to hold that state for 
each zone. The first state is when the zone calls for heating; we will 
assign a value of 1.0. The second is when the zone calls for cooling; we 
assign 2.0.

To sense the state we will use EMS sensors associated with the output 
variable called “Zone/Sys Sensible Load Predicted.” We will set up one of 
these for each zone and use it as input data. If this value is less than 
zero, the zone is in the cooling state. If it is greater than zero, the zone 
is in the heating state. This predicted load is calculated during the 
predictor part of the model, so we choose the EMS calling point called 
“AfterPredictorAfterHVACManagers.”

An EMS actuator is available for the ideal loads air system that overrides 
the air mass flow rate (kg/s) delivered by the system when it is on. The 
override is not absolute in that the model will still apply the limits 
defined in the input object and overrides only if the system is “on.” 
The internal logic will turn off the air system if the zone is in the 
thermostat dead band or scheduled “off” by availability managers. This “off” 
state is modeled inside the ideal loads air system so it does not need to 
be calculated in Erl. This control leads to a constant volume system that 
cycles in an attempt to control the zone conditions. In practice, it can 
achieve relatively good control when loads do not exceed the available capacity.
"""