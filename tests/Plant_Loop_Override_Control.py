"""
# Example 10. Plant Loop Override Control

## Problem Statement

A common occurrence in EnergyPlus central plant simulations is for a component 
to be designed well, but during the course of an annual simulation, it is 
operated outside of its allowable region. This is due to the governing 
control strategies (operating schemes). These operation schemes may not have 
the intelligence to say, turn off a cooling tower when the outdoor 
temperature is too low.

Within the EnergyPlus example files, the cooling tower offers warnings stating 
that the tower temperature is going below a low temperature limit. We should 
ask, can we use a simple EMS addition to an input file to override the loop 
and turn off the cooling tower to avoid these situations and therefore the 
warnings?

## EMS Design Discussion

For this example, we will start with the example file that is packaged with 
EnergyPlus called EcoRoofOrlando.idf. This is one example of an input file 
where a cooling tower throws warnings due to operating at too low of a 
temperature. Although the input file has many aspects related to zone and 
HVAC, we will only be interested in the loop containing the tower, which is a 
CondenserLoop named Chiller Plant Condenser Loop. The loop has a minimum loop 
temperature of 5 degrees Celsius, as specified by the CondenserLoop object.

In order to avoid these warnings and properly shut off the tower, EMS will be 
used to check the outdoor temperature and shut down the whole loop. Special 
care must be taken when manipulating plant and condenser loops with EMS. The 
most robust way found is to both disable the loop equipment and also override 
(turn off) the loop. Skipping either of these can cause mismatches where 
either components are still expecting flow but the pump is not running, or the 
pump is trying to force flow through components which are disabled. Either 
of these cases can cause unstable conditions and possibly fatal flow errors.

The outdoor air temperature must be checked in order to determine what the 
EMS needs to do at a particular point in the simulation. This is handled by 
use of an EMS sensor monitoring the Outdoor Dry Bulb standard E+ output 
variable.

To manage the loop and pump, actuators are employed on both. The pump 
actuator is a mass flow rate override. This can be used to set the flow to 
zero, effectively shutting off the pump. The loop actuator is an on/off 
supervisory control, which allows you to “shut the entire loop down.” This 
actuator will not actually shut the loop down, it effectively empties the 
equipment availability list, so that there is no equipment available to 
reject/absorb load on the supply side. If you use this actuator alone 
to “shut down the loop,” you may find that the pump is still flowing fluid 
around the loop, but the equipment will not be running.

The EMS calling point chosen is “InsideHVACSystemIterationLoop,” so that the 
operation will be updated every time the plant loops are simulated.

The Erl program is quite simple for this case. If the outdoor dry bulb 
temperature goes below a certain value, the loop and pump actuators are 
set to zero. If the outdoor temperature is equal to or above this value, the 
actuators are set to Null, relinquishing control back to the regular 
operation schemes. In modifying this specific input file it was found that 
the outdoor dry bulb temperature which removed these warnings was six 
degrees Celsius. We also create a custom output variable called “EMS 
Condenser Flow Override On” to easily record when the overrides have occurred.
"""