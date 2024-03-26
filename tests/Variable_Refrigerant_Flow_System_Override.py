"""
# Example 12. Variable Refrigerant Flow System Override

## Problem Statement

The variable refrigerant flow heat pump air conditioner has several available 
thermostat control options. These operation control schemes may not provide 
the type of control desired. How can we use a simple EMS addition to an input 
file that can override the specified thermostat control logic and set an 
alternate mode of operation?

## EMS Design Discussion

Depending on the type of thermostat control logic, the EnergyPlus program 
will review the loads in each zone, the number of zones in cooling or 
heating, the deviation from set point temperature, etc. to determine the 
mode of operation for the heat pump condenser. Alternate control logic may 
be developed to more accurately reflect the operation of a specific 
manufacturers product, or investigate other control techniques. This control 
logic may be added to an input file and used as the operating control 
logic of the heat pump.

This simple example shows how to use EMS actuators to SET the operating 
mode and cause a specific terminal unit to operate at a specified part-load 
ratio (PLR). When setting the terminal unit PLR, the terminal unit will 
turn on only if the condenser is allowed to operate according to the minimum 
and maximum outdoor temperature limits.
"""