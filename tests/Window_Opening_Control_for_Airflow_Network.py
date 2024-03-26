"""
# Example 3. Hygro-thermal Window Opening Control for Airflow Network

## Problem Statement

A user of EnergyPlus version 3.1 posted the following question on the Yahoo! 
list (circa spring 2009):

    I am currently trying to model a simple ventilation system based on an 
    exhaust fan and outdoor air variable aperture paths that open according 
    to the indoor relative humidity.

    As I didn’t find any object to directly do this, I am trying to use an 
    AirflowNetwork: MultiZone: Component: DetailedOpening object and its 
    AirflowNetwork: multizone: Surface object to model the variable aperture. 
    But the Ventilation Control Mode of the surface object can only be done 
    via Temperature or Enthalpy controls (or other not interesting for my 
    purpose), and not via humidity.

    So my questions are:

    1. is it possible to make the surface object variable according to the 
    relative humidity? (maybe adapting the program?)
    2. or is there an other way to make it?

Because the traditional EnergyPlus controls for window openings do not support 
humidity-based controls (or did not as of Version 3.1), the correct 
response to Question #1 was “No.” But with the EMS, we can now answer 
Question #2 as “Yes.” How can we take the example file called 
HybridVentilationControl.idf and implement humidity-based control for a 
detailed opening in the airflow network model?

## EMS Design Discussion

The main EMS sensor will be the zone air humidity, so we use an 
EnergyManagementSystem:Sensor object that maps to the output variable 
called System Node Relative Humidity for the zone’s air node. This zone 
has the detailed opening.

The EMS will actuate the opening in an airflow network that is defined by 
the input object AirflowNetwork:MultiZone:Component:DetailedOpening. The 
program will setup the actuator for this internally, but we need to use 
an EnergyManagementSystem:Actuator object to declare that we want to use 
the actuator and provide the variable name we want for the Erl programs.

Because we do not know the exactly what the user had in mind, for this example 
we assume that the desired behavior for the opening area is that the opening 
should vary linearly with room air relative humidity. When the humidity 
increases, we want the opening to be larger. When the humidity decreases, 
we want the opening to be smaller. For relative humidity below 25% we close 
the opening. At 60% or higher relative humidity, the opening should be 
completely open. We formulate a model equation for opening factor as:

    ```
    if RH < 0.25:
        F_open = 0.0
    elif RH > 0.6:
        F_open = 1.0
    else:
        F_open = (RH - 0.25)/(0.6 - 0.25)
    ```
"""