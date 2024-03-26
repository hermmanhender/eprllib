"""
# Example 6. Window Shade Control

## Problem Statement

EnergyPlus offers a wide range of control options in the WindowShadingControl 
object, but it is easy to imagine custom schemes for controlling shades or 
blinds that are not available. We need to ask, Can we use the EMS to override 
the shading controls?

## EMS Design Discussion

We will take the example file PurchAirWindowBlind.idf and use EMS to add a 
new control scheme. This file has an interior blind that can be either “on” 
or “off.” The control scheme has three parts:

* Deploy the blind whenever too much direct sun would enter the zone and cause 
discomfort for the occupants.
* Deploy the blind whenever there is a significant cooling load.
* Leave the blind open whenever the first two constraints have not triggered.

We assume that a model for the direct sun control is based on incidence angle, 
where the angle is defined as zero for normal incidence relative to the 
plane of the window. When the direct solar incidence angle is less than 45 
degrees, we want to draw the blind. EnergyPlus has a report variable called 
“Surface Ext Solar Beam Cosine Of Incidence Angle,” for which we will use a 
sensor in our EnergyManagementSystem:Sensor input object. This sensor is a 
cosine value that we turn into an angle value with the built-in function 
@ArcCos. Then we will use the built-in function @RadToDeg to convert from 
radians to degrees. This new window/solar incidence angle in degree may be 
an interesting report variable, so we use an 
EnergyManagementSystem:OutputVariable input object to create custom output.

Because the transmitted solar is a problem only when there is a cooling load, 
we also trigger the blind based on the current data for cooling. The report 
variable called “Zone/Sys Sensible Cooling Rate” is used in an EMS sensor 
to obtain an Erl variable with the most recent data about zone cooling load 
required to meet setpoint. When this value is positive, we know the zone 
cannot make good use of passive solar heating, so we close the blind.

The EMS actuator will override the operation of a WindowShadingControl input 
object. Related to this, the EDD file shows

! <EnergyManagementSystem:Actuator Available>, Component Unique Name, 
Component Type, Control Type

EnergyManagementSystem:Actuator Available,ZN001:WALL001:WIN001,Window 
Shading Control,Control Status

Although the user-defined name for the WindowShadingControl is “INCIDENT 
SOLAR ON BLIND,” the component unique name of the actuator that is available 
is called “ZN001:WALL001:WIN001.” There could be multiple windows, all with 
shades, and each is governed by a single WindowShadingControl input object. 
The EMS actuator could override each window separately. The Control Type is 
called “Control Status,” and requires you to set the status to one of a set 
of possible control flags. For this case, with only an interior shade, there 
are two states for the actuator to take. The first shows the shade is “off,” 
and corresponds to a value of 0.0. The second shows the interior shade is 
“on,” and corresponds to a value of 6.0.
"""