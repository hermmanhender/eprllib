

def reward_function(occupancy:float, T_zone:float, T_confort:float=23.5) -> float:
    """This function returns the reward calcualted as the absolute value of the cube in the 
    difference between set point temperatur for comfort and the temperature measured in the 
    thermal zone when there are people in the zone but zero when is not.

    Args:
        occupancy (float): Zone People Occupant Count for the Thermal Zone.
        T_zone (float): Zone Mean Air Temperature for the Thermal Zone in °C.
        T_confort (float, optional): Setpoint of Zone Mean Air Temperature for Comfort in °C. Defaults to 23.5.

    Returns:
        float: reward value
    """
    if occupancy > 0: # When there are people in the thermal zone, a reward is calculated.
        reward = -(abs((T_confort - T_zone)**3))
    else:
        # If there are not people, only the reward is calculated when the environment is far away
        # from the comfort temperature ranges. This limits are recommended in EnergyPlus documentation:
        # InputOutput Reference p.522
        if T_zone > 29.4:
            reward = -(abs((T_confort - T_zone)**3))
        elif T_zone < 16.7:
            reward = -(abs((T_confort - T_zone)**3))
        else:
            reward = 0.
    return reward