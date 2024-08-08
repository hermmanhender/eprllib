"""This module contains the InternalGains generator file for EnergyPlus. It considers the
People, Light, Electricity or plugin Loads, and GasEquipment for Cooking objects in the model.
"""
import pandas as pd

# Define a class for the type of user living in the building
# The types of users are: DayAwayEveningHome, MostlyHomeEarlyReiser, DayAwayEveningAway, and MostlyHome
class UserType:
    def __init__(self, name, type):
        self.name = name
        self.type = type
        # The types of users are: DayAwayEveningHome, MostlyHomeEarlyReiser, DayAwayEveningAway, and MostlyHome
        
    # The following funtions define the occupancy profile, the activity level, the light use, the electric plugin use, and the gas equipment use.
    def occupancy_profile(self):
        """This function defines the occupancy profile of the user. For that, it read the 
        occupancy profile from the file allocated on eprllib.ExampleFiles.users and apply then
        a random modification based on stadistics.
        """
        # Read the example file with pandas using the self.type variable as a reference for the file name.
        occupancy_profile = pd.read_csv(f'../ExampleFiles/users/{self.type}.csv')
        # Apply a random modification based on stadistics.
        occupancy_profile = occupancy_profile.sample(frac=1).reset_index(drop=True)
        
    def activity_level(self):
        pass
    def light_use(self):
        pass
    def electric_plugin_use(self):
        pass
    def gas_equipment_use(self):
        pass
    def cooking_use(self):
        pass
            

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name