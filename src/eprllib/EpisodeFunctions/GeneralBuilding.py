"""
Generalization Building Model
=============================

The `GeneralBuilding` class is a part of the `EpisodeFunction` module, which is responsible for 
defining the properties and configurations of a building episode. The class has the following main 
functionalities:

1. **Building Dimensions** : The code generates random dimensions for the building, including height, 
width, and length, within a specified range. It also calculates the building area and aspect ratio 
based on these dimensions.
2. **Window Area Ratio** : The code randomly assigns window area ratios for each side of the building 
(north, east, south, and west) within a specified range.
3. **Construction Materials** : The code defines the properties of the construction materials used 
for the building's walls, roof, and internal mass. These properties include thickness, conductivity, 
density, specific heat, thermal absorptance, and solar absorptance. The values for these properties 
are randomly generated within specified ranges.
4. **Window Properties** : The code randomly assigns values for the window's U-factor and solar heat
gain coefficient within specified ranges.
5. **Internal Mass** : The code randomly assigns surface areas for the internal mass components of the 
building.
6. **HVAC Capacity** : The code randomly sets the maximum sensible heating and total cooling capacities 
for the HVAC system within specified ranges.

The code reads an initial EnergyPlus JSON (epJSON) file and modifies its properties based on the randomly 
generated values. This modified epJSON object can then be used to simulate the building's energy performance 
using EnergyPlus.

It's important to note that the code uses random number generation extensively to create variations in the 
building properties for each episode. This approach is likely used for training or testing purposes in the 
context of a reinforcement learning environment.
"""
import os
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from eprllib.EpisodeFunctions.EpisodeFunctions import EpisodeFunction
from eprllib.Tools.Utils import building_dimension, inertial_mass, u_factor, random_weather_config

class GeneralBuilding(EpisodeFunction):
    def __init__(
        self, episode_fn_config:Dict[str,Any]
    ):
        """
        This method initialize the GeneralBuilding class. It reads the initial epJSON file and
        modifies its properties based on the randomly generated values.

        The episode_fn_config dictionary should contain the following keys:
        - epjson_files_folder_path (str): The path to the folder containing the epJSON files.
        - epw_files_folder_path (str): The path to the folder containing the EPW weather files.

        The method calls the parent class's __init__ method and initializes the class attributes
        based on the provided episode_fn_config dictionary.
        
        If you set `EnvConfig.generals(epjson_path='auto')` the default GeneralBuildingModel epJSON
        file is called.

        Example:
            episode_fn_config = {
                'epjson_files_folder_path': 'path/to/epjson/files',
                'epw_files_folder_path': 'path/to/epw/files'
            }
            episode_fn = GeneralBuilding(episode_fn_config)

        Args:
            episode_fn_config (Dict[str,Any]): Dictionary to configurate the episode_fn.
        """
        super().__init__(episode_fn_config)
        self.epjson_files_folder_path: str = episode_fn_config['epjson_files_folder_path']
        self.epw_files_folder_path: str = episode_fn_config['epw_files_folder_path']
    
    def get_episode_config(self, env_config:Dict[str,Any]) -> Dict[str,Any]:
        """
        This method define the properties of the episode. Changing some properties as weather or 
        Run Time Period, and defining others fix properties as volumen or window area relation.
        
        Return:
            dict: The method returns the env_config with modifications.
        """
        
        epjson_path: str = env_config['epjson_path']
        if epjson_path in ["auto", None]:
            epjson_path = "src/eprllib/files/GeneralBuildinModel/model_1.epJSON"
        # Establish the epJSON Object, it will be manipulated to modify the building model.
        with open(epjson_path) as file:
            epJSON_object: dict = json.load(file)
        
        # == BUILDING ==
        # The building volume is V=h(high)*w(weiht)*l(large) m3
        h = 3
        w = (10-3) * np.random.random_sample() + 3
        l = (10-3) * np.random.random_sample() + 3
        
        # Calculate the aspect relation
        env_config['building_properties']['building_area'] = w*l
        env_config['building_properties']['aspect_ratio'] = w/l
        
        # Change the dimension of the building and windows
        window_area_relation = []
        env_config['building_properties']['window_area_relation_north'] = (0.9-0.05) * np.random.random_sample() + 0.05
        env_config['building_properties']['window_area_relation_east'] = (0.9-0.05) * np.random.random_sample() + 0.05
        env_config['building_properties']['window_area_relation_south'] = (0.9-0.05) * np.random.random_sample() + 0.05
        env_config['building_properties']['window_area_relation_west'] = (0.9-0.05) * np.random.random_sample() + 0.05
        window_area_relation.append(env_config['building_properties']['window_area_relation_north'])
        window_area_relation.append(env_config['building_properties']['window_area_relation_east'])
        window_area_relation.append(env_config['building_properties']['window_area_relation_south'])
        window_area_relation.append(env_config['building_properties']['window_area_relation_west'])
        window_area_relation = np.array(window_area_relation)
        building_dimension(epJSON_object, h, w, l,window_area_relation)
        
        # Define the type of construction (construction properties for each three layers)
        # Walls
        # Exterior Finish
        # Se toman las propiedades en el rango de valores utilizados en BOpt
        epJSON_object["Material"]["wall_exterior"]["thickness"] = (0.1017-0.0095) * np.random.random_sample() + 0.0095
        epJSON_object["Material"]["wall_exterior"]["conductivity"] = (0.7934-0.0894) * np.random.random_sample() + 0.0894
        epJSON_object["Material"]["wall_exterior"]["density"] = (1762.3-177.822) * np.random.random_sample() + 177.822
        epJSON_object["Material"]["wall_exterior"]["specific_heat"] = (1172.37-795.53) * np.random.random_sample() + 795.53
        epJSON_object["Material"]["wall_exterior"]["thermal_absorptance"] = (0.97-0.82) * np.random.random_sample() + 0.82
        epJSON_object["Material"]["wall_exterior"]["solar_absorptance"] = epJSON_object["Material"]["wall_exterior"]["visible_absorptance"] = (0.89-0.3) * np.random.random_sample() + 0.3
        # Inter layers
        # Se realizó un equivalente de los diferentes materiales utilizados en BOpt para resumirlos en una sola capa equivalente.
        epJSON_object["Material"]["wall_inter"]["thickness"] = (0.29-0.1524) * np.random.random_sample() + 0.1524
        epJSON_object["Material"]["wall_inter"]["conductivity"] = (0.8656-0.0474) * np.random.random_sample() + 0.0474
        epJSON_object["Material"]["wall_inter"]["density"] = (1822.35-84.57) * np.random.random_sample() + 84.57
        epJSON_object["Material"]["wall_inter"]["specific_heat"] = (1214.24-852.57) * np.random.random_sample() + 852.57
        epJSON_object["Material"]["wall_inter"]["thermal_absorptance"] = 0.9
        epJSON_object["Material"]["wall_inter"]["solar_absorptance"] = epJSON_object["Material"]["wall_inter"]["visible_absorptance"] = (0.71-0.65) * np.random.random_sample() + 0.65
        
        # Ceiling/Roof (exterior, inter layers)
        # Exterior Finish
        epJSON_object["Material"]["roof_exterior"]["thickness"] = (0.01906-0.0005) * np.random.random_sample() + 0.0005
        epJSON_object["Material"]["roof_exterior"]["conductivity"] = (50.04-0.1627) * np.random.random_sample() + 0.1627
        epJSON_object["Material"]["roof_exterior"]["density"] = (7801.75-1121.4) * np.random.random_sample() + 1121.4
        epJSON_object["Material"]["roof_exterior"]["specific_heat"] = (1465.46-460.57) * np.random.random_sample() + 460.57
        epJSON_object["Material"]["roof_exterior"]["thermal_absorptance"] = (0.95-0.88) * np.random.random_sample() + 0.88
        epJSON_object["Material"]["roof_exterior"]["solar_absorptance"] = epJSON_object["Material"]["roof_exterior"]["visible_absorptance"] = (0.93-0.6) * np.random.random_sample() + 0.6
        # Inter layers
        epJSON_object["Material"]["roof_inner"]["thickness"] = (0.1666-0.0936) * np.random.random_sample() + 0.0936
        epJSON_object["Material"]["roof_inner"]["conductivity"] = 0.029427
        epJSON_object["Material"]["roof_inner"]["density"] = 32.04
        epJSON_object["Material"]["roof_inner"]["specific_heat"] = 1214.23
        epJSON_object["Material"]["roof_inner"]["thermal_absorptance"] = 0.9
        epJSON_object["Material"]["roof_inner"]["solar_absorptance"] = epJSON_object["Material"]["roof_inner"]["visible_absorptance"] = 0.7

        # Internal Mass layer (interior layer)
        epJSON_object["Material"]["wall_inner"]["thickness"] = epJSON_object["Material"]["roof_inner"]["thickness"] = (0.0318-0.0127) * np.random.random_sample() + 0.0127
        epJSON_object["Material"]["wall_inner"]["conductivity"] = epJSON_object["Material"]["roof_inner"]["conductivity"] = 0.1602906
        epJSON_object["Material"]["wall_inner"]["density"] = epJSON_object["Material"]["roof_inner"]["density"] = 801
        epJSON_object["Material"]["wall_inner"]["specific_heat"] = epJSON_object["Material"]["roof_inner"]["specific_heat"] = 837.4
        epJSON_object["Material"]["wall_inner"]["thermal_absorptance"] = epJSON_object["Material"]["roof_inner"]["thermal_absorptance"] = 0.9
        epJSON_object["Material"]["wall_inner"]["solar_absorptance"] = epJSON_object["Material"]["wall_inner"]["visible_absorptance"] = epJSON_object["Material"]["roof_inner"]["solar_absorptance"] = epJSON_object["Material"]["roof_inner"]["visible_absorptance"] = 0.6

        # Windows
        # Change the window thermal properties
        epJSON_object['WindowMaterial:SimpleGlazingSystem']['WindowMaterial']['u_factor'] = (4.32-0.9652) * np.random.random_sample() + 0.9652
        epJSON_object['WindowMaterial:SimpleGlazingSystem']['WindowMaterial']['solar_heat_gain_coefficient'] = (0.67-0.26) * np.random.random_sample() + 0.26
        
        # The internal thermal mass is modified.
        for key in [key for key in epJSON_object["InternalMass"].keys()]:
            epJSON_object["InternalMass"][key]["surface_area"] = np.random.randint(10,40)
        
        # RunPeriod of a random date
        beging_month, beging_day, end_month, end_day = self.run_period(np.random.randint(1, 365-90), 90)
        epJSON_object['RunPeriod']['Run Period 1']['begin_month'] = beging_month
        epJSON_object['RunPeriod']['Run Period 1']['begin_day_of_month'] = beging_day
        epJSON_object['RunPeriod']['Run Period 1']['end_month'] = end_month
        epJSON_object['RunPeriod']['Run Period 1']['end_day_of_month'] = end_day
        
        # The total inertial thermal mass is calculated.
        # env_config['building_properties']['inercial_mass'] = inertial_mass(epJSON_object)
        
        # The global U factor is calculated.
        # env_config['building_properties']['construction_u_factor'] = u_factor(epJSON_object)
        
        # The limit capacity of bouth cooling and heating are changed.
        E_cool_ref = env_config['building_properties']['E_cool_ref'] = (3000 - 100)*np.random.random_sample() + 100
        E_heat_ref = env_config['building_properties']['E_heat_ref'] = (3000 - 100)*np.random.random_sample() + 100
        HVAC_names = [key for key in epJSON_object["ZoneHVAC:IdealLoadsAirSystem"].keys()]
        for hvac in range(len(HVAC_names)):
            epJSON_object["ZoneHVAC:IdealLoadsAirSystem"][HVAC_names[hvac]]["maximum_sensible_heating_capacity"] = E_heat_ref
            epJSON_object["ZoneHVAC:IdealLoadsAirSystem"][HVAC_names[hvac]]["maximum_total_cooling_capacity"] = E_cool_ref
        
        number_of_timesteps_per_hour = epJSON_object['Timestep']['Timestep 1']['number_of_timesteps_per_hour']
        for agent in [agent for agent in env_config['agents_config'].keys()]:
            env_config['reward_fn'].reward_fn_config[agent]['floor_size'] = w*l
            env_config['reward_fn'].reward_fn_config[agent]['cooling_energy_ref'] = E_cool_ref*3600/number_of_timesteps_per_hour
            env_config['reward_fn'].reward_fn_config[agent]['heating_energy_ref'] = E_heat_ref*3600/number_of_timesteps_per_hour
        
        # Implementation of a random number of agent indicator and thermal zone
        agent_indicator_init = np.random.randint(0,50)
        thermal_zone_indicator = np.random.randint(0,50)
        for agent in [agent for agent in env_config['agents_config'].keys()]:
            env_config['agents_config'][agent]['agent_indicator'] = agent_indicator_init
            env_config['agents_config'][agent]['thermal_zone_indicator'] = thermal_zone_indicator
            agent_indicator_init += 1

        # Select the weather site
        env_config = random_weather_config(env_config, self.epw_files_folder_path)
                
        # The new modify epjson file is writed in the results folder created by RLlib
        # If don't exist, reate a folder call 'models' into env_config['episode_config']['epjson_files_folder_path']
        if not os.path.exists(f"{self.epjson_files_folder_path}/models"):
            os.makedirs(f"{self.epjson_files_folder_path}/models")
        
        env_config["epjson_path"] = f"{self.epjson_files_folder_path}/models/model-{env_config['episode']:08}-{os.getpid():05}.epJSON"
        with open(env_config["epjson_path"], 'w') as fp:
            json.dump(epJSON_object, fp, sort_keys=False, indent=4)
            # The new modify epjson file is writed.
        
        return env_config
    
    def run_period(self, julian_day:int, days_period:int=28) -> Tuple[int,int,int,int]:
        """
        This function returns the begin and end date of the 'days_period' of simulation given a 'julian_day' between 1 and 365.
        """
        if julian_day < 1 or julian_day+days_period > 365:
            raise ValueError('Julian day must be between 1 and (365-days_period)')
        if days_period < 1:
            raise ValueError('Days period must be greater than 0')
        
        # Declaration of variables
        beging_month = 1
        beging_day = 0
        check_day = 0
        max_day = self.max_day_in_month(beging_month)
        
        # Initial date
        while True:
            beging_day += 1
            check_day += 1
            if julian_day == check_day:
                break
            if beging_day >= max_day:
                beging_day = 0
                beging_month += 1
                max_day = self.max_day_in_month(beging_month)
        
        # End date
        end_month = beging_month
        end_day = beging_day + days_period
        while True:
            if end_day > max_day:
                end_day -= max_day
                end_month += 1
                max_day = self.max_day_in_month(end_month)
            else:
                break
            
        return beging_month, beging_day, end_month, end_day
    
    
    def max_day_in_month(self, month:int) -> int:
        """
        This function returns the maximum number of days in a given month.
        """
        months_31 = [1,3,5,7,8,10,12]
        months_30 = [4,6,9,11]
        months_28 = [2]
        if month in months_31:
            max_day = 31
        elif month in months_30:
            max_day = 30
        elif month in months_28:
            max_day = 28
        else:
            raise ValueError('Invalid month')
        return max_day
    