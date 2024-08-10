"""
Generalization Building Model
=============================

This method is to create random simlpe buildings to train a policy that generalizate
the control of different devices in a building.
"""
import os
import json
import numpy as np
from typing import Dict, Any
from eprllib.EpisodeFunctions.EpisodeFunctions import EpisodeFunction
from eprllib.Tools.Utils import building_dimension, inertial_mass, u_factor, random_weather_config

class GeneralBuilding(EpisodeFunction):
    def __init__(
        self, episode_fn_config:Dict[str,Any]
    ):
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
        
        # The total inertial thermal mass is calculated.
        # env_config['building_properties']['inercial_mass'] = inertial_mass(epJSON_object)
        
        # The global U factor is calculated.
        # env_config['building_properties']['construction_u_factor'] = u_factor(epJSON_object)
        
        # The limit capacity of bouth cooling and heating are changed.
        E_cool_ref = env_config['building_properties']['E_cool_ref'] = (3000 - 100)*np.random.random_sample() + 100
        E_heat_ref = env_config['building_properties']['E_heat_ref'] = (3000 - 100)*np.random.random_sample() + 100
        HVAC_names = [key for key in epJSON_object["ZoneHVAC:IdealLoadsAirSystem"].keys()]
        number_of_timesteps_per_hour = epJSON_object['Timestep']['Timestep 1']['number_of_timesteps_per_hour']
        for hvac in range(len(HVAC_names)):
            epJSON_object["ZoneHVAC:IdealLoadsAirSystem"][HVAC_names[hvac]]["maximum_sensible_heating_capacity"] = E_heat_ref
            epJSON_object["ZoneHVAC:IdealLoadsAirSystem"][HVAC_names[hvac]]["maximum_total_cooling_capacity"] = E_cool_ref
        if env_config.get('reward_fn_config', False):
            if env_config['reward_fn_config'].get('cooling_energy_ref', False):
                env_config['reward_fn_config']['cooling_energy_ref'] = E_cool_ref*number_of_timesteps_per_hour*3600
            if env_config['reward_fn_config'].get('heating_energy_ref', False):
                env_config['reward_fn_config']['heating_energy_ref'] = E_heat_ref*number_of_timesteps_per_hour*3600
        
                
        # Select the schedule file for loads
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
    