"""
Random Simple Building Episode
===============================

The ``RandomSimpleBuildingEpisode`` class is a part of the ``EpisodeFunction`` module, which is responsible for 
defining the properties and configurations of a building episode. The class has the following main 
functionalities:

1. **Building Dimensions**: The code generates random dimensions for the building, including height, 
   width, and length, within a specified range. It also calculates the building area and aspect ratio 
   based on these dimensions.
2. **Window Area Ratio**: The code randomly assigns window area ratios for each side of the building 
   (north, east, south, and west) within a specified range.
3. **Construction Materials**: The code defines the properties of the construction materials used 
   for the building's walls, roof, and internal mass. These properties include thickness, conductivity, 
   density, specific heat, thermal absorptance, and solar absorptance. The values for these properties 
   are randomly generated within specified ranges.
4. **Window Properties**: The code randomly assigns values for the window's U-factor and solar heat
   gain coefficient within specified ranges.
5. **Internal Mass**: The code randomly assigns surface areas for the internal mass components of the 
   building.
6. **HVAC Capacity**: The code randomly sets the maximum sensible heating and total cooling capacities 
   for the HVAC system within specified ranges.

The code reads an initial EnergyPlus JSON (epJSON) file and modifies its properties based on the randomly 
generated values. This modified epJSON object can then be used to simulate the building's energy performance 
using EnergyPlus.

It's important to note that the code uses random number generation extensively to create variations in the 
building properties for each episode. This approach is likely used for training or testing purposes in the 
context of a reinforcement learning environment.
"""
import os
import tempfile
import numpy as np
from numpy.typing import NDArray
from numpy import float32
from typing import Dict, Any, List # type: ignore
from eprllib.Episodes.BaseEpisode import BaseEpisode
from eprllib.Utils.episode_fn_utils import (
    load_ep_model,
    save_ep_model,
    get_random_weather,
    run_period,
    building_dimension,
)
from eprllib.Utils.annotations import override
from eprllib import logger

class RandomSimpleBuildingEpisode(BaseEpisode):
    def __init__(
        self, episode_fn_config:Dict[str,Any]
    ):
        """
        Initializes the RandomSimpleBuildingEpisode class. It reads the initial epJSON file and
        modifies its properties based on the randomly generated values.

        The episode_fn_config dictionary should contain the following keys:
        - epjson_files_folder_path (str): The path to the folder containing the epJSON files.
        - epw_files_folder_path (str): The path to the folder containing the EPW weather files.

        Args:
            episode_fn_config (Dict[str, Any]): Configuration dictionary for the episode function.
        """
        super().__init__(episode_fn_config)
        
        # check that 'epjson_files_folder_path', 'epw_files_folder_path' and 'load_profiles_folder_path' exist in the episode_fn_config
        if 'epjson_files_folder_path' not in self.episode_fn_config:
            msg = "The 'epjson_files_folder_path' must be defined in the episode_fn_config."
            logger.error(msg)
            raise ValueError(msg)
        if 'epw_files_folder_path' not in self.episode_fn_config:
            msg = "The 'epw_files_folder_path' must be defined in the episode_fn_config."
            logger.error(msg)
            raise ValueError(msg)
        if 'load_profiles_folder_path' not in self.episode_fn_config:
            msg = "The 'load_profiles_folder_path' must be defined in the episode_fn_config."
            logger.error(msg)
            raise ValueError(msg)
        
        # When modifying the epjson file, a temporary folder is created to save the update models.
        self.folder_for_models = tempfile.gettempdir()
    
    @override(BaseEpisode)
    def get_episode_config(self, env_config:Dict[str,Any]) -> Dict[str,Any]:
        """
        Returns the episode configuration for the EnergyPlus environment.

        Args:
            env_config (Dict[str, Any]): The environment configuration.

        Returns:
            Dict[str, Any]: The episode configuration.
        """
        # select the model:
        model_window_configs = {
          "1": [1,1,1,1], # all windows: [n,e,s,w]
          "2": [1,1,1,0],
          "3": [1,1,0,1],
          "4": [1,0,1,1],
          "5": [0,1,1,1],
          "6": [1,1,0,0],
          "7": [1,0,1,0],
          "8": [0,1,1,0],
          "9": [1,0,0,1],
          "10": [0,1,0,1],
          "11": [0,0,1,1],
          "12": [1,0,0,0],
          "13": [0,1,0,0],
          "14": [0,0,1,0],
          "15": [0,0,0,1],
        }
        model = np.random.randint(1, 16)
        epjson_path: str = env_config['epjson_path']
        if epjson_path in ["auto", None]:
            epjson_path = f"{self.episode_fn_config['epjson_files_folder_path']}/model_{model}.epJSON"
        
        # Establish the epJSON Object, it will be manipulated to modify the building model.
        epJSON_object: Dict[str, Any] = load_ep_model(epjson_path)
        
        # == BUILDING ==
        # The building volume is V=h(high)*w(weiht)*l(large) m3
        h = 3
        w = (10-3) * np.random.random_sample() + 3
        l = (10-3) * np.random.random_sample() + 3
        
        # Calculate the aspect relation
        env_config['building_properties']['building_area'] = w*l
        env_config['building_properties']['aspect_ratio'] = w/l
        
        # Change the dimension of the building and windows
        window_area_relation_list: List[float] = []
        model_window_config = model_window_configs[str(model)]
        for i in range(4):
            if model_window_config[i] == 1:
                window_area_relation_list.append((0.9-0.05) * np.random.random_sample() + 0.05)
            else:
                window_area_relation_list.append(0)
        env_config['building_properties']['window_area_relation_north'] = window_area_relation_list[0]
        env_config['building_properties']['window_area_relation_east'] = window_area_relation_list[1]
        env_config['building_properties']['window_area_relation_south'] = window_area_relation_list[2]
        env_config['building_properties']['window_area_relation_west'] = window_area_relation_list[3]
        
        window_area_relation: NDArray[float32] = np.array(window_area_relation_list)
        epJSON_object = building_dimension(epJSON_object, h, w, l,window_area_relation)
        
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
        beging_month, beging_day, end_month, end_day = run_period(np.random.randint(1, 365-14), 14)
        epJSON_object['RunPeriod']['Run Period 1']['begin_month'] = beging_month
        epJSON_object['RunPeriod']['Run Period 1']['begin_day_of_month'] = beging_day
        epJSON_object['RunPeriod']['Run Period 1']['end_month'] = end_month
        epJSON_object['RunPeriod']['Run Period 1']['end_day_of_month'] = end_day
        
        # The total inertial thermal mass is calculated.
        # for agent in [agent for agent in env_config['agents_config'].keys()]:
        #     env_config['reward_fn'].reward_fn_config[agent]['Cdyn'] = effective_thermal_capacity(epJSON_object)
        
        # The global U factor is calculated.
        # env_config['building_properties']['construction_u_factor'] = u_factor(epJSON_object)
        
        # The limit capacity of bouth cooling and heating are changed.
        E_cool_ref = env_config['building_properties']['E_cool_ref'] = (3000 - 100)*np.random.random_sample() + 100
        E_heat_ref = env_config['building_properties']['E_heat_ref'] = (3000 - 100)*np.random.random_sample() + 100
        HVAC_names = [key for key in epJSON_object["ZoneHVAC:IdealLoadsAirSystem"].keys()]
        for hvac in range(len(HVAC_names)):
            epJSON_object["ZoneHVAC:IdealLoadsAirSystem"][HVAC_names[hvac]]["maximum_sensible_heating_capacity"] = E_heat_ref
            epJSON_object["ZoneHVAC:IdealLoadsAirSystem"][HVAC_names[hvac]]["maximum_total_cooling_capacity"] = E_cool_ref
        
        # number_of_timesteps_per_hour = epJSON_object['Timestep']['Timestep 1']['number_of_timesteps_per_hour']
        # for agent in [agent for agent in env_config['agents_config'].keys()]:
        #     env_config['reward_fn'].reward_fn_config[agent]['floor_size'] = w*l
        #     env_config['reward_fn'].reward_fn_config[agent]['cooling_energy_ref'] = E_cool_ref*3600/number_of_timesteps_per_hour
        #     env_config['reward_fn'].reward_fn_config[agent]['heating_energy_ref'] = E_heat_ref*3600/number_of_timesteps_per_hour
        
        # Change the load file profiles
        schedule_file_keys = [key for key in epJSON_object["Schedule:File"].keys()]
        for key in schedule_file_keys:
            assert isinstance(self.episode_fn_config['load_profiles_folder_path'], str), "The 'load_profiles_folder_path' must be a string."
            epJSON_object["Schedule:File"][key]["file_name"] = self.episode_fn_config['load_profiles_folder_path'] + "/" + np.random.choice(os.listdir(self.episode_fn_config['load_profiles_folder_path']))

        # Select the weather site
        env_config["epw_path"] = get_random_weather(self.episode_fn_config['epw_files_folder_path'])
                
        # Save the model and update the path in the env_config.
        env_config["epjson_path"] = save_ep_model(epJSON_object, self.folder_for_models)
        
        return env_config
        
    @override(BaseEpisode)
    def get_episode_agents(self, env_config: Dict[str, Any], possible_agents: List[str]) -> List[str]:
        """
        Returns the agents for the episode configuration in the EnergyPlus environment.

        Args:
            env_config (Dict[str, Any]): The environment configuration.
            possible_agents (List[str]): List of possible agents.

        Returns:
            Dict[str, Any]: The agents that are acting for the episode configuration. Default: possible_agents list.
        """
        # Implement the logic to select the agents for the episode
        return super().get_episode_agents(env_config, possible_agents)
    
    @override(BaseEpisode)
    def get_timestep_agents(self, env_config: Dict[str, Any], possible_agents: List[str]) -> List[str]:
        """
        Returns the agents for the timestep configuration in the EnergyPlus environment.

        Args:
            env_config (Dict[str, Any]): The environment configuration.
            possible_agents (List[str]): List of possible agents.

        Returns:
            Dict[str, Any]: The agents that are acting for the timestep configuration. Default: possible_agents list.
        """
        # Implement the logic to select the agents for each timestep
        return super().get_timestep_agents(env_config, possible_agents)
    