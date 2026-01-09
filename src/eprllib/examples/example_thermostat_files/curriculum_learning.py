import os
import json
import numpy as np
import tempfile
from typing import Any, Dict, List, Tuple
from eprllib.Episodes.BaseEpisode import BaseEpisode
from eprllib.Utils.episode_fn_utils import building_dimension, run_period, extract_epw_location_data, select_epjson_model, get_random_parameter
from eprllib import logger


class episode_fn(BaseEpisode):
    def __init__(self, episode_fn_config:Dict[str,Any]):
        
        # Initialize the BaseEpisode
        super().__init__(episode_fn_config)
        
        # Generate a temporary directory to save the episode files, like EP models configurations.
        self.temp_dir = tempfile.gettempdir()
        #  = os.path.join(sys_temp, 'eprllib-epjson-models')
        logger.info(f"Temporary directory for EP models: {self.temp_dir}")
        
        # General variables.
        self.agents: List[str] = []
        self.episode_agents: List[str] = []
        
        # Create a random generator.
        self.rng = np.random.default_rng()
        
        # Model files
        self.epjson_files_folder_path: str = episode_fn_config['epjson_files_folder_path']
        
        # Dimensions of the building.
        # If float, the dimension is fixed. If tuple, the dimension is random between the tuple index values [0] and [1] with a step of [2].
        self.building_h: float|List[float] = episode_fn_config["building_h"]
        self.building_w: float|List[float] = episode_fn_config["building_w"]
        self.building_l: float|List[float] = episode_fn_config["building_l"]
        
        self.war_north: float|List[float] = episode_fn_config["war_north"]
        self.war_east: float|List[float] = episode_fn_config["war_east"]
        self.war_south: float|List[float] = episode_fn_config["war_south"]
        self.war_west: float|List[float] = episode_fn_config["war_west"]
        
        # Building material properties
        self.wall_properties: Dict[str, Dict[str, float|List[float]]] = episode_fn_config["wall_properties"]
        self.roof_properties: Dict[str, Dict[str, float|List[float]]] = episode_fn_config["roof_properties"]
        
        self.window_u_factor: float|List[float] = episode_fn_config["window_u_factor"]
        self.window_solar_heat_gain_coefficient: float|List[float] = episode_fn_config["window_solar_heat_gain_coefficient"]
        
        self.internal_mass_surface_area_factor: float|List[float] = episode_fn_config["internal_mass_surface_area_factor"]
        
        # Infiltrations
        self.infiltrations: float = episode_fn_config["infiltrations"]
        
        # Controls
        ## Shading
        self.shading_setpoint: float = episode_fn_config["shading_setpoint"]
        
        ## HVAC
        self.heating_specific_power: float|List[float] = episode_fn_config["heating_specific_power"]
        self.cooling_heating_ratio: float|List[float] = episode_fn_config["cooling_heating_ratio"]
        
        # RunPeriod
        self.julian_day: int|List[int] = episode_fn_config.get("julian_day", 1)
        self.days_period: int = episode_fn_config.get("days_period", 28)
        
        # Weather file
        self.epw_files_folder_path: str = episode_fn_config['epw_files_folder_path']
        self.epw_file: str|List[str] = episode_fn_config['epw_file']
        
        # Load profiles
        self.load_profiles_folder_path: str = episode_fn_config['load_profiles_folder_path']
        self.load_profil_file: str = episode_fn_config["load_profil_file"]
        
        # Occupancy
        self.user_type: str|List[str] = episode_fn_config['user_type']
        self.zone_type: str|List[str] = episode_fn_config['zone_type']
            
    def get_episode_config(self, env_config: Dict[str,Any]) -> Dict[str,Any]:
        """
        This method define the properties of the episode. Changing some properties as weather or 
        Run Time Period, and defining others fix properties as volumen or window area relation.
        
        Return:
            dict: The method returns the env_config with modifications.
        """
        # ================
        # === GENERALS ===
        # ================
        # The first time, full episode_agents list is defined.
        if len(self.agents) == 0:
            self.agents = [agent for agent in env_config['agents_config'].keys()]
        
        # Clean de episode_agents list each time the method is called.
        self.episode_agents: List[str] = []
        
        # Append the Setpoint agent that it is always used.
        self.episode_agents.append("HVAC")
        
        
        # ====================
        # === SELECT MODEL ===
        # ====================
        # To determine the model, we need to know the WAR to each orientation.
        window_area_relation_list: List[float] = []
        for war in [self.war_north, self.war_east, self.war_south, self.war_west]:
            if type(war) == list:
                window_area_relation_list.append(get_random_parameter(war[0], war[1], war[2]))
            else:
                assert type(war) == float, f"Window area relation must be float or tuple, but is {type(war)}."
                window_area_relation_list.append(war)
        
        model, model_window_config = select_epjson_model(window_area_relation_list)
        # Establish the epJSON Object, it will be manipulated to modify the building model.
        with open(f"{self.epjson_files_folder_path}/model_{model}.epJSON") as file:
            epJSON_object: Dict[str,Any] = json.load(file)
        
        # Add the war properties to the observation space of each agent.
        # for agent in self.agents:
        #     env_config["agents_config"][agent]["observation"]["other_obs"]["war_north"] = window_area_relation_list[0]
        #     env_config["agents_config"][agent]["observation"]["other_obs"]["war_east"] = window_area_relation_list[1]
        #     env_config["agents_config"][agent]["observation"]["other_obs"]["war_south"] = window_area_relation_list[2]
        #     env_config["agents_config"][agent]["observation"]["other_obs"]["war_west"] = window_area_relation_list[3]
            
        # ============================
        # === CHANGE BUILDING SIZE ===
        # ============================
        volume: List[float] = []
        for size in [self.building_h, self.building_w, self.building_l]:
            if type(size) == list:
                volume.append(get_random_parameter(size[0], size[1], size[2]))
            else:
                assert type(size) == float, f"Building size parameter must be float or tuple, but is {type(size)}."
                volume.append(size)
        
        # Change the dimension of the building and windows
        building_dimension(epJSON_object, volume[0], volume[1], volume[2], np.array(window_area_relation_list))
        
        
        # ===================
        # === VENTILATION ===
        # ===================
        surface_number = 0
        for i in range(4):
            if model_window_config[i] == 1:
                # set ventilation_control_mode to NoVent to avoid natural ventilation
                surface_number += 1
                # add the respective agents to the list of episode_agents
                if i == 0:
                    epJSON_object["AirflowNetwork:MultiZone:Surface"][f"AirflowNetwork:MultiZone:Surface {surface_number}"]["ventilation_control_mode"] = "NoVent"
                        
                elif i == 1:
                    epJSON_object["AirflowNetwork:MultiZone:Surface"][f"AirflowNetwork:MultiZone:Surface {surface_number}"]["ventilation_control_mode"] = "NoVent"
                        
                elif i == 2:
                    epJSON_object["AirflowNetwork:MultiZone:Surface"][f"AirflowNetwork:MultiZone:Surface {surface_number}"]["ventilation_control_mode"] = "NoVent"
                        
                elif i == 3:
                    epJSON_object["AirflowNetwork:MultiZone:Surface"][f"AirflowNetwork:MultiZone:Surface {surface_number}"]["ventilation_control_mode"] = "NoVent"                    
        
        # Infiltrations
        epJSON_object["AirflowNetwork:MultiZone:Component:SimpleOpening"]["SimpleOpening"].update({"air_mass_flow_coefficient_when_opening_is_closed": self.infiltrations})
        
        # ======================
        # === WINDOW SHADING ===
        # ======================
        if window_area_relation_list[0] > 0:
            epJSON_object["WindowShadingControl"]["window_north"]["shading_control_type"] = "OnIfHighSolarOnWindow"
            epJSON_object["WindowShadingControl"]["window_north"]["setpoint"] = self.shading_setpoint
        
            
        # ===========================
        # === ENVELOPE PROPERTIES ===
        # ===========================
        # Define the type of construction (construction properties for each three layers)
        
        wall_properties: Dict[str, Dict[str, float]] = {}
        for shape in self.wall_properties.keys():
            wall_properties[shape] = {}
            for prop in self.wall_properties[shape].keys():
                if type(self.wall_properties[shape][prop]) == list:
                    wall_properties[shape][prop] = get_random_parameter(self.wall_properties[shape][prop][0], self.wall_properties[shape][prop][1], self.wall_properties[shape][prop][2])
                else:
                    assert type(self.wall_properties[shape][prop]) == float, f"wall_properties parameter must be float or tuple, but is {type(self.wall_properties[shape][prop])}."
                    wall_properties[shape][prop] = self.wall_properties[shape][prop]
        
        roof_properties: Dict[str, Dict[str, float]] = {}
        for shape in self.roof_properties.keys():
            roof_properties[shape] = {}
            for prop in self.roof_properties[shape].keys():
                if type(self.roof_properties[shape][prop]) == list:
                    roof_properties[shape][prop] = get_random_parameter(self.roof_properties[shape][prop][0], self.roof_properties[shape][prop][1], self.roof_properties[shape][prop][2])
                else:
                    assert type(self.roof_properties[shape][prop]) == float, f"roof_properties parameter must be float or tuple, but is {type(self.roof_properties[shape][prop])}."
                    roof_properties[shape][prop] = self.roof_properties[shape][prop]
        
        
        if type(self.window_u_factor) == list:
            window_u_factor = get_random_parameter(self.window_u_factor[0], self.window_u_factor[1], self.window_u_factor[2])
        else:
            assert type(self.window_u_factor) == float, f"window_u_factor parameter must be float or tuple, but is {type(self.window_u_factor)}."
            window_u_factor = self.window_u_factor
        
        if type(self.window_solar_heat_gain_coefficient) == list:
            window_solar_heat_gain_coefficient = get_random_parameter(self.window_solar_heat_gain_coefficient[0], self.window_solar_heat_gain_coefficient[1], self.window_solar_heat_gain_coefficient[2])
        else:
            assert type(self.window_solar_heat_gain_coefficient) == float, f"window_solar_heat_gain_coefficient parameter must be float or tuple, but is {type(self.window_solar_heat_gain_coefficient)}."
            window_solar_heat_gain_coefficient = self.window_solar_heat_gain_coefficient
        
        # Walls
        for shape in wall_properties.keys():
            for prop in wall_properties[shape].keys():
                epJSON_object["Material"][shape][prop] = wall_properties[shape][prop]
        
        # Roof
        for shape in roof_properties.keys():
            for prop in roof_properties[shape].keys():
                epJSON_object["Material"][shape][prop] = roof_properties[shape][prop]
        
        # Windows
        epJSON_object['WindowMaterial:SimpleGlazingSystem']['WindowMaterial']['u_factor'] = window_u_factor
        epJSON_object['WindowMaterial:SimpleGlazingSystem']['WindowMaterial']['solar_heat_gain_coefficient'] = window_solar_heat_gain_coefficient
        
        
        # =====================
        # === INTERNAL MASS ===
        # =====================
        if type(self.internal_mass_surface_area_factor) == list:
            internal_mass_surface_area_factor: float = get_random_parameter(self.internal_mass_surface_area_factor[0], self.internal_mass_surface_area_factor[1], self.internal_mass_surface_area_factor[2])
        else:
            assert type(self.internal_mass_surface_area_factor) == float, "internal_mass_surface_area_factor must be float or tuple"
            internal_mass_surface_area_factor: float = self.internal_mass_surface_area_factor
        
        for key in [key for key in epJSON_object["InternalMass"].keys()]:
            epJSON_object["InternalMass"][key]["surface_area"] = (volume[1]*volume[2]) * internal_mass_surface_area_factor
        
        
        # =======================
        # === HVAC PROPERTIES ===
        # =======================
        # The limit capacity of bouth cooling and heating are changed.
        if type(self.heating_specific_power) == list:
            heating_specific_power: float = get_random_parameter(self.heating_specific_power[0], self.heating_specific_power[1], self.heating_specific_power[2])
        else:
            assert type(self.heating_specific_power) == float, "heating_specific_power must be float or tuple"
            heating_specific_power: float = self.heating_specific_power
        
        if type(self.cooling_heating_ratio) == list:
            cooling_heating_ratio: float = get_random_parameter(self.cooling_heating_ratio[0], self.cooling_heating_ratio[1], self.cooling_heating_ratio[2])
        else:
            assert type(self.cooling_heating_ratio) == float, "cooling_heating_ratio must be float or tuple"
            cooling_heating_ratio: float = self.cooling_heating_ratio
            
        HVAC_names = [key for key in epJSON_object["ZoneHVAC:IdealLoadsAirSystem"].keys()]
        number_of_timesteps_per_hour = epJSON_object['Timestep']['Timestep 1']['number_of_timesteps_per_hour']
        for hvac in range(len(HVAC_names)):
            epJSON_object["ZoneHVAC:IdealLoadsAirSystem"][HVAC_names[hvac]]["maximum_sensible_heating_capacity"] = (volume[1]*volume[2]) * heating_specific_power
            epJSON_object["ZoneHVAC:IdealLoadsAirSystem"][HVAC_names[hvac]]["maximum_total_cooling_capacity"] = epJSON_object["ZoneHVAC:IdealLoadsAirSystem"][HVAC_names[hvac]]["maximum_sensible_heating_capacity"] * cooling_heating_ratio
            # Change the energy reference for the reward function.
            for agent in self.agents:
                env_config['agents_config'][agent]["reward"]['reward_fn_config']['heating_energy_ref'] = epJSON_object["ZoneHVAC:IdealLoadsAirSystem"][HVAC_names[hvac]]["maximum_sensible_heating_capacity"] * 3600/number_of_timesteps_per_hour
                env_config['agents_config'][agent]["reward"]['reward_fn_config']['cooling_energy_ref'] = epJSON_object["ZoneHVAC:IdealLoadsAirSystem"][HVAC_names[hvac]]["maximum_total_cooling_capacity"] * 3600/number_of_timesteps_per_hour
                env_config['agents_config'][agent]["reward"]['reward_fn_config']['number_of_timesteps_per_hour'] = number_of_timesteps_per_hour
        
        
        # ==================
        # === RUN PERIOD ===
        # ==================
        # RunPeriod (use cut_period_len to define the period length).
        if type(self.julian_day) == int:
            julian_day = self.julian_day
            days_period = self.days_period
        
        elif type(self.julian_day) == list:
            a = self.julian_day
            p: List[float] = []
            for _ in range(len(a)):
                p.append((1)/(len(a)))
            # p.append(0.7)
            p_array = np.array(p)
            a_array = np.array(a)
            
            julian_day = self.rng.choice(
                a=a_array,
                p=p_array)
            days_period = self.days_period
        else:
            raise ValueError('julian_day must be int or list of int')
        
        runperiod_begin_month, runperiod_begin_day, runperiod_end_month, runperiod_end_day = run_period(julian_day, days_period)
        
        epJSON_object['RunPeriod']['Run Period 1']['begin_month'] = runperiod_begin_month
        epJSON_object['RunPeriod']['Run Period 1']['begin_day_of_month'] = runperiod_begin_day
        epJSON_object['RunPeriod']['Run Period 1']['end_month'] = runperiod_end_month
        epJSON_object['RunPeriod']['Run Period 1']['end_day_of_month'] = runperiod_end_day
        
        
        # ====================
        # === WEATHER FILE ===
        # ====================
        # Establecer un clima para el entrenamiento
        # epw_file: str = np.random.choice(os.listdir(self.epw_files_folder_path))
        # epw_file = os.path.join(self.epw_files_folder_path, epw_file)
        if type(self.epw_file) == str:
            epw_file = self.epw_file
        
        elif type(self.epw_file) == list:
            a = self.epw_file
            p: List[float] = []
            for _ in range(len(a)):
                p.append((1)/(len(a)))
            # p.append(0.7)
            p_array = np.array(p)
            a_array = np.array(a)
            
            epw_file = self.rng.choice(
                a=a_array,
                p=p_array)
            
        else:
            raise ValueError('julian_day must be int or list of int')
        
        
        env_config["epw_path"] = epw_file
        # la, lo, tz, al = extract_epw_location_data(epw_file)
        # for agent in self.agents:
        #     env_config["agents_config"][agent]["observation"]["other_obs"]["latitude"] = np.sin(2 * np.pi * la / 90)
        #     env_config["agents_config"][agent]["observation"]["other_obs"]["longitude"] = np.sin(2 * np.pi * lo / 180)
        #     env_config["agents_config"][agent]["observation"]["other_obs"]["time_zone"] = np.sin(2 * np.pi * tz / 12)
        #     env_config["agents_config"][agent]["observation"]["other_obs"]["elevation"] = al
        
        
        # =================
        # === OCCUPANCY ===
        # =================
        if type(self.user_type) == str:
            user_type = self.user_type
            
        elif type(self.user_type) == list:
            a = self.user_type
            p: List[float] = []
            for _ in range(len(a)):
                p.append((1)/(len(a)))
            # p.append(0.7)
            p_array = np.array(p)
            a_array = np.array(a)
            
            user_type = self.rng.choice(
                a=a_array,
                p=p_array)
        
        else:
            user_type = "Typical family, office job"
    
        if type(self.zone_type) == str:
            zone_type = self.zone_type
            
        elif type(self.zone_type) == list:
            a = self.zone_type
            p: List[float] = []
            for _ in range(len(a)):
                p.append((1)/(len(a)))
            # p.append(0.7)
            p_array = np.array(p)
            a_array = np.array(a)
            
            zone_type = self.rng.choice(
                a=a_array,
                p=p_array)
        else:
            zone_type = "daytime"
        
        for agent in self.agents:
            env_config["agents_config"][agent]["observation"]["user_type"] = user_type
            env_config["agents_config"][agent]["observation"]["zone_type"] = zone_type
        
        
        # =============
        # === LOADS ===
        # =============
        # Change the load file profiles names to the new copy of schedule
        schedule_file_keys = [key for key in epJSON_object["Schedule:File"].keys()]
        for key in schedule_file_keys:
            epJSON_object["Schedule:File"][key]["file_name"] = self.load_profil_file
        
        
        # ==================
        # === EVALUATION ===
        # ==================
        if env_config["evaluation"]:        
            for metric in epJSON_object['OutputControl:Files']['OutputControl:Files 1'].keys():
                epJSON_object['OutputControl:Files']['OutputControl:Files 1'][metric] = "Yes"
            
        else:
            for metric in epJSON_object['OutputControl:Files']['OutputControl:Files 1'].keys():
                epJSON_object['OutputControl:Files']['OutputControl:Files 1'][metric] = "No"
        
        
        # ========================
        # === SAVE EPJSON FILE ===
        # ========================
        # The new modify epjson file is writed.
        env_config["epjson_path"] = os.path.join(self.temp_dir, f"temp-{os.getpid()}.epJSON")
        # The new modify epjson file is writed.
        with open(env_config["epjson_path"], 'w') as fp:
            json.dump(epJSON_object, fp, sort_keys=False, indent=4)

        return env_config

    def get_episode_agents(self, env_config: Dict[str,Any], possible_agents: List[str]) -> List[str]:
        """
        This method returns the agents for the episode configuration in the EnergyPlus environment.

        Returns:
            List[str]: The agent that are acting for the episode configuration. Default: possible_agent list.
        """
        return self.episode_agents
    
    def get_timestep_agents(self, env_config: Dict[str,Any], possible_agents: List[str]) -> List[str]:
        """
        This method returns the agents for the episode configuration in the EnergyPlus environment.

        Returns:
            List[str]: The agent that are acting for the episode configuration. Default: possible_agent list.
        """
        return self.get_episode_agents(env_config, possible_agents)
