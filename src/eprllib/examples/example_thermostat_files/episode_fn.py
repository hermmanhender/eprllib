"""
Tarea 1-d: Control de termostatos
=======================================

Se analizan este caso considerando la tarea de control de termostato de los sistemas activos de una
vivienda y disponibilidad. En este caso, una habitación prismática con un sistema de calefacción y refrigeración ideal 
con un agente que opera los siguientes actuadores:

1. ("Schedule:Compact", "Schedule Value", "heating_setpoint"),
2. ("Schedule:Compact", "Schedule Value", "cooling_setpoint"),
3. ("Schedule:Constant", "Schedule Value", "HVAC_OnOff")

Se definen características térmicas de la envolvente constantes.

Es importante considerar los perfiles de ocupación y perfiles de cargas internas, ya que ambos impactan 
en el balance térmico de la vivienda y en la recompensa. En este estudio se ha optado por un perfil de 
ocupación de una persona que trabaja fuera de casa con un calendario ajustado al horario de comercio e industria.

La potencia del equipo de acondicionamiento de aire, tanto para frio como para calor, se han dimensionado según simulación
con reglas convencionales en EnergyPlus y se han utilizado para el resto de los experimentos.

La operación de las ventanas es nula y solo se consideran los efectos de infiltraciones.

El control de sombra en la ventana norte se basa en reglas simples de control de sombra. Se analizó con simulaciones
las reglas implementadas de forma nativa en EnergyPlus para el caso convencional y se utilizó para el resto de los experimentos
la de mejor rendimiento.

Se establece que cada episodio tenga una longitud de 7 días, donde se pueden apreciar los
fenómenos de inercia térmica y se aumenta la cantidad de episodios considerablemente para la generalización
del problema. Para ello se plantearon dos escenarios: uno con las semanas consecutivas (empezando en enero y finalizando
en diciembre) y otro con semanas aleatorias.

Los climas utilizados corresponden a una misma región climática dentro de la Provincia de Mendoza. Para la 
evaluación se utiliza un clima similar, pero no utilizado en el entrenamiento.

La evaluación de las políticas para los diferentes agentes se realizan con métricas energéticas y de violación
de temperaturas de confort.
"""
import os
import json
import numpy as np
import tempfile
from typing import Any, Dict, List
from eprllib.Episodes.BaseEpisode import BaseEpisode
from eprllib.Utils.episode_fn_utils import building_dimension, run_period, extract_epw_location_data, select_epjson_model

class task_cofiguration(BaseEpisode):
    def __init__(
        self, episode_fn_config:Dict[str,Any]
    ):
        
        super().__init__(episode_fn_config)
        
        
        self.temp_dir = tempfile.gettempdir()
        print(f"Temporary directory for models: {self.temp_dir}")
        self.model_window_configs = {
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
        self.agents = []
        self.episode_agents = []
        self.first = True
        
        # Asignation of the configuration
        self.epjson_files_folder_path: str = episode_fn_config['epjson_files_folder_path']
        self.epw_files_folder_path: str = episode_fn_config['epw_files_folder_path']
        self.epw_file = episode_fn_config['epw_file']
        self.load_profiles_folder_path: str = episode_fn_config['load_profiles_folder_path']
        
        
        self.model = episode_fn_config["model"]
        self.building_h = episode_fn_config["building_h"]
        self.building_w = episode_fn_config["building_w"]
        self.building_l = episode_fn_config["building_l"]
        self.war_north = episode_fn_config["war_north"]
        self.war_east = episode_fn_config["war_east"]
        self.war_south = episode_fn_config["war_south"]
        self.war_west = episode_fn_config["war_west"]
        self.shading_setpoint = episode_fn_config["shading_setpoint"]
        
        self.Material_wall_exterior_thickness = episode_fn_config["Material_wall_exterior_thickness"]
        self.Material_wall_exterior_conductivity = episode_fn_config["Material_wall_exterior_conductivity"]
        self.Material_wall_exterior_density = episode_fn_config["Material_wall_exterior_density"]
        self.Material_wall_exterior_specific_heat = episode_fn_config["Material_wall_exterior_specific_heat"]
        self.Material_wall_exterior_thermal_absorptance = episode_fn_config["Material_wall_exterior_thermal_absorptance"]
        self.Material_wall_exterior_solar_absorptance = episode_fn_config["Material_wall_exterior_solar_absorptance"]

        self.Material_wall_inter_thickness = episode_fn_config["Material_wall_inter_thickness"]
        self.Material_wall_inter_conductivity = episode_fn_config["Material_wall_inter_conductivity"]
        self.Material_wall_inter_density = episode_fn_config["Material_wall_inter_density"]
        self.Material_wall_inter_specific_heat = episode_fn_config["Material_wall_inter_specific_heat"]
        self.Material_wall_inter_thermal_absorptance = episode_fn_config["Material_wall_inter_thermal_absorptance"]
        self.Material_wall_inter_solar_absorptance = episode_fn_config["Material_wall_inter_solar_absorptance"]

        self.Material_wall_inner_thickness = episode_fn_config["Material_wall_inner_thickness"]
        self.Material_wall_inner_conductivity = episode_fn_config["Material_wall_inner_conductivity"]
        self.Material_wall_inner_density = episode_fn_config["Material_wall_inner_density"]
        self.Material_wall_inner_specific_heat = episode_fn_config["Material_wall_inner_specific_heat"]
        self.Material_wall_inner_thermal_absorptance = episode_fn_config["Material_wall_inner_thermal_absorptance"]
        self.Material_wall_inner_solar_absorptance = episode_fn_config["Material_wall_inner_solar_absorptance"]
        
        self.Material_roof_exterior_thickness = episode_fn_config["Material_roof_exterior_thickness"]
        self.Material_roof_exterior_conductivity = episode_fn_config["Material_roof_exterior_conductivity"]
        self.Material_roof_exterior_density = episode_fn_config["Material_roof_exterior_density"]
        self.Material_roof_exterior_specific_heat = episode_fn_config["Material_roof_exterior_specific_heat"]
        self.Material_roof_exterior_thermal_absorptance = episode_fn_config["Material_roof_exterior_thermal_absorptance"]
        self.Material_roof_exterior_solar_absorptance = episode_fn_config["Material_roof_exterior_solar_absorptance"]

        self.Material_roof_inner_thickness = episode_fn_config["Material_roof_inner_thickness"]
        self.Material_roof_inner_conductivity = episode_fn_config["Material_roof_inner_conductivity"]
        self.Material_roof_inner_density = episode_fn_config["Material_roof_inner_density"]
        self.Material_roof_inner_specific_heat = episode_fn_config["Material_roof_inner_specific_heat"]
        self.Material_roof_inner_thermal_absorptance = episode_fn_config["Material_roof_inner_thermal_absorptance"]
        self.Material_roof_inner_solar_absorptance = episode_fn_config["Material_roof_inner_solar_absorptance"]
        
        self.WindowMaterial_SimpleGlazingSystem_WindowMaterial_u_factor = episode_fn_config["WindowMaterial:SimpleGlazingSystem_WindowMaterial_u_factor"]
        self.WindowMaterial_SimpleGlazingSystem_WindowMaterial_solar_heat_gain_coefficient = episode_fn_config["WindowMaterial:SimpleGlazingSystem_WindowMaterial_solar_heat_gain_coefficient"]
        
        self.im_surface_area_factor = episode_fn_config["im_surface_area_factor"]
        
        self.heating_specific_power = episode_fn_config["heating_specific_power"]
        self.cooling_heating_ratio = episode_fn_config["cooling_heating_ratio"]
        
        self.runperiod_begin_month = episode_fn_config["runperiod_begin_month"]
        self.runperiod_begin_day = episode_fn_config["runperiod_begin_day"]
        self.runperiod_end_month = episode_fn_config["runperiod_end_month"]
        self.runperiod_end_day = episode_fn_config["runperiod_end_day"]
        
        self.load_profil_file = episode_fn_config["load_profil_file"]
        
        self.julian_day: int|List[int] = episode_fn_config.get("julian_day", 1)
        self.days_period: int = episode_fn_config.get("days_period", 28)
        
        self.rng = np.random.default_rng()
        
            
    def get_episode_config(self, env_config: Dict[str,Any]) -> Dict[str,Any]:
        """
        This method define the properties of the episode. Changing some properties as weather or 
        Run Time Period, and defining others fix properties as volumen or window area relation.
        
        Return:
            dict: The method returns the env_config with modifications.
        """    
        if len(self.agents) == 0:
            self.agents = [agent for agent in env_config['agents_config'].keys()]
        # Clean de episode_agents list
        self.episode_agents: List[str] = []
        
        # Append the Setpoint agent that it is always used.
        self.episode_agents.append("HVAC")
        
        # === Superficies vidriadas ===
        # Selección del modelo. Esto define las Superficies vidriadas que el modelo posee.
        # Adicionalmente, hay que establecer los agentes que van a actuar en el episodio de 
        # acurdo con el modelo que se seleccione.
        model = self.model
        
        # Establish the epJSON Object, it will be manipulated to modify the building model.
        with open(f"{self.epjson_files_folder_path}/model_{model}.epJSON") as file:
            epJSON_object: Dict[str,Any] = json.load(file)
        
        # === Tamaño de la zona térmica equivalente ===
        # The building volume is V=h(high)*w(weiht)*l(large) m3
        h = self.building_h
        w = self.building_w
        l = self.building_l
        
        # === Proporción de area vidriada ===
        window_area_relation_list: List[float] = []
        model_window_config = self.model_window_configs[str(model)]
        surface_number = 0
        
        epJSON_object["AirflowNetwork:MultiZone:Component:SimpleOpening"]["SimpleOpening"]["air_mass_flow_coefficient_when_opening_is_closed"] = 0.5
        epJSON_object["People"]["People"]["number_of_people_schedule_name"] = "occupancy_schedule"

        for i in range(4):
            if model_window_config[i] == 1:
                # set ventilation_control_mode to NoVent to avoid natural ventilation
                surface_number += 1
                # add the respective agents to the list of episode_agents
                if i == 0:
                    window_area_relation_list.append(self.war_north)
                    epJSON_object["AirflowNetwork:MultiZone:Surface"][f"AirflowNetwork:MultiZone:Surface {surface_number}"]["ventilation_control_mode"] = "NoVent"

                elif i == 1:
                    window_area_relation_list.append(self.war_east)
                    epJSON_object["AirflowNetwork:MultiZone:Surface"][f"AirflowNetwork:MultiZone:Surface {surface_number}"]["ventilation_control_mode"] = "NoVent"
                
                elif i == 2:
                    window_area_relation_list.append(self.war_south)
                    epJSON_object["AirflowNetwork:MultiZone:Surface"][f"AirflowNetwork:MultiZone:Surface {surface_number}"]["ventilation_control_mode"] = "NoVent"
                
                elif i == 3:
                    window_area_relation_list.append(self.war_west)
                    epJSON_object["AirflowNetwork:MultiZone:Surface"][f"AirflowNetwork:MultiZone:Surface {surface_number}"]["ventilation_control_mode"] = "NoVent"
                
            else:
                window_area_relation_list.append(0)
        
        # === Window shading control ===
        epJSON_object["WindowShadingControl"]["window_north"]["shading_control_type"] = "OnIfHighSolarOnWindow"
        epJSON_object["WindowShadingControl"]["window_north"]["setpoint"] = self.shading_setpoint
        
        # Change the dimension of the building and windows
        window_area_relation = np.array(window_area_relation_list)
        building_dimension(epJSON_object, h, w, l, window_area_relation)
        
        # === Masa térmica interior | Aislación exterior ===
        # Define the type of construction (construction properties for each three layers)
        # Walls
        # Exterior Finish
        epJSON_object["Material"]["wall_exterior"]["thickness"] = self.Material_wall_exterior_thickness
        epJSON_object["Material"]["wall_exterior"]["conductivity"] = self.Material_wall_exterior_conductivity
        epJSON_object["Material"]["wall_exterior"]["density"] = self.Material_wall_exterior_density
        epJSON_object["Material"]["wall_exterior"]["specific_heat"] = self.Material_wall_exterior_specific_heat
        epJSON_object["Material"]["wall_exterior"]["thermal_absorptance"] = self.Material_wall_exterior_thermal_absorptance
        epJSON_object["Material"]["wall_exterior"]["solar_absorptance"] = epJSON_object["Material"]["wall_exterior"]["visible_absorptance"] = self.Material_wall_exterior_solar_absorptance
        # Inter layers
        epJSON_object["Material"]["wall_inter"]["thickness"] = self.Material_wall_inter_thickness
        epJSON_object["Material"]["wall_inter"]["conductivity"] = self.Material_wall_inter_conductivity
        epJSON_object["Material"]["wall_inter"]["density"] = self.Material_wall_inter_density
        epJSON_object["Material"]["wall_inter"]["specific_heat"] = self.Material_wall_inter_specific_heat
        epJSON_object["Material"]["wall_inter"]["thermal_absorptance"] = self.Material_wall_inter_thermal_absorptance
        epJSON_object["Material"]["wall_inter"]["solar_absorptance"] = epJSON_object["Material"]["wall_inter"]["visible_absorptance"] = self.Material_wall_inter_solar_absorptance
        # Internal Mass layer (interior layer)
        epJSON_object["Material"]["wall_inner"]["thickness"] = self.Material_wall_inner_thickness
        epJSON_object["Material"]["wall_inner"]["conductivity"] = self.Material_wall_inner_conductivity
        epJSON_object["Material"]["wall_inner"]["density"] = self.Material_wall_inner_density
        epJSON_object["Material"]["wall_inner"]["specific_heat"] = self.Material_wall_inner_specific_heat
        epJSON_object["Material"]["wall_inner"]["thermal_absorptance"] = self.Material_wall_inner_thermal_absorptance
        epJSON_object["Material"]["wall_inner"]["solar_absorptance"] = epJSON_object["Material"]["wall_inner"]["visible_absorptance"] = self.Material_wall_inner_solar_absorptance
        
        # Ceiling/Roof (exterior, inter layers)
        # Exterior Finish
        epJSON_object["Material"]["roof_exterior"]["thickness"] = self.Material_roof_exterior_thickness
        epJSON_object["Material"]["roof_exterior"]["conductivity"] = self.Material_roof_exterior_conductivity
        epJSON_object["Material"]["roof_exterior"]["density"] = self.Material_roof_exterior_density
        epJSON_object["Material"]["roof_exterior"]["specific_heat"] = self.Material_roof_exterior_specific_heat
        epJSON_object["Material"]["roof_exterior"]["thermal_absorptance"] = self.Material_roof_exterior_thermal_absorptance
        epJSON_object["Material"]["roof_exterior"]["solar_absorptance"] = epJSON_object["Material"]["roof_exterior"]["visible_absorptance"] = self.Material_roof_exterior_solar_absorptance
        # Inter layers
        epJSON_object["Material"]["roof_inner"]["thickness"] = self.Material_roof_inner_thickness
        epJSON_object["Material"]["roof_inner"]["conductivity"] = self.Material_roof_inner_conductivity
        epJSON_object["Material"]["roof_inner"]["density"] = self.Material_roof_inner_density
        epJSON_object["Material"]["roof_inner"]["specific_heat"] = self.Material_roof_inner_specific_heat
        epJSON_object["Material"]["roof_inner"]["thermal_absorptance"] = self.Material_roof_inner_thermal_absorptance
        epJSON_object["Material"]["roof_inner"]["solar_absorptance"] = epJSON_object["Material"]["roof_inner"]["visible_absorptance"] = self.Material_roof_inner_solar_absorptance
        
        # Windows
        # Change the window thermal properties
        epJSON_object['WindowMaterial:SimpleGlazingSystem']['WindowMaterial']['u_factor'] = self.WindowMaterial_SimpleGlazingSystem_WindowMaterial_u_factor
        epJSON_object['WindowMaterial:SimpleGlazingSystem']['WindowMaterial']['solar_heat_gain_coefficient'] = self.WindowMaterial_SimpleGlazingSystem_WindowMaterial_solar_heat_gain_coefficient
        
        # The internal thermal mass is modified.
        for key in [key for key in epJSON_object["InternalMass"].keys()]:
            epJSON_object["InternalMass"][key]["surface_area"] = (w*l) * self.im_surface_area_factor
        
        # The limit capacity of bouth cooling and heating are changed.
        HVAC_names = [key for key in epJSON_object["ZoneHVAC:IdealLoadsAirSystem"].keys()]
        number_of_timesteps_per_hour = epJSON_object['Timestep']['Timestep 1']['number_of_timesteps_per_hour']
        for hvac in range(len(HVAC_names)):
            epJSON_object["ZoneHVAC:IdealLoadsAirSystem"][HVAC_names[hvac]]["maximum_sensible_heating_capacity"] = (w*l) * self.heating_specific_power
            epJSON_object["ZoneHVAC:IdealLoadsAirSystem"][HVAC_names[hvac]]["maximum_total_cooling_capacity"] = epJSON_object["ZoneHVAC:IdealLoadsAirSystem"][HVAC_names[hvac]]["maximum_sensible_heating_capacity"] * self.cooling_heating_ratio
            # Change the energy reference for the reward function.
            for agent in self.agents:
                env_config['agents_config'][agent]["reward"]['reward_fn_config']['heating_energy_ref'] = epJSON_object["ZoneHVAC:IdealLoadsAirSystem"][HVAC_names[hvac]]["maximum_sensible_heating_capacity"] * 3600/number_of_timesteps_per_hour
                env_config['agents_config'][agent]["reward"]['reward_fn_config']['cooling_energy_ref'] = epJSON_object["ZoneHVAC:IdealLoadsAirSystem"][HVAC_names[hvac]]["maximum_total_cooling_capacity"] * 3600/number_of_timesteps_per_hour
                env_config['agents_config'][agent]["reward"]['reward_fn_config']['number_of_timesteps_per_hour'] = number_of_timesteps_per_hour
        
        # RunPeriod (use cut_period_len to define the period length).
        if type(self.julian_day) == int:
            runperiod_begin_month, runperiod_begin_day, runperiod_end_month, runperiod_end_day = run_period(self.julian_day, self.days_period)
        
        elif type(self.julian_day) == list:
            a = self.julian_day
            p: List[float] = []
            for _ in range(len(a)):
                p.append((1)/(len(a)))
            # p.append(0.7)
            p_array = np.array(p)
            a_array = np.array(a)
            
            j_day = self.rng.choice(
                a=a_array,
                p=p_array)
            
            runperiod_begin_month, runperiod_begin_day, runperiod_end_month, runperiod_end_day = run_period(j_day, self.days_period)
        else:
            raise ValueError('julian_day must be int or list of int')
        
        epJSON_object['RunPeriod']['Run Period 1']['begin_month'] = runperiod_begin_month
        epJSON_object['RunPeriod']['Run Period 1']['begin_day_of_month'] = runperiod_begin_day
        epJSON_object['RunPeriod']['Run Period 1']['end_month'] = runperiod_end_month
        epJSON_object['RunPeriod']['Run Period 1']['end_day_of_month'] = runperiod_end_day
        
        # Establecer un clima aleatorio durante el entrenamiento
        env_config["epw_path"] = self.epw_file
            
        if env_config["evaluation"]:
            # Change the load file profiles names to the new copy of schedule
            schedule_file_keys = [key for key in epJSON_object["Schedule:File"].keys()]
            for key in schedule_file_keys:
                epJSON_object["Schedule:File"][key]["file_name"] = self.load_profil_file

            for metric in epJSON_object['OutputControl:Files']['OutputControl:Files 1'].keys():
                epJSON_object['OutputControl:Files']['OutputControl:Files 1'][metric] = "Yes"
            
        else:
            # Change the load file profiles names to the new copy of schedule
            schedule_file_keys = [key for key in epJSON_object["Schedule:File"].keys()]
            for key in schedule_file_keys:
                epJSON_object["Schedule:File"][key]["file_name"] = self.load_profiles_folder_path + "/" + np.random.choice(os.listdir(self.load_profiles_folder_path))

            for metric in epJSON_object['OutputControl:Files']['OutputControl:Files 1'].keys():
                epJSON_object['OutputControl:Files']['OutputControl:Files 1'][metric] = "No"
        
        # The new modify epjson file is writed.
        env_config["epjson_path"] = os.path.join(self.temp_dir, f"temp-{os.getpid()}.epJSON")
        # The new modify epjson file is writed.
        with open(env_config["epjson_path"], 'w') as fp:
            json.dump(epJSON_object, fp, sort_keys=False, indent=4)
        
        if self.first:
            self.first = False
            epjson_path = os.path.join("C:/Users/grhen/Documents/GitHub/SimpleCases/configurations", "task1.epJSON")
            with open(epjson_path, 'w') as fp:
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


class task2_cofiguration(BaseEpisode):
    def __init__(
        self, episode_fn_config:Dict[str,Any]
    ):
        
        super().__init__(episode_fn_config)
        
        
        self.temp_dir = tempfile.gettempdir()
        print(f"Temporary directory for models: {self.temp_dir}")
        self.model_window_configs = {
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
        self.agents = []
        self.episode_agents = []
        self.first = True
        
        # Asignation of the configuration
        self.epjson_files_folder_path: str = episode_fn_config['epjson_files_folder_path']
        self.epw_files_folder_path: str = episode_fn_config['epw_files_folder_path']
        self.epw_file:str = episode_fn_config['epw_file']
        self.load_profiles_folder_path: str = episode_fn_config['load_profiles_folder_path']
        
        
        self.model = episode_fn_config["model"]
        self.building_h = episode_fn_config["building_h"]
        self.building_w = episode_fn_config["building_w"]
        self.building_l = episode_fn_config["building_l"]
        self.war_north: float = episode_fn_config["war_north"]
        self.war_east: float = episode_fn_config["war_east"]
        self.war_south: float = episode_fn_config["war_south"]
        self.war_west: float = episode_fn_config["war_west"]
        self.shading_setpoint = episode_fn_config["shading_setpoint"]
        
        self.Material_wall_exterior_thickness = episode_fn_config["Material_wall_exterior_thickness"]
        self.Material_wall_exterior_conductivity = episode_fn_config["Material_wall_exterior_conductivity"]
        self.Material_wall_exterior_density = episode_fn_config["Material_wall_exterior_density"]
        self.Material_wall_exterior_specific_heat = episode_fn_config["Material_wall_exterior_specific_heat"]
        self.Material_wall_exterior_thermal_absorptance = episode_fn_config["Material_wall_exterior_thermal_absorptance"]
        self.Material_wall_exterior_solar_absorptance = episode_fn_config["Material_wall_exterior_solar_absorptance"]

        self.Material_wall_inter_thickness = episode_fn_config["Material_wall_inter_thickness"]
        self.Material_wall_inter_conductivity = episode_fn_config["Material_wall_inter_conductivity"]
        self.Material_wall_inter_density = episode_fn_config["Material_wall_inter_density"]
        self.Material_wall_inter_specific_heat = episode_fn_config["Material_wall_inter_specific_heat"]
        self.Material_wall_inter_thermal_absorptance = episode_fn_config["Material_wall_inter_thermal_absorptance"]
        self.Material_wall_inter_solar_absorptance = episode_fn_config["Material_wall_inter_solar_absorptance"]

        self.Material_wall_inner_thickness = episode_fn_config["Material_wall_inner_thickness"]
        self.Material_wall_inner_conductivity = episode_fn_config["Material_wall_inner_conductivity"]
        self.Material_wall_inner_density = episode_fn_config["Material_wall_inner_density"]
        self.Material_wall_inner_specific_heat = episode_fn_config["Material_wall_inner_specific_heat"]
        self.Material_wall_inner_thermal_absorptance = episode_fn_config["Material_wall_inner_thermal_absorptance"]
        self.Material_wall_inner_solar_absorptance = episode_fn_config["Material_wall_inner_solar_absorptance"]
        
        self.Material_roof_exterior_thickness = episode_fn_config["Material_roof_exterior_thickness"]
        self.Material_roof_exterior_conductivity = episode_fn_config["Material_roof_exterior_conductivity"]
        self.Material_roof_exterior_density = episode_fn_config["Material_roof_exterior_density"]
        self.Material_roof_exterior_specific_heat = episode_fn_config["Material_roof_exterior_specific_heat"]
        self.Material_roof_exterior_thermal_absorptance = episode_fn_config["Material_roof_exterior_thermal_absorptance"]
        self.Material_roof_exterior_solar_absorptance = episode_fn_config["Material_roof_exterior_solar_absorptance"]

        self.Material_roof_inner_thickness = episode_fn_config["Material_roof_inner_thickness"]
        self.Material_roof_inner_conductivity = episode_fn_config["Material_roof_inner_conductivity"]
        self.Material_roof_inner_density = episode_fn_config["Material_roof_inner_density"]
        self.Material_roof_inner_specific_heat = episode_fn_config["Material_roof_inner_specific_heat"]
        self.Material_roof_inner_thermal_absorptance = episode_fn_config["Material_roof_inner_thermal_absorptance"]
        self.Material_roof_inner_solar_absorptance = episode_fn_config["Material_roof_inner_solar_absorptance"]
        
        self.WindowMaterial_SimpleGlazingSystem_WindowMaterial_u_factor = episode_fn_config["WindowMaterial:SimpleGlazingSystem_WindowMaterial_u_factor"]
        self.WindowMaterial_SimpleGlazingSystem_WindowMaterial_solar_heat_gain_coefficient = episode_fn_config["WindowMaterial:SimpleGlazingSystem_WindowMaterial_solar_heat_gain_coefficient"]
        
        self.im_surface_area_factor = episode_fn_config["im_surface_area_factor"]
        
        self.heating_specific_power = episode_fn_config["heating_specific_power"]
        self.cooling_heating_ratio = episode_fn_config["cooling_heating_ratio"]
        
        self.runperiod_begin_month = episode_fn_config["runperiod_begin_month"]
        self.runperiod_begin_day = episode_fn_config["runperiod_begin_day"]
        self.runperiod_end_month = episode_fn_config["runperiod_end_month"]
        self.runperiod_end_day = episode_fn_config["runperiod_end_day"]
        
        self.load_profil_file = episode_fn_config["load_profil_file"]
        
        self.julian_day: int|List[int] = episode_fn_config.get("julian_day", 1)
        self.days_period: int = episode_fn_config.get("days_period", 28)
        
        self.rng = np.random.default_rng()
        
            
    def get_episode_config(self, env_config: Dict[str,Any]) -> Dict[str,Any]:
        """
        This method define the properties of the episode. Changing some properties as weather or 
        Run Time Period, and defining others fix properties as volumen or window area relation.
        
        Return:
            dict: The method returns the env_config with modifications.
        """    
        if len(self.agents) == 0:
            self.agents = [agent for agent in env_config['agents_config'].keys()]
        # Clean de episode_agents list
        self.episode_agents: List[str] = []
        
        # Append the Setpoint agent that it is always used.
        self.episode_agents.append("HVAC")
        
        # === Superficies vidriadas ===
        # Selección del modelo. Esto define las Superficies vidriadas que el modelo posee.
        # Adicionalmente, hay que establecer los agentes que van a actuar en el episodio de 
        # acurdo con el modelo que se seleccione.
        model = self.model
        war_list: List[float] = [self.war_north, self.war_east, self.war_south, self.war_west]
        model, model_window_config = select_epjson_model(war_list)
        # Establish the epJSON Object, it will be manipulated to modify the building model.
        with open(f"{self.epjson_files_folder_path}/model_{model}.epJSON") as file:
            epJSON_object: Dict[str,Any] = json.load(file)
        
        # === Tamaño de la zona térmica equivalente ===
        # The building volume is V=h(high)*w(weiht)*l(large) m3
        h = self.building_h
        w = self.building_w
        l = self.building_l
        
        # === Proporción de area vidriada ===
        window_area_relation_list: List[float] = []
        model_window_config = self.model_window_configs[str(model)]
        surface_number = 0
        
        epJSON_object["AirflowNetwork:MultiZone:Component:SimpleOpening"]["SimpleOpening"]["air_mass_flow_coefficient_when_opening_is_closed"] = 0.5
        epJSON_object["People"]["People"]["number_of_people_schedule_name"] = "occupants schedule"
        
        for i in range(4):
            if model_window_config[i] == 1:
                # set ventilation_control_mode to NoVent to avoid natural ventilation
                surface_number += 1
                # add the respective agents to the list of episode_agents
                if i == 0:
                    window_area_relation_list.append(self.war_north)
                    epJSON_object["AirflowNetwork:MultiZone:Surface"][f"AirflowNetwork:MultiZone:Surface {surface_number}"]["ventilation_control_mode"] = "NoVent"
                    for agent in self.agents:
                        env_config["agents_config"][agent]["observation"]["other_obs"]["war_north"] = self.war_north
                        
                elif i == 1:
                    window_area_relation_list.append(self.war_east)
                    epJSON_object["AirflowNetwork:MultiZone:Surface"][f"AirflowNetwork:MultiZone:Surface {surface_number}"]["ventilation_control_mode"] = "NoVent"
                    for agent in self.agents:
                        env_config["agents_config"][agent]["observation"]["other_obs"]["war_east"] = self.war_east
                        
                elif i == 2:
                    window_area_relation_list.append(self.war_south)
                    epJSON_object["AirflowNetwork:MultiZone:Surface"][f"AirflowNetwork:MultiZone:Surface {surface_number}"]["ventilation_control_mode"] = "NoVent"
                    for agent in self.agents:
                        env_config["agents_config"][agent]["observation"]["other_obs"]["war_south"] = self.war_south
                        
                elif i == 3:
                    window_area_relation_list.append(self.war_west)
                    epJSON_object["AirflowNetwork:MultiZone:Surface"][f"AirflowNetwork:MultiZone:Surface {surface_number}"]["ventilation_control_mode"] = "NoVent"
                    for agent in self.agents:
                        env_config["agents_config"][agent]["observation"]["other_obs"]["war_west"] = self.war_west
                        
            else:
                window_area_relation_list.append(0)
                if i == 0:
                    for agent in self.agents:
                        env_config["agents_config"][agent]["observation"]["other_obs"]["war_north"] = 0
                elif i == 1:
                    for agent in self.agents:
                        env_config["agents_config"][agent]["observation"]["other_obs"]["war_east"] = 0
                elif i == 2:
                    for agent in self.agents:
                        env_config["agents_config"][agent]["observation"]["other_obs"]["war_south"] = 0
                elif i == 3:
                    for agent in self.agents:
                        env_config["agents_config"][agent]["observation"]["other_obs"]["war_west"] = 0
        
        
        # === Window shading control ===
        epJSON_object["WindowShadingControl"]["window_north"]["shading_control_type"] = "OnIfHighSolarOnWindow"
        epJSON_object["WindowShadingControl"]["window_north"]["setpoint"] = self.shading_setpoint
        
        # Change the dimension of the building and windows
        window_area_relation = np.array(window_area_relation_list)
        building_dimension(epJSON_object, h, w, l, window_area_relation)
        
        # === Masa térmica interior | Aislación exterior ===
        # Define the type of construction (construction properties for each three layers)
        # Walls
        # Exterior Finish
        epJSON_object["Material"]["wall_exterior"]["thickness"] = self.Material_wall_exterior_thickness
        epJSON_object["Material"]["wall_exterior"]["conductivity"] = self.Material_wall_exterior_conductivity
        epJSON_object["Material"]["wall_exterior"]["density"] = self.Material_wall_exterior_density
        epJSON_object["Material"]["wall_exterior"]["specific_heat"] = self.Material_wall_exterior_specific_heat
        epJSON_object["Material"]["wall_exterior"]["thermal_absorptance"] = self.Material_wall_exterior_thermal_absorptance
        epJSON_object["Material"]["wall_exterior"]["solar_absorptance"] = epJSON_object["Material"]["wall_exterior"]["visible_absorptance"] = self.Material_wall_exterior_solar_absorptance
        # Inter layers
        epJSON_object["Material"]["wall_inter"]["thickness"] = self.Material_wall_inter_thickness
        epJSON_object["Material"]["wall_inter"]["conductivity"] = self.Material_wall_inter_conductivity
        epJSON_object["Material"]["wall_inter"]["density"] = self.Material_wall_inter_density
        epJSON_object["Material"]["wall_inter"]["specific_heat"] = self.Material_wall_inter_specific_heat
        epJSON_object["Material"]["wall_inter"]["thermal_absorptance"] = self.Material_wall_inter_thermal_absorptance
        epJSON_object["Material"]["wall_inter"]["solar_absorptance"] = epJSON_object["Material"]["wall_inter"]["visible_absorptance"] = self.Material_wall_inter_solar_absorptance
        # Internal Mass layer (interior layer)
        epJSON_object["Material"]["wall_inner"]["thickness"] = self.Material_wall_inner_thickness
        epJSON_object["Material"]["wall_inner"]["conductivity"] = self.Material_wall_inner_conductivity
        epJSON_object["Material"]["wall_inner"]["density"] = self.Material_wall_inner_density
        epJSON_object["Material"]["wall_inner"]["specific_heat"] = self.Material_wall_inner_specific_heat
        epJSON_object["Material"]["wall_inner"]["thermal_absorptance"] = self.Material_wall_inner_thermal_absorptance
        epJSON_object["Material"]["wall_inner"]["solar_absorptance"] = epJSON_object["Material"]["wall_inner"]["visible_absorptance"] = self.Material_wall_inner_solar_absorptance
        
        # Ceiling/Roof (exterior, inter layers)
        # Exterior Finish
        epJSON_object["Material"]["roof_exterior"]["thickness"] = self.Material_roof_exterior_thickness
        epJSON_object["Material"]["roof_exterior"]["conductivity"] = self.Material_roof_exterior_conductivity
        epJSON_object["Material"]["roof_exterior"]["density"] = self.Material_roof_exterior_density
        epJSON_object["Material"]["roof_exterior"]["specific_heat"] = self.Material_roof_exterior_specific_heat
        epJSON_object["Material"]["roof_exterior"]["thermal_absorptance"] = self.Material_roof_exterior_thermal_absorptance
        epJSON_object["Material"]["roof_exterior"]["solar_absorptance"] = epJSON_object["Material"]["roof_exterior"]["visible_absorptance"] = self.Material_roof_exterior_solar_absorptance
        # Inter layers
        epJSON_object["Material"]["roof_inner"]["thickness"] = self.Material_roof_inner_thickness
        epJSON_object["Material"]["roof_inner"]["conductivity"] = self.Material_roof_inner_conductivity
        epJSON_object["Material"]["roof_inner"]["density"] = self.Material_roof_inner_density
        epJSON_object["Material"]["roof_inner"]["specific_heat"] = self.Material_roof_inner_specific_heat
        epJSON_object["Material"]["roof_inner"]["thermal_absorptance"] = self.Material_roof_inner_thermal_absorptance
        epJSON_object["Material"]["roof_inner"]["solar_absorptance"] = epJSON_object["Material"]["roof_inner"]["visible_absorptance"] = self.Material_roof_inner_solar_absorptance
        
        # Windows
        # Change the window thermal properties
        epJSON_object['WindowMaterial:SimpleGlazingSystem']['WindowMaterial']['u_factor'] = self.WindowMaterial_SimpleGlazingSystem_WindowMaterial_u_factor
        epJSON_object['WindowMaterial:SimpleGlazingSystem']['WindowMaterial']['solar_heat_gain_coefficient'] = self.WindowMaterial_SimpleGlazingSystem_WindowMaterial_solar_heat_gain_coefficient
        
        # The internal thermal mass is modified.
        for key in [key for key in epJSON_object["InternalMass"].keys()]:
            epJSON_object["InternalMass"][key]["surface_area"] = (w*l) * self.im_surface_area_factor
        
        # The limit capacity of bouth cooling and heating are changed.
        HVAC_names = [key for key in epJSON_object["ZoneHVAC:IdealLoadsAirSystem"].keys()]
        number_of_timesteps_per_hour = epJSON_object['Timestep']['Timestep 1']['number_of_timesteps_per_hour']
        for hvac in range(len(HVAC_names)):
            epJSON_object["ZoneHVAC:IdealLoadsAirSystem"][HVAC_names[hvac]]["maximum_sensible_heating_capacity"] = (w*l) * self.heating_specific_power
            epJSON_object["ZoneHVAC:IdealLoadsAirSystem"][HVAC_names[hvac]]["maximum_total_cooling_capacity"] = epJSON_object["ZoneHVAC:IdealLoadsAirSystem"][HVAC_names[hvac]]["maximum_sensible_heating_capacity"] * self.cooling_heating_ratio
            # Change the energy reference for the reward function.
            for agent in self.agents:
                env_config['agents_config'][agent]["reward"]['reward_fn_config']['heating_energy_ref'] = epJSON_object["ZoneHVAC:IdealLoadsAirSystem"][HVAC_names[hvac]]["maximum_sensible_heating_capacity"] * 3600/number_of_timesteps_per_hour
                env_config['agents_config'][agent]["reward"]['reward_fn_config']['cooling_energy_ref'] = epJSON_object["ZoneHVAC:IdealLoadsAirSystem"][HVAC_names[hvac]]["maximum_total_cooling_capacity"] * 3600/number_of_timesteps_per_hour
                env_config['agents_config'][agent]["reward"]['reward_fn_config']['number_of_timesteps_per_hour'] = number_of_timesteps_per_hour
        
        # RunPeriod (use cut_period_len to define the period length).
        if type(self.julian_day) == int:
            runperiod_begin_month, runperiod_begin_day, runperiod_end_month, runperiod_end_day = run_period(self.julian_day, self.days_period)
        
        elif type(self.julian_day) == list:
            a = self.julian_day
            p: List[float] = []
            for _ in range(len(a)):
                p.append((1)/(len(a)))
            # p.append(0.7)
            p_array = np.array(p)
            a_array = np.array(a)
            
            j_day = self.rng.choice(
                a=a_array,
                p=p_array)
            
            runperiod_begin_month, runperiod_begin_day, runperiod_end_month, runperiod_end_day = run_period(j_day, self.days_period)
        else:
            raise ValueError('julian_day must be int or list of int')
        
        epJSON_object['RunPeriod']['Run Period 1']['begin_month'] = runperiod_begin_month
        epJSON_object['RunPeriod']['Run Period 1']['begin_day_of_month'] = runperiod_begin_day
        epJSON_object['RunPeriod']['Run Period 1']['end_month'] = runperiod_end_month
        epJSON_object['RunPeriod']['Run Period 1']['end_day_of_month'] = runperiod_end_day
        
        # Establecer un clima para el entrenamiento
        env_config["epw_path"] = self.epw_file
        la, lo, tz, al = extract_epw_location_data(self.epw_file)
        for agent in self.agents:
            env_config["agents_config"][agent]["observation"]["other_obs"]["latitude"] = la
            env_config["agents_config"][agent]["observation"]["other_obs"]["longitude"] = lo
            env_config["agents_config"][agent]["observation"]["other_obs"]["time_zone"] = tz
            env_config["agents_config"][agent]["observation"]["other_obs"]["elevation"] = al
            
        
        if env_config["evaluation"]:
            # Change the load file profiles names to the new copy of schedule
            # schedule_file_keys = [key for key in epJSON_object["Schedule:File"].keys()]
            # for key in schedule_file_keys:
            #     epJSON_object["Schedule:File"][key]["file_name"] = self.load_profil_file

            for metric in epJSON_object['OutputControl:Files']['OutputControl:Files 1'].keys():
                epJSON_object['OutputControl:Files']['OutputControl:Files 1'][metric] = "Yes"
            
        else:
            # Change the load file profiles names to the new copy of schedule
            # schedule_file_keys = [key for key in epJSON_object["Schedule:File"].keys()]
            # for key in schedule_file_keys:
            #     epJSON_object["Schedule:File"][key]["file_name"] = self.load_profiles_folder_path + "/" + np.random.choice(os.listdir(self.load_profiles_folder_path))

            for metric in epJSON_object['OutputControl:Files']['OutputControl:Files 1'].keys():
                epJSON_object['OutputControl:Files']['OutputControl:Files 1'][metric] = "No"
        
        # The new modify epjson file is writed.
        env_config["epjson_path"] = os.path.join(self.temp_dir, f"temp-{os.getpid()}.epJSON")
        # The new modify epjson file is writed.
        with open(env_config["epjson_path"], 'w') as fp:
            json.dump(epJSON_object, fp, sort_keys=False, indent=4)
        
        if self.first:
            self.first = False
            epjson_path = os.path.join("C:/Users/grhen/Documents/GitHub/SimpleCases/configurations", "task1.epJSON")
            with open(epjson_path, 'w') as fp:
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
