"""Utilities and methods to configurate the execution of the episode in EnergyPlus with RLlib.
"""
import numpy as np
import os
import json
from typing import List, Dict

def random_building_config(env_config:dict):
    """This method define the path to the epJSON file.

    Args:
        env_config (dict): Environment configuration.

    Return:
        dict: The method returns the env_config with modifications.
        
    Example:
    
        env_config = {
            ...
            'episode_config': {
                'epjson_files_folder_path': 'path/to/epjson/files',
                'id_epjson_file': None
            }
        }
    """
    if env_config.get('epjson', False) == False:
        env_config['epjson'] = ''
    
    # Define the properties of the method
    epjson_files_folder_path: str = None # This is mandatory
    id_epjson_file: int = None # The default is a random int, but the user can indicated a diferent number
    
    # For each property, assign the default or the user input
    epjson_files_folder_path = env_config['episode_config'].get('epjson_files_folder_path', None)
    if epjson_files_folder_path is None:
        ValueError('epjson_files_folder_path is not defined')
    
    id_epjson_file = env_config['episode_config'].get('id_epjson_file', None)
    if id_epjson_file is None:
        id_epjson_file = np.random.randint(0, len(os.listdir(epjson_files_folder_path)))
    
    # The path to the epjson file is defined
    env_config['epjson'] = os.path.join(epjson_files_folder_path, os.listdir(epjson_files_folder_path)[id_epjson_file])
    
    return env_config

def random_weather_config(env_config:Dict) -> str:
    """This method define the path to the epJSON file.

    Args:
        env_config (dict): Environment configuration.

    Return:
        str: The method returns the epw path.
        
    Example:
    
        env_config = {
            ...
            'episode_config': {
                'epw_files_folder_path': 'path/to/epw/files',
                'id_epw_file': None
            }
        }
    """
    if env_config.get('epw', False) == False:
        env_config['epw'] = ''
    
    # Define the properties of the method
    epw_files_folder_path: str = None # This is mandatory
    id_epw_file: int = None # The default is a random int, but the user can indicated a diferent number
    
    # For each property, assign the default or the user input
    epw_files_folder_path = env_config['episode_config'].get('epw_files_folder_path', None)
    if epw_files_folder_path is None:
        ValueError('epw_files_folder_path is not defined')
    
    id_epw_file = env_config['episode_config'].get('id_epw_file', None)
    if id_epw_file is None:
        id_epw_file = np.random.randint(0, len(os.listdir(epw_files_folder_path)))
    
    # The path to the epjson file is defined
    epw_path = os.path.join(epw_files_folder_path, os.listdir(epw_files_folder_path)[id_epw_file])
    
    return epw_path


def episode_epJSON(env_config:Dict) -> Dict:
    """This method define the properties of the episode. Changing some properties as weather or 
    Run Time Period, and defining others fix properties as volumen or window area relation.
    
    Args:
        env_config (dict): Environment configuration.
        
    Return:
        dict: The method returns the env_config with modifications.
    """
    # If the path to epjson is not set, arraise a error.
    if not env_config['episode_config'].get('epjson', False):
        raise ValueError('epjson is not defined')
    
    # Establish the epJSON Object, it will be manipulated to modify the building model.
    with open(env_config['episode_config']['epjson']) as file:
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
    env_config['building_properties']['inercial_mass'] = inertial_mass(epJSON_object)
    
    # The global U factor is calculated.
    env_config['building_properties']['construction_u_factor'] = u_factor(epJSON_object)
    
    # The limit capacity of bouth cooling and heating are changed.
    E_cool_ref = env_config['building_properties']['E_cool_ref'] = (3000 - 100)*np.random.random_sample() + 100
    E_heat_ref = env_config['building_properties']['E_heat_ref'] = (3000 - 100)*np.random.random_sample() + 100
    HVAC_names = [key for key in epJSON_object["ZoneHVAC:IdealLoadsAirSystem"].keys()]
    number_of_timesteps_per_hour = epJSON_object['Timestep']['Timestep 1']['number_of_timesteps_per_hour']
    for hvac in range(len(HVAC_names)):
        epJSON_object["ZoneHVAC:IdealLoadsAirSystem"][HVAC_names[hvac]]["maximum_sensible_heating_capacity"] = E_heat_ref
        epJSON_object["ZoneHVAC:IdealLoadsAirSystem"][HVAC_names[hvac]]["maximum_total_cooling_capacity"] = E_cool_ref
    if env_config.get('reward_function_config', False):
        if env_config['reward_function_config'].get('cooling_energy_ref', False):
            env_config['reward_function_config']['cooling_energy_ref'] = E_cool_ref*number_of_timesteps_per_hour*3600
        if env_config['reward_function_config'].get('heating_energy_ref', False):
            env_config['reward_function_config']['heating_energy_ref'] = E_heat_ref*number_of_timesteps_per_hour*3600
      
            
    # Select the schedule file for loads
    env_config['epw'] = random_weather_config(env_config)
    
    # The new modify epjson file is writed in the results folder created by RLlib
    # If don't exist, reate a folder call 'models' into env_config['episode_config']['epjson_files_folder_path']
    if not os.path.exists(f"{env_config['episode_config']['epjson_files_folder_path']}/models"):
        os.makedirs(f"{env_config['episode_config']['epjson_files_folder_path']}/models")
    
    env_config["epjson"] = f"{env_config['episode_config']['epjson_files_folder_path']}/models/model-{env_config['episode']:08}-{os.getpid():05}.epJSON"
    with open(env_config["epjson"], 'w') as fp:
        json.dump(epJSON_object, fp, sort_keys=False, indent=4)
        # The new modify epjson file is writed.
    
    return env_config

def inertial_mass_calculation(env_config:Dict) -> float:
    """The function reads the epjson file path from the env_config dictionary. If the epjson
    key is not present, it raises a ValueError. Otherwise, it opens the specified epjson
    file, loads its contents as a Python dictionary (epJSON_object), and then calls the 
    inertial_mass function (not shown in the provided code) with epJSON_object
    as an argument. The result of inertial_mass is returned as a float value.

    Args:
        env_config (dict): A dictionary containing the environment configuration, including the 
            episode_config key, which should have an epjson key with the path to the 
            EnergyPlus JSON file.

    Returns:
        float: The inertial mass calculated based on the contents of the epjson file.
    
    Example:
        env_config = {
            'episode_config': {
                'epjson': 'path/to/epjson_file.json'
            }
        }

        inertial_mass_value = inertial_mass_calculation(env_config)
        print(f"Inertial mass: {inertial_mass_value}")

    """
    # If the path to epjson is not set, arraise a error.
    if not env_config.get('epjson', False):
        raise ValueError('epjson is not defined')
    
    with open(env_config['epjson']) as file:
        epJSON_object: dict = json.load(file)

    return inertial_mass(epJSON_object)

def inertial_mass(epJSON_object:Dict[str,Dict]) -> float:
    """The inertial_mass function calculates the total thermal mass of a building based 
    on the construction materials and surface areas specified in an EnergyPlus JSON 
    object.

    Args:
        epJSON_object (dict[str,dict]): A dictionary containing information about the 
        building surfaces, materials, and constructions in the EnergyPlus JSON format. 
        The keys of the outer dictionary represent different object types (e.g., 
        "BuildingSurface:Detailed", "Material", "Construction"), and the values are 
        dictionaries containing the properties of each object instance.

    Returns:
        float: The total thermal mass of the building in J/°C (Joules per degree Celsius). 
        The thermal mass represents the amount of heat energy required to raise the 
        temperature of the building's construction materials by one degree Celsius.
    """
    # se define una lista para almacenar
    masas_termicas = []
    
    building_surfaces = []
    if epJSON_object.get("BuildingSurface:Detailed", False):
        building_surfaces = [key for key in epJSON_object["BuildingSurface:Detailed"].keys()]
    # se obtienen los nombres de las superficies de la envolvente
    internal_mass_surfaces = []
    if epJSON_object.get("InternalMass", False):
        internal_mass_surfaces = [key for key in epJSON_object["InternalMass"].keys()]
    
    all_building_keys = [key for key in epJSON_object.keys()]
    all_material_list = ['Material','Material:NoMass','Material:InfraredTransparent','Material:AirGap',
        'Material:RoofVegetation','WindowMaterial:SimpleGlazingSystem','WindowMaterial:Glazing',
        'WindowMaterial:GlazingGroup:Thermochromic','WindowMaterial:Glazing:RefractionExtinctionMethod',
        'WindowMaterial:Gas','WindowGap:SupportPillar','WindowGap:DeflectionState',
        'WindowMaterial:GasMixture','WindowMaterial:Gap'
    ]
    materials_dict = {}
    for material in all_material_list:
        if material in all_building_keys:
            materials_dict[material] = epJSON_object[material].keys()
    # se obtienen los nombres de los diferentes tipos de materiales
    
    # lazo para consultar cada superficie de la envolvente
    for surface in building_surfaces:
        # se calcula el área de la superficie
        area = material_area(epJSON_object,surface)
        # se identifica la consutrucción
        s_construction = epJSON_object['BuildingSurface:Detailed'][surface]['construction_name']
        
        # se obtiene la densidad del materia: \rho[kg/m3]
        # se obtiene el calor específico del material: C[J/kg°C]
        # se calcula el volumen que ocupa el material: V[m3]=area*thickness
        # se calcula la masa térmica: M[J/°C] = \rho[kg/m3] * C[J/kg°C] * V[m3]
        
        # se establece un lazo para calcular la masa térmica de cada capa
        m_surface = 0
        layers = [key for key in epJSON_object['Construction'][s_construction].keys()]
        for layer in layers:
            material = epJSON_object['Construction'][s_construction][layer]
            material_list = find_dict_key_by_nested_key(
                material,
                materials_dict
            )
            # se obtiene el espesor y la conductividad térmica del material de la capa
            if material_list == 'Material:NoMass' or material_list == 'Material:AirGap' or material_list == 'Material:InfraredTransparent' or material_list == 'WindowMaterial:Gas':
                m_capa = 0
            else:
                espesor_capa = epJSON_object[material_list][material]['thickness']
                calor_especifico_capa = epJSON_object[material_list][material]['specific_heat']
                densidad_capa = epJSON_object[material_list][material]['density']
                m_capa = area * espesor_capa * calor_especifico_capa * densidad_capa

            # se suma la resistencia de la superficie
            m_surface += m_capa
        # se guarda la resistencia de la superficie
        masas_termicas.append(m_surface)
    
    # se suma la masa interna asignada
    for surface in internal_mass_surfaces:
        # se calcula el área de la superficie
        area = epJSON_object['InternalMass'][surface]['surface_area']
        # se identifica la consutrucción
        s_construction = epJSON_object['InternalMass'][surface]['construction_name']
        
        # se obtiene la densidad del materia: \rho[kg/m3]
        # se obtiene el calor específico del material: C[J/kg°C]
        # se calcula el volumen que ocupa el material: V[m3]=area*thickness
        # se calcula la masa térmica: M[J/°C] = \rho[kg/m3] * C[J/kg°C] * V[m3]
        
        # se establece un lazo para calcular la masa térmica de cada capa
        m_surface = 0
        layers = [key for key in epJSON_object['Construction'][s_construction].keys()]
        for layer in layers:
            material = epJSON_object['Construction'][s_construction][layer]
            material_list = find_dict_key_by_nested_key(
                material,
                materials_dict
            )
            # se obtiene el espesor y la conductividad térmica del material de la capa
            if material_list == 'Material:NoMass' or material_list == 'Material:AirGap' or material_list == 'Material:InfraredTransparent' or material_list == 'WindowMaterial:Gas':
                m_capa = 0
            else:
                espesor_capa = epJSON_object[material_list][material]['thickness']
                calor_especifico_capa = epJSON_object[material_list][material]['specific_heat']
                densidad_capa = epJSON_object[material_list][material]['density']
                m_capa = area * espesor_capa * calor_especifico_capa * densidad_capa

            # se suma la resistencia de la superficie
            m_surface += m_capa
        # se guarda la resistencia de la superficie
        masas_termicas.append(m_surface)
    
    # Cálculo de la masa termica total
    M_total = 0.
    for m in range(0,len(masas_termicas)-1,1):
        M_total += masas_termicas[m]
    
    return M_total

def u_factor_calculation(env_config:Dict) -> float:
    """The u_factor_calculation function calculates the U-factor (a measure of heat 
    transfer through a building element) based on the provided environment configuration 
    (env_config).

    Args:
        env_config (dict): A dictionary containing the environment configuration, including 
        the episode_config key, which should have an epjson key with the path to the 
        EnergyPlus JSON file.

    Returns:
        float: The calculated U-factor value.
        
    Example:
        env_config = {
            'episode_config': {
                'epjson': 'path/to/epjson_file.json'
            }
        }

        u_factor_value = u_factor_calculation(env_config)
        print(f"U-factor: {u_factor_value}")

    """
    # If the path to epjson is not set, arraise a error.
    if not env_config.get('epjson', False):
        raise ValueError('epjson is not defined')
    with open(env_config['epjson']) as file:
        epJSON_object: dict = json.load(file)
    #  Calculate the u_factor
    return u_factor(epJSON_object)

def u_factor(epJSON_object:Dict[str,Dict]) -> float:
    """This function select all the building surfaces and fenestration surfaces and calculate the
    global U-factor of the building, like EnergyPlus does.
    """
    # se define una lista para almacenar las resistencias de cada supercie
    resistences = []
    areas = []
    # se obtienen los nombres de las superficies de la envolvente
    building_surfaces = [key for key in epJSON_object['BuildingSurface:Detailed'].keys()]
    fenestration_surfaces = [key for key in epJSON_object['FenestrationSurface:Detailed'].keys()]
    # se obtienen los nombres de los diferentes tipos de materiales
    
    all_building_keys = [key for key in epJSON_object.keys()]
    all_material_list = ['Material','Material:NoMass','Material:InfraredTransparent','Material:AirGap',
        'Material:RoofVegetation','WindowMaterial:SimpleGlazingSystem','WindowMaterial:Glazing',
        'WindowMaterial:GlazingGroup:Thermochromic','WindowMaterial:Glazing:RefractionExtinctionMethod',
        'WindowMaterial:Gas','WindowGap:SupportPillar','WindowGap:DeflectionState',
        'WindowMaterial:GasMixture','WindowMaterial:Gap'
    ]
    materials_dict = {}
    for material in all_material_list:
        if material in all_building_keys:
            materials_dict[material] = epJSON_object[material].keys()
    # se obtienen los nombres de los diferentes tipos de materiales
    
    # lazo para consultar cada superficie de la envolvente
    for surface in building_surfaces:
        # se calcula el área de la superficie
        areas.append(material_area(epJSON_object,surface))
        # se identifica la consutrucción
        s_construction = epJSON_object['BuildingSurface:Detailed'][surface]['construction_name']
        # se establece un lazo para calcular la resistencia de cada capa
        r_surface = 0
        layers = [key for key in epJSON_object['Construction'][s_construction].keys()]
        for layer in layers:
            material = epJSON_object['Construction'][s_construction][layer]
            material_list = find_dict_key_by_nested_key(
                material,
                materials_dict
            )
            # se obtiene el espesor y la conductividad térmica del material de la capa
            if material_list == 'Material:NoMass' or material_list == 'Material:AirGap':
                r_capa = epJSON_object[material_list][material]['thermal_resistance']
            elif material_list == 'Material:InfraredTransparent':
                r_capa = 0
            elif material_list == 'WindowMaterial:Gas':
                espesor_capa = epJSON_object[material_list][material]['thickness']
                if epJSON_object[material_list][material]['gas_type'] == 'Air':
                    conductividad_capa = 0.0257
                elif epJSON_object[material_list][material]['gas_type'] == 'Argon':
                    conductividad_capa = 0.0162
                elif epJSON_object[material_list][material]['gas_type'] == 'Xenon':
                    conductividad_capa = 0.00576
                elif epJSON_object[material_list][material]['gas_type'] == 'Krypton':
                    conductividad_capa = 0.00943
                else:
                    print('El nombre del gas no corresponde con los que pueden utilizarse: Air, Argon, Xenon, Krypton.')
                    NameError
                r_capa = espesor_capa/conductividad_capa
            else:
                espesor_capa = epJSON_object[material_list][material]['thickness']
                conductividad_capa = epJSON_object[material_list][material]['conductivity']
                r_capa = espesor_capa/conductividad_capa

            # se suma la resistencia de la superficie
            r_surface += r_capa
        # se guarda la resistencia de la superficie
        resistences.append(r_surface)
        
    # lazo para consultar cada superfice de fenestración
    for fenestration in fenestration_surfaces:
        # se calcula el área de la superficie
        areas.append(fenestration_area(epJSON_object, fenestration))
        # se identifica la consutrucción
        s_construction = epJSON_object['FenestrationSurface:Detailed'][fenestration]['construction_name']
        # se establece un lazo para calcular la resistencia de cada capa
        r_surface = 0
        u_factor_windows = 0
        u_factor_window = 0
        layers = [key for key in epJSON_object['Construction'][s_construction].keys()]
        for layer in layers:
            material = epJSON_object['Construction'][s_construction][layer]
            material_list = find_dict_key_by_nested_key(
                material,
                materials_dict
            )
            # se obtiene el espesor y la conductividad térmica del material de la capa
            if material_list == 'Material:NoMass' or material_list == 'Material:AirGap':
                r_capa = epJSON_object[material_list][material]['thermal_resistance']
            elif material_list == 'Material:InfraredTransparent':
                r_capa = 0
            elif material_list == 'WindowMaterial:Gas':
                espesor_capa = epJSON_object[material_list][material]['thickness']
                if epJSON_object[material_list][material]['gas_type'] == 'Air':
                    conductividad_capa = 0.0257
                elif epJSON_object[material_list][material]['gas_type'] == 'Argon':
                    conductividad_capa = 0.0162
                elif epJSON_object[material_list][material]['gas_type'] == 'Xenon':
                    conductividad_capa = 0.00576
                elif epJSON_object[material_list][material]['gas_type'] == 'Krypton':
                    conductividad_capa = 0.00943
                else:
                    print('El nombre del gas no corresponde con los que pueden utilizarse: Air, Argon, Xenon, Krypton.')
                    NameError
                r_capa = espesor_capa/conductividad_capa
            elif material_list == 'WindowMaterial:SimpleGlazingSystem':
                u_factor_window = epJSON_object[material_list][material]['u_factor']
            else:
                espesor_capa = epJSON_object[material_list][material]['thickness']
                conductividad_capa = epJSON_object[material_list][material]['conductivity']
                r_capa = espesor_capa/conductividad_capa

            # se suma la resistencia de la superficie
            r_surface += r_capa
            u_factor_windows += u_factor_window
        # se guarda la resistencia de la superficie
        resistences.append(r_surface)

    # Cálculo de U-Factor en W/°C
    u_factor = 0.
    for n in range(0, len(areas)-1,1):
        u_factor =+ areas[n]/resistences[n]
    
    return u_factor + u_factor_windows

def material_area(epJSON_object, nombre_superficie):
    """_summary_

    Args:
        epJSON_object (_type_): _description_
        nombre_superficie (_type_): _description_

    Returns:
        _type_: _description_
    """
    # Calcula dos vectores que forman dos lados del cuadrilátero
    vector1 = [
        epJSON_object['BuildingSurface:Detailed'][nombre_superficie]['vertices'][1]['vertex_x_coordinate'] - epJSON_object['BuildingSurface:Detailed'][nombre_superficie]['vertices'][0]['vertex_x_coordinate'],
        epJSON_object['BuildingSurface:Detailed'][nombre_superficie]['vertices'][1]['vertex_y_coordinate'] - epJSON_object['BuildingSurface:Detailed'][nombre_superficie]['vertices'][0]['vertex_y_coordinate'],
        epJSON_object['BuildingSurface:Detailed'][nombre_superficie]['vertices'][1]['vertex_z_coordinate'] - epJSON_object['BuildingSurface:Detailed'][nombre_superficie]['vertices'][0]['vertex_z_coordinate']
    ]
    vector2 = [
        epJSON_object['BuildingSurface:Detailed'][nombre_superficie]['vertices'][2]['vertex_x_coordinate'] - epJSON_object['BuildingSurface:Detailed'][nombre_superficie]['vertices'][0]['vertex_x_coordinate'],
        epJSON_object['BuildingSurface:Detailed'][nombre_superficie]['vertices'][2]['vertex_y_coordinate'] - epJSON_object['BuildingSurface:Detailed'][nombre_superficie]['vertices'][0]['vertex_y_coordinate'],
        epJSON_object['BuildingSurface:Detailed'][nombre_superficie]['vertices'][2]['vertex_z_coordinate'] - epJSON_object['BuildingSurface:Detailed'][nombre_superficie]['vertices'][0]['vertex_z_coordinate']
    ]

    # Calcula el producto vectorial entre los dos vectores
    producto_vectorial = [
        vector1[1] * vector2[2] - vector1[2] * vector2[1],
        vector1[2] * vector2[0] - vector1[0] * vector2[2],
        vector1[0] * vector2[1] - vector1[1] * vector2[0]
    ]

    # Calcula el módulo del producto vectorial como el área del cuadrilátero
    area = 0.5 * (abs(producto_vectorial[0]) + abs(producto_vectorial[1]) + abs(producto_vectorial[2]))
    return area

def fenestration_area(epJSON_object, fenestration):
    """_summary_

    Args:
        epJSON_object (_type_): _description_
        fenestration (_type_): _description_

    Returns:
        _type_: _description_
    """
    
    # Calcula dos vectores que forman dos lados del cuadrilátero
    vector1 = [
        epJSON_object['FenestrationSurface:Detailed'][fenestration]['vertex_2_x_coordinate'] - epJSON_object['FenestrationSurface:Detailed'][fenestration]['vertex_1_x_coordinate'],
        epJSON_object['FenestrationSurface:Detailed'][fenestration]['vertex_2_y_coordinate'] - epJSON_object['FenestrationSurface:Detailed'][fenestration]['vertex_1_y_coordinate'],
        epJSON_object['FenestrationSurface:Detailed'][fenestration]['vertex_2_z_coordinate'] - epJSON_object['FenestrationSurface:Detailed'][fenestration]['vertex_1_z_coordinate']
    ]
    vector2 = [
        epJSON_object['FenestrationSurface:Detailed'][fenestration]['vertex_3_x_coordinate'] - epJSON_object['FenestrationSurface:Detailed'][fenestration]['vertex_1_x_coordinate'],
        epJSON_object['FenestrationSurface:Detailed'][fenestration]['vertex_3_y_coordinate'] - epJSON_object['FenestrationSurface:Detailed'][fenestration]['vertex_1_y_coordinate'],
        epJSON_object['FenestrationSurface:Detailed'][fenestration]['vertex_3_z_coordinate'] - epJSON_object['FenestrationSurface:Detailed'][fenestration]['vertex_1_z_coordinate']
    ]

    # Calcula el producto vectorial entre los dos vectores
    producto_vectorial = [
        vector1[1] * vector2[2] - vector1[2] * vector2[1],
        vector1[2] * vector2[0] - vector1[0] * vector2[2],
        vector1[0] * vector2[1] - vector1[1] * vector2[0]
    ]

    # Calcula el módulo del producto vectorial como el área del cuadrilátero
    area = 0.5 * (abs(producto_vectorial[0]) + abs(producto_vectorial[1]) + abs(producto_vectorial[2]))
    return area

def find_dict_key_by_nested_key(key, lists_dict):
    """_summary_

    Args:
        key (_type_): _description_
        lists_dict (_type_): _description_

    Returns:
        _type_: _description_
    """
    for dict_key, lst in lists_dict.items():
        if key in lst:
            return dict_key
    return None

def window_size_epJSON(epJSON_object, window:str, area_ventana:float):
    """Given a epJSON_object, return another epJSON_object with diferent size of windows.

    Args:
        epJSON_object (json): _description_
        window_name (str): _description_
        factor (float): _description_

    Returns:
        json: Devuelve el objeto epJSON modificado.
    """
    window_vertexs = {}
    # se extraen los valores de los vértices de cada ventana según el epJSON
    window_vertexs['v1x'] = epJSON_object['FenestrationSurface:Detailed'][window]['vertex_1_x_coordinate']
    window_vertexs['v1y'] = epJSON_object['FenestrationSurface:Detailed'][window]['vertex_1_y_coordinate']
    window_vertexs['v1z'] = epJSON_object['FenestrationSurface:Detailed'][window]['vertex_1_z_coordinate']
    window_vertexs['v2x'] = epJSON_object['FenestrationSurface:Detailed'][window]['vertex_2_x_coordinate']
    window_vertexs['v2y'] = epJSON_object['FenestrationSurface:Detailed'][window]['vertex_2_y_coordinate']
    window_vertexs['v2z'] = epJSON_object['FenestrationSurface:Detailed'][window]['vertex_2_z_coordinate']
    window_vertexs['v3x'] = epJSON_object['FenestrationSurface:Detailed'][window]['vertex_3_x_coordinate']
    window_vertexs['v3y'] = epJSON_object['FenestrationSurface:Detailed'][window]['vertex_3_y_coordinate']
    window_vertexs['v3z'] = epJSON_object['FenestrationSurface:Detailed'][window]['vertex_3_z_coordinate']
    window_vertexs['v4x'] = epJSON_object['FenestrationSurface:Detailed'][window]['vertex_4_x_coordinate']
    window_vertexs['v4y'] = epJSON_object['FenestrationSurface:Detailed'][window]['vertex_4_y_coordinate']
    window_vertexs['v4z'] = epJSON_object['FenestrationSurface:Detailed'][window]['vertex_4_z_coordinate']
    
    L = [] # agrupa los vertices en forma de lista: [[x1,y1,z1],[x2,y2,z2],[x3,y3,z3],[x4,y4,z4]]
    for l in range(1,5,1):
        vertex_x = 'v'+str(l)+'x'
        vertex_y = 'v'+str(l)+'y'
        vertex_z = 'v'+str(l)+'z'
        L.append([window_vertexs[vertex_x], window_vertexs[vertex_y], window_vertexs[vertex_z]])

    # Se calcula el factor de escala de la ventana
    area_ventana_old = fenestration_area(epJSON_object, window)
    factor_escala = area_ventana/area_ventana_old
    centro = calcular_centro(L)
    ventana_escalada = []
    for punto in L:
        nuevo_punto = [centro[0] + (punto[0] - centro[0]) * factor_escala**(1/2),
                    centro[1] + (punto[1] - centro[1]) * factor_escala**(1/2),
                    centro[2] + (punto[2] - centro[2]) * factor_escala**(1/2)]
        ventana_escalada.append(nuevo_punto)
    
    for l in range(1,5,1):
        epJSON_object['FenestrationSurface:Detailed'][window]['vertex_'+str(l)+'_x_coordinate'] = ventana_escalada[l-1][0]
        epJSON_object['FenestrationSurface:Detailed'][window]['vertex_'+str(l)+'_y_coordinate'] = ventana_escalada[l-1][1]
        epJSON_object['FenestrationSurface:Detailed'][window]['vertex_'+str(l)+'_z_coordinate'] = ventana_escalada[l-1][2]

def calcular_centro(ventana):
    # Calcula el centro de la ventana
    centro = [(ventana[0][0] + ventana[1][0] + ventana[2][0] + ventana[3][0]) / 4,
            (ventana[0][1] + ventana[1][1] + ventana[2][1] + ventana[3][1]) / 4,
            (ventana[0][2] + ventana[1][2] + ventana[2][2] + ventana[3][2]) / 4]
    return centro


def building_dimension(epJSON_object, h:float, w:float, l:float, window_area_relation:List):
    """_summary_

    Args:
        epJSON_object (_type_): _description_
        nombre_superficie (_type_): _description_

    Returns:
        _type_: _description_
    """
    # Reshape the position vectors
    surface_coordenates = {
        'wall_north': [[0,0,h],[0,0,0],[w,0,0],[w,0,h]],
        'wall_east': [[w,0,h],[w,0,0],[w,l,0],[w,l,h]],
        'wall_south': [[w,l,h],[w,l,0],[0,l,0],[0,l,h]],
        'wall_west': [[0,l,h],[0,l,0],[0,0,0],[0,0,h]],
        'roof': [[w,0,h],[w,l,h],[0,l,h],[0,0,h]],
        'floor': [[w,l,0],[w,0,0],[0,0,0],[0,l,0]]
    }
    coordinate = ['x', 'y', 'z']
    for surface_name in [key for key in surface_coordenates.keys()]:
        # Iterate over the four vertices of the surface
        for _ in range(4):
            for xyz in range(3):
                epJSON_object['BuildingSurface:Detailed'][surface_name]['vertices'][_]['vertex_'+coordinate[xyz]+'_coordinate'] = surface_coordenates[surface_name][_][xyz]
    
    north_window_proportion = window_area_relation[0]
    east_window_proportion = window_area_relation[1]
    south_window_proportion = window_area_relation[2]
    west_window_proportion = window_area_relation[3]
    
    window_coordinates = {
        'window_north': [
            [0+(w*north_window_proportion)/2,0,h*0.9],
            [0+(w*north_window_proportion)/2,0,0.1],
            [w-(w*north_window_proportion)/2,0,0.1],
            [w-(w*north_window_proportion)/2,0,h*0.9]
        ],
        'window_east': [
            [w,0+(l*east_window_proportion)/2,h*0.9],
            [w,0+(l*east_window_proportion)/2,0.1],
            [w,l-(l*east_window_proportion)/2,0.1],
            [w,l-(l*east_window_proportion)/2,h*0.9]
        ],
        'window_south': [
            [w-(w*south_window_proportion)/2,l,h*0.9],
            [w-(w*south_window_proportion)/2,l,0.1],
            [0+(w*south_window_proportion)/2,l,0.1],
            [0+(w*south_window_proportion)/2,l,h*0.9]
        ],
        'window_west': [
            [0,l-(l*west_window_proportion)/2,h*0.9],
            [0,l-(l*west_window_proportion)/2,0.1],
            [0,0+(l*west_window_proportion)/2,0.1],
            [0,0+(l*west_window_proportion)/2,h*0.9]
        ],
    }
    
    for window_name in [key for key in window_coordinates.keys()]:
        # Iterate over the four vertices of the surface
        for _ in range(1,5,1):
            for xyz in range(3):
                epJSON_object['FenestrationSurface:Detailed'][window_name]['vertex_'+str(_)+'_'+coordinate[xyz]+'_coordinate'] = window_coordinates[window_name][_-1][xyz]
        