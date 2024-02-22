"""Utilities and methods to configurate the execution of the episode in EnergyPlus with RLlib.
"""
import numpy as np
import os
import json
try:
    from tools.weather_utils import weather_file
except ModuleNotFoundError: # This alternative is used when the code is used in Google Colab
    from natural_ventilation_EP_RLlib.tools.weather_utils import weather_file

def episode_epJSON(env_config: dict):
    """This method define the properties of the episode. Changing some properties as weather or 
    Run Time Period, and defining others fix properties as volumen or window area relation.
    
    Args:
        env_config (dict): Environment configuration.
        
    Return:
        dict: The method returns the env_config with modifications.
    """
    if env_config.get('epjson', False) == False:
        env_config = epJSON_path(env_config)
        # If the path to epjson is not already set, it is set here.
    with open(env_config['epjson']) as file:
        epJSON_object: dict = json.load(file)
        # Establish the epJSON Object, it will be manipulated to modify the building model.
    
    epJSON_object['Building'][next(iter(epJSON_object['Building']))]['north_axis'] = env_config['rotation']
    # The building is oriented as is possitioned in the land.
    
    ObjectName = next(iter(epJSON_object['RunPeriod']))  
    epJSON_object["RunPeriod"][ObjectName]["begin_month"] = 1
    epJSON_object["RunPeriod"][ObjectName]["begin_day_of_month"] = 1
    epJSON_object["RunPeriod"][ObjectName]["end_month"] = 12
    epJSON_object["RunPeriod"][ObjectName]["end_day_of_month"] = 31
    
    env_config['construction_u_factor'] = u_factor(epJSON_object)
    # The global U factor is calculated.
    
    for key in [key for key in epJSON_object["InternalMass"].keys()]:
        epJSON_object["InternalMass"][key]["surface_area"] = np.random.randint(10,40) if not env_config['is_test'] else 15
        # The internal thermal mass is modified.
    
    env_config['inercial_mass'] = inertial_mass(epJSON_object)
    # The total inertial thermal mass is calculated.
    
    env_config['E_max'] = (0.5+(0.5 - 0.08)*np.random.random_sample())*6 if not env_config['is_test'] else 2.5
    HVAC_names = [key for key in epJSON_object["ZoneHVAC:IdealLoadsAirSystem"].keys()]
    for hvac in range(len(HVAC_names)):
        epJSON_object["ZoneHVAC:IdealLoadsAirSystem"][HVAC_names[hvac]]["maximum_sensible_heating_capacity"] = env_config['E_max']
        epJSON_object["ZoneHVAC:IdealLoadsAirSystem"][HVAC_names[hvac]]["maximum_total_cooling_capacity"] = env_config['E_max']
        # The limit capacity of bouth cooling and heating are changed.
    
    env_config["epjson"] = f"{env_config['epjson_output_folder']}/model-{env_config['episode']:08}-{os.getpid():05}.epJSON"
    with open(env_config["epjson"], 'w') as fp:
        json.dump(epJSON_object, fp, sort_keys=False, indent=4)
        # The new modify epjson file is writed.
    
    env_config['epw'],env_config['latitud'], env_config['longitud'], env_config['altitud'] = weather_file(env_config)
    # Assign the epw path and the correspondent latitude, longitude, and altitude.
    return env_config

def epJSON_path(env_config: dict):
    """This method define the path to the epJSON file to be simulated.
    
    Args:
        env_config (dict): Environment configuration that must contain:
            'epjson_folderpath'
            'building_name'
        
    Return:
        dict: The method returns the env_config with modifications.
    """
    env_config['epjson'] = env_config['epjson_folderpath']+'/'+env_config['building_name']+'.epJSON'
    return env_config

def inertial_mass(epJSON_object: dict[str,dict]):
    """_summary_

    Args:
        epJSON_object (dict[str,dict]): _description_

    Returns:
        _type_: _description_
    """
    # se define una lista para almacenar
    masas_termicas = []
    
    building_surfaces = [key for key in epJSON_object["BuildingSurface:Detailed"].keys()]
    # se obtienen los nombres de las superficies de la envolvente
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
    M_total = 0
    for m in range(0,len(masas_termicas)-1,1):
        M_total += masas_termicas[m]
    
    return M_total
    
def u_factor(epJSON_object: dict[str,dict]):
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

    # Cálculo de U-Factor en W/°C
    u_factor = 0
    for n in range(0, len(areas)-1,1):
        u_factor =+ areas[n]/resistences[n]
    
    return u_factor

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