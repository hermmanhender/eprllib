"""
General utilities
==================

Work in progress...
"""
from typing import Set, Dict, List, get_origin, get_args, Union, _SpecialGenericAlias

import os
import json
import pandas as pd
import numpy as np
import datetime


def variable_checking(
    epJSON_file:str,
) -> Set:
    """
    This function check if the epJSON file has the required variables.

    Args:
        epJSON_file(str): path to the epJSON file.

    Return:
        set: list of missing variables.
    """
    pass

from typing import get_origin, get_args, Union, _SpecialGenericAlias

# Importar las clases que deseas validar
from eprllib.AgentsConnectors.BaseConnector import BaseConnector
from eprllib.Episodes.BaseEpisode import BaseEpisode
from eprllib.Agents.Rewards.BaseReward import BaseReward
from eprllib.Agents.Triggers.BaseTrigger import BaseTrigger
from eprllib.Agents.Filters.BaseFilter import BaseFilter

def validate_properties(obj, expected_types):
    """
    Enhanced version that supports Union types and optional properties
    
    Args:
        obj: The object to validate
        expected_types: A dictionary mapping property names to either:
                       - A type
                       - A tuple of types (Union)
                       - (type, bool) tuple where bool indicates if property is optional
    """
    errors = []
    is_valid = True
    
    for prop_name, type_spec in expected_types.items():
        # Handle optional properties
        is_optional = False
        expected_type = type_spec
        
        if isinstance(type_spec, tuple) and len(type_spec) == 2 and isinstance(type_spec[1], bool):
            expected_type, is_optional = type_spec
            
        # Check if property exists
        if not hasattr(obj, prop_name):
            if not is_optional:
                errors.append(f"Missing required property: {prop_name}")
                is_valid = False
            continue
            
        actual_value = getattr(obj, prop_name)
        
        # Handle None for optional properties
        if actual_value is None and is_optional:
            continue
            
        # Handle union types
        valid_types = (expected_type,) if isinstance(expected_type, type) else expected_type
        
        def is_instance_of_type(value, types):
            if isinstance(types, _SpecialGenericAlias):
                types = (types,)
            for typ in types:
                origin = get_origin(typ)
                if origin:
                    if origin is Union:
                        if any(is_instance_of_type(value, get_args(typ))):
                            return True
                    elif isinstance(value, origin):
                        return True
                elif isinstance(value, typ):
                    return True
            return False

        if not is_instance_of_type(actual_value, valid_types):
            errors.append(
                f"Property '{prop_name}' has incorrect type. "
                f"Expected {', '.join(t.__name__ if isinstance(t, type) else str(t) for t in valid_types)}, "
                f"got {type(actual_value).__name__}"
            )
            is_valid = False
    
    return is_valid, errors

def inertial_mass_calculation(env_config:Dict) -> float:
    """
    The function reads the epjson file path from the env_config dictionary. If the epjson
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
    if not env_config.get('epjson_path', False):
        raise ValueError('epjson_path is not defined')
    
    with open(env_config['epjson_path']) as file:
        epJSON_object: dict = json.load(file)

    return effective_thermal_capacity(epJSON_object)

def effective_thermal_capacity(epJSON_object:Dict[str,Dict]) -> float:
    """
    The `effective_thermal_capacity` function calculates the total effective thermal capacity of a building based 
    on the construction materials and surface areas specified in an EnergyPlus JSON 
    object.

    Args:
        epJSON_object (dict[str,dict]): A dictionary containing information about the 
        building surfaces, materials, and constructions in the EnergyPlus JSON format. 
        The keys of the outer dictionary represent different object types (e.g., 
        "BuildingSurface:Detailed", "Material", "Construction"), and the values are 
        dictionaries containing the properties of each object instance.

    Returns:
        float: The total thermal capacity of the building in J/°C (Joules per degree Celsius). 
        The thermal mass represents the amount of heat energy required to raise the 
        temperature of the building's construction materials by one degree Celsius.
    """
    # se define una lista para almacenar
    Cdyn = []
    
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
        # se calcula la capacidad térmica: M[J/°C] = \rho[kg/m3] * C[J/kg°C] * V[m3]
        
        # se establece un lazo para calcular la masa térmica de cada capa
        Cdyn_surface = 0
        layers = [key for key in epJSON_object['Construction'][s_construction].keys()]
        for layer in layers:
            material = epJSON_object['Construction'][s_construction][layer]
            material_list = find_dict_key_by_nested_key(
                material,
                materials_dict
            )
            # se obtiene el espesor y la conductividad térmica del material de la capa
            if material_list == 'Material:NoMass' or material_list == 'Material:AirGap' or material_list == 'Material:InfraredTransparent' or material_list == 'WindowMaterial:Gas':
                Cdyn_capa = 0
            else:
                espesor_capa = epJSON_object[material_list][material]['thickness']
                calor_especifico_capa = epJSON_object[material_list][material]['specific_heat']
                densidad_capa = epJSON_object[material_list][material]['density']
                Cdyn_capa = (area * espesor_capa * densidad_capa) * calor_especifico_capa

            # se suma la resistencia de la superficie
            Cdyn_surface += Cdyn_capa
        # se guarda la resistencia de la superficie
        Cdyn.append(Cdyn_surface)
    
    # se suma la masa interna asignada
    for surface in internal_mass_surfaces:
        # se calcula el área de la superficie
        area = epJSON_object['InternalMass'][surface]['surface_area']
        # se identifica la consutrucción
        s_construction = epJSON_object['InternalMass'][surface]['construction_name']
        
        # se obtiene la densidad del materia: \rho[kg/m3]
        # se obtiene el calor específico del material: C[J/kg°C]
        # se calcula el volumen que ocupa el material: V[m3]=area*thickness
        # se calcula la capacidad térmica: M[J/°C] = \rho[kg/m3] * C[J/kg°C] * V[m3]
        
        # se establece un lazo para calcular la masa térmica de cada capa
        Cdyn_surface = 0
        layers = [key for key in epJSON_object['Construction'][s_construction].keys()]
        for layer in layers:
            material = epJSON_object['Construction'][s_construction][layer]
            material_list = find_dict_key_by_nested_key(
                material,
                materials_dict
            )
            # se obtiene el espesor y la conductividad térmica del material de la capa
            if material_list == 'Material:NoMass' or material_list == 'Material:AirGap' or material_list == 'Material:InfraredTransparent' or material_list == 'WindowMaterial:Gas':
                Cdyn_capa = 0
            else:
                espesor_capa = epJSON_object[material_list][material]['thickness']
                calor_especifico_capa = epJSON_object[material_list][material]['specific_heat']
                densidad_capa = epJSON_object[material_list][material]['density']
                Cdyn_capa = (area * espesor_capa * densidad_capa) * calor_especifico_capa

            # se suma la resistencia de la superficie
            Cdyn_surface += Cdyn_capa
        # se guarda la resistencia de la superficie
        Cdyn.append(Cdyn_surface)
    
    # Cálculo de la masa termica total
    Cdyn_total = 0.
    for m in range(0,len(Cdyn)-1,1):
        Cdyn_total += Cdyn[m]
    
    return Cdyn_total

def u_factor_calculation(env_config:Dict) -> float:
    """
    The u_factor_calculation function calculates the U-factor (a measure of heat 
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
    if not env_config.get('epjson_path', False):
        raise ValueError('epjson_path is not defined')
    with open(env_config['epjson_path']) as file:
        epJSON_object: dict = json.load(file)
    #  Calculate the u_factor
    return u_factor(epJSON_object)

def u_factor(epJSON_object:Dict[str,Dict]) -> float:
    """
    This function select all the building surfaces and fenestration surfaces and calculate the
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
    """
    _summary_

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
    """
    _summary_

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
    """
    _summary_

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
    """
    Given a epJSON_object, return another epJSON_object with diferent size of windows.

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
    """
    _summary_

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
    
    window_coordinates = {}
    
    # add the new coordinates to the windows that exist in the model
    if north_window_proportion > 0:
        window_coordinates.update(
            {
                'window_north': [
                    [0+(w*north_window_proportion)/2,0,h*0.9],
                    [0+(w*north_window_proportion)/2,0,0.1],
                    [w-(w*north_window_proportion)/2,0,0.1],
                    [w-(w*north_window_proportion)/2,0,h*0.9]
                ],
            }
        )
    if east_window_proportion > 0:
        window_coordinates.update(
            {
                'window_east': [
                    [w,0+(l*east_window_proportion)/2,h*0.9],
                    [w,0+(l*east_window_proportion)/2,0.1],
                    [w,l-(l*east_window_proportion)/2,0.1],
                    [w,l-(l*east_window_proportion)/2,h*0.9]
                ],
            }
        )
    if south_window_proportion > 0:
        window_coordinates.update(
            {
                'window_south': [
                    [w-(w*south_window_proportion)/2,l,h*0.9],
                    [w-(w*south_window_proportion)/2,l,0.1],
                    [0+(w*south_window_proportion)/2,l,0.1],
                    [0+(w*south_window_proportion)/2,l,h*0.9]
                ],
            }
        )
    if west_window_proportion > 0:
        window_coordinates.update(
            {
                'window_west': [
                    [0,l-(l*west_window_proportion)/2,h*0.9],
                    [0,l-(l*west_window_proportion)/2,0.1],
                    [0,0+(l*west_window_proportion)/2,0.1],
                    [0,0+(l*west_window_proportion)/2,h*0.9]
                ],
            }
        )
    
    
    for window_name in [key for key in window_coordinates.keys()]:
        # Iterate over the four vertices of the surface
        for _ in range(1,5,1):
            for xyz in range(3):
                epJSON_object['FenestrationSurface:Detailed'][window_name]['vertex_'+str(_)+'_'+coordinate[xyz]+'_coordinate'] = window_coordinates[window_name][_-1][xyz]
        

def random_building_config(env_config:Dict):
    """
    This method define the path to the epJSON file.

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
    if env_config.get('epjson_path', False) is False:
        env_config['epjson_path'] = ''
    
    # Define the properties of the method
    epjson_files_folder_path: str = None # This is mandatory
    id_epjson_file: int = None # The default is a random int, but the user can indicated a diferent number
    
    # For each property, assign the default or the user input
    epjson_files_folder_path = env_config['episode_fn_config'].get('epjson_files_folder_path', None)
    if epjson_files_folder_path is None:
        ValueError('epjson_files_folder_path is not defined')
    
    id_epjson_file = env_config['episode_fn_config'].get('id_epjson_file', None)
    if id_epjson_file is None:
        id_epjson_file = np.random.randint(0, len(os.listdir(epjson_files_folder_path)))
    
    # The path to the epjson file is defined
    env_config['epjson_path'] = os.path.join(epjson_files_folder_path, os.listdir(epjson_files_folder_path)[id_epjson_file])
    
    return env_config

def generate_variated_schedule(input_path, output_path, variation_percentage=10):
    """
    Generates a new CSV with variations in schedule data.
    
    Parameters:
    - input_path: str, path to the input CSV file.
    - output_path: str, path to save the output CSV file.
    - variation_percentage: float, maximum percentage variation for each value (default: 10).
    """
    # Load the input CSV
    data = pd.read_csv(input_path)
    
    # Calculate variation range
    variation_factor = variation_percentage / 100.0
    
    # Apply random variations within the specified range
    def apply_variation(value):
        variation = np.random.uniform(-variation_factor, variation_factor)
        new_value = value * (1 + variation)
        return np.clip(new_value, 0, 1)  # Ensure values stay within [0, 1]

    # Apply variation to all columns
    variated_data = data.applymap(apply_variation)
    
    # Save the new dataset to the specified output path
    variated_data.to_csv(output_path, index=False)

def generate_variated_schedule_with_shift(input_path, output_path, variation_percentage=10, shift_hours=0):
    """
    Generates a new CSV with variations and optional time shift in schedule data.
    
    Parameters:
    - input_path: str, path to the input CSV file.
    - output_path: str, path to save the output CSV file.
    - variation_percentage: float, maximum percentage variation for each value (default: 10).
    - shift_hours: int, number of hours to shift the schedules (default: 0).
    """
    # Load the input CSV
    data = pd.read_csv(input_path)
    
    # Calculate variation range
    variation_factor = variation_percentage / 100.0
    
    # Apply random variations within the specified range
    def apply_variation(value):
        variation = np.random.uniform(-variation_factor, variation_factor)
        new_value = value * (1 + variation)
        return np.clip(new_value, 0, 1)  # Ensure values stay within [0, 1]
    
    # Apply variation to all columns
    variated_data = data.applymap(apply_variation)
    
    # Apply temporal shift if specified
    if shift_hours != 0:
        rows_to_shift = shift_hours  # Number of rows to shift (1 hour per row)
        variated_data = variated_data.shift(periods=rows_to_shift, fill_value=0).reset_index(drop=True)

    # Save the new dataset to the specified output path
    variated_data.to_csv(output_path, index=False)

def generate_occupancy_schedule(input_path, output_path, random_variation=False, seed=None):
    """
    Generates an updated occupancy schedule based on typical residential patterns.
    
    Parameters:
    - input_path: str, path to the input CSV file (with "occupancy" column).
    - output_path: str, path to save the output CSV file.
    - random_variation: bool, whether to introduce random variations to the schedule.
    - seed: int, random seed for reproducibility (default: None).
    """
    # Set random seed if needed
    if seed is not None:
        np.random.seed(seed)
    
    # Load the input CSV
    data = pd.read_csv(input_path)
    assert "occupancy" in data.columns, "Input file must have an 'occupancy' column."
    
    # Generate timestamps assuming hourly data for a year (if not provided)
    num_rows = len(data)
    start_date = datetime.datetime(2024, 1, 1)
    timestamps = [start_date + datetime.timedelta(hours=i) for i in range(num_rows)]
    data["timestamp"] = timestamps

    # Define occupancy rules based on weekdays and hours
    def occupancy_pattern(timestamp):
        hour = timestamp.hour
        day = timestamp.weekday()  # 0=Monday, 6=Sunday
        
        if day < 5:  # Weekdays
            if 7 <= hour < 9 or 17 <= hour < 23:
                return 1  # Morning/evening peak
            elif 9 <= hour < 17:
                return 0  # Work hours
            else:
                return 1  # Night hours (sleeping)
        else:  # Weekends
            if 9 <= hour < 23:
                return 1  # Occupied during the day
            else:
                return 1  # Night hours (sleeping)

    # Apply patterns
    data["occupancy"] = data["timestamp"].apply(occupancy_pattern)
    
    # Introduce random variations if enabled
    if random_variation:
        def add_randomness(occ_value):
            return occ_value if np.random.rand() > 0.1 else 1 - occ_value  # Flip value with 10% probability
        
        data["occupancy"] = data["occupancy"].apply(add_randomness)

    # Drop the timestamp column before saving
    data.drop(columns=["timestamp"], inplace=True)

    # Save the updated dataset to the specified output path
    data.to_csv(output_path, index=False)