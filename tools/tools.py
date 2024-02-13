"""# TOOLS FOR THE PROJECT
"""
import numpy as np
import os
import pickle
import json

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
    
    if not env_config['is_test']:
        epJSON_object = run_period(
            epJSON_object = epJSON_object,
            ObjectName = next(iter(epJSON_object['RunPeriod'])),
            day = np.random.randint(1,355),
            longitud_episodio = env_config['episode_len']
        )
    else:
        epJSON_object = run_period(
            epJSON_object = epJSON_object,
            ObjectName = next(iter(epJSON_object['RunPeriod'])),
            day = env_config['test_init_day'],
            longitud_episodio = env_config['episode_len']
        )
    
    # Se calcula el factor global de pérdidas U
    env_config['construction_u_factor'] = u_factor(epJSON_object)
    
    # Modificación de la masa térmica interna
    for key in [key for key in epJSON_object["InternalMass"].keys()]:
        epJSON_object["InternalMass"][key]["surface_area"] = np.random.randint(10,40)
    
    # Se calcula la masa térmica total del edificio
    env_config['inercial_mass'] = masa_inercial(epJSON_object)
    
    # Se cambia el límite de capacidad máxima del HVAC Ideal
    env_config['E_max'] = 0.5+(0.5 - 0.08)*np.random.random_sample()  #2500/1000/6
    HVAC_names = [key for key in epJSON_object["ZoneHVAC:IdealLoadsAirSystem"].keys()]
    for hvac in range(len(HVAC_names)):
        epJSON_object["ZoneHVAC:IdealLoadsAirSystem"][HVAC_names[hvac]]["maximum_sensible_heating_capacity"] = env_config['E_max']
        epJSON_object["ZoneHVAC:IdealLoadsAirSystem"][HVAC_names[hvac]]["maximum_total_cooling_capacity"] = env_config['E_max']
    
    # Se escribe el nuevo epJSON modificado
    env_config["epjson"] = f"{env_config['idf_output_folder']}/model-{env_config['episode']:08}-{os.getpid():05}.epJSON"
    
    with open(env_config["epjson"], 'w') as fp:
        json.dump(epJSON_object, fp, sort_keys=False, indent=4)
    
    env_config['epw'], CLIMATIC_STADS_PATH,env_config['latitud'], env_config['longitud'], env_config['altitud'] = weather_file(
        env_config
    )
    with open(CLIMATIC_STADS_PATH, 'rb') as fp:
        env_config['climatic_stads'] = pickle.load(fp)

    return env_config

def epJSON_path(env_config: dict):
    """This method define the path to the epJSON file to be simulated.
    
    Args:
        env_config (dict): Environment configuration.
        
    Return:
        dict: The method returns the env_config with modifications.
    """
    env_config['epjson'] = env_config['epjson_folderpath']+env_config['building_name']+'.epjson'
    return env_config

def plus_day(day:int, month:int, day_p:int):
    """This method take a date in the form of `day` and `month` and calculate the date `day_p` ahead.

    Args:
        day (int): Day of reference.
        month (int): Month of reference.
        day_p (int): Quantity of days ahead to calculate the new date.

    Returns:
        tuple[int, int]: Return a tuple of `day, month` for the new date.
    """
    if month in [1,3,5,7,8,10,12]:
            day_max = 31
    elif month == 2:
        day_max = 28
    else:
        day_max = 30
    # Calculate the maximum days in the especified month of reference.
    if day_p != 0:
        if (day + day_p) > day_max:
            day += (day_p - day_max)
            if month != 12:
                month += 1
            else:
                month = 1
        else:
            day += day_p
    return day, month

def natural_ventilation_action(central_action: int):
    """_summary_

    Args:
        central_action (int): _description_

    Returns:
        _type_: _description_
    """
    action_space = [
        [0,0],
        [0,1],
        [1,0],
        [1,1]
    ]
    return action_space[central_action]
    
def weather_file(env_config: dict, weather_choice:int = np.random.randint(0,24)):
    """_summary_

    Args:
        env_config (dict): _description_
        weather_choice (int, optional): _description_. Defaults to np.random.randint(0,24).

    Returns:
        _type_: _description_
    """
    folder_path = env_config['weather_folder']
    # clima aleatorio
    if not env_config['is_test']:
        weather_path = [
            ['GEF_Formosa-hour-H1',-26.0,-58.2,64],
            ['GEF_Lujan_de_cuyo-hour-H1',-32.985,-68.93,1043],
            ['GEF_Rawson-hour-H1',-43.300,-65.075,6],
            ['GEF_Salta-hour-H1',-24.770,-65.470,1280],
            ['GEF_San_Miguel_de_Tucuman-hour-H1',-26.839,-65.209,435],
            ['GEF_San_Nicolas_de_los_Arroyos-hour-H1',-33.362,-60.245,29],
            ['GEF_Ushuaia-hour-H1',-54.800,-68.317,14],
            ['GEF_Zapalla-hour-H1',-38.871,-70.112,1051],
            
            ['GEF_Formosa-hour-H2',-26.0,-58.2,64],
            ['GEF_Lujan_de_cuyo-hour-H2',-32.985,-68.93,1043],
            ['GEF_Rawson-hour-H2',-43.300,-65.075,6],
            ['GEF_Salta-hour-H2',-24.770,-65.470,1280],
            ['GEF_San_Miguel_de_Tucuman-hour-H2',-26.839,-65.209,435],
            ['GEF_San_Nicolas_de_los_Arroyos-hour-H2',-33.362,-60.245,29],
            ['GEF_Ushuaia-hour-H2',-54.800,-68.317,14],
            ['GEF_Zapalla-hour-H2',-38.871,-70.112,1051],
            
            ['GEF_Formosa-hour-H3',-26.0,-58.2,64],
            ['GEF_Lujan_de_cuyo-hour-H3',-32.985,-68.93,1043],
            ['GEF_Rawson-hour-H3',-43.300,-65.075,6],
            ['GEF_Salta-hour-H3',-24.770,-65.470,1280],
            ['GEF_San_Miguel_de_Tucuman-hour-H3',-26.839,-65.209,435],
            ['GEF_San_Nicolas_de_los_Arroyos-hour-H3',-33.362,-60.245,29],
            ['GEF_Ushuaia-hour-H3',-54.800,-68.317,14],
            ['GEF_Zapalla-hour-H3',-38.871,-70.112,1051],
        ]
        
        latitud = weather_path[weather_choice][1]
        longitud = weather_path[weather_choice][2]
        altitud = weather_path[weather_choice][3]
        return folder_path+'/'+weather_path[weather_choice][0]+'.epw', folder_path+'/'+weather_path[weather_choice][0]+'.pkl', latitud, longitud, altitud
    
    else:
        return folder_path+'/GEF_Lujan_de_cuyo-hour-H4.epw', folder_path+'/GEF_Lujan_de_cuyo-hour-H4.pkl', -32.985,-68.93,1043

def run_period(epJSON_object, ObjectName: str, day: int, longitud_episodio: int):
    """Función que modifica el periodo de ejecución del objeto epJSON.

    Args:
        epJSON_objent (json): Un objeto de epJSON.
        ObjectName (str): Nombre del objeto de RunPeriod existente en el archivo IDF (es decir, en el objeto epJSON).
        day (int): Día juliano que puede tomar los valores de 1 a (365-28), debido a que
        el periodo de ejecución de EnergyPlus será de 28 días.
        longitud_episodio (int): cantidad de días que tendrá cada episodio.
        
    Returns:
        json: Devuelve el objeto epJSON modificado.
    """
    days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    # se calcula el día inicial
    sum = 0
    for k, item in enumerate(days):
        sum += item
        if sum >= day:
            init_day = item - (sum - day)
            init_month = k+1
            break
    # se calcula el día final
    final_date = day + longitud_episodio - 1
    sum = 0
    for k, item in enumerate(days):
        sum += item
        if sum >= final_date:
            final_day = item - (sum - final_date)
            final_month = k+1
            break
    epJSON_object["RunPeriod"][ObjectName]["begin_month"] = init_month
    epJSON_object["RunPeriod"][ObjectName]["begin_day_of_month"] = init_day
    epJSON_object["RunPeriod"][ObjectName]["end_month"] = final_month
    epJSON_object["RunPeriod"][ObjectName]["end_day_of_month"] = final_day
    
    return epJSON_object

def masa_inercial(epJSON_object: dict[str,dict]):
    """_summary_

    Args:
        epJSON_object (dict[str,dict]): _description_

    Returns:
        _type_: _description_
    """
    # se define una lista para almacenar
    masas_termicas = []
    # se obtienen los nombres de las superficies de la envolvente
    building_surfaces = [key for key in epJSON_object["BuildingSurface:Detailed"].keys()]
    internal_mass_surfaces = [key for key in epJSON_object["InternalMass"].keys()]
    # se obtienen los nombres de los diferentes tipos de materiales
    materials_dict = {
        'Material': epJSON_object['Material'].keys(),
        #'material_nomass_names': epJSON_object['Material:NoMass'].keys(),
        #'material_infrared_names': epJSON_object['Material:InfraredTransparent'].keys(),
        #'Material:AirGap': epJSON_object['Material:AirGap'].keys(),
        #'material_roofveg_names': epJSON_object['Material:RoofVegetation'].keys(),
        #'material_wsimpleglassis_names': epJSON_object['WindowMaterial:SimpleGlazingSystem'].keys(),
        'WindowMaterial:Glazing': epJSON_object['WindowMaterial:Glazing'].keys(),
        #'material_wthermocrom_names': epJSON_object['WindowMaterial:GlazingGroup:Thermochromic'].keys(),
        #'material_wrefraction_names': epJSON_object['WindowMaterial:Glazing:RefractionExtinctionMethod'].keys(),
        'WindowMaterial:Gas': epJSON_object['WindowMaterial:Gas'].keys(),
        #'material_wpillar_names': epJSON_object['WindowGap:SupportPillar'].keys(),
        #'material_wdeflection_names': epJSON_object['WindowGap:DeflectionState'].keys(),
        #'material_wgasmix_names': epJSON_object['WindowMaterial:GasMixture'].keys(),
        #'material_wgap_names': epJSON_object['WindowMaterial:Gap'].keys(),
    }
    
    # lazo para consultar cada superficie de la envolvente
    for surface in building_surfaces:
        # se calcula el área de la superficie
        area = calculo_area_material(epJSON_object,surface)
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
    materials_dict = {
        'Material': epJSON_object['Material'].keys(),
        #'material_nomass_names': epJSON_object['Material:NoMass'].keys(),
        #'material_infrared_names': epJSON_object['Material:InfraredTransparent'].keys(),
        #'Material:AirGap': epJSON_object['Material:AirGap'].keys(),
        #'material_roofveg_names': epJSON_object['Material:RoofVegetation'].keys(),
        #'material_wsimpleglassis_names': epJSON_object['WindowMaterial:SimpleGlazingSystem'].keys(),
        'WindowMaterial:Glazing': epJSON_object['WindowMaterial:Glazing'].keys(),
        #'material_wthermocrom_names': epJSON_object['WindowMaterial:GlazingGroup:Thermochromic'].keys(),
        #'material_wrefraction_names': epJSON_object['WindowMaterial:Glazing:RefractionExtinctionMethod'].keys(),
        'WindowMaterial:Gas': epJSON_object['WindowMaterial:Gas'].keys(),
        #'material_wpillar_names': epJSON_object['WindowGap:SupportPillar'].keys(),
        #'material_wdeflection_names': epJSON_object['WindowGap:DeflectionState'].keys(),
        #'material_wgasmix_names': epJSON_object['WindowMaterial:GasMixture'].keys(),
        #'material_wgap_names': epJSON_object['WindowMaterial:Gap'].keys(),
    }
    
    # lazo para consultar cada superficie de la envolvente
    for surface in building_surfaces:
        # se calcula el área de la superficie
        areas.append(calculo_area_material(epJSON_object,surface))
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
        areas.append(calculo_area_fenestracion(epJSON_object, fenestration))
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

def calculo_area_material(epJSON_object, nombre_superficie):
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

def calculo_area_fenestracion(epJSON_object, fenestration):
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