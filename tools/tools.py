"""# TOOLS FOR THE PROJECT
"""
import numpy as np
import pandas as pd
import os
import pickle
import json
from numpy.typing import NDArray

def episode_epJSON(env_config: dict):
    """This method define the properties of the episode. Changing some properties as weather or 
    Run Time Period, and defining others fix properties as volumen or window area relation.
    
    Args:
        env_config (dict): Environment configuration.
        
    Return:
        dict: The method returns the env_config with modifications.
    """
    if env_config.get('idf', False) == False:
        env_config = epJSON_path(env_config)
    with open(env_config['idf']) as file:
        epJSON_object: dict = json.load(file)
    
    # determinación del volumen
    env_config['volumen'] = 50.32
    # se orienta el edificio según plano
    epJSON_object['Building'][next(iter(epJSON_object['Building']))]['north_axis'] = 0
    # relación de superficies de ventanas con respecto a la superficie cubierta
    env_config['window_area_relation_north'] = 0
    env_config['window_area_relation_west'] = 0
    env_config['window_area_relation_south'] = 0.595/9.125
    env_config['window_area_relation_east'] = 1.43/11.5
    
    # se establece el periodo de ejecución
    env_config['longitud_episodio'] = 7
    epJSON_object = run_period(
        epJSON_object = epJSON_object,
        ObjectName = next(iter(epJSON_object['RunPeriod'])),
        day = np.random.randint(1,355),
        longitud_episodio = env_config['longitud_episodio']
    )
    
    # Se calcula el factor global de pérdidas U
    env_config['construction_config'] = u_factor(epJSON_object)
    
    # Modificación de la masa térmica interna
    for key in [key for key in epJSON_object["InternalMass"].keys()]:
        epJSON_object["InternalMass"][key]["surface_area"] = np.random.randint(10,40)
    
    # Se calcula la masa térmica total del edificio
    env_config['internal_mass'] = masa_inercial(epJSON_object)
    
    # Se cambia el límite de capacidad máxima del HVAC Ideal
    env_config['E_max'] = 0.5+(0.5 - 0.08)*np.random.random_sample()  #2500/1000/6
    HVAC_names = [key for key in epJSON_object["ZoneHVAC:IdealLoadsAirSystem"].keys()]
    for hvac in range(len(HVAC_names)):
        epJSON_object["ZoneHVAC:IdealLoadsAirSystem"][HVAC_names[hvac]]["maximum_sensible_heating_capacity"] = env_config['E_max']
        epJSON_object["ZoneHVAC:IdealLoadsAirSystem"][HVAC_names[hvac]]["maximum_total_cooling_capacity"] = env_config['E_max']
    
    # Se escribe el nuevo epJSON modificado
    env_config["idf"] = f"{env_config['idf_output_folder']}/model-{env_config['episode']:08}-{os.getpid():05}.epJSON"
    
    with open(env_config["idf"], 'w') as fp:
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
    env_config['idf'] = env_config['idf_folderpath']+'/prot_1.epJSON'
    
    return env_config

def model_library(env_config: dict):
    """The method chose a random GEF building and modify the env_config dictionary to assign the following
    parameters for the simulation:
        'idf'
        'volumen'
        'window_area_relation_north'
        'window_area_relation_west'
        'window_area_relation_south'
        'window_area_relation_east'

    Args:
        env_config (dict): Dictionary that contains the environment configuration for the DRL experiment.
        
    Return:
        Returns the env_config dictionary with modifications
    """
    models_info = {
        'GEF_Formosa':{
            'volumen':0,
            'window_area_relation_north':0,
            'window_area_relation_west':0,
            'window_area_relation_south':0,
            'window_area_relation_east':0,
        },
        'GEF_MDZ_PROT_1':{
            'volumen':131.6565,
            'window_area_relation_north':0,
            'window_area_relation_west':0,
            'window_area_relation_south':0.595/9.125,
            'window_area_relation_east':1.43/11.5,
        },
        'GEF_MDZ_PROT_2':{
            'volumen':131.6565,
            'window_area_relation_north':0,
            'window_area_relation_west':0,
            'window_area_relation_south':0.595/9.125,
            'window_area_relation_east':1.43/11.5,
        },
        'GEF_MDZ_PROT_3':{
            'volumen':131.6565,
            'window_area_relation_north':0,
            'window_area_relation_west':0,
            'window_area_relation_south':0.595/9.125,
            'window_area_relation_east':1.43/11.5,
        },
        'GEF_MDZ_PROT_4':{
            'volumen':131.6565,
            'window_area_relation_north':0,
            'window_area_relation_west':0,
            'window_area_relation_south':0.595/9.125,
            'window_area_relation_east':1.43/11.5,
        },
        'GEF_Rawson':{
            'volumen':0,
            'window_area_relation_north':0,
            'window_area_relation_west':0,
            'window_area_relation_south':0,
            'window_area_relation_east':0,
        },
        'GEF_Salta':{
            'volumen':0,
            'window_area_relation_north':0,
            'window_area_relation_west':0,
            'window_area_relation_south':0,
            'window_area_relation_east':0,
        },
        'GEF_San_Miguel_de_Tucuman':{
            'volumen':0,
            'window_area_relation_north':0,
            'window_area_relation_west':0,
            'window_area_relation_south':0,
            'window_area_relation_east':0,
        },
        'GEF_San_Nicolas_de_los_Arroyos':{
            'volumen':0,
            'window_area_relation_north':0,
            'window_area_relation_west':0,
            'window_area_relation_south':0,
            'window_area_relation_east':0,
        },
        'GEF_Ushuaia':{
            'volumen':0,
            'window_area_relation_north':0,
            'window_area_relation_west':0,
            'window_area_relation_south':0,
            'window_area_relation_east':0,
        },
        'GEF_Zapalla':{
            'volumen':0,
            'window_area_relation_north':0,
            'window_area_relation_west':0,
            'window_area_relation_south':0,
            'window_area_relation_east':0,
        },
    }
    
    if not env_config['is_test']:
        x = 'Número de modelos disponibles para todas las provincias de afectadas por el proyecto del IPV'
        model = np.random.randint(0,x)
        models_keys = [key for key in models_info.keys()]
        model_key = models_keys[model]
        
        env_config['idf'] = env_config['idf_folderpath'] + model_key + '.epJSON'
        env_config['volumen'] = models_info[model_key]['volumen']
        env_config['window_area_relation_north'] = models_info[model_key]['window_area_relation_north']
        env_config['window_area_relation_west'] = models_info[model_key]['window_area_relation_west']
        env_config['window_area_relation_south'] = models_info[model_key]['window_area_relation_south']
        env_config['window_area_relation_east'] = models_info[model_key]['window_area_relation_east']
        
    else: # Apply GEF_MDZ_PROT_1 for evaluation
        env_config['idf'] = env_config['idf_folderpath'] + models_info['GEF_MDZ_PROT_1']['name'] + '.epJSON'
        env_config['volumen'] = models_info['GEF_MDZ_PROT_1']['volumen']
        env_config['window_area_relation_north'] = models_info['GEF_MDZ_PROT_1']['window_area_relation_north']
        env_config['window_area_relation_west'] = models_info['GEF_MDZ_PROT_1']['window_area_relation_west']
        env_config['window_area_relation_south'] = models_info['GEF_MDZ_PROT_1']['window_area_relation_south']
        env_config['window_area_relation_east'] = models_info['GEF_MDZ_PROT_1']['window_area_relation_east']

    return env_config

def plus_day(day, month, day_p):
    """_summary_

    Args:
        day (_type_): _description_
        month (_type_): _description_
        day_p (_type_): _description_

    Returns:
        _type_: _description_
    """
    if month in [1,3,5,7,8,10,12]:
            day_max = 31
    elif month == 2:
        day_max = 28
    else:
        day_max = 30
        
    if day_p != 0:
        
        if day + day_p > day_max:
            day += day_p - day_max
            if month != 12:
                month += 1
            else:
                month = 1
        else:
            day += day_p
    return day, month
    
def final_month_day(day, month):
    """_summary_

    Args:
        day (_type_): _description_
        month (_type_): _description_

    Returns:
        _type_: _description_
    """
    if month in [1,3,5,7,8,10,12]:
            day_max = 31
    elif month == 2:
        day_max = 28
    else:
        day_max = 30
        
    day_p = 10
        
    if day + day_p > day_max:
        day += day_p - day_max
        if month != 12:
            month += 1
        else:
            month = 1
    else:
        day += day_p
        
    return day, month

class weather_function():
    """_summary_
    """
    def climatic_stads(epw_file_path, month):
        """_summary_

        Args:
            epw_file_path (_type_): _description_
            month (_type_): _description_

        Returns:
            _type_: _description_
        """
        print('Calculando los datos climáticos estadísticos para el archivo de clima EPW.')
        epw_file = pd.read_csv(epw_file_path,
                            header = None, #['Year', 'Month', 'Day', 'Hour', 'Minutes', 'Data Source and Uncertainty Flags', 'Dry Bulb Temperature', 'Dew Point Temperature', 'Relative Humidity', 'Atmospheric Station Pressure', 'Extraterrestrial Horizontal Radiation', 'Extraterrestrial Direct Normal Radiation', 'Horizontal Infrared Radiation Intensity', 'Global Horizontal Radiation', 'Direct Normal Radiation', 'Diffuse Horizontal Radiation', 'Global Horizontal Illuminance', 'Direct Normal Illuminance', 'Diffuse Horizontal Illuminance', 'Zenith Luminance', 'Wind Direction', 'Wind Speed', 'Total Sky Cover', 'Opaque Sky Cover', 'Visibility', 'Ceiling Height', 'Present Weather Observation', 'Present Weather Codes', 'Precipitable Water', 'Aerosol Optical Depth', 'Snow Depth', 'Days Since Last Snowfall', 'Albedo', 'Liquid Precipitation Depth', 'Liquid Precipitation Quantity'],
                            skiprows = 8
                            )
        output = {}
        if month == 2:
            days = range(1,29,1)
        elif month in [4,6,9,11]:
            days = range(1,31,1)
        else:
            days = range(1,32,1)
        for day in days:
            dict = {
                    str(day):
                        {
                        'T_max_0': weather_function.tmax(epw_file, day, month),
                        'T_min_0': weather_function.tmin(epw_file, day, month),
                        'RH_0': weather_function.rh_avg(epw_file, day, month),
                        'raining_total_0': weather_function.rain_tot(epw_file, day, month),
                        'wind_avg_0': weather_function.wind_avg(epw_file, day, month),
                        'wind_max_0': weather_function.wind_max(epw_file, day, month),
                        'total_sky_cover_0': weather_function.total_sky_cover(epw_file, day, month),
                        }
                    }
            output.update(dict)
        print('Se finalizó la creación de los estadísticos.')
        return output
        
    def tmax(epw_file, day, month, day_p=0):
        """_summary_

        Args:
            epw_file (_type_): _description_
            day (_type_): _description_
            month (_type_): _description_
            day_p (int, optional): _description_. Defaults to 0.

        Returns:
            _type_: _description_
        """
        day, month = plus_day(day, month, day_p)
        array = []
        for _ in range(0, 8760,1):
            if epw_file[1][_] == month and epw_file[2][_] == day:
                array.append(epw_file[6][_])
        tmax = max(array)
        return tmax

    def tmin(epw_file, day, month, day_p=0):
        """_summary_

        Args:
            epw_file (_type_): _description_
            day (_type_): _description_
            month (_type_): _description_
            day_p (int, optional): _description_. Defaults to 0.

        Returns:
            _type_: _description_
        """
        day, month = plus_day(day, month, day_p)
        array = []
        for _ in range(0, 8760,1):
            if epw_file[1][_] == month and epw_file[2][_] == day:
                array.append(epw_file[6][_])
        tmin = min(array)
        return tmin

    def rh_avg(epw_file, day, month, day_p=0):
        """_summary_

        Args:
            epw_file (_type_): _description_
            day (_type_): _description_
            month (_type_): _description_
            day_p (int, optional): _description_. Defaults to 0.

        Returns:
            _type_: _description_
        """
        day, month = plus_day(day, month, day_p)
        array = []
        for _ in range(0, 8760,1):
            if epw_file[1][_] == month and epw_file[2][_] == day:
                array.append(epw_file[8][_])
        rh_avg = sum(array)/len(array)
        return rh_avg

    def rain_tot(epw_file, day, month, day_p=0):
        """_summary_

        Args:
            epw_file (_type_): _description_
            day (_type_): _description_
            month (_type_): _description_
            day_p (int, optional): _description_. Defaults to 0.

        Returns:
            _type_: _description_
        """
        day, month = plus_day(day, month, day_p)
        array = []
        for _ in range(0, 8760,1):
            if epw_file[1][_] == month and epw_file[2][_] == day:
                array.append(epw_file[33][_])
        rain_tot = sum(array)
        return rain_tot

    def wind_avg(epw_file, day, month, day_p=0):
        """_summary_

        Args:
            epw_file (_type_): _description_
            day (_type_): _description_
            month (_type_): _description_
            day_p (int, optional): _description_. Defaults to 0.

        Returns:
            _type_: _description_
        """
        day, month = plus_day(day, month, day_p)
        array = []
        for _ in range(0, 8760,1):
            if epw_file[1][_] == month and epw_file[2][_] == day:
                array.append(epw_file[21][_])
        wind_avg = sum(array)/len(array)
        return wind_avg

    def wind_max(epw_file, day, month, day_p=0):
        """_summary_

        Args:
            epw_file (_type_): _description_
            day (_type_): _description_
            month (_type_): _description_
            day_p (int, optional): _description_. Defaults to 0.

        Returns:
            _type_: _description_
        """
        day, month = plus_day(day, month, day_p)
        array = []
        for _ in range(0, 8760,1):
            if epw_file[1][_] == month and epw_file[2][_] == day:
                array.append(epw_file[21][_])
        wind_max = max(array)
        return wind_max
    
    def total_sky_cover(epw_file, day, month, day_p=0):
        """_summary_

        Args:
            epw_file (_type_): _description_
            day (_type_): _description_
            month (_type_): _description_
            day_p (int, optional): _description_. Defaults to 0.

        Returns:
            _type_: _description_
        """
        day, month = plus_day(day, month, day_p)
        array = []
        for _ in range(0, 8760,1):
            if epw_file[1][_] == month and epw_file[2][_] == day:
                array.append(epw_file[22][_])
        total_sky_cover = sum(array)/len(array)
        return total_sky_cover

def transform_centralized_action(central_action: int, centralized_action_space: NDArray):
    """This method transform a centralized action into descentralized actions for the agents in an environment that learn
        a joint action. The actions are:
        
        * `set_point_temperature`: two temperatures, for heating and cooling set points, are defined. This must have 1 °C of difference
        to works good in EnergyPlus.
        * `blind_action_1`: binary action: 0 -> off and 1 -> on.
        * `blind_action_2`: binary action: 0 -> off and 1 -> on.
        * `window_action_1`: binary action: 0 -> close and 1 -> open.
        * `window_action_2`: binary action: 0 -> close and 1 -> open.

        Args:
            central_action (int): Centralized action returned from step() method in gym.Env configuration environment.
            centralized_action_space (NDArray): action space represented in a numpy array.

        Returns:
            dict: dictionary that contain the format of {agent_id:action}
        """
    descentralized_action = [
        centralized_action_space[central_action][1],
        centralized_action_space[central_action][2],
        centralized_action_space[central_action][3],
        centralized_action_space[central_action][4],
        centralized_action_space[central_action][5],
        centralized_action_space[central_action][6]
    ]
    
    return descentralized_action

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

def shadow_control_actions(central_action: int):
    """_summary_

    Args:
        central_action (int): _description_

    Returns:
        _type_: _description_
    """
    return natural_ventilation_action(central_action)

def DSP_actions(central_action: int):
    """_summary_

    Args:
        central_action (int): _description_

    Returns:
        _type_: _description_
    """
    action_space = [
        [17,18],[17,19],[17,20],[17,21],[17,22],[17,23],[17,24],[17,25],[17,26],[17,27],[17,28],
        [18,19],[18,20],[18,21],[18,22],[18,23],[18,24],[18,25],[18,26],[18,27],[18,28],
        [19,20],[19,21],[19,22],[19,23],[19,24],[19,25],[19,26],[19,27],[19,28],
        [20,21],[20,22],[20,23],[20,24],[20,25],[20,26],[20,27],[20,28],
        [21,22],[21,23],[21,24],[21,25],[21,26],[21,27],[21,28],
        [22,23],[22,24],[22,25],[22,26],[22,27],[22,28],
        [23,24],[23,25],[23,26],[23,27],[23,28],
        [24,25],[24,26],[24,27],[24,28],
        [25,26],[25,27],[25,28],
        [26,27],[26,28],
        [27,28]
    ]
    return action_space[central_action]

def transformin_central_binary_action(central_action, agents_ids):
    """_summary_

    Args:
        central_action (_type_): _description_
        agents_ids (_type_): _description_

    Returns:
        _type_: _description_
    """
    modulos = [] # la lista para guardar los módulos
    while central_action != 0: # mientras el número de entrada sea diferente de cero
        # paso 1: dividimos entre 2
        modulo = central_action % 2
        cociente = central_action // 2
        modulos.append(modulo) # guardamos el módulo calculado
        central_action = cociente # el cociente pasa a ser el número de entrada
    while len(modulos) < len(agents_ids):
        modulos.append(0)
    return dict(zip(agents_ids,modulos))
    
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

def run_test_period(epJSON_object, ObjectName: str):
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
    epJSON_object["RunPeriod"][ObjectName]["begin_month"] = 1
    epJSON_object["RunPeriod"][ObjectName]["begin_day_of_month"] = 1
    epJSON_object["RunPeriod"][ObjectName]["end_month"] = 12
    epJSON_object["RunPeriod"][ObjectName]["end_day_of_month"] = 31
    
    return epJSON_object

def run_summer_period(epJSON_object, ObjectName: str):
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
    epJSON_object["RunPeriod"][ObjectName]["begin_month"] = 12
    epJSON_object["RunPeriod"][ObjectName]["begin_day_of_month"] = 1
    epJSON_object["RunPeriod"][ObjectName]["end_month"] = 12
    epJSON_object["RunPeriod"][ObjectName]["end_day_of_month"] = 31
    
    return epJSON_object

def change_vertex_volumen(epJSON_object, env_config: dict, h:float, w:float, l:float):
    """Given a epJSON_object, return another epJSON_object with diferent size of surfaces.

    Args:
        epJSON_object (json): _description_
        env_config (dict): _description_
        h (float): _description_
        w (float): _description_
        l (float): _description_
    Returns:
        json: Devuelve el objeto epJSON modificado.
        volumen: A float that represent the new volumen of the room.
    """
    # se modifican los valores de los vértices de la superficie para {h,l,w}
    surfaces = ['muro_oeste', 'muro_norte', 'muro_este', 'muro_sur','techo', 'piso']
    vertex = [
        [0,l,h, 0,l,0, 0,0,0, 0,0,h], #muro_oeste
        [w,l,h, w,l,0, 0,l,0, 0,l,h], #muro_norte
        [w,0,h, w,0,0, w,l,0, w,l,h], #muro_este
        [0,0,h, 0,0,0, w,0,0, w,0,h], #muro_sur
        [w,0,h, w,l,h, 0,l,h, 0,0,h], #techo
        [w,l,0, w,0,0, 0,0,0, 0,l,0]  #piso
        ]
    for _ in range(0,len(surfaces),1):
        epJSON_object['BuildingSurface:Detailed'][surfaces[_]]["vertices"][0]['vertex_x_coordinate'] = vertex[_][0]
        epJSON_object['BuildingSurface:Detailed'][surfaces[_]]["vertices"][0]['vertex_y_coordinate'] = vertex[_][1]
        epJSON_object['BuildingSurface:Detailed'][surfaces[_]]["vertices"][0]['vertex_z_coordinate'] = vertex[_][2]
        epJSON_object['BuildingSurface:Detailed'][surfaces[_]]["vertices"][1]['vertex_x_coordinate'] = vertex[_][3]
        epJSON_object['BuildingSurface:Detailed'][surfaces[_]]["vertices"][1]['vertex_y_coordinate'] = vertex[_][4]
        epJSON_object['BuildingSurface:Detailed'][surfaces[_]]["vertices"][1]['vertex_z_coordinate'] = vertex[_][5]
        epJSON_object['BuildingSurface:Detailed'][surfaces[_]]["vertices"][2]['vertex_x_coordinate'] = vertex[_][6]
        epJSON_object['BuildingSurface:Detailed'][surfaces[_]]["vertices"][2]['vertex_y_coordinate'] = vertex[_][7]
        epJSON_object['BuildingSurface:Detailed'][surfaces[_]]["vertices"][2]['vertex_z_coordinate'] = vertex[_][8]
        epJSON_object['BuildingSurface:Detailed'][surfaces[_]]["vertices"][3]['vertex_x_coordinate'] = vertex[_][9]
        epJSON_object['BuildingSurface:Detailed'][surfaces[_]]["vertices"][3]['vertex_y_coordinate'] = vertex[_][10]
        epJSON_object['BuildingSurface:Detailed'][surfaces[_]]["vertices"][3]['vertex_z_coordinate'] = vertex[_][11]
    
    # se modifican los vértices de las ventanas para que correspondan con las nuevas dimensiones de los muros
    windows = ["ventana_1", "ventana_2"]
    if env_config["model"] == 1:
        windows_vertex = [
            [0.1,0,h-0.1, 0.1,0,0.1, w-0.1,0,0.1, w-0.1,0,h-0.1], #muro_sur
            [w-0.1,l,h-0.1, w-0.1,l,0.1, 0.1,l,0.1, 0.1,l,h-0.1] #muro_norte
        ]
    else:
        windows_vertex = [
            [0.1,0,h-0.1, 0.1,0,0.1, w-0.1,0,0.1, w-0.1,0,h-0.1], #muro_sur
            [w,0.1,h-0.1, w,0.1,0.1, w,l-0.1,0.1, w,l-0.1,h-0.1] #muro_este
        ]
    
    for _ in range(0,len(windows),1):
        epJSON_object['FenestrationSurface:Detailed'][windows[_]]['vertex_1_x_coordinate'] = windows_vertex[_][0]
        epJSON_object['FenestrationSurface:Detailed'][windows[_]]['vertex_1_y_coordinate'] = windows_vertex[_][1]
        epJSON_object['FenestrationSurface:Detailed'][windows[_]]['vertex_1_z_coordinate'] = windows_vertex[_][2]
        epJSON_object['FenestrationSurface:Detailed'][windows[_]]['vertex_2_x_coordinate'] = windows_vertex[_][3]
        epJSON_object['FenestrationSurface:Detailed'][windows[_]]['vertex_2_y_coordinate'] = windows_vertex[_][4]
        epJSON_object['FenestrationSurface:Detailed'][windows[_]]['vertex_2_z_coordinate'] = windows_vertex[_][5]
        epJSON_object['FenestrationSurface:Detailed'][windows[_]]['vertex_3_x_coordinate'] = windows_vertex[_][6]
        epJSON_object['FenestrationSurface:Detailed'][windows[_]]['vertex_3_y_coordinate'] = windows_vertex[_][7]
        epJSON_object['FenestrationSurface:Detailed'][windows[_]]['vertex_3_z_coordinate'] = windows_vertex[_][8]
        epJSON_object['FenestrationSurface:Detailed'][windows[_]]['vertex_4_x_coordinate'] = windows_vertex[_][9]
        epJSON_object['FenestrationSurface:Detailed'][windows[_]]['vertex_4_y_coordinate'] = windows_vertex[_][10]
        epJSON_object['FenestrationSurface:Detailed'][windows[_]]['vertex_4_z_coordinate'] = windows_vertex[_][11]
    
    # se calcula en volumen de la habitación
    volumen = w*l*h
    
    return volumen
    
def window_size_epJSON(epJSON_object, window:str, factor_escala:float, env_config: dict):
    """Given a epJSON_object, return another epJSON_object with diferent size of windows.

    Args:
        epJSON_object (json): _description_
        window_name (str): _description_
        factor (float): _description_

    Returns:
        json: Devuelve el objeto epJSON modificado.
    """
    # se extraen los valores de los vértices de cada ventana según el epJSON
    window_vertexs = [
        [
            epJSON_object['FenestrationSurface:Detailed'][window]['vertex_1_x_coordinate'],
            epJSON_object['FenestrationSurface:Detailed'][window]['vertex_1_y_coordinate'],
            epJSON_object['FenestrationSurface:Detailed'][window]['vertex_1_z_coordinate']
        ],
        [
            epJSON_object['FenestrationSurface:Detailed'][window]['vertex_2_x_coordinate'],
            epJSON_object['FenestrationSurface:Detailed'][window]['vertex_2_y_coordinate'],
            epJSON_object['FenestrationSurface:Detailed'][window]['vertex_2_z_coordinate']
        ],
        [
            epJSON_object['FenestrationSurface:Detailed'][window]['vertex_3_x_coordinate'],
            epJSON_object['FenestrationSurface:Detailed'][window]['vertex_3_y_coordinate'],
            epJSON_object['FenestrationSurface:Detailed'][window]['vertex_3_z_coordinate']
        ],
        [
            epJSON_object['FenestrationSurface:Detailed'][window]['vertex_4_x_coordinate'],
            epJSON_object['FenestrationSurface:Detailed'][window]['vertex_4_y_coordinate'],
            epJSON_object['FenestrationSurface:Detailed'][window]['vertex_4_z_coordinate']
        ]
    ]

    centro = calcular_centro(window_vertexs)
    ventana_escalada = []
    for punto in window_vertexs:
        nuevo_punto = [centro[0] + (punto[0] - centro[0]) * factor_escala**(1/2),
                    centro[1] + (punto[1] - centro[1]) * factor_escala**(1/2),
                    centro[2] + (punto[2] - centro[2]) * factor_escala**(1/2)]
        ventana_escalada.append(nuevo_punto)
    
    # Se calcula el factor de area de la ventana escalada
    area_ventana = calcular_area(ventana_escalada)
    if env_config['model'] == 1 or env_config['model'] == 2 or env_config['model'] == 3:
        area_factor = area_ventana/9
    elif env_config['model'] == 4 or env_config['model'] == 5 or env_config['model'] == 6:
        area_factor = area_ventana/5
    else:
        area_factor = area_ventana/15
    
    for l in range(1,5,1):
        epJSON_object["FenestrationSurface:Detailed"][window]["vertex_"+str(l)+"_x_coordinate"] = ventana_escalada[l-1][0]
        epJSON_object["FenestrationSurface:Detailed"][window]["vertex_"+str(l)+"_y_coordinate"] = ventana_escalada[l-1][1]
        epJSON_object["FenestrationSurface:Detailed"][window]["vertex_"+str(l)+"_z_coordinate"] = ventana_escalada[l-1][2]
        
    return area_factor

def calcular_centro(ventana):
    """_summary_

    Args:
        ventana (_type_): _description_

    Returns:
        _type_: _description_
    """
    # Calcula el centro de la ventana
    centro = [(ventana[0][0] + ventana[1][0] + ventana[2][0] + ventana[3][0]) / 4,
            (ventana[0][1] + ventana[1][1] + ventana[2][1] + ventana[3][1]) / 4,
            (ventana[0][2] + ventana[1][2] + ventana[2][2] + ventana[3][2]) / 4]
    return centro

def calcular_area(ventana):
    """_summary_

    Args:
        ventana (_type_): _description_

    Returns:
        _type_: _description_
    """
    # Calcula dos vectores que forman dos lados del cuadrilátero
    vector1 = [ventana[1][0] - ventana[0][0], ventana[1][1] - ventana[0][1], ventana[1][2] - ventana[0][2]]
    vector2 = [ventana[2][0] - ventana[0][0], ventana[2][1] - ventana[0][1], ventana[2][2] - ventana[0][2]]

    # Calcula el producto vectorial entre los dos vectores
    producto_vectorial = [
        vector1[1] * vector2[2] - vector1[2] * vector2[1],
        vector1[2] * vector2[0] - vector1[0] * vector2[2],
        vector1[0] * vector2[1] - vector1[1] * vector2[0]
    ]

    # Calcula el módulo del producto vectorial como el área del cuadrilátero
    area = 0.5 * (abs(producto_vectorial[0]) + abs(producto_vectorial[1]) + abs(producto_vectorial[2]))
    return area

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
        M_total =+ masas_termicas[m]
    
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