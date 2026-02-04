"""
Generación de los escenarios de entrenamiento general
======================================================

Se consideran los siguientes grupos de variables:

1. Orientación de ventanas.
2. Área de la zona térmica.
3. Tamaño de las superficies vidreadas.
4. Propiedades de la envolvente.
5. Propiedades de las ventanas.
6. Masa térmica interior.
7. Periodo de simulación.
8. Potencia de los sistemas activos de climatización.
9. Perfil climático.
10. Perfil de carga.
11. Perfil de ocupación.


1. Orientación de ventanas.
----------------------------

Se han definido 15 modelos unitarios que poseen las siguientes características:

    | Model | [n,e,s,w] |
    +-------+-----------+
    | "1"   | [1,1,1,1] |
    | "2"   | [1,1,1,0] |
    | "3"   | [1,1,0,1] |
    | "4"   | [1,0,1,1] |
    | "5"   | [0,1,1,1] |
    | "6"   | [1,1,0,0] |
    | "7"   | [1,0,1,0] |
    | "8"   | [0,1,1,0] |
    | "9"   | [1,0,0,1] |
    | "10"  | [0,1,0,1] |
    | "11"  | [0,0,1,1] |
    | "12"  | [1,0,0,0] |
    | "13"  | [0,1,0,0] |
    | "14"  | [0,0,1,0] |
    | "15"  | [0,0,0,1] |
  
Estos modelos unitarios permiten simular escenarios de una habitación que posee
ventanas en las diferentes orientaciones. Cada uno de estos modelos base luego serán
modificados en los puntos siguientes. Sin embargo, su selección determina los agentes 
que estarán disponibles en el episodio, debido a que parte de ellos dependen directamente
de la existencia o no de ventanas.

No se han considerado modelos en los que algunos de los muros se encuentran colindando
con otras zonas térmicas (lo que requiere la utilización de muros no expuestos al sol
o al viento y que sean adiaváticos). Esto se podría integrar en un futuro.
    
2. Área de la zona térmica.
----------------------------

Los modelos entrenados en esta política corresponden a edificios simplificados a una 
sola zona térmica. Esta zona térmica es a su vez geométricamente simple al ser 
estructurada como un prisma equivalente. Para poder trabajar con diferentes edificios
equivalentes, se modifica el tamaño de la zona térmica con los parámetros:

* h: rango de alturas, en metros.
* w: rango de anchos (este-oeste), en metros.
* l: rango de largos (norte-sur), en metros.

Para esta configuración de entrenamiento se han adoptado los siguientes rangos de
valores:

building_size = [
    [2,4], # Rango de alturas (m)
    [5,20], # Rango de anchos (m)
    [5,20], # Rango de largos (m)
]

Esto permite establecer zonas térmicas de 25 m2 a 400 m2.

3. Tamaño de las superficies vidreadas.
----------------------------------------

El tamaño de las ventanas influye en el balance térmico de la zona térmica y en la 
posibilidad de controlar la ganancia solar por medio de sombras y la ventilación
natural con la apertura de ventanas.

Se ha decidido variar el tamñano de cada una de las ventanas de manera independiente
en tamaños que van desde un 10% a un 90% de la superficie del muro en la que se encuentran.
Las ventanas son posicionadas en el centro del muro para simplificar el cálculo.

Todas las ventanas cuentan con dos actuadores, uno para operar el sombreado y otro para 
operar el nivel de apertura.

4. Propiedades de la envolvente.
---------------------------------

La envolvente está compuesta por 6 superficies: 4 muros, un techo y un piso.
Todas estas superficies pueden tener diferentes propiedades térmicas. Además, no solo
importa su valor final de conductividad y masa térmica, sino que también en qué 
orden se encuentra la masa térmica y la aislación. Disponer de masa térmica interior
o exterior cambia el comportamiento del entorno.

Para ello se ha constituido la envolvente, para sus muros, en tres capas:

* capa exterior: que principalmente cumple con funciones radiativas y de masa térmica.
* capa intermedia: que representa la propiedad aislante de la envolvente.
* capa interior: que cumple funciones radiativas y de masa térmica.

En el caso del techo, solo se consideran dos capas, una interior y otra exterior.

En el caso del piso, esta superficie no cambiará.

El espesor y las propiedades térmicas de cada capa, para cada tipo de superficie, se 
establecen dentro de un rango de valores.

Los rangos utilizados se muestran en la siguiente tabla:

| **Componente** | **Capa**     | **Espesor [m]** | **Conductividad [W/m·K]** | **Densidad [kg/m³]** | **Calor específico [J/kg·K]** | **Absortancia térmica** | **Absortancia solar** |
| -------------- | ------------ | ---------------- | -------------------------- | --------------------- | ------------------------------ | ----------------------- | --------------------- |
| **Muro**       | Interior (1) | 0.01 - 0.15      | 0.3 - 1.8                  | 800 - 2400            | 800 - 1200                     | 0.8 - 0.95              | 0.3 - 0.6             |
|                | Media (2)    | 0.02 - 0.20      | 0.02 - 0.08                | 10 - 150              | 900 - 1500                     | 0.3 - 0.6               | 0.2 - 0.4             |
|                | Exterior (3) | 0.01 - 0.20      | 0.2 - 2.0                  | 1000 - 2500           | 700 - 1100                     | 0.7 - 0.95              | 0.4 - 0.9             |
| **Techo**      | Inferior (1) | 0.01 - 0.20      | 0.3 - 1.5                  | 600 - 2200            | 800 - 1200                     | 0.7 - 0.95              | 0.3 - 0.6             |
|                | Superior (2) | 0.02 - 0.25      | 0.02 - 0.10                | 20 - 300              | 900 - 1600                     | 0.3 - 0.6               | 0.5 - 0.95            |


5. Propiedades de las ventanas.
---------------------------------

Las tipologías de sistemas de aberturas permiten configurar con precisión el funcionamiento
térmico de las ventanas. Sin embargo, para simplificar el modelado de diferentes escenarios
se controla la calidad y tipo de sistema con las siguientes variables:

* u_factor: Coeficiente global de pérdidas térmicas de las ventanas.
* solar_heat_gain_coefficient: Coeficiente de ganancia térmica de la superficie vidreada.

Se considera que todas las ventanas de la zona térmica están compuestas por la
misma calidad de abertura.

| **Propiedad**                 | **Símbolo** | **Unidad**   | **Rango típico** | **Descripción**                                                                  |
| ----------------------------- | ----------- | ------------ | ---------------- | -------------------------------------------------------------------------------- |
| Coeficiente de pérdidas       | $U$         | W/m²·K       | 0.8 - 6.0        | Valores bajos indican buen aislamiento térmico. Varía según vidrio y marco.      |
| Coeficiente de ganancia solar | $SHGC$      | adimensional | 0.1 - 0.87       | Fracción de radiación solar transmitida al interior. Depende del tipo de vidrio. |


| **Tipo de ventana**                           | $U$ [W/m²·K] | $SHGC$      |
| --------------------------------------------- | ------------- | ----------- |
| Vidrio simple, sin tratamiento                | 5.5 - 6.0     | 0.75 - 0.87 |
| Vidrio simple con película de control solar   | 4.5 - 5.5     | 0.4 - 0.6   |
| Vidrio doble sin baja emisividad              | 2.5 - 3.5     | 0.6 - 0.7   |
| Vidrio doble con baja emisividad (Low-E)      | 1.3 - 2.5     | 0.3 - 0.5   |
| Triple acristalamiento                        | 0.8 - 1.5     | 0.2 - 0.4   |
| Vidrio espectral selectivo (high-performance) | 0.9 - 1.8     | 0.1 - 0.35  |



6. Masa térmica interior.
--------------------------

La masa térmica interior que se encuentra en la zona térmica equivalente prismática
permite considerar los efectos de inercia térmica debido a los muros interiores
y mobiliario que aporta almacenamiento de energía térmica en la zona.

Esta variable considera un material constante y se varía la cantidad de superficie
que existe de este material. Para ello, se toma como referencia el área de la zona térmica
y se la multiplica por un valor en el rango de 0.2 a 0.4.

7. Periodo de simulación.
--------------------------

Se establece que cada episodio tenga una longitud de 7 días, donde se pueden apreciar los
fenómenos de inercia térmica y se aumenta la cantidad de episodios considerablemente para la generalización
del problema.

8. Potencia de los sistemas activos de climatización.
------------------------------------------------------

La potencia del equipo de acondicionamiento de aire, tanto para frio como para calor, pueden ser diferentes 
en los diferentes escenarios planteados. Se establecen tres niveles de potencias, considerando la de 
calefacción un 30% superior a la de enfriamiento por el tipo de clima utilizado. Estas potencias dependeran del
tamaño del edificio.


9. Perfil climático.
---------------------

Los climas utilizados corresponden a una misma región climática dentro de la Provincia 
de Mendoza.


10. Perfil de carga.
--------------------

Diferentes perfiles de carga se configuran.


11. Perfil de ocupación.
-------------------------

Se considera que la zona térmica está simpre ocupada.


Agentes considerados en el entrenamiento
=========================================

Los agentes entrenados son:

    "Setpoint", 
    "North Windows", 
    "South Windows", 
    "East Windows", 
    "West Windows", 
    "North Shades", 
    "South Shades", 
    "East Shades", 
    "West Shades"


Arquitectura de la política
============================

Se utiliza una política de parámetros totalmente compartidos. Se utilizan 3 capas totalmente conectadas
de 128 neuronas cada capa. La función de activación es del tipo tangente hiperbólica.


Entrenamiento por currículum
=============================

Se establece un entrenamiento basado en tareas, desde las más simples hasta las más complejas de forma
paulatina.

Las tareas son:

1. Control de termostato dual.
2. Control de sombreado: una ventana.
3. Control de sombreado: dos ventanas.
4. Control de sombreado: tres ventanas.
5. Control de sombreado: cuatro ventanas.
6. Control de ventilación natural: una ventana.
7. Control de ventilación natural: dos ventanas.
8. Control de ventilación natural: tres ventanas.
9. Control de ventilación natural: cuatro ventanas.
10. Control de termostato dual + Control de sombreado.
11. Control de termostato dual + Control de ventilación natural.
12. Control de sombreado + Control de ventilación natural.
13. Control de termostato dual + Control de sombreado + Control de ventilación natural.

Cada vez que se domine una de estas actividades se pasa al entrenamiento de la siguiente. Sin embargo, para 
evitar que la política "olvide" como actuar en una tarea previa, un porcentaje del tiempo del entrenamiento
se utilizarán tareas previamente aprendidas, aunque con una tasa mayor de la tarea que se está queriendo 
aprender.


Escenarios de evaluación
=========================

La evaluación de las políticas para los diferentes agentes se realizan con métricas energéticas y de violación
de temperaturas en la vivienda bioclimática del IPV para Mendoza (prototipo 3). Esta vivienda se encuentra
diseñada para diferentes orientaciones, por lo que se comparan las estrategias seguidas en cada caso.

Los resultados son las bases para un manual del usuario.
"""

import os
import json
import numpy as np
import tempfile
from typing import Any, Dict, Tuple, List
from numpy import float32
from numpy.typing import NDArray
from eprllib.Episodes.BaseEpisode import BaseEpisode
from eprllib.Utils.episode_fn_utils import get_random_weather, building_dimension
from eprllib.Utils.annotations import override


class task_cofiguration(BaseEpisode):
    REQUIRED_KEYS: Dict[str,Any] = {
        # 1. Orientación de ventanas. (Modelos)
        "epjson_files_folder_path": str,
        
        # 2. Área de la zona térmica.
        "building_size_h": List[float|int], #[2,4], # Rango de alturas (m)
        "building_size_w": List[float|int], #[5,20], # Rango de anchos (m)
        "building_size_l": List[float|int], #[5,20], # Rango de largos (m)
        
        # 3. Tamaño de las superficies vidreadas.
        "war_factor_range": List[float|int], #[0.1,0.9]
        
        # 4. Propiedades de la envolvente.
        # Muros
        "muro_exterior_thickness": List[float],
        "muro_exterior_conductivity": List[float],
        "muro_exterior_density": List[float],
        "muro_exterior_specific_heat": List[float],
        "muro_exterior_thermal_absorptance": List[float],
        "muro_exterior_solar_absorptance": List[float],
        "muro_intermedio_thickness": List[float],
        "muro_intermedio_conductivity": List[float],
        "muro_intermedio_density": List[float],
        "muro_intermedio_specific_heat": List[float],
        "muro_intermedio_thermal_absorptance": List[float],
        "muro_intermedio_solar_absorptance": List[float],
        "muro_interior_thickness": List[float],
        "muro_interior_conductivity": List[float],
        "muro_interior_density": List[float],
        "muro_interior_specific_heat": List[float],
        "muro_interior_thermal_absorptance": List[float],
        "muro_interior_solar_absorptance": List[float],
        
        # Techo
        "techo_exterior_thickness": List[float],
        "techo_exterior_conductivity": List[float],
        "techo_exterior_density": List[float],
        "techo_exterior_specific_heat": List[float],
        "techo_exterior_thermal_absorptance": List[float],
        "techo_exterior_solar_absorptance": List[float],
        "techo_interior_thickness": List[float],
        "techo_interior_conductivity": List[float],
        "techo_interior_density": List[float],
        "techo_interior_specific_heat": List[float],
        "techo_interior_thermal_absorptance": List[float],
        "techo_interior_solar_absorptance": List[float],
        
        # 5. Propiedades de las ventanas.
        "window_u_factor": List[float],
        "window_solar_heat_gain_coefficient": List[float],
        
        # 6. Masa térmica interior.
        "im_surface_area_factor": List[float],
        
        # 7. Periodo de simulación.
        "episode_len": int, # 7
        
        # 8. Potencia de los sistemas activos de climatización.
        "specific_power_hvac": List[float],
        "cooling_heating_ratio": List[float],
        
        # 9. Perfil climático.
        "epw_files_folder_path": str,
        
        # 10. Perfil de carga.
        "load_profiles_folder_path": str,
        
        # 11. Perfil de ocupación.
        "occupancy_profile_folder_path": str,
        
        # 12. Exhaust fan.
        "exhaust_fan_use": bool
    }
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
        
        Process:
            1)	Open IDF EnergyPlus file
            2)	Change the large and weight between 3 and 10 meters, while the high is constant and equal to 3 meters.
            3)	Calculate the building floor area and the aspect ratio as w*l and w/l respectively.
            4)	Define the window area relation for each orientation and change the dimensions of windows in the model.
            5)	Change the properties of the envelope shapes.
            6)	Change the properties of the windows.
            7)	Change the internal mass.
            8)	Calculate the inertial mass of the building.
            9)	Calculate the global construction U-factor for the building.
            10)	Define the maximum refrigeration power in the building.
            11)	Define the maximum heating power in the building.
            12)	Save the new random building model.
            13)	Select a random weather file.


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
        
        # Verificación de los tipos de las claves solicitadas.
        # config_validation(episode_fn_config, self.REQUIRED_KEYS)
        
        self.agents = None
        self.episode_agents = []
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
        
        # 1. Orientación de ventanas. (Modelos)
        self.epjson_files_folder_path: str = episode_fn_config['epjson_files_folder_path']
        
        # 2. Área de la zona térmica.
        self.building_size: List[float] = []
        self.building_size_h = episode_fn_config['building_size_h']
        self.building_size_w = episode_fn_config['building_size_w']
        self.building_size_l = episode_fn_config['building_size_l']
        
        # 3. Tamaño de las superficies vidreadas.
        self.war_factor_range = episode_fn_config['war_factor_range']
        
        # 4. Propiedades de la envolvente.
        # Muros
        self.muro_exterior_thickness = episode_fn_config['muro_exterior_thickness']
        self.muro_exterior_conductivity = episode_fn_config['muro_exterior_conductivity']
        self.muro_exterior_density = episode_fn_config['muro_exterior_density']
        self.muro_exterior_specific_heat = episode_fn_config['muro_exterior_specific_heat']
        self.muro_exterior_thermal_absorptance = episode_fn_config['muro_exterior_thermal_absorptance']
        self.muro_exterior_solar_absorptance = episode_fn_config['muro_exterior_solar_absorptance']
        self.muro_intermedio_thickness = episode_fn_config['muro_intermedio_thickness']
        self.muro_intermedio_conductivity = episode_fn_config['muro_intermedio_conductivity']
        self.muro_intermedio_density = episode_fn_config['muro_intermedio_density']
        self.muro_intermedio_specific_heat = episode_fn_config['muro_intermedio_specific_heat']
        self.muro_intermedio_thermal_absorptance = episode_fn_config['muro_intermedio_thermal_absorptance']
        self.muro_intermedio_solar_absorptance = episode_fn_config['muro_intermedio_solar_absorptance']
        self.muro_interior_thickness = episode_fn_config['muro_interior_thickness']
        self.muro_interior_conductivity = episode_fn_config['muro_interior_conductivity']
        self.muro_interior_density = episode_fn_config['muro_interior_density']
        self.muro_interior_specific_heat = episode_fn_config['muro_interior_specific_heat']
        self.muro_interior_thermal_absorptance = episode_fn_config['muro_interior_thermal_absorptance']
        self.muro_interior_solar_absorptance = episode_fn_config['muro_interior_solar_absorptance']
        # Techo
        self.techo_exterior_thickness = episode_fn_config['techo_exterior_thickness']
        self.techo_exterior_conductivity = episode_fn_config['techo_exterior_conductivity']
        self.techo_exterior_density = episode_fn_config['techo_exterior_density']
        self.techo_exterior_specific_heat = episode_fn_config['techo_exterior_specific_heat']
        self.techo_exterior_thermal_absorptance = episode_fn_config['techo_exterior_thermal_absorptance']
        self.techo_exterior_solar_absorptance = episode_fn_config['techo_exterior_solar_absorptance']
        self.techo_interior_thickness = episode_fn_config['techo_interior_thickness']
        self.techo_interior_conductivity = episode_fn_config['techo_interior_conductivity']
        self.techo_interior_density = episode_fn_config['techo_interior_density']
        self.techo_interior_specific_heat = episode_fn_config['techo_interior_specific_heat']
        self.techo_interior_thermal_absorptance = episode_fn_config['techo_interior_thermal_absorptance']
        self.techo_interior_solar_absorptance = episode_fn_config['techo_interior_solar_absorptance']

        # 5. Propiedades de las ventanas.
        self.window_u_factor = episode_fn_config['window_u_factor']
        self.window_solar_heat_gain_coefficient = episode_fn_config['window_solar_heat_gain_coefficient']

        # 6. Masa térmica interior.
        self.im_surface_area_factor = episode_fn_config['im_surface_area_factor']
        
        # 7. Periodo de simulación.
        self.episode_len: int = episode_fn_config['episode_len']
        
        # 8. Potencia de los sistemas activos de climatización.
        self.specific_power_hvac: List[float] = episode_fn_config['specific_power_hvac']
        self.cooling_heating_ratio: List[float] = episode_fn_config['cooling_heating_ratio']
        
        # 9. Perfil climático.
        self.epw_files_folder_path: str = episode_fn_config['epw_files_folder_path']
        
        # 10. Perfil de carga.
        self.load_profiles_folder_path: str = episode_fn_config['load_profiles_folder_path']
        
        # 11. Perfil de ocupación.
        self.occupancy_profile_folder_path: str = episode_fn_config['occupancy_profile_folder_path']
        
        # 12. Exhaust fan.
        self.exhaust_fan_use: bool = episode_fn_config['exhaust_fan_use']
        
        self.training_task: List[str] = episode_fn_config['training_task']

        
    @override(BaseEpisode)
    def get_episode_config(self, env_config: Dict[str,Any]) -> Dict[str,Any]:
        """
        This method define the properties of the episode. Changing some properties as weather or 
        Run Time Period, and defining others fix properties as volumen or window area relation.
        
        Return:
            dict: The method returns the env_config with modifications.
        """    
        # print(f"DEBUG: Entering get_episode_config. env_config keys: {list(env_config.keys())}")
        if self.agents is None:
            self.agents = [agent for agent in env_config['agents_config'].keys()]
        
        # Se crea una lista vacía para almacenar los agentes que serán parte en el episodio
        self.episode_agents: List[str] = []
        
        # Se asigna el agente HVAC que siempre está presente para poder establecer
        # un cosumo energético y los termostatos establecidos.
        if "HVAC" in self.training_task:
            self.episode_agents.append("HVAC")
        # Verificar que el agente agregado exista en la configuración.
            if not "HVAC" in self.agents:
                NameError("El agente HVAC no está presente en la configuración de eprllib utilizada.")
            
        # === 1. Orientación de ventanas ===
        # ===================================================================================
        
        # Selección del modelo en el rango [1, 16)
        model: int = np.random.randint(1,16)
        # print(f"DEBUG: Selected model: {model}")
        
        # Una vez seleccionado el modelo, se lee su configuración para poder 
        # manipularlo y establecer el modelo particular para el episodio.
        with open(f"{self.epjson_files_folder_path}/model_{model}.epJSON") as file:
            epJSON_object: Dict[str,Any] = json.load(file)
        
        # === 2. Área de la zona térmica ===
        # ===================================================================================
        
        # Se asignan las dimensiones de la zona térmica de acuerdo a los rangos
        # establecidos.
        # print(f"DEBUG: Building size ranges: \n- h: {self.building_size_h}, type: {type(self.building_size_h)}\n- w: {self.building_size_w}, type: {type(self.building_size_w)}\n- l: {self.building_size_l}, type: {type(self.building_size_l)}")
        h = self.value_from_range(self.building_size_h)
        w = self.value_from_range(self.building_size_w)
        l = self.value_from_range(self.building_size_l)
        # print(f"DEBUG: Building dimensions - h: {h:.2f}, w: {w:.2f}, l: {l:.2f}")
        
        # === 3. Tamaño de las superficies vidreadas ===
        # ===================================================================================
        
        # Se crea una lista que guardará la relación de superficie vidreada / superficie de muro
        # para cada orientación.
        window_area_relation_list: List[float] = []
        # Se lee del modelo la orientación de las ventanas presentes.
        model_window_config = self.model_window_configs[str(model)]
        # Variable para modificar la condición de ventilación en la ventana.
        surface_number = 0
        # Lazo para iterar en las cuatro orientaciones
        for i in range(4):
            # Condición de presencia de la ventana en la orientación iterada.
            if model_window_config[i] == 1:
                # Incremento de la ventana presente en el modelo.
                surface_number += 1
                # Selección aleatoria del tamaño de la ventana.
                window_area_relation_list.append(self.value_from_range(self.war_factor_range))
                # add the respective agents to the list of episode_agents
                if i == 0:
                    if "NaturalVentilation" in self.training_task:
                        epJSON_object["AirflowNetwork:MultiZone:Surface"][f"AirflowNetwork:MultiZone:Surface {surface_number}"]["ventilation_control_mode"] = "Constant"
                        self.episode_agents.append("North Windows")
                        if not "North Windows" in self.agents:
                            NameError("El agente North Windows no está presente en la configuración de eprllib utilizada.")
                    else:
                        epJSON_object["AirflowNetwork:MultiZone:Surface"][f"AirflowNetwork:MultiZone:Surface {surface_number}"]["ventilation_control_mode"] = "NoVent"
                    
                    if "ShadingControl" in self.training_task:
                        epJSON_object["WindowShadingControl"]["window_north"]["shading_control_type"] = "OnIfScheduleAllows"
                        epJSON_object["WindowShadingControl"]["window_north"]["shading_control_is_scheduled"] = "Yes"
                        epJSON_object["WindowShadingControl"]["window_north"]["schelude_name"] = "north_shading_control"
                        self.episode_agents.append("North Shades")
                        if not "North Shades" in self.agents:
                            NameError("El agente North Shades no está presente en la configuración de eprllib utilizada.")
                        
                elif i == 1:
                    if "NaturalVentilation" in self.training_task:
                        epJSON_object["AirflowNetwork:MultiZone:Surface"][f"AirflowNetwork:MultiZone:Surface {surface_number}"]["ventilation_control_mode"] = "Constant"
                        self.episode_agents.append("East Windows")
                        if not "East Windows" in self.agents:
                            NameError("El agente East Windows no está presente en la configuración de eprllib utilizada.")
                    else:
                        epJSON_object["AirflowNetwork:MultiZone:Surface"][f"AirflowNetwork:MultiZone:Surface {surface_number}"]["ventilation_control_mode"] = "NoVent"
                    if "ShadingControl" in self.training_task:
                        epJSON_object["WindowShadingControl"]["window_north"]["shading_control_type"] = "OnIfScheduleAllows"
                        epJSON_object["WindowShadingControl"]["window_north"]["shading_control_is_scheduled"] = "Yes"
                        epJSON_object["WindowShadingControl"]["window_north"]["schelude_name"] = "east_shading_control"
                        self.episode_agents.append("East Shades")
                        if not "East Windows" in self.agents:
                            NameError("El agente East Windows no está presente en la configuración de eprllib utilizada.")
                
                elif i == 2:
                    if "NaturalVentilation" in self.training_task:
                        epJSON_object["AirflowNetwork:MultiZone:Surface"][f"AirflowNetwork:MultiZone:Surface {surface_number}"]["ventilation_control_mode"] = "Constant"
                        self.episode_agents.append("South Windows")
                        if not "South Windows" in self.agents:
                            NameError("El agente South Windows no está presente en la configuración de eprllib utilizada.")
                    else:
                        epJSON_object["AirflowNetwork:MultiZone:Surface"][f"AirflowNetwork:MultiZone:Surface {surface_number}"]["ventilation_control_mode"] = "NoVent"
                    if "ShadingControl" in self.training_task:
                        epJSON_object["WindowShadingControl"]["window_north"]["shading_control_type"] = "OnIfScheduleAllows"
                        epJSON_object["WindowShadingControl"]["window_north"]["shading_control_is_scheduled"] = "Yes"
                        epJSON_object["WindowShadingControl"]["window_north"]["schelude_name"] = "south_shading_control"
                        self.episode_agents.append("South Shades")
                        if not "South Windows" in self.agents:
                            NameError("El agente South Windows no está presente en la configuración de eprllib utilizada.")
                
                elif i == 3:
                    if "NaturalVentilation" in self.training_task:
                        epJSON_object["AirflowNetwork:MultiZone:Surface"][f"AirflowNetwork:MultiZone:Surface {surface_number}"]["ventilation_control_mode"] = "Constant"
                        self.episode_agents.append("West Windows")
                        if not "West Windows" in self.agents:
                            NameError("El agente West Windows no está presente en la configuración de eprllib utilizada.")
                    else:
                        epJSON_object["AirflowNetwork:MultiZone:Surface"][f"AirflowNetwork:MultiZone:Surface {surface_number}"]["ventilation_control_mode"] = "NoVent"
                    if "ShadingControl" in self.training_task:
                        epJSON_object["WindowShadingControl"]["window_north"]["shading_control_type"] = "OnIfScheduleAllows"
                        epJSON_object["WindowShadingControl"]["window_north"]["shading_control_is_scheduled"] = "Yes"
                        epJSON_object["WindowShadingControl"]["window_north"]["schelude_name"] = "west_shading_control"
                        self.episode_agents.append("West Shades")
                        if not "West Windows" in self.agents:
                            NameError("El agente West Windows no está presente en la configuración de eprllib utilizada.")
            else:
                window_area_relation_list.append(0)
        
        # Se agregan actualizan variables del espacio de observaciones de los agentes
        # que tienen en cuenta el tamaño de las ventanas.
        for agent in self.agents:
            env_config["agents_config"][agent]["observation"]["other_obs"]["WWR-North"] = window_area_relation_list[0]
            env_config["agents_config"][agent]["observation"]["other_obs"]["WWR-East"] = window_area_relation_list[1]
            env_config["agents_config"][agent]["observation"]["other_obs"]["WWR-South"] = window_area_relation_list[2]
            env_config["agents_config"][agent]["observation"]["other_obs"]["WWR-West"] = window_area_relation_list[3]
        # Se transforma la lista en un NDArray.
        # print(f"DEBUG: Window area relation (N,E,S,W): {window_area_relation}")
        window_area_relation: NDArray[float32] = np.array(window_area_relation_list, dtype=np.float32)
        # Se modifica el tamaño del modelo y de las ventanas.
        epJSON_object = building_dimension(epJSON_object, h, w, l, window_area_relation)
        
        # === 4. Propiedades de la envolvente ===
        # ===================================================================================
        
        # Se cambian las propiedades térmicas de cada material que componen los materiales
        # de las contrucciones.
        
        # Muros.
        # ------
        # Capa exterior.
        epJSON_object["Material"]["wall_exterior"]["thickness"] = self.value_from_range(self.muro_exterior_thickness)
        epJSON_object["Material"]["wall_exterior"]["conductivity"] = self.value_from_range(self.muro_exterior_conductivity)
        epJSON_object["Material"]["wall_exterior"]["density"] = self.value_from_range(self.muro_exterior_density)
        epJSON_object["Material"]["wall_exterior"]["specific_heat"] = self.value_from_range(self.muro_exterior_specific_heat)
        epJSON_object["Material"]["wall_exterior"]["thermal_absorptance"] = self.value_from_range(self.muro_exterior_thermal_absorptance)
        epJSON_object["Material"]["wall_exterior"]["solar_absorptance"] = epJSON_object["Material"]["wall_exterior"]["visible_absorptance"] = self.value_from_range(self.muro_exterior_solar_absorptance)
        # Capa intermedia.
        epJSON_object["Material"]["wall_inter"]["thickness"] = self.value_from_range(self.muro_intermedio_thickness)
        epJSON_object["Material"]["wall_inter"]["conductivity"] = self.value_from_range(self.muro_intermedio_conductivity)
        epJSON_object["Material"]["wall_inter"]["density"] = self.value_from_range(self.muro_intermedio_density)
        epJSON_object["Material"]["wall_inter"]["specific_heat"] = self.value_from_range(self.muro_intermedio_specific_heat)
        epJSON_object["Material"]["wall_inter"]["thermal_absorptance"] = self.value_from_range(self.muro_intermedio_thermal_absorptance)
        epJSON_object["Material"]["wall_inter"]["solar_absorptance"] = epJSON_object["Material"]["wall_inter"]["visible_absorptance"] = self.value_from_range(self.muro_intermedio_solar_absorptance)
        # Capa interior.
        epJSON_object["Material"]["wall_inner"]["thickness"] = self.value_from_range(self.muro_interior_thickness)
        epJSON_object["Material"]["wall_inner"]["conductivity"] = self.value_from_range(self.muro_interior_conductivity)
        epJSON_object["Material"]["wall_inner"]["density"] = self.value_from_range(self.muro_interior_density)
        epJSON_object["Material"]["wall_inner"]["specific_heat"] = self.value_from_range(self.muro_interior_specific_heat)
        epJSON_object["Material"]["wall_inner"]["thermal_absorptance"] = self.value_from_range(self.muro_interior_thermal_absorptance)
        epJSON_object["Material"]["wall_inner"]["solar_absorptance"] = epJSON_object["Material"]["wall_inter"]["visible_absorptance"] = self.value_from_range(self.muro_interior_solar_absorptance)
        
        # Techo.
        # ------
        # Capa exterior.
        epJSON_object["Material"]["roof_exterior"]["thickness"] = self.value_from_range(self.techo_exterior_thickness)
        epJSON_object["Material"]["roof_exterior"]["conductivity"] = self.value_from_range(self.techo_exterior_conductivity)
        epJSON_object["Material"]["roof_exterior"]["density"] = self.value_from_range(self.techo_exterior_density)
        epJSON_object["Material"]["roof_exterior"]["specific_heat"] = self.value_from_range(self.techo_exterior_specific_heat)
        epJSON_object["Material"]["roof_exterior"]["thermal_absorptance"] = self.value_from_range(self.techo_exterior_thermal_absorptance)
        epJSON_object["Material"]["roof_exterior"]["solar_absorptance"] = epJSON_object["Material"]["roof_exterior"]["visible_absorptance"] = self.value_from_range(self.techo_exterior_solar_absorptance)
        # Capa interior.
        epJSON_object["Material"]["roof_inner"]["thickness"] = self.value_from_range(self.techo_interior_thickness)
        epJSON_object["Material"]["roof_inner"]["conductivity"] = self.value_from_range(self.techo_interior_conductivity)
        epJSON_object["Material"]["roof_inner"]["density"] = self.value_from_range(self.techo_interior_density)
        epJSON_object["Material"]["roof_inner"]["specific_heat"] = self.value_from_range(self.techo_interior_specific_heat)
        epJSON_object["Material"]["roof_inner"]["thermal_absorptance"] = self.value_from_range(self.techo_interior_thermal_absorptance)
        epJSON_object["Material"]["roof_inner"]["solar_absorptance"] = epJSON_object["Material"]["roof_inner"]["visible_absorptance"] = self.value_from_range(self.techo_interior_solar_absorptance)
        
        
        # === 5. Propiedades de las ventanas ===
        # ===================================================================================
        
        # Windows
        # Change the window thermal properties
        epJSON_object['WindowMaterial:SimpleGlazingSystem']['WindowMaterial']['u_factor'] = self.value_from_range(self.window_u_factor)
        epJSON_object['WindowMaterial:SimpleGlazingSystem']['WindowMaterial']['solar_heat_gain_coefficient'] = self.value_from_range(self.window_solar_heat_gain_coefficient)
        
        
        #  === 6. Masa térmica interior ===
        # ===================================================================================
        
        # The internal thermal mass is modified.
        for key in [key for key in epJSON_object["InternalMass"].keys()]:
            epJSON_object["InternalMass"][key]["surface_area"] = (w*l) * self.value_from_range(self.im_surface_area_factor)
        
        
        #  === 7. Periodo de simulación ===
        # ===================================================================================
        
        # RunPeriod in winter time (south hemisphere)
        run_period_dates = self.run_period(np.random.randint(1,364-self.episode_len),self.episode_len)
        epJSON_object['RunPeriod']['Run Period 1']['begin_month'] = run_period_dates[0]
        epJSON_object['RunPeriod']['Run Period 1']['begin_day_of_month'] = run_period_dates[1]
        epJSON_object['RunPeriod']['Run Period 1']['end_month'] = run_period_dates[2]
        epJSON_object['RunPeriod']['Run Period 1']['end_day_of_month'] = run_period_dates[3]
        
        
        # === 8. Potencia de los sistemas activos de climatización ===
        # ===================================================================================
        
        # The limit capacity of bouth cooling and heating are changed.
        HVAC_names = [key for key in epJSON_object["ZoneHVAC:IdealLoadsAirSystem"].keys()]
        number_of_timesteps_per_hour = epJSON_object['Timestep']['Timestep 1']['number_of_timesteps_per_hour']
        for hvac in HVAC_names:
            epJSON_object["ZoneHVAC:IdealLoadsAirSystem"][hvac]["maximum_sensible_heating_capacity"] = (w*l) * self.value_from_range(self.specific_power_hvac)
            epJSON_object["ZoneHVAC:IdealLoadsAirSystem"][hvac]["maximum_total_cooling_capacity"] = epJSON_object["ZoneHVAC:IdealLoadsAirSystem"][hvac]["maximum_sensible_heating_capacity"] * (self.value_from_range(self.cooling_heating_ratio))
            for agent in self.agents:
                env_config['agents_config'][agent]["reward"]['reward_fn_config']['heating_energy_ref'] = epJSON_object["ZoneHVAC:IdealLoadsAirSystem"][hvac]["maximum_sensible_heating_capacity"] * 3600/number_of_timesteps_per_hour
                env_config['agents_config'][agent]["reward"]['reward_fn_config']['cooling_energy_ref'] = epJSON_object["ZoneHVAC:IdealLoadsAirSystem"][hvac]["maximum_total_cooling_capacity"] * 3600/number_of_timesteps_per_hour
        
        
        # === 9. Perfil climático ===
        # ===================================================================================
        
        # Establecer un clima aleatorio durante el entrenamiento
        env_config["epw_path"] = get_random_weather(self.epw_files_folder_path)
        # print(f"DEBUG: Selected EPW weather file: {env_config['epw_path']}")
        
        
        # === 10. Perfil de carga ===
        # ===================================================================================
        
        # Change the load file profiles names to the new copy of schedule
        # print(f"DEBUG: Available load profiles in {self.load_profiles_folder_path}: {os.listdir(self.load_profiles_folder_path)}")
        schedule_file_keys = [key for key in epJSON_object["Schedule:File"].keys()]
        for key in schedule_file_keys:
            selected_load_profile = np.random.choice(os.listdir(self.load_profiles_folder_path))
            epJSON_object["Schedule:File"][key]["file_name"] = os.path.join(self.load_profiles_folder_path, selected_load_profile)
            # print(f"DEBUG: Schedule:File '{key}' set to use load profile: {selected_load_profile}")
        
        # === 11. Perfil de ocupación ===
        # ===================================================================================
        
        # TODO: Implementar rutina para cambiar el perfil de ocupación.
        # Para esta primera etapa se considera que el lugar estará siempre ocupado. De esta manera
        # se espera aprender las tareas necesarias para cada agente en el caso de que se quiera
        # confort térmico siempre.
        people_names = [key for key in epJSON_object["People"].keys()]
        for key in people_names:
            epJSON_object["People"][key]["number_of_people_schedule_name"] = "Always On"
        
        # === 12. Ventilador de extracción ===
        # ===================================================================================
        
        if self.exhaust_fan_use:
            exhaust_availability: bool = np.random.choice([True, False])
        else:
            exhaust_availability = False
        # print(f"DEBUG: Exhaust fan availability: {exhaust_availability}")
            
        ExhaustFan_names: List[str] = [key for key in epJSON_object["Fan:ZoneExhaust"].keys()]
        for fan in ExhaustFan_names:
            # Change the availability
            epJSON_object["Fan:ZoneExhaust"][fan]["availability_schedule_name"] = "Always On" if exhaust_availability else "Always Off"
            for surface in range(1,len(epJSON_object["AirflowNetwork:MultiZone:Surface"])+1):
                if epJSON_object["AirflowNetwork:MultiZone:Surface"][f"AirflowNetwork:MultiZone:Surface {surface}"]["leakage_component_name"] == "ExhaustFan":
                    # Change the control mode to allow ventilation or not.
                    epJSON_object["AirflowNetwork:MultiZone:Surface"][f"AirflowNetwork:MultiZone:Surface {surface}"]["ventilation_control_mode"] = "Constant" if exhaust_availability else "NoVent"
                    # Change the surface where the fan is transfering the heat.
                    epJSON_object["AirflowNetwork:MultiZone:Surface"][f"AirflowNetwork:MultiZone:Surface {surface}"]["surface_name"] = np.random.choice(["wall_north", "wall_east", "wall_south", "wall_west"])
                    # Add the agent that control the exhaust fan.
                    # Se asigna al agente de control.
                    if exhaust_availability:
                        # print(f"DEBUG: Added agent: Exhaust Fan")
                        self.episode_agents.append("Exhaust Fan")
                        # Verificar que el agente agregado exista en la configuración.
                        if not "Exhaust Fan" in self.agents:
                            NameError("El agente Exhaust Fan no está presente en la configuración de eprllib utilizada.")
    
            
        # === FIN DE LA CONFIGURACION === (exportación del modelo para el episodio)
        # ===================================================================================
        # The new modify epjson file is writed.
        env_config["epjson_path"] = os.path.join(self.temp_dir, f"temp-{os.getpid()}.epJSON")
        # The new modify epjson file is writed.
        with open(env_config["epjson_path"], 'w') as fp:
            json.dump(epJSON_object, fp, sort_keys=False, indent=4)
        # print(f"DEBUG: Generated epJSON file at: {env_config['epjson_path']}")
        # print(f"DEBUG: Final episode agents: {self.episode_agents}")
        return env_config
    
    
    @override(BaseEpisode)
    def get_episode_agents(self, env_config: Dict[str,Any], possible_agents: List[str]) -> List[str]:
        """
        This method returns the agents for the episode configuration in the EnergyPlus environment.

        Returns:
            List[str]: The agent that are acting for the episode configuration. Default: possible_agent list.
        """
        return self.episode_agents
    
    
    @override(BaseEpisode)
    def get_timestep_agents(self, env_config: Dict[str,Any], possible_agents: List[str]) -> List[str]:
        """
        This method returns the agents for the episode configuration in the EnergyPlus environment.

        Returns:
            List[str]: The agent that are acting for the episode configuration. Default: possible_agent list.
        """
        return self.get_episode_agents(env_config, possible_agents)
    
    
    
    # === MÉTODOS DE UTILIDAD PARA LA FUNCIÓN DE EPISODIO ===
    # ===================================================================================
    
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
    
    
    @staticmethod
    def max_day_in_month(month:int) -> int:
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
    
    @staticmethod
    def value_from_range(range_list: List[float]) -> float:
        """
        This function returns a random value from a given range.
        """
        return min(range_list) + np.random.sample() * (max(range_list) - min(range_list))