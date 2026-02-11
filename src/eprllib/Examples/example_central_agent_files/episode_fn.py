"""
Tarea: Control de Ventilación Mixta (ventanas y extractor)
===========================================================

Se analiza en este caso el control de los termostato de los sistemas activos (sistema de aire acondicionado) de una
oficina localizada en España. La oficina cuenta con un sistema de HVAC, un estractor de aire y dos ventanas operables
en su muro sur. Además, un control de sombras en estas ventanas posibilita la gestión de la ganancia solar.
La zona térmica es modelada como una única zona térmia prismática.
con un agente que opera los siguientes actuadores:

1. ("Schedule:Compact", "Schedule Value", "heating_setpoint"),
2. ("Schedule:Compact", "Schedule Value", "cooling_setpoint"),
3. ("Schedule:Constant", "Schedule Value", "HVAC_OnOff")
4. ("Schedule:Constant", "Schedule Value", "ExhaustFanMode")
5. ("AirFlow Network Window/Door Opening", "Venting Opening Factor", "window_south")
6. ("Window Shading Control", "Control Status", "window_south"),

El tamaño del edificio se lo hace variar aleatoriamente dentro de un rango definido. Esto permite extender el espacio de
observaciones y preparar el modelo para condiciones reales.

Se definen características térmicas de la envolvente variables en un rango apropiado para abarcar las posibles condiciones
reales de funcionamiento. En la evaluación se consideran las siguientes condiciones específicas para establecer un marco de
referencia:

[TABLA DE CONSTRUCCIONES Y MATERIALES]

Los perfiles de ocupación son prestablecidos según los datos recopilados en la Universidad Vasca. Se considera un perfil de ocupación
de una persona que trabaja fuera de casa con un calendario ajustado al horario de comercio e industria.

Para el relevo de las cargas internas, se consideran los siguientes perfiles de carga (y se les asigna una variación aleatoria para
extender el espacio de observaciones):

[PERFILES DE CARGAS INTERNAS]

La potencia del equipo de acondicionamiento de aire, tanto para frio como para calor, se han establecido según los datos
reales del equipo instalado en la oficina. 

Además de la operación de las ventanas, se consideran los efectos de las infiltraciones. Para aumentar el espacio de observaciones
y preparar el modelo para condiciones reales, el nivel de infiltraciones se hace variar sutilmente para cada episodio de manera 
aleatoria.

El control de sombra en la ventana sur se hace con un agente integrado en la política de parámetros totalmente compartidos.

Se establece que cada episodio tenga una longitud de 7 días, donde se pueden apreciar los
fenómenos de inercia térmica y se aumenta la cantidad de episodios considerablemente para la generalización
del problema. Para ello se plantearon dos escenarios: uno con las semanas consecutivas (empezando en enero y finalizando
en diciembre) y otro con semanas aleatorias.

Los climas utilizados corresponden a una misma región climática. Para la 
evaluación se utiliza un clima similar, pero no utilizado en el entrenamiento.

La evaluación de las políticas para los diferentes agentes se realizan con métricas energéticas y de violación
de temperaturas de confort.
"""
import logging
import os
import json
import numpy as np
import tempfile
from typing import Any, Dict, Tuple, List
from eprllib.Episodes.BaseEpisode import BaseEpisode
from eprllib.Utils.episode_fn_utils import get_random_weather

logger = logging.getLogger("ray.rllib")

class task_cofiguration(BaseEpisode):
    def __init__(
        self, episode_fn_config:Dict[str,Any]
    ):
        
        super().__init__(episode_fn_config)
        
        self.agents = ["Central Agent"]
        self.episode_agents = ["Central Agent"]
        self.epjson_model = episode_fn_config['epjson_model']
        self.epw_files_folder_path: str = episode_fn_config['epw_files_folder_path']
        self.episode_len = episode_fn_config['episode_len']
        self.temp_dir = tempfile.gettempdir()
        logger.info(f"Temporary directory for models: {self.temp_dir}")
        
        self.first = True
            
    def get_episode_config(self, env_config: Dict[str,Any]) -> Dict[str,Any]:
        """
        This method define the properties of the episode. Changing some properties as weather or 
        Run Time Period, and defining others fix properties as volumen or window area relation.
        
        Return:
            dict: The method returns the env_config with modifications.
        """    
        # Establish the epJSON Object, it will be manipulated to modify the building model.
        logger.debug(f"epJSON model opening is starting...")
        with open(self.epjson_model) as file:
            epJSON_object: Dict[str,Any] = json.load(file)
        logger.debug(f"epJSON model opened.")
        
        # === For evaluation ===
        if env_config["evaluation"]:
            logger.debug("Evaluation mode")
            # RunPeriod (use cut_period_len to define the period length).
            epJSON_object['RunPeriod']['Run Period 1']['begin_month'] = 1
            epJSON_object['RunPeriod']['Run Period 1']['begin_day_of_month'] = 1
            epJSON_object['RunPeriod']['Run Period 1']['end_month'] = 12
            epJSON_object['RunPeriod']['Run Period 1']['end_day_of_month'] = 31
            logger.debug(f"EpJSON RunPeriod modified.")
            # Establecer el clima para evaluación
            env_config["epw_path"] = self.epw_files_folder_path + "/" +os.listdir(self.epw_files_folder_path)[0]
            logger.debug(f"Epw path: {env_config['epw_path']}")
            # Permitir la escritura de resultados
            for metric in epJSON_object['OutputControl:Files']['OutputControl:Files 1'].keys():
                epJSON_object['OutputControl:Files']['OutputControl:Files 1'][metric] = "Yes"
        
        # === For training ===
        else:
            logger.debug("Training mode")
            # RunPeriod in winter time (south hemisphere)
            run_period_dates = self.run_period(np.random.randint(1,(365-self.episode_len)),self.episode_len)
            epJSON_object['RunPeriod']['Run Period 1']['begin_month'] = run_period_dates[0]
            epJSON_object['RunPeriod']['Run Period 1']['begin_day_of_month'] = run_period_dates[1]
            epJSON_object['RunPeriod']['Run Period 1']['end_month'] = run_period_dates[2]
            epJSON_object['RunPeriod']['Run Period 1']['end_day_of_month'] = run_period_dates[3]
            logger.debug(f"EpJSON RunPeriod modified.")
            # epJSON_object['RunPeriod']['Run Period 1']['begin_month'] = 11
            # epJSON_object['RunPeriod']['Run Period 1']['begin_day_of_month'] = 1
            # epJSON_object['RunPeriod']['Run Period 1']['end_month'] = 11
            # epJSON_object['RunPeriod']['Run Period 1']['end_day_of_month'] = 7
            # logger.debug(f"EpJSON RunPeriod modified.")
            # Establecer un clima aleatorio durante el entrenamiento
            env_config["epw_path"] = get_random_weather(self.epw_files_folder_path)
            logger.debug(f"Epw path: {env_config['epw_path']}")
            
            for metric in epJSON_object['OutputControl:Files']['OutputControl:Files 1'].keys():
                epJSON_object['OutputControl:Files']['OutputControl:Files 1'][metric] = "No"
        
        # The new modify epjson file is writed.
        env_config["epjson_path"] = os.path.join(self.temp_dir, f"temp-{os.getpid()}.epJSON")
        logger.debug(f"EpJSON path: {env_config['epjson_path']}")
        # The new modify epjson file is writed.
        with open(env_config["epjson_path"], 'w') as fp:
            json.dump(epJSON_object, fp, sort_keys=False, indent=4)
        logger.debug(f"EpJSON file writed: {env_config['epjson_path']}")
        if self.first:
            self.first = False
            epjson_path = os.path.join("C:/Users/grhen/Documents/GitHub/SimpleCases/configurations", "task_esp.epJSON")
            with open(epjson_path, 'w') as fp:
                json.dump(epJSON_object, fp, sort_keys=False, indent=4)
            logger.debug(f"EpJSON file writed: {epjson_path}")
        return env_config

    def get_episode_agents(self, env_config: Dict[str,Any], possible_agents: List[str]) -> List[str]:
        """
        This method returns the agents for the episode configuration in the EnergyPlus environment.

        Returns:
            List[str]: The agent that are acting for the episode configuration. Default: possible_agent list.
        """
        logger.debug(f"Episode agents: {self.episode_agents}")
        return self.episode_agents
    
    def get_timestep_agents(self, env_config: Dict[str,Any], possible_agents: List[str]) -> List[str]:
        """
        This method returns the agents for the episode configuration in the EnergyPlus environment.

        Returns:
            List[str]: The agent that are acting for the episode configuration. Default: possible_agent list.
        """
        logger.debug(f"Timestep agents: {self.episode_agents}")
        return self.get_episode_agents(env_config, possible_agents)



    
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
    
    def max_day_in_month(self, month:int) -> int:
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
    
