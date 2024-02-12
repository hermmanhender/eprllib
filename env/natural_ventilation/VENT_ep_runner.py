"""# ENERGYPLUS RUNNER

"""

import os
import sys
import tools
import threading
import numpy as np
os_platform = sys.platform
if os_platform == "linux":
    sys.path.insert(0, '/usr/local/EnergyPlus-23-2-0')
else:
    sys.path.insert(0, 'C:/EnergyPlusV23-2-0')
    
from pyenergyplus.api import EnergyPlusAPI
api = EnergyPlusAPI()

from queue import Queue
from time import sleep
from typing import Any, Dict, List, Optional

class EnergyPlusRunner:
    """_summary_
    """
    def __init__(
        self,
        episode: int,
        env_config: Dict[str, Any],
        agents_ids: list,
        obs_queue: Queue,
        act_queue: Queue,
        cooling_queue: Queue,
        heating_queue: Queue,
        pmv_queue: Queue,
        PPD_queue: Queue
        ) -> None:
        """_summary_

        Args:
            episode (int): _description_
            env_config (Dict[str, Any]): _description_
            agents_ids (list): _description_
            obs_queue (Queue): _description_
            act_queue (Queue): _description_
            cooling_queue (Queue): _description_
            heating_queue (Queue): _description_
            pmv_queue (Queue): _description_
            PPD_queue (Queue): _description_
        """
        # Se establecen las variables generales de configuración
        self.episode = episode
        self.env_config = env_config
        self.agents_ids = agents_ids
        # Se asignan las variables con las cuales se coordina la ejecución 
        # del hilo principal con este hilo secundario donde se ejecuta E+.
        self.obs_queue = obs_queue
        self.act_queue = act_queue
        self.cooling_queue = cooling_queue
        self.heating_queue = heating_queue
        self.pmv_queue = pmv_queue
        self.PPD_queue = PPD_queue
        
        # Controles de sincronización
        self.obs_event = threading.Event()
        self.act_event = threading.Event()
        self.cooling_event = threading.Event()
        self.heating_event = threading.Event()
        self.pmv_event = threading.Event()
        self.PPD_event = threading.Event()
        
       # Se asigna la variable que será utilizada para inicial el hilo de simulación
        self.energyplus_exec_thread: Optional[threading.Thread] = None
        self.energyplus_state: Any = None
        self.sim_results: int = 0
        self.initialized = False
        self.init_handles = False
        self.simulation_complete = False
        
        # below is declaration of variables, meters and actuators
        # this simulation will interact with
        self.variables = {
            "To": ("Site Outdoor Air Drybulb Temperature", "Environment"), #0
            "Ti": ("Zone Mean Air Temperature", "Thermal Zone"), #1
            "v": ("Site Wind Speed", "Environment"), #2
            "d": ("Site Wind Direction", "Environment"), #3
            "RHo": ("Site Outdoor Air Relative Humidity", "Environment"), #4
            "RHi": ("Zone Air Relative Humidity", "Thermal Zone"), #5
            "T_rad": ("Zone Mean Radiant Temperature", "Thermal Zone"), #del
            "Fanger_PMV":("Zone Thermal Comfort Fanger Model PMV", "People"), #del
            "Fanger_PPD":("Zone Thermal Comfort Fanger Model PPD", "People"), #del
        }
        self.var_handles: Dict[str, int] = {}

        self.meters = {
            "dh": "Heating:DistrictHeatingWater", #6
            "dc": "Cooling:DistrictCooling" #7
        }
        self.meter_handles: Dict[str, int] = {}
        
        # Actuadores
        # Zone HVAC Control
        # Two actuators called “Zone Temperature Control” are available with the control types of
        # “Heating Setpoint” and “Cooling Setpoint.” These enable you to directly override the zoneby-
        # zone heating and cooling setpoints. The units are in degrees Celsius. The unique identifier
        # is the zone name.
        
        # Airflow Network Openings
        # An actuator called “AirFlow Network Window/Door Opening” is available with a control type
        # called “Venting Opening Factor.” It is available in models that have operable openings in the Airflow
        # Network model and that are entered by using either AirflowNetwork:MultiZone:Component:DetailedOpening,
        # AirflowNetwork:MultiZone:Component:SimpleOpening, or AirflowNetwork:MultiZone:Component:HorizontalOpening
        # input objects. This control allows you to use EMS to vary the size of the opening during the
        # airflow model calculations, such as for natural and hybrid ventilation.
        # The unique identifier is the name of the surface (window, door or air boundary), not the name of
        # the associated airflow network input objects. The actuator control involves setting the value of the
        # opening factor between 0.0 and 1.0. Use of this actuator with an air boundary surface is allowed,
        # but will generate a warning since air boundaries are typically always open.
        
        self.actuators = {
            "opening_window_1": ("AirFlow Network Window/Door Opening", "Venting Opening Factor", "window_2"), # 8: opening factor between 0.0 and 1.0
            "opening_window_2": ("AirFlow Network Window/Door Opening", "Venting Opening Factor", "window_3"), # 9: opening factor between 0.0 and 1.0
        }
        self.actuator_handles: Dict[str, int] = {}

    def start(self) -> None:
        """_summary_
        """
        self.env_config = tools.episode_epJSON(self.env_config)
        
        self.energyplus_state = api.state_manager.new_state()
    
        # register callback used to collect the first observation
        api.runtime.callback_end_zone_timestep_after_zone_reporting(self.energyplus_state, self._collect_obs)
        # The calling point called “EndOfZoneTimestepAfterZoneReporting” occurs at the end of a zone
        # timestep after output variable reporting is finalized. It is useful for preparing calculations that
        # will go into effect the next timestep. Its capabilities are similar to BeginTimestepBeforePredictor,
        # except that input data for current time, date, and weather data align with different timesteps.
        
        # register callback used to send actions
        api.runtime.callback_begin_zone_timestep_after_init_heat_balance(self.energyplus_state, self._send_actions)
        # The calling point called “BeginZoneTimestepBeforeInitHeatBalance” occurs at the beginning of each
        # timestep before “InitHeatBalance” executes but after the weather manager and exterior energy use
        # manager. “InitHeatBalance” refers to the step in EnergyPlus modeling when the solar shading and
        # daylighting coefficients are calculated. This calling point is useful for controlling components that
        # affect the building envelope including surface constructions, window shades, and shading surfaces.
        # Programs called from this point might actuate the building envelope or internal gains based on
        # current weather or on the results from the previous timestep. Demand management routines might
        # use this calling point to operate window shades, change active window constructions, activate
        # exterior shades, etc.
        
        # Se controla la impresión de EnergyPlus en la consola
        api.runtime.set_console_output_status(self.energyplus_state, self.env_config['ep_terminal_output'])
        
        # run EnergyPlus in a non-blocking way
        def _run_energyplus():
            """_summary_
            """
            # start one episode simulation
            cmd_args = self.make_eplus_args()
            print(f"running EnergyPlus with args: {cmd_args}")
            self.sim_results = api.runtime.run_energyplus(self.energyplus_state, cmd_args)
            self.simulation_complete = True
            
        self.energyplus_exec_thread = threading.Thread(
            target=_run_energyplus,
            args=()
        )
        self.energyplus_exec_thread.start() # En este punto el hilo se divide en dos

    def stop(self) -> None:
        """_summary_
        """
        if not self.simulation_complete:
            self.simulation_complete = True
        sleep(3)
        self._flush_queues()
        self.energyplus_exec_thread.join()
        self.energyplus_exec_thread = None
        api.runtime.clear_callbacks()
        api.state_manager.delete_state(self.energyplus_state)
    
    def stop_failed(self) -> None:
        """_summary_
        """
        self.simulation_complete = True      

    def failed(self) -> bool:
        """_summary_

        Returns:
            bool: _description_
        """
        return self.sim_results != 0

    def make_eplus_args(self) -> List[str]:
        """make command line arguments to pass to EnergyPlus
        """
        eplus_args = ["-r"] if self.env_config.get("csv", False) else []
        eplus_args += [
            "-w",
            self.env_config["epw"],
            "-d",
            f"{self.env_config['output']}/episode-{self.episode:08}-{os.getpid():05}",
            self.env_config["idf"]
        ]
        return eplus_args

    def _collect_obs(self, state_argument) -> None:
        """EnergyPlus callback that collects output variables/meters
        values and enqueue them
        """
        if self.simulation_complete or not self._init_callback(state_argument):
            return
        
        # Se calculan algunas variables de paso de tiempo
        time_step = api.exchange.zone_time_step_number(state_argument)
        hour = api.exchange.hour(state_argument)
        day = api.exchange.day_of_month(state_argument)
        simulation_day = api.exchange.day_of_year(state_argument)
        month = api.exchange.month(state_argument)
        day_p1, month_p1 = tools.plus_day(day, month, 1)
        day_p2, month_p2 = tools.plus_day(day, month, 2)
        # Se realiza la observación del entorno
        obs = {
            **{
                key: api.exchange.get_variable_value(state_argument, handle)
                for key, handle
                in self.var_handles.items()
            },
            **{
                key: api.exchange.get_meter_value(state_argument, handle)
                for key, handle
                in self.meter_handles.items()
            },
            **{
                key: api.exchange.get_actuator_value(state_argument, handle)
                for key, handle
                in self.actuator_handles.items()
            }
        }
        if hour < 23:
            To_p1h = api.exchange.today_weather_outdoor_dry_bulb_at_time(state_argument, hour+1, time_step)
        else:
            To_p1h = api.exchange.tomorrow_weather_outdoor_dry_bulb_at_time(state_argument, hour-23, time_step)
        if hour < 22:
            To_p2h = api.exchange.today_weather_outdoor_dry_bulb_at_time(state_argument, hour+2, time_step)
        else:
            To_p2h = api.exchange.tomorrow_weather_outdoor_dry_bulb_at_time(state_argument, hour-22, time_step)
        if hour < 21:
            To_p3h = api.exchange.today_weather_outdoor_dry_bulb_at_time(state_argument, hour+3, time_step)
        else:
            To_p3h = api.exchange.tomorrow_weather_outdoor_dry_bulb_at_time(state_argument, hour-21, time_step)
        obs.update(
            {
            "To_p1h": To_p1h, #10
            "To_p2h": To_p2h, #11
            "To_p3h": To_p3h, #12
            "T_max_0": self.env_config['climatic_stads'][str(month)][str(day)]['T_max_0'],#13
            "T_min_0": self.env_config['climatic_stads'][str(month)][str(day)]['T_min_0'],#14
            "RH_0": self.env_config['climatic_stads'][str(month)][str(day)]['RH_0'],#15
            "raining_total_0": self.env_config['climatic_stads'][str(month)][str(day)]['raining_total_0'],#16
            "wind_avg_0": self.env_config['climatic_stads'][str(month)][str(day)]['wind_avg_0'],#17
            "wind_max_0": self.env_config['climatic_stads'][str(month)][str(day)]['wind_max_0'],#18
            "total_sky_cover_0": self.env_config['climatic_stads'][str(month)][str(day)]['total_sky_cover_0'],#19
            "T_max_1": self.env_config['climatic_stads'][str(month_p1)][str(day_p1)]['T_max_0'],#20
            "T_min_1": self.env_config['climatic_stads'][str(month_p1)][str(day_p1)]['T_min_0'],#21
            "RH_1": self.env_config['climatic_stads'][str(month_p1)][str(day_p1)]['RH_0'],#22
            "raining_total_1": self.env_config['climatic_stads'][str(month_p1)][str(day_p1)]['raining_total_0'],#23
            "wind_avg_1": self.env_config['climatic_stads'][str(month_p1)][str(day_p1)]['wind_avg_0'],#24
            "wind_max_1": self.env_config['climatic_stads'][str(month_p1)][str(day_p1)]['wind_max_0'],#25
            "total_sky_cover_1": self.env_config['climatic_stads'][str(month_p1)][str(day_p1)]['total_sky_cover_0'],#26
            "T_max_2": self.env_config['climatic_stads'][str(month_p2)][str(day_p2)]['T_max_0'],#27
            "T_min_2": self.env_config['climatic_stads'][str(month_p2)][str(day_p2)]['T_min_0'],#28
            "RH_2": self.env_config['climatic_stads'][str(month_p2)][str(day_p2)]['RH_0'],#29
            "raining_total_2": self.env_config['climatic_stads'][str(month_p2)][str(day_p2)]['raining_total_0'],#30
            "wind_avg_2": self.env_config['climatic_stads'][str(month_p2)][str(day_p2)]['wind_avg_0'],#31
            "wind_max_2": self.env_config['climatic_stads'][str(month_p2)][str(day_p2)]['wind_max_0'],#32
            "total_sky_cover_2": self.env_config['climatic_stads'][str(month_p2)][str(day_p2)]['total_sky_cover_0'],#33
            'hora': hour,#34
            'simulation_day': simulation_day,#35
            'volumen': self.env_config['volumen'],#36
            'window_area_relation_north': self.env_config['window_area_relation_north'],#37
            'window_area_relation_west': self.env_config['window_area_relation_west'],#38
            'window_area_relation_south': self.env_config['window_area_relation_south'],#39
            'window_area_relation_east': self.env_config['window_area_relation_east'],#40
            'construction_config': self.env_config['construction_config'], #41: TODO: hacer que esta variable tome el valor del coeficiente global de pérdidas de la vivienda.
            'internal_mass': self.env_config['internal_mass'], #42: TODO: hacer que esta variable tome un valor normalizado de masa interna para permitir la integración de diversos materiales que conforman la masa interna de una vivienda particular.
            'latitud': self.env_config['latitud'], #43
            'longitud':self.env_config['longitud'], #44
            'altitud': self.env_config['altitud'], #45
            'beta': self.env_config['beta'], #46
            'E_max': self.env_config['E_max'], #47
            "rad": api.exchange.today_weather_beam_solar_at_time(state_argument, hour, time_step), #48
            }
        )
        # se asignan estas variables al queue porque luego se eliminarán de la observación
        self.pmv_queue.put(obs["Fanger_PMV"])
        self.pmv_event.set()
        self.PPD_queue.put(obs["Fanger_PPD"])
        self.PPD_event.set()
        
        del obs["T_rad"]
        del obs["Fanger_PMV"]
        del obs["Fanger_PPD"]
        
        self.next_obs = np.array(list(obs.values()),dtype=float)
        
        # se comunican la observación, métricas y variables
        self.obs_queue.put(self.next_obs)
        self.obs_event.set()
        self.cooling_queue.put(obs['dc'])
        self.cooling_event.set()
        self.heating_queue.put(obs['dh'])
        self.heating_event.set()

    def _send_actions(self, state_argument):
        """EnergyPlus callback that sets actuator value from last decided action
        """
        if self.simulation_complete or not self._init_callback(state_argument):
            return
        
        # Se ejecuta la acción en el simulador
        self.act_event.wait(10)
        if self.act_queue.empty():
            return
        next_central_action = self.act_queue.get()
        next_action = tools.natural_ventilation_action(next_central_action)
        # Acciones en EnergyPlus
        api.exchange.set_actuator_value(
            state=state_argument,
            actuator_handle=self.actuator_handles["opening_window_1"],
            actuator_value=next_action[0]#['agent_w1']
        )
        api.exchange.set_actuator_value(
            state=state_argument,
            actuator_handle=self.actuator_handles["opening_window_2"],
            actuator_value=next_action[1]#['agent_w2']
        )
    
    def _init_callback(self, state_argument) -> bool:
        """Initialize EnergyPlus handles and checks if simulation runtime is ready"""
        self.init_handles = self._init_handles(state_argument)
        self.initialized = self.init_handles \
            and not api.exchange.warmup_flag(state_argument)
        return self.initialized

    def _init_handles(self, state_argument):
        """Initialize sensors/actuators handles to interact with during simulation"""
        if not self.init_handles:
            if not api.exchange.api_data_fully_ready(state_argument):
                return False
                
            self.var_handles = {
                key: api.exchange.get_variable_handle(state_argument, *var)
                for key, var in self.variables.items()
            }

            self.meter_handles = {
                key: api.exchange.get_meter_handle(state_argument, meter)
                for key, meter in self.meters.items()
            }

            self.actuator_handles = {
                key: api.exchange.get_actuator_handle(state_argument, *actuator)
                for key, actuator in self.actuators.items()
            }

            for handles in [
                self.var_handles,
                self.meter_handles,
                self.actuator_handles
            ]:
                if any([v == -1 for v in handles.values()]):
                    available_data = api.exchange.list_available_api_data_csv(state_argument).decode('utf-8')
                    print(
                        f"got -1 handle, check your var/meter/actuator names:\n"
                        f"> variables: {self.var_handles}\n"
                        f"> meters: {self.meter_handles}\n"
                        f"> actuators: {self.actuator_handles}\n"
                        f"> available E+ API data: {available_data}"
                    )
                
            self.init_handles = True
        return True

    def _flush_queues(self):
        """_summary_
        """
        for q in [self.obs_queue, self.act_queue, self.cooling_queue, 
                  self.heating_queue, self.pmv_queue, self.PPD_queue]:
            while not q.empty():
                q.get()