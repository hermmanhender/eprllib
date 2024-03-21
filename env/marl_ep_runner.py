"""# ENERGYPLUS RUNNER

This script contain the EnergyPlus Runner that execute EnergyPlus from its Python API in the version
23.2.0.
"""

import os
import sys
import threading
import numpy as np
from queue import Queue
from time import sleep
from typing import Any, Dict, List, Optional
from tools import ep_episode_config, devices_space_action as dsa, weather_utils
from agents.conventional import Conventional


os_platform = sys.platform
if os_platform == "linux":
    sys.path.insert(0, '/usr/local/EnergyPlus-23-2-0')
else:
    sys.path.insert(0, 'C:/EnergyPlusV23-2-0')
    
from pyenergyplus.api import EnergyPlusAPI
api = EnergyPlusAPI()

class EnergyPlusRunner:
    """This object have the particularity of `start` EnergyPlus, `_collect_obs` and `_send_actions` to
    send it trhougt queue to the EnergyPlus Environment thread.
    """
    def __init__(
        self,
        episode: int,
        env_config: Dict[str, Any],
        obs_queue: Queue,
        act_queue: Queue,
        infos_queue: Queue
        ) -> None:
        """The object has an intensive interaction with EnergyPlus Environment script, exchange information
        between two threads. For a good coordination queue events are stablished and different canals of
        information are defined.

        Args:
            episode (int): Episode number.
            env_config (Dict[str, Any]): Environment configuration defined in the call to the EnergyPlus Environment.
            obs_queue (Queue): Queue object definition.
            act_queue (Queue): Queue object definition.
            infos_queue (Queue): Queue object definition.
        
        Return:
            None.
        """
        self.episode = episode
        self.env_config = env_config
        self.env_config['episode'] = self.episode
        self.obs_queue = obs_queue
        self.act_queue = act_queue
        self.infos_queue = infos_queue
        # Asignation of variables.
        
        self.obs_event = threading.Event()
        self.act_event = threading.Event()
        self.infos_event = threading.Event()
        # The queue events are generated.
        
        self.energyplus_exec_thread: Optional[threading.Thread] = None
        self.energyplus_state: Any = None
        self.sim_results: int = 0
        self.initialized = False
        self.init_handles = False
        self.simulation_complete = False
        self.first_observation = True
        self.dc_sum = 0
        self.dh_sum = 0
        self.obs = {}
        self.infos = {}
        # Variables to be used in this thread.

        self.env_config = ep_episode_config.epJSON_path(self.env_config)
        # The path for the epjson file is defined.
        
        self.variables = {
            "To": ("Site Outdoor Air Drybulb Temperature", "Environment"), #1
            "Ti": ("Zone Mean Air Temperature", "Thermal Zone: Living"), #2
            "v": ("Site Wind Speed", "Environment"), #3
            "d": ("Site Wind Direction", "Environment"), #4
            "RHo": ("Site Outdoor Air Relative Humidity", "Environment"), #5
            "RHi": ("Zone Air Relative Humidity", "Thermal Zone: Living"), #6
            "pres": ("Site Outdoor Air Barometric Pressure", "Environment"), #7
            "occupancy": ("Zone People Occupant Count", "Thermal Zone: Living"), #8
            "ppd": ("Zone Thermal Comfort Fanger Model PPD", "Living Occupancy") # infos
        }
        self.var_handles: Dict[str, int] = {}
        # Declaration of variables this simulation will interact with.

        self.meters = {
            "electricity": "Electricity:Zone:THERMAL ZONE: LIVING", #9
            "gas": "NaturalGas:Zone:THERMAL ZONE: LIVING", #10
        }
        self.meter_handles: Dict[str, int] = {}
        # Declaration of meters this simulation will interact with.

        self.actuators = {
            "opening_window_1": ("AirFlow Network Window/Door Opening", "Venting Opening Factor", "living_NW_window"), # 11: opening factor between 0.0 and 1.0
            "opening_window_2": ("AirFlow Network Window/Door Opening", "Venting Opening Factor", "living_E_window"), # 12: opening factor between 0.0 and 1.0
        }
        self.actuator_handles: Dict[str, int] = {}
        # Declaration of actuators this simulation will interact with.
        # Airflow Network Openings (EnergyPlus Documentation)
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

    def start(self) -> None:
        """This method inicialize EnergyPlus. First the episode is configurate, the calling functions
        established and the thread is generated here.
        """
        self.env_config['epw'], _, _, _ = ep_episode_config.weather_file(self.env_config)
        # Configurate the episode.
        
        self.weather_stats = weather_utils.Probabilities(self.env_config)
        # Specify the weather statisitical file.
        
        self.energyplus_state = api.state_manager.new_state()
        # Start a new EnergyPlus state (condition for execute EnergyPlus Python API).
        
        api.runtime.callback_begin_system_timestep_before_predictor(self.energyplus_state, self._collect_first_obs)
        # Collect the first observation. This is execute only once at the begginig of the episode.
        # The calling point called “BeginTimestepBeforePredictor” occurs near the beginning of each timestep
        # but before the predictor executes. “Predictor” refers to the step in EnergyPlus modeling when the
        # zone loads are calculated. This calling point is useful for controlling components that affect the
        # thermal loads the HVAC systems will then attempt to meet. Programs called from this point
        # might actuate internal gains based on current weather or on the results from the previous timestep.
        # Demand management routines might use this calling point to reduce lighting or process loads,
        # change thermostat settings, etc.
        
        api.runtime.callback_begin_zone_timestep_after_init_heat_balance(self.energyplus_state, self._send_actions)
        # Execute the actions in the environment.
        # The calling point called “BeginZoneTimestepAfterInitHeatBalance” occurs at the beginning of each
        # timestep after “InitHeatBalance” executes and before “ManageSurfaceHeatBalance”. “InitHeatBalance” refers to the step in EnergyPlus modeling when the solar shading and daylighting coefficients
        # are calculated. This calling point is useful for controlling components that affect the building envelope including surface constructions and window shades. Programs called from this point might
        # actuate the building envelope or internal gains based on current weather or on the results from the
        # previous timestep. Demand management routines might use this calling point to operate window
        # shades, change active window constructions, etc. This calling point would be an appropriate place
        # to modify weather data values.
        
        api.runtime.callback_end_zone_timestep_after_zone_reporting(self.energyplus_state, self._collect_obs)
        # Collect the observations after the action executions and use them to provide new actions.
        # The calling point called “EndOfZoneTimestepAfterZoneReporting” occurs at the end of a zone
        # timestep after output variable reporting is finalized. It is useful for preparing calculations that
        # will go into effect the next timestep. Its capabilities are similar to BeginTimestepBeforePredictor,
        # except that input data for current time, date, and weather data align with different timesteps.
        
        api.runtime.set_console_output_status(self.energyplus_state, self.env_config['ep_terminal_output'])
        # Control of the console printing process.
        
        def _run_energyplus():
            """Run EnergyPlus in a non-blocking way with Threads.
            """
            cmd_args = self.make_eplus_args()
            print(f"running EnergyPlus with args: {cmd_args}")
            self.sim_results = api.runtime.run_energyplus(self.energyplus_state, cmd_args)
            self.simulation_complete = True
            
        self.energyplus_exec_thread = threading.Thread(
            target=_run_energyplus,
            args=()
        )
        self.energyplus_exec_thread.start()
        # Here the thread is divide in two.

    def _collect_obs(self, state_argument) -> None:
        """EnergyPlus callback that collects output variables, meters and actuator actions
        values and enqueue them to the EnergyPlus Environment thread.
        """
        if self.simulation_complete or not self._init_callback(state_argument):
            # To not perform observations when the episode is ended or if the callbacks and the 
            # warming period are not complete.
            return
        
        time_step = api.exchange.zone_time_step_number(state_argument)
        hour = api.exchange.hour(state_argument)
        simulation_day = api.exchange.day_of_year(state_argument)
        # Timestep variables.
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
        # Variables, meters and actuatos conditions as observation.
        obs.update(
            {
            'day_of_the_week': api.exchange.day_of_week(state_argument), #13
            'is_raining': api.exchange.is_raining(state_argument), #14
            'sun_is_up': api.exchange.sun_is_up(state_argument), #15
            'hora': hour, #16
            'simulation_day': simulation_day, #17
            "rad": api.exchange.today_weather_beam_solar_at_time(state_argument, hour, time_step), #18
            }
        )
        # Upgrade of the timestep observation.
        infos_dict = {'ppd': obs['ppd'], 'Ti': obs['Ti'], "occupancy": obs['occupancy']}
        infos = {
            'window_opening_1': infos_dict,
            'window_opening_2': infos_dict,
        }
        self.infos_queue.put(infos)
        self.infos_event.set()
        # Set the variables to communicate with queue before to delete the following.
        del obs['ppd']
        
        self.obs = obs
        self.infos = infos
        next_obs = np.array(list(obs.values()))
        # Transform the observation in a numpy array to meet the condition expected in a RLlib Environment
        weather_prob = self.weather_stats.n_days_predictions(simulation_day, 2)
        # Consult the stadistics of the weather to put into the obs array. This add 1440 elements to the observation.
        next_obs = np.concatenate([next_obs, weather_prob])
        
        next_obs_dict = {
            'window_opening_1': np.concatenate(([10], next_obs)),
            'window_opening_2': np.concatenate(([20], next_obs)),
        }
        
        self.obs_queue.put(next_obs_dict)
        self.obs_event.set()
        # Set the observation to communicate with queue.

    def _collect_first_obs(self, state_argument):
        """This method is used to collect only the first observation of the environment when the episode beggins.

        Args:
            state_argument (c_void_p): EnergyPlus state pointer. This is created with `api.state_manager.new_state()`.
        """
        if self.first_observation:
            self._collect_obs(state_argument)
            self.first_observation = False
        else:
            return

    def _send_actions(self, state_argument):
        """EnergyPlus callback that sets actuator value from last decided action
        """
        if self.simulation_complete or not self._init_callback(state_argument):
            # To not perform actions when the episode is ended or if the callbacks and the 
            # warming period are not complete.
            return
        
        self.act_event.wait(20)
        # Wait for an action.
        if self.act_queue.empty():
            # Return in the first timestep.
            return
        dict_action = self.act_queue.get()
        # Get the central action from the EnergyPlus Environment `step` method.
        # In the case of simple agent a int value and for multiagents a dictionary.
        opening_window_1_action = dict_action['window_opening_1']
        opening_window_2_action = dict_action['window_opening_2']
        # Execute the same action during an hour.
        
        api.exchange.set_actuator_value(
            state=state_argument,
            actuator_handle=self.actuator_handles["opening_window_1"],
            actuator_value=opening_window_1_action
        )
        api.exchange.set_actuator_value(
            state=state_argument,
            actuator_handle=self.actuator_handles["opening_window_2"],
            actuator_value=opening_window_2_action
        )
        # Perform the actions in EnergyPlus simulation.
    
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

    def stop(self) -> None:
        """Method to stop EnergyPlus simulation and joint the threads.
        """
        if not self.simulation_complete:
            self.simulation_complete = True
        sleep(3)
        self._flush_queues()
        self.energyplus_exec_thread.join()
        self.energyplus_exec_thread = None
        self.first_observation = True
        api.runtime.clear_callbacks()
        api.state_manager.delete_state(self.energyplus_state)  

    def failed(self) -> bool:
        """This method tells if a EnergyPlus simulations was finished successfully or not.

        Returns:
            bool: Boolean value of the success of the simulation.
        """
        return self.sim_results != 0

    def make_eplus_args(self) -> List[str]:
        """Make command line arguments to pass to EnergyPlus
        """
        eplus_args = ["-r"] if self.env_config.get("csv", False) else []
        eplus_args += [
            "-w",
            self.env_config["epw"],
            "-d",
            f"{self.env_config['output']}/episode-{self.episode:08}-{os.getpid():05}",
            self.env_config["epjson"]
        ]
        return eplus_args
    
    def _flush_queues(self):
        """Method to liberate the space in the different queue objects.
        """
        for q in [self.obs_queue, self.act_queue, self.infos_queue]:
            while not q.empty():
                q.get()
