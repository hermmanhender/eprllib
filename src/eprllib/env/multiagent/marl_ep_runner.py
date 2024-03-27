"""# ENERGYPLUS RUNNER

This script contain the EnergyPlus Runner that execute EnergyPlus from its 
Python API in the version 23.2.0.
"""

import sys
import threading
import numpy as np
from queue import Queue
from time import sleep
from typing import Any, Dict, List, Optional
from eprllib.tools import weather_utils

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
        # Asignation of variables.
        self.episode = episode
        self.env_config = env_config
        self.obs_queue = obs_queue
        self.act_queue = act_queue
        self.infos_queue = infos_queue
        
        # saving the episode in the env_config to use across functions.
        self.env_config['episode'] = self.episode
        
        # The queue events are generated.
        self.obs_event = threading.Event()
        self.act_event = threading.Event()
        self.infos_event = threading.Event()
        
        # Variables to be used in this thread.
        self.energyplus_exec_thread: Optional[threading.Thread] = None
        self.energyplus_state: Any = None
        self.sim_results: int = 0
        self.initialized = False
        self.init_handles = False
        self.simulation_complete = False
        self.first_observation = True
        self.obs = {}
        self.infos = {}    
        
        # Declaration of variables this simulation will interact with.
        self.variables: dict = self.env_config['ep_variables']
        self.var_handles: Dict[str, int] = {}
        
        # Declaration of meters this simulation will interact with.
        self.meters: dict = self.env_config['ep_meters']
        self.meter_handles: Dict[str, int] = {}
        
        # Declaration of actuators this simulation will interact with.
        self.actuators: dict = self.env_config['ep_actuators']
        self.actuator_handles: Dict[str, int] = {}
        """Example:
        
        >>> "opening_window": ("AirFlow Network Window/Door Opening", "Venting Opening Factor", "window_name")
        
        Airflow Network Openings (EnergyPlus Documentation)
        
        An actuator called “AirFlow Network Window/Door Opening” is available with a control type
        called “Venting Opening Factor.” It is available in models that have operable openings in the Airflow
        Network model and that are entered by using either AirflowNetwork:MultiZone:Component:DetailedOpening,
        AirflowNetwork:MultiZone:Component:SimpleOpening, or AirflowNetwork:MultiZone:Component:HorizontalOpening
        input objects. This control allows you to use EMS to vary the size of the opening during the
        airflow model calculations, such as for natural and hybrid ventilation.
        The unique identifier is the name of the surface (window, door or air boundary), not the name of
        the associated airflow network input objects. The actuator control involves setting the value of the
        opening factor between 0.0 and 1.0. Use of this actuator with an air boundary surface is allowed,
        but will generate a warning since air boundaries are typically always open.
        """
        
    def start(self) -> None:
        """This method inicialize EnergyPlus. First the episode is configurate, the calling functions
        established and the thread is generated here.
        """
        # Specify the weather statisitical file.        
        self.weather_stats = weather_utils.Probabilities(self.env_config)
        
        # Start a new EnergyPlus state (condition for execute EnergyPlus Python API).
        self.energyplus_state = api.state_manager.new_state()
        
        
        api.runtime.callback_begin_system_timestep_before_predictor(self.energyplus_state, self._collect_first_obs)
        """Collect the first observation.
        This is execute only once at the begginig of the episode.
        The calling point called “BeginTimestepBeforePredictor” occurs near the beginning of each timestep
        but before the predictor executes. “Predictor” refers to the step in EnergyPlus modeling when the
        zone loads are calculated. This calling point is useful for controlling components that affect the
        thermal loads the HVAC systems will then attempt to meet. Programs called from this point
        might actuate internal gains based on current weather or on the results from the previous timestep.
        Demand management routines might use this calling point to reduce lighting or process loads,
        change thermostat settings, etc."""
        
        api.runtime.callback_begin_zone_timestep_after_init_heat_balance(self.energyplus_state, self._send_actions)
        """Execute the actions in the environment.
        The calling point called “BeginZoneTimestepAfterInitHeatBalance” occurs at the beginning of each
        timestep after “InitHeatBalance” executes and before “ManageSurfaceHeatBalance”. “InitHeatBalance” refers to the step in EnergyPlus modeling when the solar shading and daylighting coefficients
        are calculated. This calling point is useful for controlling components that affect the building envelope including surface constructions and window shades. Programs called from this point might
        actuate the building envelope or internal gains based on current weather or on the results from the
        previous timestep. Demand management routines might use this calling point to operate window
        shades, change active window constructions, etc. This calling point would be an appropriate place
        to modify weather data values."""
        
        api.runtime.callback_end_zone_timestep_after_zone_reporting(self.energyplus_state, self._collect_obs)
        """Collect the observations after the action executions and use them to provide new actions.
        The calling point called “EndOfZoneTimestepAfterZoneReporting” occurs at the end of a zone
        timestep after output variable reporting is finalized. It is useful for preparing calculations that
        will go into effect the next timestep. Its capabilities are similar to BeginTimestepBeforePredictor,
        except that input data for current time, date, and weather data align with different timesteps."""
        
        # Control of the console printing process.
        api.runtime.set_console_output_status(self.energyplus_state, self.env_config['ep_terminal_output'])
                
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
        # Here the thread is divide in two.
        self.energyplus_exec_thread.start()

    def _collect_obs(self, state_argument) -> None:
        """EnergyPlus callback that collects output variables, meters and actuator actions
        values and enqueue them to the EnergyPlus Environment thread.
        """
        # To not perform observations when the callbacks and the 
        # warming period are not complete.
        if not self._init_callback(state_argument):
            return
        # To not perform observations when the episode is ended
        if self.simulation_complete:
            return
        
        # Variables, meters and actuatos conditions as observation.
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
        
        # Upgrade of the timestep observation with other variables.
        time_step = api.exchange.zone_time_step_number(state_argument)
        hour = api.exchange.hour(state_argument)
        simulation_day = api.exchange.day_of_year(state_argument)
        obs.update(
            {
            'day_of_the_week': api.exchange.day_of_week(state_argument),
            'is_raining': api.exchange.is_raining(state_argument),
            'sun_is_up': api.exchange.sun_is_up(state_argument),
            'hora': hour,
            'simulation_day': simulation_day,
            "rad": api.exchange.today_weather_beam_solar_at_time(state_argument, hour, time_step),
            }
        )
        # Set the variables in the infos dict before to delete from the obs dict.
        infos_dict = {}
        for variable in self.env_config['infos_variables']:
            infos_dict[variable] = obs[variable]
        
        infos = {}
        for agent in self.env_config['agent_ids']:
            infos[agent] = infos_dict
        
        self.infos_queue.put(infos)
        self.infos_event.set()
        
        for variable in self.env_config['no_observable_variables']:
            del obs[variable]
        
        # save the last obs and infos dicts.
        self.obs = obs
        self.infos = infos
        
        # Transform the observation in a numpy array to meet the condition expected in a RLlib Environment
        next_obs = np.array(list(obs.values()))
        
        # Consult the stadistics of the weather to put into the obs array. This add 1440 elements 
        # to the observation.
        weather_prob = self.weather_stats.n_days_predictions(simulation_day, self.env_config.get('weather_prob_days', 2))
        next_obs = np.concatenate([next_obs, weather_prob])
        
        next_obs_dict = {}
        agent_indicator = 10
        for agent in self.env_config['agent_ids']:
            next_obs_dict[agent] = np.concatenate(([agent_indicator], next_obs))
            agent_indicator += 10
        
        # Set the observation to communicate with the MDP.
        self.obs_queue.put(next_obs_dict)
        self.obs_event.set()

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
          
        # To not perform actions if the callbacks and the warming period are not complete.
        if not self._init_callback(state_argument):
            return
        # To not perform actions when the episode is ended or is the first timestep
        # and there are not observations.
        if self.simulation_complete or self.first_observation:
            return
        
        # Wait for an action.
        event_flag = self.act_event.wait(10)
        if not event_flag:
            return
        
        # Get the central action from the EnergyPlus Environment `step` method.
        # In the case of simple agent a int value and for multiagents a dictionary.
        dict_action = self.act_queue.get()
        
        # Validate if the action must be transformed
        if self.env_config.get('action_transformer', False):
            action_transformer = self.env_config['action_transformer']
            dict_action_transformed = {}
            # Transform all the actions
            for agent in self.env_config['agent_ids']:
                dict_action_transformed[agent] = action_transformer(agent, dict_action[agent])
            dict_action = dict_action_transformed
        
        # Perform the actions in EnergyPlus simulation.       
        for agent in self.env_config['agent_ids']:
            api.exchange.set_actuator_value(
                state=state_argument,
                actuator_handle=self.actuator_handles[agent],
                actuator_value=dict_action[agent]
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
                        f"> available EnergyPlus API data: {available_data}"
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
            self.env_config["epw"] if self.env_config['is_test'] else self.env_config["epw_training"],
            "-d",
            f"{self.env_config['output']}/episode-{self.episode:08}",
            self.env_config["epjson"]
        ]
        return eplus_args
    
    def _flush_queues(self):
        """Method to liberate the space in the different queue objects.
        """
        for q in [self.obs_queue, self.act_queue, self.infos_queue]:
            while not q.empty():
                q.get()
