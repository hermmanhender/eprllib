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
from eprllib.tools.ep_episode_config import inertial_mass_calculation, u_factor_calculation

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
        # autozise properties
        if self.env_config['episode_config']['inercial_mass'] == 'auto':
            self.env_config['episode_config']['inercial_mass'] = inertial_mass_calculation(self.env_config)
        if self.env_config['episode_config']['construction_u_factor'] == 'auto':
            self.env_config['episode_config']['construction_u_factor'] = u_factor_calculation(self.env_config)
        
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
        # Start a new EnergyPlus state (condition for execute EnergyPlus Python API).
        self.energyplus_state = api.state_manager.new_state()
        
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
        
        hour = api.exchange.hour(state_argument)
        zone_time_step_number = api.exchange.zone_time_step_number(state_argument)
        
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
        # Upgrade the observation with the building general properties
        building_properties = {
            'building_area': self.env_config['episode_config']['building_area'],
            'aspect_ratio': self.env_config['episode_config']['aspect_ratio'],
            'window_area_relation_north': self.env_config['episode_config']['window_area_relation_north'],
            'window_area_relation_east': self.env_config['episode_config']['window_area_relation_east'],
            'window_area_relation_south': self.env_config['episode_config']['window_area_relation_south'],
            'window_area_relation_west': self.env_config['episode_config']['window_area_relation_west'],
            'inercial_mass': self.env_config['episode_config']['inercial_mass'],
            'construction_u_factor': self.env_config['episode_config']['construction_u_factor'],
            'E_cool_ref': self.env_config['episode_config']['E_cool_ref'],
            'E_heat_ref': self.env_config['episode_config']['E_heat_ref'],
        }
        obs.update(building_properties)
        
        # Upgrade of the timestep observation with other variables.
        if self.env_config.get('time_variables', False):
            time_variables_methods = {
                'actual_date_time': api.exchange.actual_date_time(state_argument), # Gets a simple sum of the values of the date/time function. Could be used in random seeding.
                'actual_time': api.exchange.actual_time(state_argument), # Gets a simple sum of the values of the time part of the date/time function. Could be used in random seeding.
                'current_time': api.exchange.current_time(state_argument), # Get the current time of day in hours, where current time represents the end time of the current time step.
                'day_of_month': api.exchange.day_of_month(state_argument), # Get the current day of month (1-31)
                'day_of_week': api.exchange.day_of_week(state_argument), # Get the current day of the week (1-7)
                'day_of_year': api.exchange.day_of_year(state_argument), # Get the current day of the year (1-366)
                'holiday_index': api.exchange.holiday_index(state_argument), # Gets a flag for the current day holiday type: 0 is no holiday, 1 is holiday type #1, etc.
                'hour': api.exchange.hour(state_argument), # Get the current hour of the simulation (0-23)
                'minutes': api.exchange.minutes(state_argument), # Get the current minutes into the hour (1-60)
                'month': api.exchange.month(state_argument), # Get the current month of the simulation (1-12)
                'num_time_steps_in_hour': api.exchange.num_time_steps_in_hour(state_argument), # Returns the number of zone time steps in an hour, which is currently a constant value throughout a simulation.
                'system_time_step': api.exchange.system_time_step(state_argument), # Gets the current system time step value in EnergyPlus. The system time step is variable and fluctuates during the simulation.
                'year': api.exchange.year(state_argument), # Get the “current” year of the simulation, read from the EPW. All simulations operate at a real year, either user specified or automatically selected by EnergyPlus based on other data (start day of week + leap year option).
                'zone_time_step': api.exchange.zone_time_step(state_argument), # Gets the current zone time step value in EnergyPlus. The zone time step is variable and fluctuates during the simulation.
                'zone_time_step_number': api.exchange.zone_time_step_number(state_argument) # The current zone time step index, from 1 to the number of zone time steps per hour
            }
            time_variables_list = self.env_config['time_variables']
            time_variables_dict = {}
            for variable in time_variables_list:
                time_variables_dict[variable] = time_variables_methods[variable]
            
            obs.update(time_variables_dict)
            
        if self.env_config.get('weather_variables', False):
            weather_variables_methods = {
                'is_raining': api.exchange.is_raining(state_argument), # Gets a flag for whether the it is currently raining. The C API returns an integer where 1 is yes and 0 is no, this simply wraps that with a bool conversion.
                'sun_is_up': api.exchange.sun_is_up(state_argument), # Gets a flag for whether the sun is currently up. The C API returns an integer where 1 is yes and 0 is no, this simply wraps that with a bool conversion.
                # Gets the specified weather data at the specified hour and time step index within that hour
                'today_weather_albedo_at_time': api.exchange.today_weather_albedo_at_time(state_argument, hour, zone_time_step_number),
                'today_weather_beam_solar_at_time': api.exchange.today_weather_beam_solar_at_time(state_argument, hour, zone_time_step_number),
                'today_weather_diffuse_solar_at_time': api.exchange.today_weather_diffuse_solar_at_time(state_argument, hour, zone_time_step_number),
                'today_weather_horizontal_ir_at_time': api.exchange.today_weather_horizontal_ir_at_time(state_argument, hour, zone_time_step_number),
                'today_weather_is_raining_at_time': api.exchange.today_weather_is_raining_at_time(state_argument, hour, zone_time_step_number),
                'today_weather_is_snowing_at_time': api.exchange.today_weather_is_snowing_at_time(state_argument, hour, zone_time_step_number),
                'today_weather_liquid_precipitation_at_time': api.exchange.today_weather_liquid_precipitation_at_time(state_argument, hour, zone_time_step_number),
                'today_weather_outdoor_barometric_pressure_at_time': api.exchange.today_weather_outdoor_barometric_pressure_at_time(state_argument, hour, zone_time_step_number),
                'today_weather_outdoor_dew_point_at_time': api.exchange.today_weather_outdoor_dew_point_at_time(state_argument, hour, zone_time_step_number),
                'today_weather_outdoor_dry_bulb_at_time': api.exchange.today_weather_outdoor_dry_bulb_at_time(state_argument, hour, zone_time_step_number),
                'today_weather_outdoor_relative_humidity_at_time': api.exchange.today_weather_outdoor_relative_humidity_at_time(state_argument, hour, zone_time_step_number),
                'today_weather_sky_temperature_at_time': api.exchange.today_weather_sky_temperature_at_time(state_argument, hour, zone_time_step_number),
                'today_weather_wind_direction_at_time': api.exchange.today_weather_wind_direction_at_time(state_argument, hour, zone_time_step_number),
                'today_weather_wind_speed_at_time': api.exchange.today_weather_wind_speed_at_time(state_argument, hour, zone_time_step_number),
                'tomorrow_weather_albedo_at_time': api.exchange.tomorrow_weather_albedo_at_time(state_argument, hour, zone_time_step_number),
                'tomorrow_weather_beam_solar_at_time': api.exchange.tomorrow_weather_beam_solar_at_time(state_argument, hour, zone_time_step_number),
                'tomorrow_weather_diffuse_solar_at_time': api.exchange.tomorrow_weather_diffuse_solar_at_time(state_argument, hour, zone_time_step_number),
                'tomorrow_weather_horizontal_ir_at_time': api.exchange.tomorrow_weather_horizontal_ir_at_time(state_argument, hour, zone_time_step_number),
                'tomorrow_weather_is_raining_at_time': api.exchange.tomorrow_weather_is_raining_at_time(state_argument, hour, zone_time_step_number),
                'tomorrow_weather_is_snowing_at_time': api.exchange.tomorrow_weather_is_snowing_at_time(state_argument, hour, zone_time_step_number),
                'tomorrow_weather_liquid_precipitation_at_time': api.exchange.tomorrow_weather_liquid_precipitation_at_time(state_argument, hour, zone_time_step_number),
                'tomorrow_weather_outdoor_barometric_pressure_at_time': api.exchange.tomorrow_weather_outdoor_barometric_pressure_at_time(state_argument, hour, zone_time_step_number),
                'tomorrow_weather_outdoor_dew_point_at_time': api.exchange.tomorrow_weather_outdoor_dew_point_at_time(state_argument, hour, zone_time_step_number),
                'tomorrow_weather_outdoor_dry_bulb_at_time': api.exchange.tomorrow_weather_outdoor_dry_bulb_at_time(state_argument, hour, zone_time_step_number),
                'tomorrow_weather_outdoor_relative_humidity_at_time': api.exchange.tomorrow_weather_outdoor_relative_humidity_at_time(state_argument, hour, zone_time_step_number),
                'tomorrow_weather_sky_temperature_at_time': api.exchange.tomorrow_weather_sky_temperature_at_time(state_argument, hour, zone_time_step_number),
                'tomorrow_weather_wind_direction_at_time': api.exchange.tomorrow_weather_wind_direction_at_time(state_argument, hour, zone_time_step_number),
                'tomorrow_weather_wind_speed_at_time': api.exchange.tomorrow_weather_wind_speed_at_time(state_argument, hour, zone_time_step_number)
            }
            weather_variables_list = self.env_config['weather_variables']
            weather_variables_dict = {}
            for variable in weather_variables_list:
                weather_variables_dict[variable] = weather_variables_methods[variable]
            
            obs.update(weather_variables_dict)
        
        # Weather prediction of 24 hours
        weather_pred = {}
        # The list sigma_max contains the standard deviation of the predictions in the following order:
        #   - Dry Bulb Temperature in °C with squer desviation of 2.05 °C, 
        #   - Relative Humidity in % with squer desviation of 20%, 
        #   - Wind Direction in degree with squer desviation of 40°, 
        #   - Wind Speed in m/s with squer desviation of 3.41 m/s, 
        #   - Barometric pressure in Pa with a standart deviation of 1000 Pa, 
        #   - Liquid Precipitation Depth in mm with desviation of 0.5 mm.
        sigma_max = np.array([1.43178211, 4.47213595, 6.32455532, 1.84661853, 31.62, 0.707107])
        for h in range(24):
            # For each hour, the sigma value goes from a minimum error of zero to the value listed in sigma_max following a linear function:
            sigma = sigma_max * (h/23)
            prediction_hour = hour+1 + h
            if prediction_hour < 24:
                weather_pred.update({
                    f'today_weather_liquid_precipitation_at_time_{prediction_hour}': np.random.normal(api.exchange.today_weather_liquid_precipitation_at_time(state_argument, prediction_hour, 0), sigma[5]),
                    f'today_weather_outdoor_barometric_pressure_at_time{prediction_hour}': np.random.normal(api.exchange.today_weather_outdoor_barometric_pressure_at_time(state_argument, prediction_hour, 0), sigma[4]),
                    f'today_weather_outdoor_dry_bulb_at_time{prediction_hour}': np.random.normal(api.exchange.today_weather_outdoor_dry_bulb_at_time(state_argument, prediction_hour, 0), sigma[0]),
                    f'today_weather_outdoor_relative_humidity_at_time{prediction_hour}': np.random.normal(api.exchange.today_weather_outdoor_relative_humidity_at_time(state_argument, prediction_hour, 0), sigma[1]),
                    f'today_weather_wind_direction_at_time{prediction_hour}': np.random.normal(api.exchange.today_weather_wind_direction_at_time(state_argument, prediction_hour, 0), sigma[2]),
                    f'today_weather_wind_speed_at_time{prediction_hour}': np.random.normal(api.exchange.today_weather_wind_speed_at_time(state_argument, prediction_hour, 0), sigma[3]),
                })
            else:
                prediction_hour_t = prediction_hour - 24
                weather_pred.update({
                    f'tomorrow_weather_liquid_precipitation_at_time{prediction_hour_t}': np.random.normal(api.exchange.tomorrow_weather_liquid_precipitation_at_time(state_argument, prediction_hour_t, 0), sigma[5]),
                    f'tomorrow_weather_outdoor_barometric_pressure_at_time{prediction_hour_t}': np.random.normal(api.exchange.tomorrow_weather_outdoor_barometric_pressure_at_time(state_argument, prediction_hour_t, 0), sigma[4]),
                    f'tomorrow_weather_outdoor_dry_bulb_at_time{prediction_hour_t}': np.random.normal(api.exchange.tomorrow_weather_outdoor_dry_bulb_at_time(state_argument, prediction_hour_t, 0), sigma[0]),
                    f'tomorrow_weather_outdoor_relative_humidity_at_time{prediction_hour_t}': np.random.normal(api.exchange.tomorrow_weather_outdoor_relative_humidity_at_time(state_argument, prediction_hour_t, 0), sigma[1]),
                    f'tomorrow_weather_wind_direction_at_time{prediction_hour_t}': np.random.normal(api.exchange.tomorrow_weather_wind_direction_at_time(state_argument, prediction_hour_t, 0), sigma[2]),
                    f'tomorrow_weather_wind_speed_at_time{prediction_hour_t}': np.random.normal(api.exchange.tomorrow_weather_wind_speed_at_time(state_argument, prediction_hour_t, 0), sigma[3]),
                })
        obs.update(weather_pred)
        
        # Set the variables in the infos dict before to delete from the obs dict.
        infos_dict = {}
        if self.env_config.get('infos_variables', False):
            for variable in self.env_config['infos_variables']:
                infos_dict[variable] = obs[variable]
        
        infos = {}
        for agent in self.env_config['agent_ids']:
            infos[agent] = infos_dict
        
        self.infos_queue.put(infos)
        self.infos_event.set()
        
        if self.env_config.get('no_observable_variables', False):
            for variable in self.env_config['no_observable_variables']:
                del obs[variable]
        
        # save the last obs and infos dicts.
        self.obs = obs
        self.infos = infos
        
        # Transform the observation in a numpy array to meet the condition expected in a RLlib Environment
        next_obs = np.array(list(obs.values()), dtype='float32')
        
        next_obs_dict = {}
        agent_indicator = 1
        for agent in self.env_config['agent_ids']:
            next_obs_dict[agent] = np.concatenate(([agent_indicator], [self.env_config['ep_actuators_type'][agent]], next_obs), dtype='float32')
            agent_indicator += 1
        
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
        if self.simulation_complete:
            return
        
        # If is the first timestep, obtain the first observation before to consult for an action
        if self.first_observation:
            self._collect_first_obs(state_argument)
            
        # Wait for an action.
        event_flag = self.act_event.wait(120)
        if not event_flag:
            print('The time waiting an action was over.')
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
