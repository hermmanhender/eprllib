"""# ENERGYPLUS RUNNER

This script contain the EnergyPlus Runner that execute EnergyPlus from its 
Python API in the version 24.1.0.
"""
import os
import threading
import numpy as np
import time
from queue import Queue
from typing import Any, Dict, List, Optional, Set, Tuple
from ctypes import c_void_p
from eprllib.ActionFunctions.ActionFunctions import ActionFunction
from eprllib.ObservationFunctions.ObservationFunctions import ObservationFunction
from eprllib.Env.MultiAgent.EnvUtils import EP_API_add_path
# EnergyPlus Python API path adding
EP_API_add_path(version="24-1-0")
from pyenergyplus.api import EnergyPlusAPI
import time
import time
api = EnergyPlusAPI()

class EnergyPlusRunner:
    """
    This object have the particularity of `start` EnergyPlus, `_collect_obs` and `_send_actions` to
    send it trhougt queue to the EnergyPlus Environment thread.
    """
    def __init__(
        self,
        env_config: Dict[str, Any],
        episode: int,
        obs_queue: Queue,
        act_queue: Queue,
        infos_queue: Queue,
        _agent_ids: Set,
        _thermal_zone_ids: Set,
        observation_fn: ObservationFunction,
        action_fn: ActionFunction,
        ) -> None:
        """
        The object has an intensive interaction with EnergyPlus Environment script, exchange information
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
        self.env_config = env_config
        self.episode = episode
        self.obs_queue = obs_queue
        self.act_queue = act_queue
        self.infos_queue = infos_queue
        self._agent_ids = _agent_ids
        self._thermal_zone_ids = _thermal_zone_ids
        
        # The queue events are generated (To sure the coordination with EnergyPlusEnvironment).
        self.obs_event = threading.Event()
        self.act_event = threading.Event()
        self.infos_event = threading.Event()
        
        # Variables to be used in this thread.
        self.energyplus_exec_thread: Optional[threading.Thread] = None
        self.energyplus_state: c_void_p = None
        self.sim_results: int = 0
        self.initialized = False
        self.init_handles = False
        self.simulation_complete = False
        self.first_observation = True
        self.obs = {}
        self.infos = {}
        self.unique_id = time.time()
        
        # create a variable to save the obs dict key to use in posprocess
        self.obs_keys = []
        self.infos_keys = []
        
        # Define the action and observation functions.
        self.action_fn = action_fn
        self.observation_fn = observation_fn
        
        # Declaration of variables, meters and actuators to use in the simulation. Handles
        # are used in _init_handle method.
        self.dict_site_variables, self.handle_site_variables = self.set_site_variables()
        self.dict_thermalzone_variables, self.handle_thermalzone_variables = self.set_thermalzone_variables()
        self.dict_object_variables, self.handle_object_variables = self.set_object_variables()
        self.dict_static_variables, self.handle_static_variables = self.set_static_variables()
        self.meters, self.meter_handles = self.set_meters()
        self.actuators, self.actuator_handles = self.set_actuators()
        
    def start(self) -> None:
        """
        This method inicialize EnergyPlus. First the episode is configurate, the calling functions
        established and the thread is generated here.
        """
        # Start a new EnergyPlus state (condition for execute EnergyPlus Python API).
        self.energyplus_state:c_void_p = api.state_manager.new_state()
        # TODO: use request_variable(state: c_void_p, variable_name: str | bytes, variable_key: str | bytes) to avoid
        # the necesity of the user to use the output definition inside the IDF/epJSON file.
        # for variable_name, variable_key in self.variables.items():
        #     api.exchange.request_variable(
        #         self.energyplus_state,
        #         variable_name = variable_name,
        #         variable_key = variable_key
        #         )
        api.runtime.callback_begin_zone_timestep_after_init_heat_balance(self.energyplus_state, self._send_actions)
        api.runtime.callback_end_zone_timestep_after_zone_reporting(self.energyplus_state, self._collect_obs)
        api.runtime.set_console_output_status(self.energyplus_state, self.env_config['ep_terminal_output'])
        
        def _run_energyplus():
            """
            Run EnergyPlus in a non-blocking way with Threads.
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

    def _collect_obs(
        self,
        state_argument: c_void_p
        ) -> None:
        """
        EnergyPlus callback that collects output variables, meters and actuator actions
        values and enqueue them to the EnergyPlus Environment thread.
        """
        dict_agents_obs, dict_agents_infos = NotImplemented
        # To not perform observations when the callbacks and the 
        # warming period are not complete.
        if not self._init_callback(state_argument) or self.simulation_complete:
            return
        # Agents observe: site state, thermal zone state (only the one that it belong), specific object variables 
        # and meters, and others parameters assigned as True in the env_config.observation object.
        # Get the state of the actuators.
        actuator_states, actuator_infos = self.get_actuators_state(state_argument)
        site_state, site_infos = self.get_site_variables_state(state_argument)
        site_state_p, site_infos_p = self.get_simulation_parameters_values(state_argument)
        site_state.update(site_state_p)
        site_infos.update(site_infos_p)
        site_state_p, site_infos_p = self.get_weather_prediction(state_argument)
        site_state.update(site_state_p)
        site_infos.update(site_infos_p)
        
        thermal_zone_states = {thermal_zone: {} for thermal_zone in self._thermal_zone_ids}
        thermal_zone_infos = {thermal_zone: {} for thermal_zone in self._thermal_zone_ids}
        for thermal_zone in self._thermal_zone_ids:
            thermal_zone_states_p, thermal_zone_infos_p = self.get_thermalzone_variables_state(state_argument, thermal_zone)
            thermal_zone_states[thermal_zone].update(thermal_zone_states_p)
            thermal_zone_infos[thermal_zone].update(thermal_zone_infos_p)
            
            thermal_zone_states_p, thermal_zone_infos_p = self.get_static_variables_state(state_argument, thermal_zone)
            thermal_zone_states[thermal_zone].update(thermal_zone_states_p)
            thermal_zone_infos[thermal_zone].update(thermal_zone_infos_p)
            
            thermal_zone_states_p, thermal_zone_infos_p = self.get_zone_simulation_parameters_values(state_argument, thermal_zone)
            thermal_zone_states[thermal_zone].update(thermal_zone_states_p)
            thermal_zone_infos[thermal_zone].update(thermal_zone_infos_p)
            
        agent_states = {agent: {} for agent in self._agent_ids}
        agent_infos = {agent: {} for agent in self._agent_ids}
        for agent in self._agent_ids:
            agent_states_p, agent_infos_p = self.get_object_variables_state(state_argument, agent)
            agent_states[agent].update(agent_states_p)
            agent_infos[agent].update(agent_infos_p)
            agent_states_p, agent_infos_p = self.get_meters_state(state_argument, agent)
            agent_states[agent].update(agent_states_p)
            agent_infos[agent].update(agent_infos_p)
        
        dict_agents_obs, dict_agents_infos = self.observation_fn.set_agent_obs_and_infos(
            self.env_config,
            self._agent_ids,
            self._thermal_zone_ids,
            actuator_states,
            actuator_infos,
            site_state,
            site_infos,
            thermal_zone_states,
            thermal_zone_infos,
            agent_states,
            agent_infos
        )
        
        for key, value in dict_agents_obs.items():
            if np.isnan(value).any() or np.isinf(value).any():
                print(f"NaN or Inf value found in {key}: {dict_agents_obs[key]}")
        for key, value in dict_agents_infos.items():
            if np.isnan(value).any() or np.isinf(value).any():
                print(f"NaN or Inf value found in {key}: {dict_agents_infos[key]}")

        # Set the agents observation and infos to communicate with the EPEnv.
        self.obs_queue.put(dict_agents_obs)
        self.obs_event.set()
        self.infos_queue.put(dict_agents_infos)
        self.infos_event.set()

    def _collect_first_obs(self, state_argument):
        """
        This method is used to collect only the first observation of the environment when the episode beggins.

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
        if not self._init_callback(state_argument) or self.simulation_complete:
            return
        
        # If is the first timestep, obtain the first observation before to consult for an action
        if self.first_observation:
            self.env_config['num_time_steps_in_hour'] = api.exchange.num_time_steps_in_hour(state_argument)
            self._collect_first_obs(state_argument)
            
        # Wait for an action.
        event_flag = self.act_event.wait() # (self.env_config["timeout"])
        if not event_flag:
            raise ValueError(f"The time waiting an action was over.\nThe observation event is {self.obs_event.is_set()}.\nThe infos event is {self.infos_event.is_set()}.")
                
        # Get and transform the action from the EnergyPlusEnvironment `step` method.
        dict_action = self.action_fn.transform_action(self.act_queue.get())
        # TODO: For centralize method, it is needed to descompose the junt action into the
        # different agents. This would be included in the action_fn.
        
        # Perform the actions in EnergyPlus simulation.
        for agent in self._agent_ids:
            api.exchange.set_actuator_value(
                state=state_argument,
                actuator_handle=self.actuator_handles[agent],
                actuator_value=dict_action[agent]
            )    
    
    def set_site_variables(self) -> Tuple[Dict[str, Tuple [str, str]], Dict[str,Dict[str, int]]]:
        """
        The EnergyPlus outdoor environment variables are defined in the environment configuration.

        Returns:
            Tuple[Dict[str, Tuple [str, str]], Dict[str,Dict[str, int]]]: The environment variables and their handles.
        """
        # Define the emptly Dict to include variables names and handles. The handles Dict return as emptly Dict to be used latter.
        var_handles: Dict[str,Dict[str, int]] = {thermal_zone: {} for thermal_zone in self._thermal_zone_ids}
        variables: Dict[str, Tuple [str, str]] = {thermal_zone: {} for thermal_zone in self._thermal_zone_ids}
        
        if not len(self.env_config['variables_env']) == 0:
            variables.update({variable: (variable, 'Environment') for variable in self.env_config['variables_env']})
        return variables, var_handles

    def set_thermalzone_variables(self) -> Tuple[Dict[str, Tuple [str, str]], Dict[str, int]]:
        """
        The EnergyPlus outdoor environment variables are defined in the environment configuration.

        Args:
            env_config (Dict[str, Any]): The EnvConfig dictionary.

        Returns:
            Tuple[Dict[str, Tuple [str, str]], Dict[str, int]]: The environment variables and their handles.
        """
        # Define the emptly Dict to include variables names and handles. The handles Dict return as emptly Dict to be used latter.
        var_handles: Dict[str,Dict[str, int]] = {thermal_zone: {} for thermal_zone in self._thermal_zone_ids}
        variables:Dict[str,Dict[str, Tuple [str, str]]] = {thermal_zone: {} for thermal_zone in self._thermal_zone_ids}
        if not len(self.env_config['variables_thz']) == 0:
            for thermal_zone in self._thermal_zone_ids:
                variables.update({thermal_zone:{variable: (variable, thermal_zone) for variable in self.env_config['variables_thz']}})
        return variables, var_handles

    def set_object_variables(self) -> Tuple[Dict[str, Tuple [str, str]], Dict[str, int]]:
        """
        The EnergyPlus outdoor environment variables are defined in the environment configuration.

        Args:
            env_config (Dict[str, Any]): The EnvConfig dictionary.

        Returns:
            Tuple[Dict[str, Tuple [str, str]], Dict[str, int]]: The environment variables and their handles.
        """
        # Define the emptly Dict to include variables names and handles. The handles Dict return as emptly Dict to be used latter.
        var_handles: Dict[str,Dict[str, int]] = {thermal_zone: {} for thermal_zone in self._thermal_zone_ids}
        variables:Dict[str,Dict[str, Tuple [str, str]]] = {thermal_zone: {} for thermal_zone in self._thermal_zone_ids}
        if not len(self.env_config['variables_obj']) == 0:
            # Check all the agents are in the variables_obj Dict.
            assert set(self.env_config['variables_obj'].keys()) == self._agent_ids, f"The variables_obj must include all agent_ids: {self._agent_ids}."
            for agent in self._agent_ids:
                variables.update({agent:{variable: (variable, object_key) for variable, object_key in self.env_config['variables_obj'][agent].items()}})
        return variables, var_handles

    def set_static_variables(self) -> Tuple[Dict[str, str], Dict[str, int]]:
        """Set the static variables to handle and get the values in the environment.

        Args:
            env_config (Dict[str, Any]): Environment configuration dictionary.

        Returns:
            Tuple[Dict[str, Tuple [str, str]], Dict[str, int]]: Return the static variables names dictionary and their handles IDs.
        """
        # Define the emptly Dict to include variables names and handles. The handles Dict return as emptly Dict to be used latter.
        var_handles: Dict[str,Dict[str, int]] = {thermal_zone: {} for thermal_zone in self._thermal_zone_ids}
        variables:Dict[str, str] = {thermal_zone: {} for thermal_zone in self._thermal_zone_ids}
        # Check the existency of static variables.
        if not len(self.env_config['static_variables']) == 0:
            for thermal_zone in self._thermal_zone_ids:
                variables.update({thermal_zone:{variable: variable for variable in self.env_config['static_variables']}})       
        return variables, var_handles

    def set_meters(self) -> Tuple[Dict[str, str], Dict[str, int]]:
        """The EnergyPlus meters are defined in the environment configuration.
        
        Args:
            env_config (Dict[str, Any]): The EnvConfig dictionary.

        Returns:
            Tuple[Dict[str, Tuple [str, str]], Dict[str, int]]: The meters and their handles.
        """
        meter_handles: Dict[str,Dict[str, int]] = {agent: {} for agent in self._agent_ids}
        meters: Dict[str, List[str]] = {agent: {} for agent in self._agent_ids}
        if not len(self.env_config['meters']) == 0:
            for agent in self._agent_ids:
                for _ in range(len(self.env_config['meters'][agent])):
                    meters.update({agent:{variable: variable for variable in self.env_config['meters'][agent]}})
        return meters, meter_handles

    def set_actuators(self) -> Tuple[Dict[str, Tuple [str, str]], Dict[str, int]]:
        """The EnergyPlus actuators are defined in the environment configuration.

        Args:
            env_config (Dict[str, Any]): The EnvConfig dictionary.

        Returns:
            Tuple[Dict[str,Tuple[str,str,str]], Dict[str,int]]: The actuators and their handles.
        """
        actuators: Dict[str,Tuple[str,str,str]] = {agent: self.env_config['agents_config'][agent]['ep_actuator_config'] for agent in self._agent_ids}
        actuator_handles: Dict[str, int] = {}
        
        return actuators, actuator_handles
     
    def get_site_variables_state(
        self,
        state_argument:c_void_p,
        ) -> Tuple[Dict[str,Any],Dict[str,Any]]:
        """This funtion takes the state_argument and return a dictionary with the agent actuator values.

        Args:
            state_argument (c_void_p): State argument from EnergyPlus callback.

        Returns:
            Dict[str,Any]: Agent actuator values for the actual timestep.
        """
        variables = {
            key: api.exchange.get_variable_value(state_argument, handle)
            for key, handle
            in self.handle_site_variables.items()
        }
        infos = self.update_infos(variables, 'variables_env')
        variables = self.delete_not_observable_variables(variables, 'variables_env')
        return variables, infos
    
    def get_thermalzone_variables_state(
        self,
        state_argument:c_void_p,
        thermal_zone:str = None
        ) -> Tuple[Dict[str,Any],Dict[str,Any]]:
        """This funtion takes the state_argument and return a dictionary with the agent actuator values.

        Args:
            state_argument (c_void_p): State argument from EnergyPlus callback.

        Returns:
            Dict[str,Any]: Agent actuator values for the actual timestep.
        """
        if thermal_zone is None:
            ValueError("The thermal zone must be defined.")
        variables = {
            key: api.exchange.get_variable_value(state_argument, handle)
            for key, handle
            in self.handle_thermalzone_variables[thermal_zone].items()
        }
        infos = self.update_infos(variables, 'variables_thz', thermal_zone)
        variables = self.delete_not_observable_variables(variables, 'variables_thz', thermal_zone)
        return variables, infos

    def get_object_variables_state(
        self,
        state_argument:c_void_p,
        agent:str = None
        ) -> Tuple[Dict[str,Any],Dict[str,Any]]:
        """This funtion takes the state_argument and return a dictionary with the agent actuator values.

        Args:
            state_argument (c_void_p): State argument from EnergyPlus callback.

        Returns:
            Dict[str,Any]: Agent actuator values for the actual timestep.
        """
        if agent is None:
            ValueError("The agent must be defined.")
        
        variables = {
            key: api.exchange.get_variable_value(state_argument, handle)
            for key, handle
            in self.handle_object_variables[agent].items()
        }
        infos = self.update_infos(variables, 'variables_obj', agent)
        variables = self.delete_not_observable_variables(variables, 'variables_obj', agent)
        return variables, infos
    
    def get_static_variables_state(
        self,
        state_argument:c_void_p,
        thermal_zone:str = None
        ) -> Tuple[Dict[str,Any],Dict[str,Any]]:
        """Get the static variable values defined in the static variable dict names and handles.

        Args:
            state_argument (c_void_p): EnergyPlus state.

        Returns:
            Dict[str,Any]: Static value dict values for the actual timestep.
        """
        if thermal_zone is None:
            ValueError("The thermal zone must be defined.")
        
        variables = {
            key: api.exchange.get_internal_variable_value(state_argument, handle)
            for key, handle
            in self.handle_static_variables[thermal_zone].items()
        }
        infos = self.update_infos(variables, 'static_variables', thermal_zone)
        variables = self.delete_not_observable_variables(variables, 'static_variables', thermal_zone)
        return variables, infos

    def get_meters_state(
        self,
        state_argument:c_void_p,
        agent:str = None
        ) -> Tuple[Dict[str,Any],Dict[str,Any]]:
        """Get the static variable values defined in the static variable dict names and handles.

        Args:
            state_argument (c_void_p): EnergyPlus state.

        Returns:
            Dict[str,Any]: Static value dict values for the actual timestep.
        """
        if agent is None:
            ValueError("The agent must be defined.")
        variables = {
            key: api.exchange.get_meter_value(state_argument, handle)
            for key, handle
            in self.meter_handles[agent].items()
        }
        infos = self.update_infos(variables, 'meters', agent)
        variables = self.delete_not_observable_variables(variables, 'meters', agent)
        return variables, infos

    def get_actuators_state(
        self,
        state_argument:c_void_p
        ) -> Tuple[Dict[str,Any],Dict[str,Any]]:
        """This funtion takes the state_argument and return a dictionary with the agent actuator values.

        Args:
            state_argument (c_void_p): State argument from EnergyPlus callback.

        Returns:
            Dict[str,Any]: Agent actuator values for the actual timestep.
        """
        return {
            key: api.exchange.get_actuator_value(state_argument, handle)
            for key, handle
            in self.actuator_handles.items()
        }, {}

    def get_simulation_parameters_values(
        self,
        state_argument: c_void_p,
        ) -> Tuple[Dict[str,Any],Dict[str,Any]]:
        """Get the simulation parameters values defined in the simulation parameter list.

        Args:
            state_argument (c_void_p): EnergyPlus state.
            simulation_parameter_list (List): List of the simulation parameters implemented in the observation space.

        Returns:
            Dict[str,int|float]: Dict of parameter names as keys and parameter values as values.
        """
        # Get timestep variables that are needed as input for some data_exchange methods.
        hour = api.exchange.hour(state_argument)
        zone_time_step_number = api.exchange.zone_time_step_number(state_argument)
        # Dict with the variables names and methods as values.
        parameter_methods = {
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
            'year': api.exchange.year(state_argument), # Get the “current” year of the simulation, read from the EPW. All simulations operate at a real year, either user specified or automatically selected by EnergyPlus based on other data (start day of week + leap year option).
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
            'tomorrow_weather_wind_speed_at_time': api.exchange.tomorrow_weather_wind_speed_at_time(state_argument, hour, zone_time_step_number),
        }
        # Return the dictionary with variables names and output values of the methods used.
        include = []
        parameters_keys = [key for key in self.env_config['simulation_parameters'].keys()]
        for paramater in parameters_keys:
            if self.env_config['simulation_parameters'][paramater]:
                include.append(paramater)
        variables = {paramater: parameter_methods[paramater] for paramater in include}
        infos = self.update_infos(variables, 'simulation_parameters')
        variables = self.delete_not_observable_variables(variables, 'simulation_parameters')
        return variables, infos

    def get_zone_simulation_parameters_values(
        self,
        state_argument: c_void_p,
        ) -> Tuple[Dict[str,Any],Dict[str,Any]]:
        """Get the simulation parameters values defined in the simulation parameter list.

        Args:
            state_argument (c_void_p): EnergyPlus state.
            simulation_parameter_list (List): List of the simulation parameters implemented in the observation space.

        Returns:
            Dict[str,int|float]: Dict of parameter names as keys and parameter values as values.
        """
        # Dict with the variables names and methods as values.
        parameter_methods = {
            'system_time_step': api.exchange.system_time_step(state_argument), # Gets the current system time step value in EnergyPlus. The system time step is variable and fluctuates during the simulation.
            'zone_time_step': api.exchange.zone_time_step(state_argument), # Gets the current zone time step value in EnergyPlus. The zone time step is variable and fluctuates during the simulation.
            'zone_time_step_number': api.exchange.zone_time_step_number(state_argument), # The current zone time step index, from 1 to the number of zone time steps per hour
        }
        # Return the dictionary with variables names and output values of the methods used.
        include = []
        parameters_keys = [key for key in self.env_config['zone_simulation_parameters'].keys()]
        for paramater in parameters_keys:
            if self.env_config['zone_simulation_parameters'][paramater]:
                include.append(paramater)
        variables = {paramater: parameter_methods[paramater] for paramater in include}
        infos = self.update_infos(variables, 'zone_simulation_parameters')
        variables = self.delete_not_observable_variables(variables, 'zone_simulation_parameters')
        return variables, infos

    def get_weather_prediction(
        self,
        state_argument: c_void_p,
    ) -> Tuple[Dict[str,Any],Dict[str,Any]]:
        if not self.env_config['use_one_day_weather_prediction']:
            return {}
        # Get timestep variables that are needed as input for some data_exchange methods.
        hour = api.exchange.hour(state_argument)
        zone_time_step_number = api.exchange.zone_time_step_number(state_argument)
        
        prediction_variables:Dict[str,bool] = self.env_config['prediction_variables']
        prediction_variables_methods: Dict[str,Any] = {
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
            'tomorrow_weather_wind_speed_at_time': api.exchange.tomorrow_weather_wind_speed_at_time(state_argument, hour, zone_time_step_number),
        }
        weather_pred = {}
        # The list sigma_max contains the standard deviation of the predictions in the following order:
        #   - Dry Bulb Temperature in °C with squer desviation of 2.05 °C, 
        #   - Relative Humidity in % with squer desviation of 20%, 
        #   - Wind Direction in degree with squer desviation of 40°, 
        #   - Wind Speed in m/s with squer desviation of 3.41 m/s, 
        #   - Barometric pressure in Pa with a standart deviation of 1000 Pa, 
        #   - Liquid Precipitation Depth in mm with desviation of 0.5 mm.
        for h in range(self.env_config['prediction_hours']):
            # For each hour, the sigma value goes from a minimum error of zero to the value listed in sigma_max following a linear function:
            prediction_hour = hour+1 + h
            if prediction_hour < 24:
                for key in prediction_variables.keys():
                    if prediction_variables[key]:
                        weather_pred.update({f'today_weather_{key}_at_time_{prediction_hour}': prediction_variables_methods[f'today_weather_{key}_at_time']})
            else:
                prediction_hour_t = prediction_hour - 24
                for key in prediction_variables.keys():
                    if prediction_variables[key]:
                        weather_pred.update({f'tomorrow_weather_{key}_at_time_{prediction_hour_t}': prediction_variables_methods[f'tomorrow_weather_{key}_at_time']})
        
        return weather_pred, {}
        
    def update_infos(
        self,
        dict_parameters: Dict[str,Any],
        belong_to: str = None, # variables_env, simulation_parameters, meters, etc.
        reference: str = None, # thermal_zone_id, agent_id
        ) -> Dict[str,Any]:
        infos_dict: Dict[str,Any] = {}
        if belong_to is None:
            raise ValueError("The 'belong_to' argument must be specified.")
        
        elif belong_to == 'variables_env':
            # add to infos_dict the variables listed in self.env_config['infos_variables']['variables_env']
            for variable in self.env_config['infos_variables']['variables_env']:
                infos_dict.update({variable: dict_parameters[variable]})
        
        elif belong_to == 'simulation_parameters':
            # add to infos_dict the variables listed in self.env_config['infos_variables']['variables_env']
            for variable in self.env_config['infos_variables']['simulation_parameters']:
                infos_dict.update({variable: dict_parameters[variable]})
        
        elif belong_to == 'variables_obj':
            # add to infos_dict the variables listed in self.env_config['infos_variables']['variables_env']
            if reference == None:
                raise ValueError("The 'reference' argument must be specified.")
            for variable in self.env_config['infos_variables']['variables_obj'][reference]:
                infos_dict.update({variable: dict_parameters[variable]})
        
        elif belong_to == 'meters':
            # add to infos_dict the variables listed in self.env_config['infos_variables']['variables_env']
            if reference == None:
                raise ValueError("The 'reference' argument must be specified.")
            for variable in self.env_config['infos_variables']['meters'][reference]:
                infos_dict.update({variable: dict_parameters[variable]})
        
        elif belong_to == 'static_variables':
            # add to infos_dict the variables listed in self.env_config['infos_variables']['variables_env']
            if reference == None:
                raise ValueError("The 'reference' argument must be specified.")
            for variable in self.env_config['infos_variables']['static_variables'][reference]:
                infos_dict.update({variable: dict_parameters[variable]})
        
        elif belong_to == 'variables_thz':
            # add to infos_dict the variables listed in self.env_config['infos_variables']['variables_env']
            if reference == None:
                raise ValueError("The 'reference' argument must be specified.")
            for variable in self.env_config['infos_variables']['variables_thz'][reference]:
                infos_dict.update({variable: dict_parameters[variable]})
        
        elif belong_to == 'zone_simulation_parameters':
            # add to infos_dict the variables listed in self.env_config['infos_variables']['variables_env']
            if reference == None:
                raise ValueError("The 'reference' argument must be specified.")
            for variable in self.env_config['infos_variables']['zone_simulation_parameters'][reference]:
                infos_dict.update({variable: dict_parameters[variable]})
        
        else:
            raise ValueError(f"The 'belong_to' argument must be one of the following: {self.env_config['infos_variables'].keys()}")
        
        return infos_dict

    def delete_not_observable_variables(
        self,
        dict_parameters: Dict[str,Any],
        belong_to: str = None, # variables_env, simulation_parameters, meters, etc.
        reference: str = None, # thermal_zone_id, agent_id
        ) -> Dict[str,Any]:
        
        if belong_to is None:
            raise ValueError("The 'belong_to' argument must be specified.")
        
        elif belong_to == 'variables_env':
            for variable in self.env_config['no_observable_variables']['variables_env']:
                del dict_parameters[variable]
        
        elif belong_to == 'simulation_parameters':
            for variable in self.env_config['no_observable_variables']['simulation_parameters']:
                del dict_parameters[variable]
        
        elif belong_to == 'variables_obj':
            if reference == None:
                raise ValueError("The 'reference' argument must be specified.")
            for variable in self.env_config['no_observable_variables']['variables_obj'][reference]:
                del dict_parameters[variable]
        
        elif belong_to == 'meters':
            if reference == None:
                raise ValueError("The 'reference' argument must be specified.")
            for variable in self.env_config['no_observable_variables']['meters'][reference]:
                del dict_parameters[variable]
        
        elif belong_to == 'static_variables':
            if reference == None:
                raise ValueError("The 'reference' argument must be specified.")
            for variable in self.env_config['no_observable_variables']['static_variables'][reference]:
                del dict_parameters[variable]
        
        elif belong_to == 'variables_thz':
            if reference == None:
                raise ValueError("The 'reference' argument must be specified.")
            for variable in self.env_config['no_observable_variables']['variables_thz'][reference]:
                del dict_parameters[variable]
        
        elif belong_to == 'zone_simulation_parameters':
            if reference == None:
                raise ValueError("The 'reference' argument must be specified.")
            for variable in self.env_config['no_observable_variables']['zone_simulation_parameters'][reference]:
                del dict_parameters[variable]
        
        else:
            raise ValueError(f"The 'belong_to' argument must be one of the following: {self.env_config['no_observable_variables'].keys()}")
        
        return dict_parameters
   
    def _init_callback(self, state_argument) -> bool:
        """
        Initialize EnergyPlus handles and checks if simulation runtime is ready.
        
        Args:
            state_argument (c_void_p): EnergyPlus state pointer. This is created with `api.state_manager.new_state()`.

        Returns:
            bool: True if the simulation is ready to perform actions.
        """
        self.init_handles = self._init_handles(state_argument)
        self.initialized = self.init_handles and not api.exchange.warmup_flag(state_argument)
        
        return self.initialized

    def _init_handles(self, state_argument) -> bool:
        """
        Initialize sensors/actuators handles to interact with during simulation.
        
        Args:
            state_argument (c_void_p): EnergyPlus state pointer. This is created with `api.state_manager.new_state()`.

        Returns:
            bool: True if all handles were initialized successfully.
        """        
        if not self.init_handles:
            if not api.exchange.api_data_fully_ready(state_argument):
                return False
        # Get handles of environment variables: agents (actuators) and site variables.
        self.actuator_handles.update({
            key: api.exchange.get_actuator_handle(state_argument, *actuator)
            for key, actuator in self.actuators.items()
        })
        if any([v == -1 for v in self.actuator_handles.values()]):
            available_data = api.exchange.list_available_api_data_csv(state_argument).decode('utf-8')
            ValueError(
                f"got -1 handle, check your actuator names:\n"
                f"> actuator_handles: {self.actuator_handles}\n"
                f"> available EnergyPlus API data: {available_data}"
            )
            return False
        
        self.handle_site_variables.update({
            key: api.exchange.get_variable_handle(state_argument, *actuator)
            for key, actuator in self.dict_site_variables.items()
        })
        if any([v == -1 for v in self.handle_site_variables.values()]):
            available_data = api.exchange.list_available_api_data_csv(state_argument).decode('utf-8')
            ValueError(
                f"got -1 handle, check your variables_env names:\n"
                f"> handle_site_variables: {self.handle_site_variables}\n"
                f"> available EnergyPlus API data: {available_data}"
            )
            return False
        
        # Object variables and meters are consider for each agent.
        for agent in self._agent_ids:
            self.handle_object_variables[agent].update({
                key: api.exchange.get_variable_handle(state_argument, *var)
                for key, var in self.dict_object_variables.items()
            })
            self.meter_handles[agent].update({
                key: api.exchange.get_meter_handle(state_argument, meter)
                for key, meter in self.meters.items()
            })
            for handles in [
                self.handle_object_variables[agent],
                self.meter_handles[agent],
            ]:
                if any([v == -1 for v in handles.values()]):
                    available_data = api.exchange.list_available_api_data_csv(state_argument).decode('utf-8')
                    ValueError(
                        f"got -1 handle, check your variables_obj names:\n"
                        f"> handle_object_variables: {self.handle_object_variables[agent]}\n"
                        f"> meter_handles: {self.meter_handles}\n"
                        f"> available EnergyPlus API data: {available_data}"
                    )
                    return False
        
        # Thermal zone and static variables are consider into the thermal zone properties.
        for thermal_zone in self._thermal_zone_ids:    
            self.handle_thermalzone_variables[thermal_zone].update({
                key: api.exchange.get_variable_handle(state_argument, *var)
                for key, var in self.dict_thermalzone_variables.items()
            })
            self.handle_static_variables[thermal_zone].update({
                key: api.exchange.get_internal_variable_handle(state_argument, *var)
                for key, var in self.dict_static_variables[thermal_zone].items()
            })
            for handles in [
                self.handle_thermalzone_variables[thermal_zone],
                self.handle_static_variables[thermal_zone],
            ]:
                if any([v == -1 for v in handles.values()]):
                    available_data = api.exchange.list_available_api_data_csv(state_argument).decode('utf-8')
                    ValueError(
                        f"got -1 handle, check your var/meter/actuator names:\n"
                        f"> handle_thermalzone_variables: {self.handle_thermalzone_variables}\n"
                        f"> handle_static_variables: {self.handle_static_variables}\n"
                        f"> available EnergyPlus API data: {available_data}"
                    )
                    return False
        return True

    def stop(self) -> None:
        """
        Method to stop EnergyPlus simulation and joint the threads.
        """
        if not self.simulation_complete:
            self.simulation_complete = True
        time.sleep(0.5)
        api.runtime.stop_simulation(self.energyplus_state)
        self._flush_queues()
        self.energyplus_exec_thread.join()
        self.energyplus_exec_thread = None
        self.first_observation = True
        api.runtime.clear_callbacks()
        api.state_manager.delete_state(self.energyplus_state)

    def failed(self) -> bool:
        """
        This method tells if a EnergyPlus simulations was finished successfully or not.

        Returns:
            bool: Boolean value of the success of the simulation.
        """
        return self.sim_results != 0

    def make_eplus_args(self) -> List[str]:
        """
        Make command line arguments to pass to EnergyPlus.
        
        Return:
            List[str]: List of arguments to pass to EnergyPlusEnv.
        """
        eplus_args = ["-r"] if self.env_config.get("csv", False) else []
        eplus_args += [
            "-w",
            self.env_config["epw_path"],
            "-d",
            f"{self.env_config['output_path']}/episode-{self.episode:08}-{os.getpid():05}-{self.unique_id}" if not self.env_config['evaluation'] else f"{self.env_config['output_path']}/evaluation-episode-{self.episode:08}-{os.getpid():05}-{self.unique_id}",
            self.env_config["epjson_path"]
        ]
        return eplus_args
    
    def _flush_queues(self):
        """
        Method to liberate the space in the different queue objects.
        """
        for q in [self.obs_queue, self.act_queue, self.infos_queue]:
            while not q.empty():
                q.get()
