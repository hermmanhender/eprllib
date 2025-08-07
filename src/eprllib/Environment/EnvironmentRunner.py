"""
Runner Base Class
==================

This script contain the EnergyPlus Runner that execute EnergyPlus from its 
Python API in the version 24.2.0.
"""
import os
import threading
import time
from queue import Queue
from typing import Any, Dict, List, Optional, Tuple
from ctypes import c_void_p
from eprllib.Agents.Triggers.BaseTrigger import BaseTrigger
from eprllib.Agents.Filters.BaseFilter import BaseFilter
from eprllib.AgentsConnectors.BaseConnector import BaseConnector
from eprllib.Utils.env_config_utils import EP_API_add_path
from eprllib.Utils.observation_utils import (
    get_actuator_name,
    get_internal_variable_name,
    get_meter_name,
    get_parameter_name,
    get_other_obs_name,
    get_variable_name,
    get_parameter_prediction_name,
    get_user_occupation_forecast_name
)
from eprllib.Utils.env_utils import calculate_occupancy_and_forecast
from eprllib import logger, EP_VERSION

# EnergyPlus Python API path adding
try:
    EP_API_add_path(EP_VERSION)
except RuntimeError as e:
    logger.error(f"Failed to add EnergyPlus API path: {e}")
    exit(1)
    
from pyenergyplus.api import EnergyPlusAPI

api = EnergyPlusAPI()

class EnvironmentRunner:
    """
    This object have the particularity of `start` EnergyPlus, `_collect_obs` and `_send_actions` to
    send it trhougt queue to the EnergyPlus Environment thread.
    """
    def __init__(
        self,
        env_config: Dict[str, Any],
        episode: int,
        obs_queue: Queue[Any],
        act_queue: Queue[Any],
        infos_queue: Queue[Any],
        agents: List[str],
        filter_fn: Dict[str,BaseFilter],
        trigger_fn: Dict[str, BaseTrigger],
        connector_fn: BaseConnector
        ) -> None:
        """
        Initializes the BaseRunner object.

        Args:
            env_config (Dict[str, Any]): Configuration settings for the environment.
            episode (int): The current episode number.
            obs_queue (Queue): Queue for sending observations.
            act_queue (Queue): Queue for receiving actions.
            infos_queue (Queue): Queue for sending additional information.
            agents (List): List of agents in the environment.
            filter_fn (Dict[str, BaseFilter]): Dictionary of filter functions for each agent.
            trigger_fn (Dict[str, BaseTrigger]): Dictionary of trigger functions for each agent.
            connector_fn (BaseConnector): Connector function for combining agent observations.
        """
        # Asignation of variables.
        self.env_config = env_config
        self.episode = episode
        self.obs_queue = obs_queue
        self.act_queue = act_queue
        self.infos_queue = infos_queue
        self.agents = agents
        
        # The queue events are generated (To sure the coordination with EnergyPlusEnvironment).
        self.obs_event = threading.Event()
        self.act_event = threading.Event()
        self.infos_event = threading.Event()
        
        # Variables to be used in this thread.
        self.energyplus_exec_thread: Optional[threading.Thread] = None
        self.energyplus_state: Optional[c_void_p] = None
        self.sim_results: int = 0
        self.initialized = False
        self.init_handles = False
        self.simulation_complete = False
        self.first_observation = True
        self.obs = {}
        self.infos = {agent: {} for agent in self.agents}
        self.unique_id = time.time()
        self.is_last_timestep: bool = False
        self.occupancy_next_timestep: int = 0
        
        # create a variable to save the obs dict key to use in posprocess
        # self.obs_keys = []
        # self.infos_keys = []
        
        # Define the action and observation functions.
        self.trigger_fn = trigger_fn
        self.filter_fn = filter_fn
        self.connector_fn = connector_fn
        
        # Declaration of variables, meters and actuators to use in the simulation. Handles
        # are used in _init_handle method.
        self.agent_variables_and_handles: Dict[str, Any] = {}
        for agent in self.agents:
            variables, variables_handles = self.set_variables(agent)
            internal_variables, internal_variables_handles = self.set_internal_variables(agent)
            meters, meters_handles = self.set_meters(agent)
            actuators, actuators_handles = self.set_actuators(agent)
            self.agent_variables_and_handles.update({
                f"{agent}_variables": [variables, variables_handles],
                f"{agent}_internal_variables": [internal_variables, internal_variables_handles],
                f"{agent}_meters": [meters, meters_handles],
                f"{agent}_actuators": [actuators, actuators_handles]
            })
        internal_actuators: List[Tuple[str,str,str]] = []
        for agent in self.agents:
            if self.env_config["agents_config"][agent]['observation']["user_occupation_forecast"]:
                internal_actuators.append(self.env_config["agents_config"][agent]['observation']["occupation_schedule"])
            # delete duplicate actuators in actuators_list
            internal_actuators = list(set(internal_actuators))
        self.internal_variables_and_handles: Dict[str, Any] = {}
        actuators, actuators_handles = self.set_internal_actuators(internal_actuators)
        self.internal_variables_and_handles.update({
                "internal_actuators": [actuators, actuators_handles]
            })
        
    
    def progress_handler(self, progress: int) -> None:
        # print(f"Simulation progress: {progress}%")
        if progress >= 99:
            self.is_last_timestep = True
            
    def start(self) -> None:
        """
        Starts the EnergyPlus simulation.
        """
        # Start a new EnergyPlus state (condition for execute EnergyPlus Python API).
        self.energyplus_state = api.state_manager.new_state()
        # Request variables.
        for agent in self.agents:
            if self.env_config["agents_config"][agent]['observation']["variables"] is not None:
                for variable in self.env_config["agents_config"][agent]['observation']["variables"]:
                    api.exchange.request_variable(
                        self.energyplus_state,
                        variable_name = variable[0],
                        variable_key = variable[1]
                    )
                    # print(f"Variable {get_variable_name(agent,variable[0],variable[1])} requested.")
        api.runtime.callback_begin_zone_timestep_after_init_heat_balance(self.energyplus_state, self._send_actions)
        api.runtime.callback_end_zone_timestep_after_zone_reporting(self.energyplus_state, self._collect_obs)
        api.runtime.callback_progress(self.energyplus_state, self.progress_handler)
        api.runtime.set_console_output_status(self.energyplus_state, self.env_config['ep_terminal_output'])
        
        def _run_energyplus():
            """
            Run EnergyPlus in a non-blocking way with Threads.
            """
            cmd_args = self.make_eplus_args()
            logger.info(f"running EnergyPlus with args: {cmd_args}")
            assert self.energyplus_state is not None, "EnergyPlus state is None."
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
        # To not perform observations when the callbacks and the 
        # warming period are not complete.
        if not self._init_callback(state_argument) or self.simulation_complete:
            return
        
        dict_agents_obs: Dict[str, Any] = {agent: {} for agent in self.agents}
        self.infos: Dict[str, Dict[str, Any]] = {agent: {} for agent in self.agents}
        
        # Agents observe: site state, thermal zone state (only the one that it belong), specific object variables 
        # and meters, and others parameters assigned as True in the env_config.observation object.
        # Get the state of the actuators.
        agent_states: Dict[str, Dict[str, Any]] = {agent: {} for agent in self.agents}
        for agent in self.agents:
            agent_states[agent].update(self.get_variables_state(state_argument, agent))
            agent_states[agent].update(self.get_internal_variables_state(state_argument, agent))
            agent_states[agent].update(self.get_meters_state(state_argument, agent))
            agent_states[agent].update(self.get_simulation_parameters_values(state_argument, agent))
            agent_states[agent].update(self.get_zone_simulation_parameters_values(state_argument, agent))
            agent_states[agent].update(self.get_weather_prediction(state_argument, agent))
            agent_states[agent].update(self.get_actuators_state(state_argument, agent))
            agent_states[agent].update(self.get_other_obs(self.env_config, agent))
            agent_states[agent].update(self.get_user_occupation_forecast(state_argument, self.env_config, agent))
        
        dict_agents_obs = {agent: None for agent in self.agents}
        for agent in self.agents:
            dict_agents_obs.update({
                agent: self.filter_fn[agent].get_filtered_obs(
                    self.env_config,
                    agent_states[agent]
                )})
        
        # First is send the observation of the top-level agents. If there is only one shape of agents, the _collect_obs
        # method is ended here. If there are more than one shape of agents, the action (goal) will be requested and the
        # observations of the next shape will be request. This is implemented until reach the lowest level of the hierarchy
        # if any.
        top_level_agents_obs, top_level_agents_infos, is_lowest_level = self.connector_fn.set_top_level_obs(
            self.env_config,
            agent_states,
            dict_agents_obs,
            self.infos,
            self.is_last_timestep
        )
        
        # Set the agents observation and infos to communicate with the EPEnv.
        self.obs_queue.put(top_level_agents_obs)
        self.obs_event.set()
        self.infos_queue.put(top_level_agents_infos)
        self.infos_event.set()
        
        # Implementation for hierarchical agents.
        if not is_lowest_level:
            while not is_lowest_level:
                # Wait for a goal selection
                event_flag = self.act_event.wait(self.env_config["timeout"])
                if not event_flag:
                    # print(f"Timeout waiting for action from agent {agent}.")
                    return
                # Get the action from the EnergyPlusEnvironment `step` method.
                actions: Dict[str, int | float] = self.act_queue.get()
                goals: Dict[str, Optional[int|float]] = {agent: None for agent in actions.keys()}
                for agent in actions.keys():
                    goals.update({agent: self.trigger_fn[agent].action_to_goal(actions[agent])})
                
                low_level_agents_obs, low_level_agents_infos, is_lowest_level = self.connector_fn.set_low_level_obs(
                    self.env_config,
                    agent_states,
                    dict_agents_obs,
                    self.infos,
                    goals
                )
            
                # Set the agents observation and infos to communicate with the EPEnv.
                self.obs_queue.put(low_level_agents_obs)
                self.obs_event.set()
                self.infos_queue.put(low_level_agents_infos)
                self.infos_event.set()
        

    def _collect_first_obs(self, state_argument: c_void_p):
        """
        This method is used to collect only the first observation of the environment when the episode beggins.

        Args:
            state_argument (): EnergyPlus state pointer. This is created with `api.state_manager.new_state()`.
        """
        if self.first_observation:
            # print("Collecting the first observation.")
            self._collect_obs(state_argument)
            self.first_observation = False
        else:
            return

    def _send_actions(self, state_argument: c_void_p):
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
        event_flag = self.act_event.wait(self.env_config["timeout"])
        if not event_flag:
            # print("Timeout waiting for action from agent.")
            return
        # Get the action from the EnergyPlusEnvironment `step` method.
        dict_action: Dict[str, Any] = self.act_queue.get()
        
        for agent in dict_action.keys():
            # Transform action must to consider the agents and actuators and transform agents actions to actuators actions,
            # considering that one agent could manage more than one actuator.
            actuator_list = [actuator for actuator in self.agent_variables_and_handles[f"{agent}_actuators"][0].keys()]
            
            actuator_actions = self.trigger_fn[agent].agent_to_actuator_action(dict_action[agent], actuator_list)
            
            # Check if there is an actuator_dict_actions value equal to None.
            for actuator in actuator_list:
                if actuator_actions[actuator] is None:
                    msg = f"The actuator {actuator} has no action defined in the action dict for agent {agent}."
                    logger.error(msg)
                    raise ValueError(msg)
                
            # Perform the actions in EnergyPlus simulation.
            for actuator in actuator_list:
                action = self.trigger_fn[agent].get_actuator_action(actuator_actions[actuator], actuator)
                api.exchange.set_actuator_value(
                    state=state_argument,
                    actuator_handle=self.agent_variables_and_handles[f"{agent}_actuators"][1][actuator],
                    actuator_value=action
                )
        
        for actuator in self.internal_variables_and_handles["internal_actuators"][0].keys():
            api.exchange.set_actuator_value(
                state=state_argument,
                actuator_handle=self.internal_variables_and_handles["internal_actuators"][1][actuator],
                actuator_value=self.occupancy_next_timestep
            )

    def set_variables(self, agent:str) -> Tuple[Dict[str, Tuple [str, str]], Dict[str, int]]:
        """
        The EnergyPlus variables are defined in the environment configuration.

        Args:
            env_config (Dict[str, Any]): The EnvConfig dictionary.

        Returns:
            Tuple[Dict[str, Tuple [str, str]], Dict[str, int]]: The environment variables and their handles.
        """
        # Define the emptly Dict to include variables names and handles. The handles Dict return as emptly Dict to be used latter.
        var_handles: Dict[str, int] = {}
        variables: Dict[str, Tuple [str, str]] = {}
        if self.env_config["agents_config"][agent]['observation']["variables"] is not None:
            variables.update({
                get_variable_name(agent,variable[0],variable[1]): variable
                for variable 
                in self.env_config["agents_config"][agent]['observation']["variables"]
            })
        return variables, var_handles

    def set_internal_variables(self, agent:str) -> Tuple[Dict[str, Tuple [str, str]], Dict[str, int]]:
        """
        Set the internal variables to handle and get the values in the environment.

        Args:
            env_config (Dict[str, Any]): The EnvConfig dictionary.

        Returns:
            Tuple[Dict[str, Tuple [str, str]], Dict[str, int]]: The environment variables and their handles.
        """
        # Define the emptly Dict to include variables names and handles. The handles Dict return as emptly Dict to be used latter.
        var_handles: Dict[str, int] = {}
        variables: Dict[str, Tuple [str, str]] = {}
        if self.env_config["agents_config"][agent]['observation']["internal_variables"] is not None:
            variables.update({
                get_internal_variable_name(agent,variable[0],variable[1]): variable
                for variable 
                in self.env_config["agents_config"][agent]['observation']["internal_variables"]
            })
        return variables, var_handles

    def set_meters(self, agent:str) -> Tuple[Dict[str, Tuple [str, str]], Dict[str, int]]:
        """The EnergyPlus meters are defined in the environment configuration.
        
        Args:
            env_config (Dict[str, Any]): The EnvConfig dictionary.

        Returns:
            Tuple[Dict[str, Tuple [str, str]], Dict[str, int]]: The meters and their handles.
        """
        var_handles: Dict[str, int] = {}
        variables: Dict[str, Tuple [str, str]] = {}
        if self.env_config["agents_config"][agent]['observation']["meters"] is not None:
            variables.update({
                get_meter_name(agent,variable): variable 
                for variable 
                in self.env_config["agents_config"][agent]['observation']["meters"]
            })
        return variables, var_handles

    def set_actuators(self, agent:str) -> Tuple[Dict[str,Tuple[str,str,str]], Dict[str, int]]:
        """The EnergyPlus actuators are defined in the environment configuration.

        Args:
            agent str: The EnvConfig dictionary.

        Returns:
            Tuple[Dict[str,Tuple[str,str,str]], Dict[str,int]]: The actuators and their handles.
        """
        actuators: Dict[str,Tuple[str,str,str]] = {}
        actuator_handles: Dict[str, int] = {}
        
        for actuator_config in self.env_config["agents_config"][agent]["action"]["actuators"]:
            actuators.update({
                get_actuator_name(agent,actuator_config[0],actuator_config[1],actuator_config[2]): actuator_config
            })
        
        return actuators, actuator_handles
    
    def set_internal_actuators(self, internal_actuators: List[Tuple[str,str,str]]) -> Tuple[Dict[str,Tuple[str,str,str]], Dict[str, int]]:
        """The EnergyPlus actuators are defined in the environment configuration.

        Args:
            agent str: The EnvConfig dictionary.

        Returns:
            Tuple[Dict[str,Tuple[str,str,str]], Dict[str,int]]: The actuators and their handles.
        """
        actuators: Dict[str,Tuple[str,str,str]] = {}
        actuator_handles: Dict[str, int] = {}
        
        for actuator_config in internal_actuators:
            actuators.update({
                get_actuator_name("internal",actuator_config[0],actuator_config[1],actuator_config[2]): actuator_config
            })
        
        return actuators, actuator_handles
    
    def get_variables_state(
        self,
        state_argument: c_void_p,
        agent: Optional[str] = None
        ) -> Dict[str,Any]:
        """This funtion takes the state_argument and return a dictionary with the agent actuator values.

        Args:
            state_argument (): State argument from EnergyPlus callback.

        Returns:
            Dict[str,Any]: Agent actuator values for the actual timestep.
        """
        if agent is None:
            msg = "The agent must be defined."
            logger.error(msg)
            raise ValueError(msg)
        variables: Dict[str,Any] = {
            key: api.exchange.get_variable_value(state_argument, handle)
            for key, handle
            in self.agent_variables_and_handles[f"{agent}_variables"][1].items()
        }        
        return variables
    
    def get_internal_variables_state(
        self,
        state_argument: c_void_p,
        agent: Optional[str] = None
        ) -> Dict[str,Any]:
        """Get the static variable values defined in the static variable dict names and handles.

        Args:
            state_argument (): EnergyPlus state.

        Returns:
            Dict[str,Any]: Static value dict values for the actual timestep.
        """
        if agent is None:
            msg = "The agent must be defined."
            logger.error(msg)
            raise ValueError(msg)
        variables: Dict[str,Any] = {
            key: api.exchange.get_internal_variable_value(state_argument, handle)
            for key, handle
            in self.agent_variables_and_handles[f"{agent}_internal_variables"][1].items()
        }
        return variables

    def get_meters_state(
        self,
        state_argument: c_void_p,
        agent: Optional[str] = None
        ) -> Dict[str,Any]:
        """Get the static variable values defined in the static variable dict names and handles.

        Args:
            state_argument (): EnergyPlus state.

        Returns:
            Dict[str,Any]: Static value dict values for the actual timestep.
        """
        if agent is None:
            msg = "The agent must be defined."
            logger.error(msg)
            raise ValueError(msg)
        
        variables: Dict[str,Any] = {
            key: api.exchange.get_meter_value(state_argument, handle)
            for key, handle
            in self.agent_variables_and_handles[f"{agent}_meters"][1].items()
        }
        return variables
        
    def get_actuators_state(
        self,
        state_argument: c_void_p,
        agent: Optional[str] = None
        ) -> Dict[str,Any]:
        """This funtion takes the state_argument and return a dictionary with the agent actuator values.

        Args:
            state_argument (): State argument from EnergyPlus callback.

        Returns:
            Dict[str,Any]: Agent actuator values for the actual timestep.
        """
        if agent is None:
            msg = "The agent must be defined."
            logger.error(msg)
            raise ValueError(msg)
        
        variables: Dict[str,Any] = {
            key: api.exchange.get_actuator_value(state_argument, handle)
            for key, handle
            in self.agent_variables_and_handles[f"{agent}_actuators"][1].items()
        }
        return variables
    
    def get_simulation_parameters_values(
        self,
        state_argument: c_void_p,
        agent: Optional[str] = None
        ) -> Dict[str,Any]:
        """Get the simulation parameters values defined in the simulation parameter list.

        Args:
            state_argument (): EnergyPlus state.
            simulation_parameter_list (List): List of the simulation parameters implemented in the observation space.

        Returns:
            Dict[str,int|float]: Dict of parameter names as keys and parameter values as values.
        """
        # Get timestep variables that are needed as input for some data_exchange methods.
        hour = api.exchange.hour(state_argument)
        zone_time_step_number = api.exchange.zone_time_step_number(state_argument)
        # Dict with the variables names and methods as values.
        parameter_methods: Dict[str,Any] = {
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
        variables = {}
        
        # Return the dictionary with variables names and output values of the methods used.
        include: List[str] = []
        parameters_keys = [key for key in parameter_methods.keys()]
        for paramater in parameters_keys:
            if self.env_config["agents_config"][agent]['observation']["simulation_parameters"][paramater]:
                include.append(paramater)
        assert isinstance(agent, str), "Agent must be a string."
        variables: Dict[str,Any] = {
            get_parameter_name(agent,paramater): parameter_methods[paramater] 
            for paramater 
            in include
        }
        return variables

    def get_zone_simulation_parameters_values(
        self,
        state_argument: c_void_p,
        agent: Optional[str]=None,
        ) -> Dict[str,Any]:
        """Get the simulation parameters values defined in the simulation parameter list.

        Args:
            state_argument (): EnergyPlus state.
            simulation_parameter_list (List): List of the simulation parameters implemented in the observation space.

        Returns:
            Dict[str,int|float]: Dict of parameter names as keys and parameter values as values.
        """
        # Dict with the variables names and methods as values.
        parameter_methods: Dict[str,Any] = {
            'system_time_step': api.exchange.system_time_step(state_argument), # Gets the current system time step value in EnergyPlus. The system time step is variable and fluctuates during the simulation.
            'zone_time_step': api.exchange.zone_time_step(state_argument), # Gets the current zone time step value in EnergyPlus. The zone time step is variable and fluctuates during the simulation.
            'zone_time_step_number': api.exchange.zone_time_step_number(state_argument), # The current zone time step index, from 1 to the number of zone time steps per hour
        }
        variables: Dict[str,Any] = {}
        
        # Return the dictionary with variables names and output values of the methods used.
        include: List[str] = []
        parameters_keys = [key for key in parameter_methods.keys()]
        for paramater in parameters_keys:
            if self.env_config["agents_config"][agent]['observation']['zone_simulation_parameters'][paramater]:
                include.append(paramater)
        assert isinstance(agent, str), "Agent must be a string."
        variables: Dict[str,Any] = {
            get_parameter_name(agent,paramater): parameter_methods[paramater] 
            for paramater 
            in include
        }
        return variables

    def get_weather_prediction(
        self,
        state_argument: c_void_p,
        agent: Optional[str]=None
    ) -> Dict[str,Any]:
        assert isinstance(agent, str), "Agent must be a string."
        if not self.env_config["agents_config"][agent]['observation']['use_one_day_weather_prediction']:
            return {}
        # Get timestep variables that are needed as input for some data_exchange methods.
        hour: int = api.exchange.hour(state_argument)
        
        prediction_variables_methods: Dict[str,Any] = {
            'today_weather_albedo_at_time': api.exchange.today_weather_albedo_at_time,
            'today_weather_beam_solar_at_time': api.exchange.today_weather_beam_solar_at_time,
            'today_weather_diffuse_solar_at_time': api.exchange.today_weather_diffuse_solar_at_time,
            'today_weather_horizontal_ir_at_time': api.exchange.today_weather_horizontal_ir_at_time,
            'today_weather_is_raining_at_time': api.exchange.today_weather_is_raining_at_time,
            'today_weather_is_snowing_at_time': api.exchange.today_weather_is_snowing_at_time,
            'today_weather_liquid_precipitation_at_time': api.exchange.today_weather_liquid_precipitation_at_time,
            'today_weather_outdoor_barometric_pressure_at_time': api.exchange.today_weather_outdoor_barometric_pressure_at_time,
            'today_weather_outdoor_dew_point_at_time': api.exchange.today_weather_outdoor_dew_point_at_time,
            'today_weather_outdoor_dry_bulb_at_time': api.exchange.today_weather_outdoor_dry_bulb_at_time,
            'today_weather_outdoor_relative_humidity_at_time': api.exchange.today_weather_outdoor_relative_humidity_at_time,
            'today_weather_sky_temperature_at_time': api.exchange.today_weather_sky_temperature_at_time,
            'today_weather_wind_direction_at_time': api.exchange.today_weather_wind_direction_at_time,
            'today_weather_wind_speed_at_time': api.exchange.today_weather_wind_speed_at_time,
            'tomorrow_weather_albedo_at_time': api.exchange.tomorrow_weather_albedo_at_time,
            'tomorrow_weather_beam_solar_at_time': api.exchange.tomorrow_weather_beam_solar_at_time,
            'tomorrow_weather_diffuse_solar_at_time': api.exchange.tomorrow_weather_diffuse_solar_at_time,
            'tomorrow_weather_horizontal_ir_at_time': api.exchange.tomorrow_weather_horizontal_ir_at_time,
            'tomorrow_weather_is_raining_at_time': api.exchange.tomorrow_weather_is_raining_at_time,
            'tomorrow_weather_is_snowing_at_time': api.exchange.tomorrow_weather_is_snowing_at_time,
            'tomorrow_weather_liquid_precipitation_at_time': api.exchange.tomorrow_weather_liquid_precipitation_at_time,
            'tomorrow_weather_outdoor_barometric_pressure_at_time': api.exchange.tomorrow_weather_outdoor_barometric_pressure_at_time,
            'tomorrow_weather_outdoor_dew_point_at_time': api.exchange.tomorrow_weather_outdoor_dew_point_at_time,
            'tomorrow_weather_outdoor_dry_bulb_at_time': api.exchange.tomorrow_weather_outdoor_dry_bulb_at_time,
            'tomorrow_weather_outdoor_relative_humidity_at_time': api.exchange.tomorrow_weather_outdoor_relative_humidity_at_time,
            'tomorrow_weather_sky_temperature_at_time': api.exchange.tomorrow_weather_sky_temperature_at_time,
            'tomorrow_weather_wind_direction_at_time': api.exchange.tomorrow_weather_wind_direction_at_time,
            'tomorrow_weather_wind_speed_at_time': api.exchange.tomorrow_weather_wind_speed_at_time,
        }
        
        variables: Dict[str,Any] = {}
        prediction_variables:Dict[str,bool] = self.env_config["agents_config"][agent]['observation']['prediction_variables']
        for h in range(self.env_config["agents_config"][agent]['observation']['prediction_hours']):
            # For each hour, the sigma value goes from a minimum error of zero to the value listed in sigma_max following a linear function:
            prediction_hour: int = hour+1 + h
            if prediction_hour < 24:
                for key in prediction_variables.keys():
                    if prediction_variables[key]:
                        variables.update({
                            get_parameter_prediction_name(agent,key,prediction_hour): prediction_variables_methods[f'today_weather_{key}_at_time'](state_argument,prediction_hour,1)
                        })
            else:
                prediction_hour_t = prediction_hour - 24
                for key in prediction_variables.keys():
                    if prediction_variables[key]:
                        variables.update({
                            get_parameter_prediction_name(agent,key,prediction_hour_t): prediction_variables_methods[f'tomorrow_weather_{key}_at_time'](state_argument,prediction_hour_t,1)
                        })
        return variables
        
    def get_other_obs(
        self,
        env_config: Dict[str,Any],
        agent: Optional[str] = None
        ) -> Dict[str,Any]:
        """Get the static variable values defined in the static variable dict names and handles.

        Args:
            env_config (Dict[str,Any]): Environment coniguration.

        Returns:
            Dict[str,Any]: Static value dict values for the actual timestep.
        """
        if agent is None:
            msg = "The agent must be defined."
            logger.error(msg)
            raise ValueError(msg)
        
        variables: Dict[str,Any] = {}
        variables.update({
            get_other_obs_name(agent,key): value 
            for key, value
            in env_config["agents_config"][agent]["observation"]["other_obs"].items()
        })
        env_config["agents_config"][agent]["observation"]["other_obs"]
        return variables
    
    def get_user_occupation_forecast(
        self,
        state_argument: c_void_p,
        env_config: Dict[str,Any],
        agent: Optional[str]=None
    ) -> Dict[str,Any]:
        assert isinstance(agent, str), "Agent must be a string."
        if not env_config["agents_config"][agent]['observation']['user_occupation_forecast']:
            return {}
        # Get timestep variables that are needed as input for some data_exchange methods.
        variables: Dict[str,Any] = {}
        self.occupancy_next_timestep, forecast_vector = calculate_occupancy_and_forecast(
            api.exchange.hour(state_argument),
            api.exchange.day_of_month(state_argument),
            api.exchange.month(state_argument),
            api.exchange.year(state_argument),
            api.exchange.holiday_index(state_argument) > 0,
            env_config["agents_config"][agent]['observation']['user_type'],
            env_config["agents_config"][agent]['observation']['zone_type'],
            env_config["agents_config"][agent]['observation']['num_simulations'],
            env_config["agents_config"][agent]['observation']['probability_variation'],
            env_config["agents_config"][agent]['observation']['probability_variation_evening_night_hours'],
            env_config["agents_config"][agent]['observation']['summer_months']
        )
        for h in range(24):
            variables.update({get_user_occupation_forecast_name(agent,h+1): forecast_vector[h]})
            
        return variables
    
    def _init_callback(self, state_argument: c_void_p) -> bool:
        """
        Initialize EnergyPlus handles and checks if simulation runtime is ready.
        
        Args:
            state_argument (): EnergyPlus state pointer. This is created with `api.state_manager.new_state()`.

        Returns:
            bool: True if the simulation is ready to perform actions.
        """
        if not self.init_handles:
            # print("Initializing handles...")
            self.init_handles = self._init_handles(state_argument)
        
        self.initialized = self.init_handles and not api.exchange.warmup_flag(state_argument)
        
        return self.initialized

    def _init_handles(self, state_argument: c_void_p) -> bool:
        """
        Initialize sensors/actuators handles to interact with during simulation.
        
        Args:
            state_argument (): EnergyPlus state pointer. This is created with `api.state_manager.new_state()`.

        Returns:
            bool: True if all handles were initialized successfully.
        """        
        if not self.init_handles:
            if not api.exchange.api_data_fully_ready(state_argument):
                return False
        
        self.agent_variables_and_handles["internal_actuators"][1].update({
            key: api.exchange.get_actuator_handle(state_argument, *actuator)
            for key, actuator in self.agent_variables_and_handles["internal_actuators"][0].items()
        })
        
        for agent in self.agents:
            self.agent_variables_and_handles[f"{agent}_variables"][1].update({
                key: api.exchange.get_variable_handle(state_argument, *actuator)
                for key, actuator in self.agent_variables_and_handles[f"{agent}_variables"][0].items()
            })
            self.agent_variables_and_handles[f"{agent}_internal_variables"][1].update({
                key: api.exchange.get_internal_variable_handle(state_argument, *var)
                for key, var in self.agent_variables_and_handles[f"{agent}_internal_variables"][0].items()
            })
            self.agent_variables_and_handles[f"{agent}_meters"][1].update({
                key: api.exchange.get_meter_handle(state_argument, meter)
                for key, meter in self.agent_variables_and_handles[f"{agent}_meters"][0].items()
            })
            self.agent_variables_and_handles[f"{agent}_actuators"][1].update({
                key: api.exchange.get_actuator_handle(state_argument, *actuator)
                for key, actuator in self.agent_variables_and_handles[f"{agent}_actuators"][0].items()
            })
            
            for handles in [
                self.agent_variables_and_handles[f"{agent}_variables"][1],
                self.agent_variables_and_handles[f"{agent}_internal_variables"][1],
                self.agent_variables_and_handles[f"{agent}_meters"][1],
                self.agent_variables_and_handles[f"{agent}_actuators"][1]
            ]:
                if any([v == -1 for v in handles.values()]):
                    available_data = api.exchange.list_available_api_data_csv(state_argument).decode('utf-8')
                    msg = (
                        f"Some handles were not initialized correctly for agent {agent}.\n"
                        f"> handles with error: {handles}\n"
                        f"> available EnergyPlus API data: {available_data}"
                    )
                    logger.error(msg)
                    raise ValueError(msg)
        
        return True

    def stop(self) -> None:
        """
        Method to stop EnergyPlus simulation and joint the threads.
        """
        # print("Stopping EnergyPlus simulation...")
        if not self.simulation_complete:
            self.simulation_complete = True
        if self.energyplus_state is not None:
            api.runtime.stop_simulation(self.energyplus_state)
        self._flush_queues()
        if self.energyplus_exec_thread is not None:
            self.energyplus_exec_thread.join()
        
        self.energyplus_exec_thread = None
        self.first_observation = True
        api.runtime.clear_callbacks()
        if self.energyplus_state is not None:
            api.state_manager.delete_state(self.energyplus_state)
            self.energyplus_state = None

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
