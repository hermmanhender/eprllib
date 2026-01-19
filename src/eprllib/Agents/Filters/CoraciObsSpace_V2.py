"""
Default Filter
===============

The state-space is composed of a series of observations provided as inputs to the agent.
The values assumed by the space-state variables influence the control action. The state-
space was made of 52 features, reported in Table 2, together with their lower and upper
bounds. The variables chosen are feasible to be collected in a real-world implementation
and provide to the agent the information necessary to predict the immediate future re-
wards. The Indoor Air Temperature information during each control step was described as dif-
ference between the indoor temperature setpoint SPINT and the actual indoor air tempera-
ture TINT as directly linked to the reward formulation, as shown in Section 5.3.3. This quan-
tity is memorised in the state-space at the current control time step ∆tcontrol and for three pre-
vious control steps, ∆tcontrol,-1 (30 min before), ∆tcontrol,-2 (1h before) and ∆tcontrol,-4 (2h be-
fore). Outdoor Air Temperature was included because it is the exogenous factor with the
most significant influence on the heating energy consumption. Occupants’ Presence status
indicates if, in a certain control time step, the zone is occupied or not (based on the occu-
pancy schedules) through a (0,1) binary variable. The two last features are saved in the
state-space at the current control time step ∆tcontrol and for all control time steps within the
following 12 hours. Observations were scaled within a range of (0, 1) in order to feed the
neural network with min-max normalisation.
"""
import numpy as np
from typing import Any, Dict, Optional, List
from numpy.typing import NDArray
from eprllib.Agents.Filters.BaseFilter import BaseFilter
from eprllib.Utils.agent_utils import get_agent_name
from eprllib.Utils.observation_utils import get_variable_name, get_parameter_prediction_name, get_meter_name, get_parameter_name, get_internal_variable_name, get_user_occupation_forecast_name
from eprllib.Utils.filter_utils import normalization_minmax, to_sin_transformation
from eprllib.Utils.annotations import override
from eprllib import logger

class CoraciObsSpaceFilter(BaseFilter):
    """
    Default filter class for preprocessing observations.

    This class extends the `BaseFilter` class and provides a basic implementation that can be used
    as-is or extended to create custom filters. The `get_filtered_obs` method returns the agent
    states as a numpy array of float32 values.
    """
    def __init__(
        self,
        filter_fn_config: Dict[str, Any]
    ):
        """
        Initializes the DefaultFilter class.

        Args:
            filter_fn_config (Dict[str, Any]): Configuration dictionary for the filter function.
            This configuration can include settings that affect how the observations are filtered.
        """
        super().__init__(filter_fn_config)
        
        self.agent_name: Optional[str] = None
        self.t_low: Optional[float] = None
        self.t_high: Optional[float] = None
        self.zone_temperature_name: Optional[str] = None
        self.dt_list: Optional[List[float]] = None
        
    @override(BaseFilter)
    def get_filtered_obs(
        self,
        env_config: Dict[str, Any],
        agent_states: Dict[str, Any],
    ) -> NDArray[np.float32]:
        """
        Returns the filtered observations for the agent based on the environment configuration
        and agent states. This method processes the raw observations according to the filter
        configuration specified in filter_fn_config.

        Args:
            env_config (Dict[str, Any]): Configuration dictionary for the environment. Can include 
            settings that affect how the observations are filtered.
            
            agent_states (Dict[str, Any], optional): Dictionary containing the states of the agent.

        Returns:
            NDarray: Filtered observations as a numpy array of float32 values.
            
        Raises:
            TypeError: If agent_states is not a dictionary.
            ValueError: If agent_states is empty or contains non-numeric values.
        """
        # Check if the agent_states dictionary is empty
        if not agent_states:
            msg = "agent_states dictionary is empty"
            logger.error(msg)
            raise ValueError(msg)
        
        # Check if all values in the agent_states dictionary are numeric
        if not all(isinstance(value, (int, float)) for value in agent_states.values()):
            msg = "All values in agent_states must be numeric"
            logger.error(msg)
            raise ValueError(msg)
        
        # Generate a copy of the agent_states to avoid conflicts with global variables.
        agent_states_copy = agent_states.copy()
        
        # As we don't know the agent that belong this filter, we auto-dectect his name form the name of the variables names
        # inside the agent_states_copy dictionary. The agent_states dict has keys with the format of "agent_name: ...".
        if self.agent_name is None:
            self.agent_name = get_agent_name(agent_states_copy)
        
        
        # === For Normalization of variables ===
        # ======================================
        t_max = 50 + 273.15
        t_min = -20 + 273.15
        e_max = 0.01 # kWh/m2 in 10 minutes
        
        # 1. Minutes
        # ==============================
        minutes = to_sin_transformation(agent_states_copy[get_parameter_name(
                self.agent_name, 
                "minutes"
            )], 1, 60)
        
        
        # 2. DT Indoor Setpoint - Mean Indoor Air
        # ========================================
        # Get the low and high temperature set points from the env_config dict.
        if self.t_low is None:
            self.t_low = env_config["agents_config"][self.agent_name]["reward"]["reward_fn_config"].get("t_low", 20.0)
        if self.t_high is None:
            self.t_high = env_config["agents_config"][self.agent_name]["reward"]["reward_fn_config"].get("t_high", 22.0)
        
        assert self.t_high is not None, "t_high is None"
        assert self.t_low is not None, "t_low is None"
        
        # Get the interior temeperature variable name index.
        if self.zone_temperature_name is None:
            self.zone_temperature_name = get_variable_name(
                self.agent_name, 
                "Zone Mean Air Temperature", 
                env_config["agents_config"][self.agent_name]["reward"]["reward_fn_config"]['thermal_zone'])
        
        # Get the interior temperature value.
        t_i = agent_states_copy[self.zone_temperature_name]
        
        # Calculate deviation from the center of the comfort band to create a continuous reward signal
        t_center = (self.t_low + self.t_high) / 2.0
        dt = np.clip(t_i - t_center, -10., 10.) / 10.0 # Normalize to [-1, 1]
        
        # Full the emptly list with the deltas of temperature.
        if self.dt_list is None:
            self.dt_list = []
            for _ in range(4):
                self.dt_list.append(dt)
        
        # If the list was already created, remove the oldest element and add the new one.
        else:
            self.dt_list.pop(0) # Remove
            self.dt_list.append(dt) # Add
        
        # 3. Outdoor Air Temperature (and future 12 h predictions)
        # ============================================================
        # Get the outdoor temperature variable name index.
        t_o = normalization_minmax(
            agent_states_copy[get_variable_name(
                self.agent_name, 
                "Site Outdoor Air Drybulb Temperature", 
                "Environment"
            )] + 273.15,
            t_min, t_max)
        
        # Create a emptly list to save the temperature predictions.
        t_o_predictions: List[float|int] = []
        weather_prediction_hours = env_config["agents_config"][self.agent_name]["observation"]["weather_prediction_hours"]
        if weather_prediction_hours != 12:
            print(f"The parameter 'weather_prediction_hours' must be 12 but is {weather_prediction_hours}. Consider to change it. 12 will be used.")
            weather_prediction_hours = 12
        
        # Save the values for the next 12 hours in the list.
        for hour in range(3):
            t_o_predictions.append(normalization_minmax(
                agent_states_copy[get_parameter_prediction_name(
                self.agent_name, 
                "outdoor_dry_bulb",
                hour)],
                t_min, t_max)
            )
        
        # 4. Occupancy status and prediction of 12 hours.
        # ==================================================

        occupancy_status = agent_states_copy[get_variable_name(
            self.agent_name, 
            "Zone People Occupant Count",
            env_config["agents_config"][self.agent_name]["reward"]["reward_fn_config"]['thermal_zone']
        )]
        
        actual_hour: int = int(agent_states_copy[get_parameter_name(
            self.agent_name, 
            "hour"
        )])
        
        occupation_prediction: List[float] = []
        for hour in range(3):
            occupation_prediction.append(float(agent_states_copy[get_user_occupation_forecast_name(
                self.agent_name,
                hour+1)])
                )
            
        
        # 5. Energy
        # ===========
        zone_floor_area = agent_states_copy[get_internal_variable_name(
            self.agent_name,
            "Zone Floor Area",
            env_config["agents_config"][self.agent_name]["reward"]["reward_fn_config"]['thermal_zone']
            )]
        
        cooling = np.clip(agent_states_copy[get_meter_name(
            self.agent_name, 
            "Cooling:DistrictCooling"
        )] / (3.6*10**6) / zone_floor_area / e_max, 0, 1)
                          
                          
        heating = np.clip(agent_states_copy[get_meter_name(
            self.agent_name, 
            "Heating:DistrictHeatingWater"
        )] / (3.6*10**6) / zone_floor_area / e_max, 0, 1)
        
        
        return np.array([to_sin_transformation(actual_hour,0,23),minutes, *self.dt_list, t_o, *t_o_predictions, cooling, heating, occupancy_status, *occupation_prediction], dtype='float32')
    