"""
eprllib
========

``eprllib`` was born out of the need to bridge the gap between building modeling with
**EnergyPlus** and Deep Reinforcement Learning (**DRL**). Traditionally, integrating these two
disciplines has been complex and laborious. ``eprllib`` aims to simplify this process,
offering an intuitive and flexible interface for developing intelligent agents that
interact with building simulations.
"""
# Version management.
from .version import __version__, EP_VERSION, ep_version_list
from .Agents.ActionSpec import ActionSpec
from .Agents.AgentSpec import AgentSpec
from .Agents.ObservationSpec import ObservationSpec

from .Connectors.BaseConnector import BaseConnector
from .Connectors.DefaultConnector import DefaultConnector

from .Environment.EnvironmentConfig import EnvironmentConfig
from .Environment.EnvironmentRunner import EnvironmentRunner
from .Environment.MultiAgentEnvironment import MultiAgentEnvironment
from .Environment.SingleAgentEnvironment import SingleAgentEnvironment

from .Episodes.BaseEpisode import BaseEpisode
from .Episodes.DefaultEpisode import DefaultEpisode


from .Utils.add_ep_to_path import EP_API_add_path
from .Utils.agent_utils import get_agent_name, config_validation
from .Utils.annotations import override, OverrideToImplementCustomLogic, trial_str_creator_for_tune
from .Utils.connector_utils import (
    set_variables_in_obs,
    set_internal_variables_in_obs,
    set_meters_in_obs,
    set_simulation_parameters_in_obs,
    set_zone_simulation_parameters_in_obs,
    set_prediction_variables_in_obs,
    set_user_occupation_forecast_in_obs,
    set_other_obs_in_obs,
    set_actuators_in_obs,
)
from .Utils.constants import (
    SIMULATION_PARAMETERS,
    ZONE_SIMULATION_PARAMETERS,
    PREDICTION_VARIABLES,
    PREDICTION_HOURS,
    OCCUPATION_PROFILES,
    VALID_USER_TYPES,
    VALID_ZONE_TYPES
)
from .Utils.env_config_utils import (
    env_config_validation,
    to_json,
    from_json,
    continuous_action_space,
    discrete_action_space,
    variable_checking,
    validate_properties,
)
from .Utils.env_utils import calculate_occupancy, calculate_occupancy_forecast
from .Utils.episode_fn_utils import (
    load_ep_model,
    save_ep_model,
    get_random_weather,
    run_period,
    max_day_in_month,
    building_dimension,
    window_size_epJSON,
    calcular_centro,
    generate_variated_schedule,
    generate_variated_schedule_with_shift,
    generate_occupancy_schedule,
    inertial_mass_calculation,
    effective_thermal_capacity,
    u_factor_calculation,
    u_factor,
    material_area,
    fenestration_area,
    find_dict_key_by_nested_key,
    run_period_change,
    from_julian_day,
    extract_epw_location_data,
    select_epjson_model,
    get_random_parameter,
)
from .Utils.filter_utils import (
    to_sin_transformation,
    to_cos_transformation,
    from_sin_cos_normalization,
    normalization_minmax,
    desnormalization_minmax,
)
from .Utils.observation_utils import (
    get_variable_name,
    get_internal_variable_name,
    get_meter_name,
    get_actuator_name,
    get_parameter_name,
    get_parameter_prediction_name,
    get_other_obs_name,
    get_user_occupation_forecast_name,
    )
from .Utils.parallel_setup import parallel_energyplus_setup
from .Utils.trial_str_creator import trial_str_creator


__all__ = [
    "__version__",
    "ep_version_list",
    "EP_VERSION",
    "ActionSpec", "AgentSpec", "ObservationSpec",
    "BaseConnector", "DefaultConnector",
    "EnvironmentConfig", "EnvironmentRunner", "MultiAgentEnvironment", "SingleAgentEnvironment",
    "BaseEpisode", "DefaultEpisode",
    "EP_API_add_path",
    "get_agent_name", "config_validation",
    "override", "OverrideToImplementCustomLogic", "trial_str_creator_for_tune",
    "set_variables_in_obs", "set_internal_variables_in_obs", "set_meters_in_obs", "set_simulation_parameters_in_obs", "set_zone_simulation_parameters_in_obs",
    "set_prediction_variables_in_obs", "set_user_occupation_forecast_in_obs", "set_other_obs_in_obs", "set_actuators_in_obs",
    "env_config_validation", "to_json", "from_json", "continuous_action_space", "discrete_action_space", "variable_checking", "validate_properties",
    "calculate_occupancy", "calculate_occupancy_forecast",
    "load_ep_model", "save_ep_model", "get_random_weather", "run_period", "max_day_in_month", "building_dimension",
    "window_size_epJSON", "calcular_centro", "generate_variated_schedule", "generate_variated_schedule_with_shift", "generate_occupancy_schedule",
    "inertial_mass_calculation", "effective_thermal_capacity", "u_factor_calculation", "u_factor", "material_area",
    "fenestration_area", "find_dict_key_by_nested_key", "run_period_change", "from_julian_day", "extract_epw_location_data",
    "select_epjson_model", "get_random_parameter",
    "to_sin_transformation", "to_cos_transformation", "from_sin_cos_normalization", "normalization_minmax", "desnormalization_minmax",
    "get_variable_name", "get_internal_variable_name", "get_meter_name", "get_actuator_name", "get_parameter_name", "get_parameter_prediction_name", "get_other_obs_name", "get_user_occupation_forecast_name",
    "parallel_energyplus_setup",
    "trial_str_creator",
    "SIMULATION_PARAMETERS",
    "ZONE_SIMULATION_PARAMETERS",
    "PREDICTION_VARIABLES",
    "PREDICTION_HOURS",
    "OCCUPATION_PROFILES",
    "VALID_USER_TYPES",
    "VALID_ZONE_TYPES"
    ]