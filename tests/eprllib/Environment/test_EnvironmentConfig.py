# """
# Test EnvironmentConfig
# =====================

# This module contains tests for the EnvironmentConfig class.
# """
# import os
# import pytest
# import tempfile
# from unittest.mock import MagicMock, patch
# from eprllib.Environment.EnvironmentConfig import EnvironmentConfig
# from eprllib.Episodes.DefaultEpisode import DefaultEpisode
# from eprllib.AgentsConnectors.DefaultConnector import DefaultConnector
# from eprllib.Agents.AgentSpec import AgentSpec
# from eprllib.Agents.Filters.DefaultFilter import DefaultFilter
# from eprllib.Agents.ActionMappers.SetpointActionMappers import AvailabilityActionMapper
# from eprllib.Agents.Rewards.EnergyRewards import EnergyWithMeters
# from eprllib.Agents.ObservationSpec import ObservationSpec
# from eprllib.Agents.ActionSpec import ActionSpec
# from eprllib.Agents.Rewards.RewardSpec import RewardSpec
# from eprllib.Agents.AgentSpec import AgentSpec
# from eprllib.Agents.Filters.FilterSpec import FilterSpec
# from eprllib.Agents.ActionMappers.ActionMapperSpec import ActionMapperSpec



# class TestEnvironmentConfig:
#     """Test class for EnvironmentConfig."""

#     def test_init(self):
#         """Test initialization of EnvironmentConfig."""
#         config = EnvironmentConfig()
        
#         # Check default values
#         assert config.epjson_path is None
#         assert config.epw_path is None
#         assert config.output_path is None
#         assert config.ep_terminal_output is True
#         assert config.evaluation is False
#         assert config.agents_config is None
#         assert config.connector_fn is DefaultConnector
#         assert config.episode_fn is DefaultEpisode
#         assert isinstance(config.connector_fn_config, dict)
#         assert isinstance(config.episode_fn_config, dict)

#     def test_generals(self):
#         """Test the generals method."""
#         config = EnvironmentConfig()
        
#         # Set general configuration
#         config.generals(
#             epjson_path="path/to/model.epJSON",
#             epw_path="path/to/weather.epw",
#             output_path="path/to/output",
#             ep_terminal_output=False,
#             timeout=20,
#             evaluation=True
#         )
        
#         # Check that values were set correctly
#         assert config.epjson_path == "path/to/model.epJSON"
#         assert config.epw_path == "path/to/weather.epw"
#         assert config.output_path == "path/to/output"
#         assert config.ep_terminal_output is False
#         assert config.timeout == 20
#         assert config.evaluation is True

#     def test_agents(self):
#         """Test the agents method."""
#         config = EnvironmentConfig()
        
#         # Create a mock agent configuration
#         agent_config = {
#             "agent1": {
#                 "observation": {
#                     "variables": None,
#                     "internal_variables": None,
#                     "meters": ["Cooling:DistrictCooling", "Heating:DistrictHeatingWater"],
#                     "simulation_parameters": {
#                         "hour": True,
#                         "current_time": True,
#                     },
#                     "zone_simulation_parameters": {
#                         "system_time_step": False,
#                     },
#                     "use_one_day_weather_prediction": False,
#                     "prediction_hours": 0,
#                     "prediction_variables": {},
#                     "other_obs": {}
#                 },
#                 "action": {
#                     "actuators": [
#                         ["Schedule:Constant", "Schedule Value", "HTGSETP_SCH"]
#                     ]
#                 },
#                 "reward": {
#                     "reward_fn": EnergyWithMeters,
#                     "reward_fn_config": {
#                         "cooling_name": "Cooling:DistrictCooling",
#                         "heating_name": "Heating:DistrictHeatingWater",
#                         "cooling_energy_ref": 100,
#                         "heating_energy_ref": 100
#                     }
#                 },
#                 "filter": {
#                     "filter_fn": DefaultFilter,
#                     "filter_fn_config": {}
#                 },
#                 "action_mapper": {
#                     "action_mapper": AvailabilityActionMapper,
#                     "action_mapper_config": {
#                         "availability_actuator": ["Schedule:Constant", "Schedule Value", "HTGSETP_SCH"]
#                     }
#                 }
#             }
#         }
        
#         # Set agent configuration
#         config.agents(agents_config=agent_config)
        
#         # Check that values were set correctly
#         assert config.agents_config == agent_config
#         assert "agent1" in config.agents_config

#     def test_connector(self):
#         """Test the connector method."""
#         config = EnvironmentConfig()
        
#         # Set connector configuration
#         mock_connector = MagicMock()
#         connector_config = {"param1": "value1"}
#         config.connector(connector_fn=mock_connector, connector_fn_config=connector_config)
        
#         # Check that values were set correctly
#         assert config.connector_fn == mock_connector
#         assert config.connector_fn_config == connector_config

#     def test_episodes(self):
#         """Test the episodes method."""
#         config = EnvironmentConfig()
        
#         # Set episode configuration
#         mock_episode = MagicMock()
#         episode_config = {"param1": "value1"}
#         config.episodes(episode_fn=mock_episode, episode_fn_config=episode_config, cut_episode_len=2)
        
#         # Check that values were set correctly
#         assert config.episode_fn == mock_episode
#         assert config.episode_fn_config == episode_config
#         assert config.cut_episode_len == 2

#     @patch('eprllib.Environment.EnvironmentConfig.TemporaryDirectory')
#     def test_build_valid_config(self, mock_temp_dir):
#         """Test the _build method with a valid configuration."""
#         mock_temp_dir.return_value.name = "/tmp/eprllib_output"
        
#         config = EnvironmentConfig()
        
#         # Set a valid configuration
#         config.generals(
#             epjson_path="path/to/model.epJSON",
#             epw_path="path/to/weather.epw",
#             output_path="path/to/output"
#         )
        
#         config.agents(
#             agents_config={
#                 "agent1": AgentSpec(
#                     observation = ObservationSpec(
#                         variables=None,
#                         internal_variables=None,
#                         meters=["Cooling:DistrictCooling", "Heating:DistrictHeatingWater"],
#                         simulation_parameters={"hour": True},
#                         zone_simulation_parameters={"system_time_step": False},
#                         use_one_day_weather_prediction=False
#                     ),
#                     action = ActionSpec(
#                         actuators=[("Schedule:Constant", "Schedule Value", "HTGSETP_SCH")]
#                     ),
#                     reward = RewardSpec(
#                         reward_fn=EnergyWithMeters,
#                         reward_fn_config={
#                                         "cooling_name": "Cooling:DistrictCooling",
#                                         "heating_name": "Heating:DistrictHeatingWater",
#                                         "cooling_energy_ref": 100,
#                                         "heating_energy_ref": 100
#                                     }
#                     ),
#                     filter = FilterSpec(
#                         filter_fn=DefaultFilter,
#                         filter_fn_config={}
#                     ),
#                     action_mapper = ActionMapperSpec(
#                         action_mapper=AvailabilityActionMapper,
#                         action_mapper_config={
#                             "availability_actuator": ["Schedule:Constant", "Schedule Value", "HTGSETP_SCH"]
#                         }
#                     )
#                 )
#             }
#         )
        
#         # Build the configuration
#         result = config._build()
        
#         # Check that the result is a dictionary with the expected keys
#         assert isinstance(result, dict)
#         assert "epjson_path" in result
#         assert "epw_path" in result
#         assert "output_path" in result
#         assert "agents_config" in result
#         assert "agent1" in result["agents_config"]
#         assert "agent_id" in result["agents_config"]["agent1"]
#         assert result["agents_config"]["agent1"]["agent_id"] == 0

#     def test_build_invalid_epjson_path(self):
#         """Test the _build method with an invalid epjson_path."""
#         config = EnvironmentConfig()
        
#         # Set an invalid epjson_path
#         config.epjson_path = 123  # Not a string
        
#         # Build should raise a ValueError
#         with pytest.raises(ValueError):
#             config._build()

#     def test_build_invalid_epw_path(self):
#         """Test the _build method with an invalid epw_path."""
#         config = EnvironmentConfig()
        
#         # Set a valid epjson_path but invalid epw_path
#         config.epjson_path = "path/to/model.epJSON"
#         config.epw_path = 123  # Not a string
        
#         # Build should raise a ValueError
#         with pytest.raises(ValueError):
#             config._build()

#     def test_build_no_agents(self):
#         """Test the _build method with no agents configured."""
#         config = EnvironmentConfig()
        
#         # Set valid paths but no agents
#         config.epjson_path = "path/to/model.epJSON"
#         config.epw_path = "path/to/weather.epw"
#         config.output_path = "path/to/output"
        
#         # Build should raise a ValueError
#         with pytest.raises(ValueError):
#             config._build()

#     def test_getitem_setitem(self):
#         """Test the __getitem__ and __setitem__ methods."""
#         config = EnvironmentConfig()
        
#         # Set a value using __setitem__
#         config["epjson_path"] = "path/to/model.epJSON"
        
#         # Get the value using __getitem__
#         assert config["epjson_path"] == "path/to/model.epJSON"
        
#         # Check that the attribute was actually set
#         assert config.epjson_path == "path/to/model.epJSON"