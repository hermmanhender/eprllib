# """
# Test EnvironmentRunner
# =====================

# This module contains tests for the EnvironmentRunner class.
# """
# import os
# import pytest
# import tempfile
# from unittest.mock import MagicMock, patch, call
# from queue import Queue
# from eprllib.Environment.EnvironmentRunner import EnvironmentRunner


# class TestEnvironmentRunner:
#     """Test class for EnvironmentRunner."""

#     @pytest.fixture
#     def mock_env_config(self):
#         """Create a mock environment configuration for testing."""
#         return {
#             "epjson_path": "tests/data/1ZoneDataCenterCRAC_wApproachTemp.idf",
#             "epw_path": "tests/data/USA_CA_San.Francisco.Intl.AP.724940_TMY3.epw",
#             "output_path": tempfile.gettempdir(),
#             "ep_terminal_output": False,
#             "timeout": 10,
#             "evaluation": False,
#             "agents_config": {
#                 "agent1": {
#                     "observation": {
#                         "variables": None,
#                         "internal_variables": None,
#                         "meters": None,
#                         "simulation_parameters": {
#                             "hour": True,
#                             "current_time": True,
#                         },
#                         "zone_simulation_parameters": {
#                             "system_time_step": False,
#                         },
#                         "use_one_day_weather_prediction": False,
#                         "prediction_hours": 0,
#                         "prediction_variables": {},
#                         "other_obs": {}
#                     },
#                     "action": {
#                         "actuators": [
#                             ["Schedule:Constant", "Schedule Value", "HTGSETP_SCH"]
#                         ]
#                     }
#                 }
#             }
#         }

#     @pytest.fixture
#     def mock_queues(self):
#         """Create mock queues for testing."""
#         return {
#             "obs_queue": MagicMock(spec=Queue),
#             "act_queue": MagicMock(spec=Queue),
#             "infos_queue": MagicMock(spec=Queue)
#         }

#     @pytest.fixture
#     def mock_functions(self):
#         """Create mock functions for testing."""
#         return {
#             "filter_fn": {"agent1": MagicMock()},
#             "trigger_fn": {"agent1": MagicMock()},
#             "connector_fn": MagicMock()
#         }

#     @patch('eprllib.Environment.EnvironmentRunner.api')
#     def test_initialization(self, mock_api, mock_env_config, mock_queues, mock_functions):
#         """Test initialization of EnvironmentRunner."""
#         # Create an instance of EnvironmentRunner
#         runner = EnvironmentRunner(
#             env_config=mock_env_config,
#             episode=0,
#             obs_queue=mock_queues["obs_queue"],
#             act_queue=mock_queues["act_queue"],
#             infos_queue=mock_queues["infos_queue"],
#             agents=["agent1"],
#             filter_fn=mock_functions["filter_fn"],
#             trigger_fn=mock_functions["trigger_fn"],
#             connector_fn=mock_functions["connector_fn"]
#         )
        
#         # Check that the runner was initialized correctly
#         assert runner.env_config == mock_env_config
#         assert runner.episode == 0
#         assert runner.obs_queue == mock_queues["obs_queue"]
#         assert runner.act_queue == mock_queues["act_queue"]
#         assert runner.infos_queue == mock_queues["infos_queue"]
#         assert runner.agents == ["agent1"]
#         assert runner.filter_fn == mock_functions["filter_fn"]
#         assert runner.trigger_fn == mock_functions["trigger_fn"]
#         assert runner.connector_fn == mock_functions["connector_fn"]
#         assert runner.energyplus_exec_thread is None
#         assert runner.energyplus_state is None
#         assert runner.sim_results == 0
#         assert runner.initialized is False
#         assert runner.init_handles is False
#         assert runner.simulation_complete is False
#         assert runner.first_observation is True
#         assert isinstance(runner.obs, dict)
#         assert isinstance(runner.infos, dict)
#         assert "agent1" in runner.infos
#         assert runner.is_last_timestep is False

#     @patch('eprllib.Environment.EnvironmentRunner.api')
#     @patch('eprllib.Environment.EnvironmentRunner.threading')
#     def test_start(self, mock_threading, mock_api, mock_env_config, mock_queues, mock_functions):
#         """Test the start method."""
#         # Create an instance of EnvironmentRunner
#         runner = EnvironmentRunner(
#             env_config=mock_env_config,
#             episode=0,
#             obs_queue=mock_queues["obs_queue"],
#             act_queue=mock_queues["act_queue"],
#             infos_queue=mock_queues["infos_queue"],
#             agents=["agent1"],
#             filter_fn=mock_functions["filter_fn"],
#             trigger_fn=mock_functions["trigger_fn"],
#             connector_fn=mock_functions["connector_fn"]
#         )
        
#         # Mock the thread
#         mock_thread = MagicMock()
#         mock_threading.Thread.return_value = mock_thread
        
#         # Call start
#         runner.start()
        
#         # Check that the EnergyPlus state was created
#         mock_api.state_manager.new_state.assert_called_once()
        
#         # Check that callbacks were registered
#         assert mock_api.runtime.callback_begin_zone_timestep_after_init_heat_balance.called
#         assert mock_api.runtime.callback_end_zone_timestep_after_zone_reporting.called
#         assert mock_api.runtime.callback_progress.called
        
#         # Check that the thread was started
#         mock_thread.start.assert_called_once()
#         assert runner.energyplus_exec_thread == mock_thread

#     @patch('eprllib.Environment.EnvironmentRunner.api')
#     def test_stop(self, mock_api, mock_env_config, mock_queues, mock_functions):
#         """Test the stop method."""
#         # Create an instance of EnvironmentRunner
#         runner = EnvironmentRunner(
#             env_config=mock_env_config,
#             episode=0,
#             obs_queue=mock_queues["obs_queue"],
#             act_queue=mock_queues["act_queue"],
#             infos_queue=mock_queues["infos_queue"],
#             agents=["agent1"],
#             filter_fn=mock_functions["filter_fn"],
#             trigger_fn=mock_functions["trigger_fn"],
#             connector_fn=mock_functions["connector_fn"]
#         )
        
#         # Mock the thread
#         runner.energyplus_exec_thread = MagicMock()
#         runner.energyplus_state = MagicMock()
        
#         # Call stop
#         runner.stop()
        
#         # Check that the simulation was stopped
#         mock_api.runtime.stop_simulation.assert_called_once_with(runner.energyplus_state)
        
#         # Check that the state was deleted
#         mock_api.state_manager.delete_state.assert_called_once_with(runner.energyplus_state)
        
#         # Check that the thread was reset
#         assert runner.energyplus_exec_thread is None
#         assert runner.first_observation is True

#     @patch('eprllib.Environment.EnvironmentRunner.api')
#     def test_failed(self, mock_api, mock_env_config, mock_queues, mock_functions):
#         """Test the failed method."""
#         # Create an instance of EnvironmentRunner
#         runner = EnvironmentRunner(
#             env_config=mock_env_config,
#             episode=0,
#             obs_queue=mock_queues["obs_queue"],
#             act_queue=mock_queues["act_queue"],
#             infos_queue=mock_queues["infos_queue"],
#             agents=["agent1"],
#             filter_fn=mock_functions["filter_fn"],
#             trigger_fn=mock_functions["trigger_fn"],
#             connector_fn=mock_functions["connector_fn"]
#         )
        
#         # Test when sim_results is 0 (success)
#         runner.sim_results = 0
#         assert runner.failed() is False
        
#         # Test when sim_results is not 0 (failure)
#         runner.sim_results = 1
#         assert runner.failed() is True

#     @patch('eprllib.Environment.EnvironmentRunner.api')
#     def test_make_eplus_args(self, mock_api, mock_env_config, mock_queues, mock_functions):
#         """Test the make_eplus_args method."""
#         # Create an instance of EnvironmentRunner
#         runner = EnvironmentRunner(
#             env_config=mock_env_config,
#             episode=0,
#             obs_queue=mock_queues["obs_queue"],
#             act_queue=mock_queues["act_queue"],
#             infos_queue=mock_queues["infos_queue"],
#             agents=["agent1"],
#             filter_fn=mock_functions["filter_fn"],
#             trigger_fn=mock_functions["trigger_fn"],
#             connector_fn=mock_functions["connector_fn"]
#         )
        
#         # Call make_eplus_args
#         args = runner.make_eplus_args()
        
#         # Check that the arguments are correct
#         assert "-w" in args
#         assert mock_env_config["epw_path"] in args
#         assert "-d" in args
#         assert mock_env_config["epjson_path"] in args
        
#         # Test with csv=True
#         runner.env_config["csv"] = True
#         args = runner.make_eplus_args()
#         assert "-r" in args

#     @patch('eprllib.Environment.EnvironmentRunner.api')
#     def test_progress_handler(self, mock_api, mock_env_config, mock_queues, mock_functions):
#         """Test the progress_handler method."""
#         # Create an instance of EnvironmentRunner
#         runner = EnvironmentRunner(
#             env_config=mock_env_config,
#             episode=0,
#             obs_queue=mock_queues["obs_queue"],
#             act_queue=mock_queues["act_queue"],
#             infos_queue=mock_queues["infos_queue"],
#             agents=["agent1"],
#             filter_fn=mock_functions["filter_fn"],
#             trigger_fn=mock_functions["trigger_fn"],
#             connector_fn=mock_functions["connector_fn"]
#         )
        
#         # Test with progress < 99
#         runner.progress_handler(50)
#         assert runner.is_last_timestep is False
        
#         # Test with progress >= 99
#         runner.progress_handler(99)
#         assert runner.is_last_timestep is True

#     @patch('eprllib.Environment.EnvironmentRunner.api')
#     def test_flush_queues(self, mock_api, mock_env_config, mock_queues, mock_functions):
#         """Test the _flush_queues method."""
#         # Create real queues for this test
#         obs_queue = Queue()
#         act_queue = Queue()
#         infos_queue = Queue()
        
#         # Add some items to the queues
#         obs_queue.put("obs")
#         act_queue.put("act")
#         infos_queue.put("info")
        
#         # Create an instance of EnvironmentRunner
#         runner = EnvironmentRunner(
#             env_config=mock_env_config,
#             episode=0,
#             obs_queue=obs_queue,
#             act_queue=act_queue,
#             infos_queue=infos_queue,
#             agents=["agent1"],
#             filter_fn=mock_functions["filter_fn"],
#             trigger_fn=mock_functions["trigger_fn"],
#             connector_fn=mock_functions["connector_fn"]
#         )
        
#         # Call _flush_queues
#         runner._flush_queues()
        
#         # Check that the queues are empty
#         assert obs_queue.empty()
#         assert act_queue.empty()
#         assert infos_queue.empty()