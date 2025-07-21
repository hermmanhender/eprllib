"""
Test Environment
===============

This module contains tests for the Environment class.
"""
import os
import pytest
import tempfile
from unittest.mock import MagicMock, patch
import numpy as np
from queue import Queue
from gymnasium import spaces
from eprllib.Environment.Environment import Environment
from eprllib.Environment.EnvironmentConfig import EnvironmentConfig
from eprllib.Episodes.DefaultEpisode import DefaultEpisode
from eprllib.AgentsConnectors.DefaultConnector import DefaultConnector
from eprllib.Agents.Filters.DefaultFilter import DefaultFilter
from eprllib.Agents.Triggers.SetpointTriggers import SetpointTrigger
from eprllib.Agents.Rewards.EnergyRewards import EnergyReward


class TestEnvironment:
    """Test class for Environment."""

    @pytest.fixture
    def mock_env_config(self):
        """Create a mock environment configuration for testing."""
        # Create a basic environment configuration
        config = {
            "epjson_path": "tests/data/1ZoneDataCenterCRAC_wApproachTemp.idf",
            "epw_path": "tests/data/USA_CA_San.Francisco.Intl.AP.724940_TMY3.epw",
            "output_path": tempfile.gettempdir(),
            "ep_terminal_output": False,
            "timeout": 10,
            "evaluation": False,
            "cut_episode_len": 0,
            "history_len": 1,
            "agents_config": {
                "agent1": {
                    "observation": {
                        "variables": None,
                        "internal_variables": None,
                        "meters": None,
                        "simulation_parameters": {
                            "actual_date_time": False,
                            "actual_time": False,
                            "current_time": True,
                            "day_of_month": False,
                            "day_of_week": False,
                            "day_of_year": False,
                            "holiday_index": False,
                            "hour": True,
                            "minutes": False,
                            "month": False,
                            "num_time_steps_in_hour": False,
                            "year": False,
                            "is_raining": False,
                            "sun_is_up": False,
                            "today_weather_albedo_at_time": False,
                            "today_weather_beam_solar_at_time": False,
                            "today_weather_diffuse_solar_at_time": False,
                            "today_weather_horizontal_ir_at_time": False,
                            "today_weather_is_raining_at_time": False,
                            "today_weather_is_snowing_at_time": False,
                            "today_weather_liquid_precipitation_at_time": False,
                            "today_weather_outdoor_barometric_pressure_at_time": False,
                            "today_weather_outdoor_dew_point_at_time": False,
                            "today_weather_outdoor_dry_bulb_at_time": True,
                            "today_weather_outdoor_relative_humidity_at_time": False,
                            "today_weather_sky_temperature_at_time": False,
                            "today_weather_wind_direction_at_time": False,
                            "today_weather_wind_speed_at_time": False,
                            "tomorrow_weather_albedo_at_time": False,
                            "tomorrow_weather_beam_solar_at_time": False,
                            "tomorrow_weather_diffuse_solar_at_time": False,
                            "tomorrow_weather_horizontal_ir_at_time": False,
                            "tomorrow_weather_is_raining_at_time": False,
                            "tomorrow_weather_is_snowing_at_time": False,
                            "tomorrow_weather_liquid_precipitation_at_time": False,
                            "tomorrow_weather_outdoor_barometric_pressure_at_time": False,
                            "tomorrow_weather_outdoor_dew_point_at_time": False,
                            "tomorrow_weather_outdoor_dry_bulb_at_time": False,
                            "tomorrow_weather_outdoor_relative_humidity_at_time": False,
                            "tomorrow_weather_sky_temperature_at_time": False,
                            "tomorrow_weather_wind_direction_at_time": False,
                            "tomorrow_weather_wind_speed_at_time": False
                        },
                        "zone_simulation_parameters": {
                            "system_time_step": False,
                            "zone_time_step": False,
                            "zone_time_step_number": False
                        },
                        "use_one_day_weather_prediction": False,
                        "prediction_hours": 0,
                        "prediction_variables": {},
                        "other_obs": {}
                    },
                    "action": {
                        "actuators": [
                            ["Schedule:Constant", "Schedule Value", "HTGSETP_SCH"]
                        ]
                    },
                    "reward": {
                        "reward_fn": EnergyReward,
                        "reward_fn_config": {}
                    },
                    "filter": {
                        "filter_fn": DefaultFilter,
                        "filter_fn_config": {}
                    },
                    "trigger": {
                        "trigger_fn": SetpointTrigger,
                        "trigger_fn_config": {
                            "min_setpoint": 18.0,
                            "max_setpoint": 24.0,
                            "num_actions": 7
                        }
                    }
                }
            },
            "connector_fn": DefaultConnector,
            "connector_fn_config": {},
            "episode_fn": DefaultEpisode,
            "episode_fn_config": {}
        }
        return config

    @patch('eprllib.Environment.Environment.EnvironmentRunner')
    def test_environment_initialization(self, mock_runner, mock_env_config):
        """Test that the Environment class initializes correctly."""
        # Create an environment instance
        env = Environment(mock_env_config)
        
        # Check that the environment was initialized correctly
        assert env.env_config == mock_env_config
        assert env.episode == -1
        assert env.timestep == 0
        assert env.terminateds is False
        assert env.truncateds is False
        assert env.possible_agents == ["agent1"]
        assert env.agents == ["agent1"]
        assert "agent1" in env.reward_fn
        assert "agent1" in env.trigger_fn
        assert "agent1" in env.filter_fn
        assert "agent1" in env.action_space
        assert isinstance(env.action_space, spaces.Dict)
        assert isinstance(env.observation_space, spaces.Dict)

    @patch('eprllib.Environment.Environment.EnvironmentRunner')
    def test_reset(self, mock_runner_class, mock_env_config):
        """Test the reset method of the Environment class."""
        # Setup mock runner
        mock_runner = MagicMock()
        mock_runner.obs_event = MagicMock()
        mock_runner.infos_event = MagicMock()
        mock_runner.simulation_complete = False
        mock_runner.failed.return_value = False
        mock_runner_class.return_value = mock_runner
        
        # Create an environment instance
        env = Environment(mock_env_config)
        
        # Mock the queue get methods
        env.obs_queue = MagicMock()
        env.obs_queue.get.return_value = {"agent1": [0.0, 0.0]}
        env.infos_queue = MagicMock()
        env.infos_queue.get.return_value = {"agent1": {}}
        
        # Call reset
        obs, infos = env.reset()
        
        # Check that reset incremented the episode counter
        assert env.episode == 0
        assert env.timestep == 0
        assert env.terminateds is False
        assert env.truncateds is False
        
        # Check that the runner was started
        mock_runner.start.assert_called_once()
        
        # Check that observations and infos were retrieved
        assert "agent1" in obs
        assert "agent1" in infos

    @patch('eprllib.Environment.Environment.EnvironmentRunner')
    def test_step(self, mock_runner_class, mock_env_config):
        """Test the step method of the Environment class."""
        # Setup mock runner
        mock_runner = MagicMock()
        mock_runner.obs_event = MagicMock()
        mock_runner.infos_event = MagicMock()
        mock_runner.act_event = MagicMock()
        mock_runner.simulation_complete = False
        mock_runner.failed.return_value = False
        mock_runner_class.return_value = mock_runner
        
        # Create an environment instance
        env = Environment(mock_env_config)
        
        # Setup the environment state as if reset has been called
        env.episode = 0
        env.timestep = 0
        env.terminateds = False
        env.truncateds = False
        env.runner = mock_runner
        
        # Mock the queue methods
        env.obs_queue = MagicMock()
        env.obs_queue.get.return_value = {"agent1": [0.0, 0.0]}
        env.act_queue = MagicMock()
        env.infos_queue = MagicMock()
        env.infos_queue.get.return_value = {"agent1": {}}
        
        # Mock the reward function
        env.reward_fn = {"agent1": MagicMock()}
        env.reward_fn["agent1"].get_reward.return_value = 0.0
        
        # Call step with a mock action
        actions = {"agent1": 3}  # Middle action (assuming 7 actions from 0-6)
        obs, rewards, terminated, truncated, infos = env.step(actions)
        
        # Check that step incremented the timestep
        assert env.timestep == 1
        
        # Check that the action was sent to the runner
        env.act_queue.put.assert_called_once_with(actions)
        mock_runner.act_event.set.assert_called_once()
        
        # Check that observations and infos were retrieved
        assert "agent1" in obs
        assert "agent1" in rewards
        assert "__all__" in terminated
        assert "__all__" in truncated
        assert "agent1" in infos
        
        # Check that the reward function was called
        env.reward_fn["agent1"].get_reward.assert_called_once()

    @patch('eprllib.Environment.Environment.EnvironmentRunner')
    def test_close(self, mock_runner_class, mock_env_config):
        """Test the close method of the Environment class."""
        # Setup mock runner
        mock_runner = MagicMock()
        mock_runner_class.return_value = mock_runner
        
        # Create an environment instance
        env = Environment(mock_env_config)
        env.runner = mock_runner
        
        # Call close
        env.close()
        
        # Check that the runner was stopped
        mock_runner.stop.assert_called_once()

    @patch('eprllib.Environment.Environment.EnvironmentRunner')
    def test_simulation_complete(self, mock_runner_class, mock_env_config):
        """Test behavior when simulation is complete."""
        # Setup mock runner
        mock_runner = MagicMock()
        mock_runner.simulation_complete = True
        mock_runner.failed.return_value = False
        mock_runner_class.return_value = mock_runner
        
        # Create an environment instance
        env = Environment(mock_env_config)
        
        # Setup the environment state as if reset has been called
        env.episode = 0
        env.timestep = 0
        env.terminateds = False
        env.truncateds = False
        env.runner = mock_runner
        env.last_obs = {"agent1": [0.0, 0.0]}
        env.last_infos = {"agent1": {}}
        
        # Mock the reward function
        env.reward_fn = {"agent1": MagicMock()}
        env.reward_fn["agent1"].get_reward.return_value = 0.0
        
        # Call step with a mock action
        actions = {"agent1": 3}
        obs, rewards, terminated, truncated, infos = env.step(actions)
        
        # Check that the environment recognized the simulation is complete
        assert terminated["__all__"] is True
        assert "agent1" in obs
        assert "agent1" in rewards
        assert "agent1" in infos

    @patch('eprllib.Environment.Environment.EnvironmentRunner')
    def test_simulation_failed(self, mock_runner_class, mock_env_config):
        """Test behavior when simulation fails."""
        # Setup mock runner
        mock_runner = MagicMock()
        mock_runner.simulation_complete = False
        mock_runner.failed.return_value = True
        mock_runner_class.return_value = mock_runner
        
        # Create an environment instance
        env = Environment(mock_env_config)
        
        # Setup the environment state as if reset has been called
        env.episode = 0
        env.timestep = 0
        env.terminateds = False
        env.truncateds = False
        env.runner = mock_runner
        env.last_obs = {"agent1": [0.0, 0.0]}
        env.last_infos = {"agent1": {}}
        
        # Mock the reward function
        env.reward_fn = {"agent1": MagicMock()}
        env.reward_fn["agent1"].get_reward.return_value = 0.0
        
        # Call step with a mock action
        actions = {"agent1": 3}
        obs, rewards, terminated, truncated, infos = env.step(actions)
        
        # Check that the environment recognized the simulation failed
        assert terminated["__all__"] is True
        assert "agent1" in obs
        assert "agent1" in rewards
        assert "agent1" in infos

    @patch('eprllib.Environment.Environment.EnvironmentRunner')
    def test_episode_truncation(self, mock_runner_class, mock_env_config):
        """Test episode truncation based on cut_episode_len."""
        # Modify config to enable episode truncation
        config = mock_env_config.copy()
        config["cut_episode_len"] = 1  # Truncate after 1 day
        config["num_time_steps_in_hour"] = 4  # 4 timesteps per hour
        
        # Setup mock runner
        mock_runner = MagicMock()
        mock_runner.simulation_complete = False
        mock_runner.failed.return_value = False
        mock_runner_class.return_value = mock_runner
        
        # Create an environment instance
        env = Environment(config)
        
        # Setup the environment state as if reset has been called
        env.episode = 0
        env.timestep = 0
        env.terminateds = False
        env.truncateds = False
        env.runner = mock_runner
        env.last_obs = {"agent1": [0.0, 0.0]}
        env.last_infos = {"agent1": {}}
        
        # Mock the queue methods
        env.obs_queue = MagicMock()
        env.obs_queue.get.return_value = {"agent1": [0.0, 0.0]}
        env.act_queue = MagicMock()
        env.infos_queue = MagicMock()
        env.infos_queue.get.return_value = {"agent1": {}}
        
        # Mock the reward function
        env.reward_fn = {"agent1": MagicMock()}
        env.reward_fn["agent1"].get_reward.return_value = 0.0
        
        # Set timestep to just before truncation point (24 hours * 4 timesteps/hour = 96 timesteps)
        env.timestep = 95
        
        # Call step with a mock action
        actions = {"agent1": 3}
        _, _, _, truncated, _ = env.step(actions)
        
        # Check that the episode is not truncated yet
        assert truncated["__all__"] is False
        
        # Call step again to reach truncation point
        _, _, _, truncated, _ = env.step(actions)
        
        # Check that the episode is now truncated
        assert truncated["__all__"] is True

    @patch('eprllib.Environment.Environment.EnvironmentRunner')
    def test_history_len_greater_than_one(self, mock_runner_class, mock_env_config):
        """Test observation history functionality when history_len > 1."""
        # Modify config to use history
        config = mock_env_config.copy()
        config["history_len"] = 3
        
        # Setup mock runner
        mock_runner = MagicMock()
        mock_runner.obs_event = MagicMock()
        mock_runner.infos_event = MagicMock()
        mock_runner.simulation_complete = False
        mock_runner.failed.return_value = False
        mock_runner_class.return_value = mock_runner
        
        # Create an environment instance with patched observation_buffers
        with patch('eprllib.Environment.Environment.collections.deque') as mock_deque:
            mock_deque_instance = MagicMock()
            mock_deque.return_value = mock_deque_instance
            mock_deque_instance.__iter__.return_value = [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
            
            env = Environment(config)
            
            # Mock the queue get methods
            env.obs_queue = MagicMock()
            env.obs_queue.get.return_value = {"agent1": [0.0, 0.0]}
            env.infos_queue = MagicMock()
            env.infos_queue.get.return_value = {"agent1": {}}
            
            # Setup observation buffers
            env.observation_buffers = {"agent1": mock_deque_instance}
            
            # Call reset
            obs, _ = env.reset()
            
            # Check that the observation has the correct shape
            assert "agent1" in obs
            # The observation should be a numpy array with shape (history_len, feature_dim)
            assert isinstance(obs["agent1"], np.ndarray)
            
            # Mock the step method to test observation history during step
            env.runner = mock_runner
            env.reward_fn = {"agent1": MagicMock()}
            env.reward_fn["agent1"].get_reward.return_value = 0.0
            
            # Call step
            actions = {"agent1": 3}
            obs, _, _, _, _ = env.step(actions)
            
            # Check that the observation has the correct shape
            assert "agent1" in obs
            assert isinstance(obs["agent1"], np.ndarray)