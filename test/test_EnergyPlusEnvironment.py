from eprllib.Env.MultiAgent.EnergyPlusEnvironment import EnergyPlusEnv_v0
from queue import Queue
from unittest.mock import MagicMock, patch
from unittest.mock import Mock, patch
import pytest

class TestEnergyplusenvironment:

    @pytest.fixture
    def mock_env_config(self):
        return {
            'agents_config': {'agent1': {'thermal_zone': 'Zone1'}},
            'action_fn': Mock(),
            'observation_fn': Mock(),
            'reward_fn': Mock(),
            'episode_fn': Mock(),
            'cut_episode_len': 0,
            'timeout': 10,
            'ep_terminal_output': False
        }

    @patch('eprllib.Env.MultiAgent.EnergyPlusEnvironment.EnergyPlusRunner')
    def test_reset_2(self, mock_runner, mock_env_config):
        """
        Test reset method when truncateds is False and energyplus_runner is None.
        """
        # Arrange
        env = EnergyPlusEnv_v0(mock_env_config)
        env.truncateds = False
        env.energyplus_runner = None
        mock_episode_fn = Mock()
        mock_episode_fn.get_episode_config.return_value = mock_env_config
        env.episode_fn = mock_episode_fn

        mock_runner_instance = Mock()
        mock_runner_instance.obs_event.wait.return_value = None
        mock_runner_instance.infos_event.wait.return_value = None
        mock_runner.return_value = mock_runner_instance

        expected_obs = {'agent1': [1.0, 2.0, 3.0]}
        expected_infos = {'agent1': {'info1': 'value1'}}
        mock_runner_instance.obs_queue.get.return_value = expected_obs
        mock_runner_instance.infos_queue.get.return_value = expected_infos

        # Act
        obs, infos = env.reset()

        # Assert
        assert env.episode == 0
        assert env.timestep == 0
        assert isinstance(env.obs_queue, Queue)
        assert isinstance(env.act_queue, Queue)
        assert isinstance(env.infos_queue, Queue)
        assert env.energyplus_runner is not None
        assert env.energyplus_runner.start.called
        assert obs == expected_obs
        assert infos == expected_infos
        assert not env.terminateds
        assert not env.truncateds

        # Verify that the EnergyPlusRunner was created with the correct arguments
        mock_runner.assert_called_once_with(
            episode=0,
            env_config=mock_env_config,
            obs_queue=env.obs_queue,
            act_queue=env.act_queue,
            infos_queue=env.infos_queue,
            _agent_ids=env._agent_ids,
            _thermal_zone_ids=env._thermal_zone_ids,
            observation_fn=env.observation_fn,
            action_fn=env.action_fn
        )

    def test_reset_3(self):
        """
        Test reset method when self.truncateds is True.
        """
        # Mock environment configuration
        env_config = {
            'agents_config': {'agent1': {}},
            'action_fn': MagicMock(),
            'observation_fn': MagicMock(),
            'reward_fn': MagicMock(),
            'episode_fn': MagicMock(),
        }

        # Create EnergyPlusEnv_v0 instance
        env = EnergyPlusEnv_v0(env_config)

        # Set truncateds to True
        env.truncateds = True

        # Mock last_obs and last_infos
        env.last_obs = {'agent1': [1, 2, 3]}
        env.last_infos = {'agent1': {'info1': 'value1'}}

        # Call reset method
        obs, infos = env.reset()

        # Assertions
        assert env.episode == 0  # Episode should be incremented
        assert env.timestep == 0  # Timestep should be reset
        assert env.terminateds == False
        assert env.truncateds == False
        assert obs == env.last_obs
        assert infos == env.last_infos

        # Verify that EnergyPlusRunner was not initialized
        assert env.energyplus_runner is None
        assert env.obs_queue is None
        assert env.act_queue is None
        assert env.infos_queue is None

        # Verify that episode_fn.get_episode_config was not called
        env.episode_fn.get_episode_config.assert_not_called()

    @patch('eprllib.Env.MultiAgent.EnergyPlusEnvironment.EnergyPlusRunner')
    def test_reset_when_not_truncated_and_runner_exists(self, mock_runner):
        """
        Test reset method when episode is not truncated and EnergyPlus runner exists.
        """
        # Arrange
        env_config = {
            'agents_config': {'agent1': {}, 'agent2': {}},
            'action_fn': MagicMock(),
            'observation_fn': MagicMock(),
            'reward_fn': MagicMock(),
            'episode_fn': MagicMock(),
            'cut_episode_len': 0,
            'timeout': 10
        }
        env = EnergyPlusEnv_v0(env_config)
        env.truncateds = False
        env.energyplus_runner = mock_runner
        env.close = MagicMock()
        
        mock_runner.obs_event = MagicMock()
        mock_runner.infos_event = MagicMock()
        
        expected_obs = {'agent1': [1, 2, 3], 'agent2': [4, 5, 6]}
        expected_infos = {'agent1': {'info1': 'value1'}, 'agent2': {'info2': 'value2'}}
        
        env.obs_queue = MagicMock()
        env.obs_queue.get.return_value = expected_obs
        env.infos_queue = MagicMock()
        env.infos_queue.get.return_value = expected_infos

        # Act
        obs, infos = env.reset()

        # Assert
        assert env.episode == 0
        assert env.timestep == 0
        env.close.assert_called_once()
        assert isinstance(env.obs_queue, Queue)
        assert isinstance(env.act_queue, Queue)
        assert isinstance(env.infos_queue, Queue)
        env.episode_fn.get_episode_config.assert_called_once_with(env.env_config)
        mock_runner.start.assert_called_once()
        mock_runner.obs_event.wait.assert_called_once()
        mock_runner.infos_event.wait.assert_called_once()
        assert obs == expected_obs
        assert infos == expected_infos
        assert env.terminateds == False
        assert env.truncateds == False