from eprllib.EpisodeFunctions.RandomWeather import RandomWeather
from unittest.mock import patch
import pytest

class TestRandomweather:

    def test___init___empty_input(self):
        """
        Test that __init__ raises a ValueError when given an empty dictionary.
        """
        with pytest.raises(ValueError, match="The 'epw_files_folder_path' must be defined in the episode_fn_config."):
            RandomWeather()

    def test___init___invalid_input_type(self):
        """
        Test that __init__ raises a TypeError when given an invalid input type.
        """
        with pytest.raises(TypeError):
            RandomWeather(episode_fn_config="not_a_dictionary")

    def test___init___missing_epw_files_folder_path(self):
        """
        Test that __init__ raises a ValueError when 'epw_files_folder_path' is not in the input dictionary.
        """
        with pytest.raises(ValueError, match="The 'epw_files_folder_path' must be defined in the episode_fn_config."):
            RandomWeather(episode_fn_config={'some_other_key': 'value'})

    def test___init___valid_input(self):
        """
        Test that __init__ correctly initializes the object with valid input.
        """
        config = {'epw_files_folder_path': '/path/to/epw/files'}
        random_weather = RandomWeather(episode_fn_config=config)
        assert random_weather.episode_fn_config == config

    def test_get_episode_config_preserves_existing_config(self):
        """
        Test that get_episode_config preserves existing configuration while adding epw_path.
        """
        episode_fn_config = {'epw_files_folder_path': '/path/to/epw/files'}
        random_weather = RandomWeather(episode_fn_config)
        env_config = {'existing_key': 'existing_value'}
        result = random_weather.get_episode_config(env_config)
        assert 'existing_key' in result
        assert result['existing_key'] == 'existing_value'
        assert 'epw_path' in result

    def test_get_episode_config_updates_epw_path(self):
        """
        Test that get_episode_config updates the 'epw_path' in env_config with a random weather file path.
        """
        # Arrange
        episode_fn_config = {'epw_files_folder_path': '/path/to/epw/files'}
        env_config = {'some_key': 'some_value'}
        random_weather = RandomWeather(episode_fn_config)
        
        # Act
        with patch('eprllib.EpisodeFunctions.RandomWeather.get_random_weather') as mock_get_random_weather:
            mock_get_random_weather.return_value = '/path/to/epw/files/random_weather.epw'
            result = random_weather.get_episode_config(env_config)

        # Assert
        assert 'epw_path' in result
        assert result['epw_path'] == '/path/to/epw/files/random_weather.epw'
        assert result['some_key'] == 'some_value'
        mock_get_random_weather.assert_called_once_with('/path/to/epw/files')

    def test_get_episode_config_with_empty_env_config(self):
        """
        Test get_episode_config with an empty environment configuration.
        """
        episode_fn_config = {'epw_files_folder_path': '/path/to/epw/files'}
        random_weather = RandomWeather(episode_fn_config)
        env_config = {}
        result = random_weather.get_episode_config(env_config)
        assert 'epw_path' in result
        assert isinstance(result['epw_path'], str)

    def test_get_episode_config_with_incorrect_type_env_config(self):
        """
        Test get_episode_config with an incorrect type for environment configuration.
        """
        episode_fn_config = {'epw_files_folder_path': '/path/to/epw/files'}
        random_weather = RandomWeather(episode_fn_config)
        env_config = "not a dictionary"
        with pytest.raises(AttributeError):
            random_weather.get_episode_config(env_config)

    def test_get_episode_config_with_invalid_env_config(self):
        """
        Test get_episode_config with an invalid environment configuration.
        """
        episode_fn_config = {'epw_files_folder_path': '/path/to/epw/files'}
        random_weather = RandomWeather(episode_fn_config)
        env_config = None
        with pytest.raises(AttributeError):
            random_weather.get_episode_config(env_config)

    def test_get_episode_config_with_nonexistent_epw_folder(self, monkeypatch):
        """
        Test get_episode_config with a non-existent EPW folder path.
        """
        def mock_get_random_weather(path):
            raise ValueError("The folder does not contain any .epw files.")

        monkeypatch.setattr("eprllib.EpisodeFunctions.RandomWeather.get_random_weather", mock_get_random_weather)

        episode_fn_config = {'epw_files_folder_path': '/nonexistent/path'}
        random_weather = RandomWeather(episode_fn_config)
        env_config = {}
        
        with pytest.raises(ValueError, match="The folder does not contain any .epw files."):
            random_weather.get_episode_config(env_config)

    def test_init_raises_value_error_when_epw_files_folder_path_missing(self):
        """
        Test that __init__ raises a ValueError when 'epw_files_folder_path' is not in episode_fn_config
        """
        with pytest.raises(ValueError) as exc_info:
            RandomWeather(episode_fn_config={})
        
        assert str(exc_info.value) == "The 'epw_files_folder_path' must be defined in the episode_fn_config."

    def test_init_success_with_valid_config(self):
        """
        Test that __init__ succeeds when 'epw_files_folder_path' is provided in episode_fn_config
        """
        config = {'epw_files_folder_path': '/path/to/epw/files'}
        random_weather = RandomWeather(episode_fn_config=config)
        assert random_weather.episode_fn_config == config

    def test_init_with_valid_config(self):
        """
        Test initialization of RandomWeather with a valid configuration.
        """
        episode_fn_config = {
            'epw_files_folder_path': '/path/to/epw/files'
        }
        random_weather = RandomWeather(episode_fn_config)
        assert random_weather.episode_fn_config == episode_fn_config

    def test_init_without_epw_files_folder_path(self):
        """
        Test initialization of RandomWeather without 'epw_files_folder_path' in config.
        """
        episode_fn_config = {}
        with pytest.raises(ValueError) as excinfo:
            RandomWeather(episode_fn_config)
        assert str(excinfo.value) == "The 'epw_files_folder_path' must be defined in the episode_fn_config."