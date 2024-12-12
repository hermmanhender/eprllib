from eprllib.EpisodeFunctions.EpisodeFunctions import EpisodeFunction
from typing import Dict, Any
import pytest

class TestEpisodefunctions:

    def test_get_episode_config_returns_input_config(self):
        """
        Test that get_episode_config returns the input env_config without modifications.
        """
        # Arrange
        episode_function = EpisodeFunction()
        env_config = {"key1": "value1", "key2": 42}

        # Act
        result = episode_function.get_episode_config(env_config)

        # Assert
        assert result == env_config, "The returned config should be identical to the input config"
        assert id(result) == id(env_config), "The returned config should be the same object as the input config"

    def test_get_episode_config_with_empty_input(self):
        """
        Test get_episode_config with an empty dictionary input.
        """
        episode_function = EpisodeFunction()
        result = episode_function.get_episode_config({})
        assert result == {}, "Expected an empty dictionary to be returned for empty input"

    def test_get_episode_config_with_invalid_type(self):
        """
        Test get_episode_config with an invalid input type.
        """
        episode_function = EpisodeFunction()
        with pytest.raises(TypeError):
            episode_function.get_episode_config("invalid_input")

    def test_get_episode_config_with_large_input(self):
        """
        Test get_episode_config with a large input dictionary.
        """
        episode_function = EpisodeFunction()
        large_input = {str(i): i for i in range(10000)}
        result = episode_function.get_episode_config(large_input)
        assert result == large_input, "Expected the large input dictionary to be returned unchanged"

    def test_get_episode_config_with_nested_dictionary(self):
        """
        Test get_episode_config with a nested dictionary input.
        """
        episode_function = EpisodeFunction()
        nested_input = {"key1": {"nested_key": "value"}, "key2": 42}
        result = episode_function.get_episode_config(nested_input)
        assert result == nested_input, "Expected the input nested dictionary to be returned unchanged"

    def test_get_episode_config_with_none_input(self):
        """
        Test get_episode_config with None as input.
        """
        episode_function = EpisodeFunction()
        with pytest.raises(TypeError):
            episode_function.get_episode_config(None)

    def test_init_with_config(self):
        """
        Test initialization of EpisodeFunction with a non-empty configuration dictionary.
        """
        config = {"key1": "value1", "key2": 42}
        episode_function = EpisodeFunction(episode_fn_config=config)
        assert isinstance(episode_function.episode_fn_config, dict)
        assert episode_function.episode_fn_config == config

    def test_init_with_empty_config(self):
        """
        Test initialization of EpisodeFunction with an empty configuration dictionary.
        """
        episode_function = EpisodeFunction()
        assert isinstance(episode_function.episode_fn_config, dict)
        assert len(episode_function.episode_fn_config) == 0

    def test_init_with_empty_input(self):
        """
        Test initializing EpisodeFunction with an empty dictionary.
        """
        episode_function = EpisodeFunction()
        assert episode_function.episode_fn_config == {}

    def test_init_with_invalid_input_type(self):
        """
        Test initializing EpisodeFunction with an invalid input type.
        """
        with pytest.raises(TypeError):
            EpisodeFunction(episode_fn_config="invalid")

    def test_init_with_large_input(self):
        """
        Test initializing EpisodeFunction with a large input dictionary.
        """
        large_dict = {str(i): i for i in range(1000000)}
        episode_function = EpisodeFunction(episode_fn_config=large_dict)
        assert episode_function.episode_fn_config == large_dict

    def test_init_with_nested_non_serializable_objects(self):
        """
        Test initializing EpisodeFunction with nested non-serializable objects.
        """
        class NonSerializable:
            pass

        with pytest.raises(TypeError):
            EpisodeFunction(episode_fn_config={"key": NonSerializable()})

    def test_init_with_non_dict_input(self):
        """
        Test initializing EpisodeFunction with a non-dictionary input.
        """
        with pytest.raises(TypeError):
            EpisodeFunction(episode_fn_config=[1, 2, 3])