from eprllib.ActionFunctions.ActionFunctions import ActionFunction
from eprllib.Env.EnvConfig import EnvConfig, env_config_to_dict
from eprllib.EpisodeFunctions.EpisodeFunctions import EpisodeFunction
from eprllib.ObservationFunctions.ObservationFunctions import ObservationFunction
from eprllib.RewardFunctions.RewardFunctions import RewardFunction
import pytest

class TestEnvconfig:

    def test_actions_2(self):
        """
        Test that actions method correctly sets action_fn and action_fn_config when a valid ActionFunction is provided
        and action_fn_config is empty.
        """
        env_config = EnvConfig()
        
        class AnotherDummyActionFunction(ActionFunction):
            def __call__(self, agent_id, action):
                pass

        another_dummy_action_fn = AnotherDummyActionFunction()

        env_config.actions(action_fn=another_dummy_action_fn)

        assert env_config.action_fn == another_dummy_action_fn
        assert env_config.action_fn_config == {}

    def test_actions_empty_action_fn_config(self):
        """
        Test that actions method accepts an empty dictionary for action_fn_config.
        """
        env_config = EnvConfig()
        mock_action_fn = ActionFunction({})
        env_config.actions(action_fn=mock_action_fn, action_fn_config={})
        assert env_config.action_fn == mock_action_fn
        assert env_config.action_fn_config == {}

    def test_actions_invalid_action_fn_config_type(self):
        """
        Test that actions method raises TypeError when action_fn_config is not a dictionary.
        """
        env_config = EnvConfig()
        mock_action_fn = ActionFunction({})
        with pytest.raises(TypeError):
            env_config.actions(action_fn=mock_action_fn, action_fn_config="invalid_type")

    def test_actions_invalid_action_fn_type(self):
        """
        Test that actions method raises TypeError when action_fn is not of type ActionFunction.
        """
        env_config = EnvConfig()
        with pytest.raises(TypeError):
            env_config.actions(action_fn="invalid_type")

    def test_actions_missing_action_fn(self):
        """
        Test that actions method raises NotImplementedError when action_fn is not provided.
        """
        env_config = EnvConfig()
        with pytest.raises(NotImplementedError, match="action_fn must be defined."):
            env_config.actions()

    def test_actions_raises_error_for_notimplemented(self):
        """
        Test that actions method raises NotImplementedError when action_fn is NotImplemented.
        """
        env_config = EnvConfig()

        with pytest.raises(NotImplementedError, match="action_fn must be defined."):
            env_config.actions(action_fn=NotImplemented)

    def test_actions_raises_error_when_action_fn_not_implemented(self):
        """
        Test that actions method raises NotImplementedError when action_fn is not provided.
        """
        env_config = EnvConfig()
        
        with pytest.raises(NotImplementedError) as excinfo:
            env_config.actions()
        
        assert str(excinfo.value) == "action_fn must be defined."

    def test_actions_sets_action_fn_and_config(self):
        """
        Test that actions method correctly sets action_fn and action_fn_config.
        """
        env_config = EnvConfig()
        mock_action_fn = ActionFunction({})
        mock_action_fn_config = {"key": "value"}

        env_config.actions(action_fn=mock_action_fn, action_fn_config=mock_action_fn_config)

        assert env_config.action_fn == mock_action_fn
        assert env_config.action_fn_config == mock_action_fn_config

    def test_actions_valid_action_fn(self):
        """
        Test that actions method correctly sets action_fn and action_fn_config when a valid ActionFunction is provided.
        """
        env_config = EnvConfig()
        
        class DummyActionFunction(ActionFunction):
            def __call__(self, agent_id, action):
                pass

        dummy_action_fn = DummyActionFunction()
        dummy_action_fn_config = {'test_key': 'test_value'}

        env_config.actions(action_fn=dummy_action_fn, action_fn_config=dummy_action_fn_config)

        assert env_config.action_fn == dummy_action_fn
        assert env_config.action_fn_config == dummy_action_fn_config

    def test_actions_with_default_config(self):
        """
        Test that actions method works with default empty config.
        """
        env_config = EnvConfig()
        mock_action_fn = ActionFunction({})

        env_config.actions(action_fn=mock_action_fn)

        assert env_config.action_fn == mock_action_fn
        assert env_config.action_fn_config == {}

    def test_actions_with_non_empty_action_fn_config(self):
        """
        Test that actions method correctly sets non-empty action_fn_config.
        """
        env_config = EnvConfig()
        mock_action_fn = ActionFunction({})
        test_config = {"key": "value"}
        env_config.actions(action_fn=mock_action_fn, action_fn_config=test_config)
        assert env_config.action_fn == mock_action_fn
        assert env_config.action_fn_config == test_config

    def test_agents_2(self):
        """
        Test that agents method correctly sets agents_config when a valid configuration is provided.
        """
        # Arrange
        env_config = EnvConfig()
        valid_agents_config = {
            "agent1": {
                "ep_actuator_config": {"type": "Schedule:Constant", "actuator_key": "Schedule Value"},
                "thermal_zone": "SPACE1-1",
                "thermal_zone_indicator": 1,
                "actuator_type": "thermostat",
                "agent_indicator": 1
            },
            "agent2": {
                "ep_actuator_config": {"type": "Schedule:Constant", "actuator_key": "Schedule Value"},
                "thermal_zone": "SPACE2-1",
                "thermal_zone_indicator": 2,
                "actuator_type": "window",
                "agent_indicator": 2
            }
        }

        # Act
        env_config.agents(agents_config=valid_agents_config)

        # Assert
        assert env_config.agents_config == valid_agents_config
        assert isinstance(env_config.agents_config, dict)
        assert len(env_config.agents_config) == 2
        assert "agent1" in env_config.agents_config
        assert "agent2" in env_config.agents_config

    def test_agents_raises_error_when_config_not_implemented(self):
        """
        Test that the agents method raises a NotImplementedError when agents_config is NotImplemented.
        """
        env_config = EnvConfig()
        
        with pytest.raises(NotImplementedError) as exc_info:
            env_config.agents(agents_config=NotImplemented)
        
        assert str(exc_info.value) == "agents_config must be defined."

    def test_agents_raises_error_when_not_implemented(self):
        """
        Test that agents method raises NotImplementedError when agents_config is NotImplemented.
        """
        # Arrange
        env_config = EnvConfig()

        # Act & Assert
        with pytest.raises(NotImplementedError, match="agents_config must be defined."):
            env_config.agents(agents_config=NotImplemented)

    def test_agents_sets_config_when_provided(self):
        """
        Test that the agents method correctly sets the agents_config when a valid configuration is provided.
        """
        env_config = EnvConfig()
        test_config = {
            "agent1": {
                "ep_actuator_config": {"type": "test", "value": 1},
                "thermal_zone": "Zone1",
                "thermal_zone_indicator": 1,
                "actuator_type": "Thermostat",
                "agent_indicator": 0
            }
        }
        
        env_config.agents(agents_config=test_config)
        
        assert env_config.agents_config == test_config

    def test_agents_with_duplicate_agent_indicators(self):
        """
        Test that agents method raises ValueError when there are duplicate agent indicators.
        """
        env_config = EnvConfig()
        invalid_config = {
            "agent1": {
                "ep_actuator_config": {},
                "thermal_zone": "Zone1",
                "thermal_zone_indicator": 1,
                "actuator_type": "ThermostatSetpoint",
                "agent_indicator": 1
            },
            "agent2": {
                "ep_actuator_config": {},
                "thermal_zone": "Zone2",
                "thermal_zone_indicator": 2,
                "actuator_type": "ThermostatSetpoint",
                "agent_indicator": 1  # Duplicate indicator
            }
        }
        with pytest.raises(ValueError, match="Duplicate agent indicators found"):
            env_config.agents(invalid_config)

    def test_agents_with_empty_agent_config(self):
        """
        Test that agents method raises ValueError when an agent's config is empty.
        """
        env_config = EnvConfig()
        invalid_config = {
            "agent1": {}
        }
        with pytest.raises(ValueError, match="Agent configuration cannot be empty"):
            env_config.agents(invalid_config)

    def test_agents_with_empty_input(self):
        """
        Test that agents method raises NotImplementedError when input is empty.
        """
        env_config = EnvConfig()
        with pytest.raises(NotImplementedError, match="agents_config must be defined."):
            env_config.agents({})

    def test_agents_with_incorrect_type(self):
        """
        Test that agents method raises TypeError when input is of incorrect type.
        """
        env_config = EnvConfig()
        with pytest.raises(TypeError):
            env_config.agents(["agent1", "agent2"])

    def test_agents_with_invalid_field_values(self):
        """
        Test that agents method raises ValueError when field values are invalid.
        """
        env_config = EnvConfig()
        invalid_config = {
            "agent1": {
                "ep_actuator_config": {},
                "thermal_zone": "Zone1",
                "thermal_zone_indicator": "invalid",  # Should be a number
                "actuator_type": "InvalidType",
                "agent_indicator": 1
            }
        }
        with pytest.raises(ValueError):
            env_config.agents(invalid_config)

    def test_agents_with_invalid_input(self):
        """
        Test that agents method raises TypeError when input is invalid.
        """
        env_config = EnvConfig()
        with pytest.raises(TypeError):
            env_config.agents("invalid_input")

    def test_agents_with_missing_required_fields(self):
        """
        Test that agents method raises KeyError when required fields are missing.
        """
        env_config = EnvConfig()
        invalid_config = {
            "agent1": {
                "ep_actuator_config": {},
                "thermal_zone": "Zone1",
                # Missing thermal_zone_indicator
                "actuator_type": "ThermostatSetpoint",
                "agent_indicator": 1
            }
        }
        with pytest.raises(KeyError):
            env_config.agents(invalid_config)

    def test_agents_with_non_dict_agent_config(self):
        """
        Test that agents method raises TypeError when an agent's config is not a dict.
        """
        env_config = EnvConfig()
        invalid_config = {
            "agent1": "Not a dict"
        }
        with pytest.raises(TypeError, match="Agent configuration must be a dictionary"):
            env_config.agents(invalid_config)

    def test_env_config_to_dict_returns_all_attributes(self):
        """
        Test that env_config_to_dict returns all attributes of EnvConfig object.
        """
        # Arrange
        env_config = EnvConfig()
        env_config.epjson_path = "path/to/epjson"
        env_config.epw_path = "path/to/epw"
        env_config.output_path = "path/to/output"
        env_config.ep_terminal_output = False
        env_config.timeout = 15.0

        # Act
        result = env_config_to_dict(env_config)

        # Assert
        assert isinstance(result, dict)
        assert result['epjson_path'] == "path/to/epjson"
        assert result['epw_path'] == "path/to/epw"
        assert result['output_path'] == "path/to/output"
        assert result['ep_terminal_output'] is False
        assert result['timeout'] == 15.0
        assert 'agents_config' in result
        assert 'observation_fn' in result
        assert 'action_fn' in result
        assert 'reward_fn' in result
        assert 'episode_fn' in result

    def test_env_config_to_dict_with_custom_attribute(self):
        """
        Test env_config_to_dict with a custom attribute added to EnvConfig.
        """
        custom_config = EnvConfig()
        custom_config.custom_attr = "custom_value"
        result = env_config_to_dict(custom_config)
        assert 'custom_attr' in result
        assert result['custom_attr'] == "custom_value"

    def test_env_config_to_dict_with_empty_input(self):
        """
        Test env_config_to_dict with an empty input.
        """
        with pytest.raises(TypeError):
            env_config_to_dict(None)

    def test_env_config_to_dict_with_incomplete_env_config(self):
        """
        Test env_config_to_dict with an incomplete EnvConfig object.
        """
        incomplete_config = EnvConfig()
        incomplete_config.epjson_path = "path/to/epjson"
        result = env_config_to_dict(incomplete_config)
        assert isinstance(result, dict)
        assert 'epjson_path' in result
        assert result['epw_path'] == NotImplemented

    def test_env_config_to_dict_with_incorrect_type(self):
        """
        Test env_config_to_dict with an input of incorrect type.
        """
        with pytest.raises(AttributeError):
            env_config_to_dict(123)

    def test_env_config_to_dict_with_invalid_input(self):
        """
        Test env_config_to_dict with an invalid input.
        """
        with pytest.raises(AttributeError):
            env_config_to_dict("invalid_input")

    def test_env_config_to_dict_with_modified_env_config(self):
        """
        Test env_config_to_dict with a modified EnvConfig object.
        """
        modified_config = EnvConfig()
        modified_config.epjson_path = "path/to/epjson"
        modified_config.epw_path = "path/to/epw"
        modified_config.timeout = 20.0
        result = env_config_to_dict(modified_config)
        assert isinstance(result, dict)
        assert result['epjson_path'] == "path/to/epjson"
        assert result['epw_path'] == "path/to/epw"
        assert result['timeout'] == 20.0

    def test_functionalities_1(self):
        """
        Test that functionalities method correctly sets episode_fn and cut_episode_len
        """
        # Arrange
        env_config = EnvConfig()
        custom_episode_fn = EpisodeFunction({"custom_key": "custom_value"})
        custom_cut_episode_len = 5

        # Act
        env_config.functionalities(episode_fn=custom_episode_fn, cut_episode_len=custom_cut_episode_len)

        # Assert
        assert env_config.episode_fn == custom_episode_fn
        assert env_config.cut_episode_len == custom_cut_episode_len
        assert env_config.episode_fn.episode_fn_config == {"custom_key": "custom_value"}

    def test_functionalities_custom_cut_episode_len(self):
        """
        Test that functionalities method correctly sets a custom cut_episode_len
        """
        # Arrange
        env_config = EnvConfig()
        custom_cut_episode_len = 10

        # Act
        env_config.functionalities(cut_episode_len=custom_cut_episode_len)

        # Assert
        assert isinstance(env_config.episode_fn, EpisodeFunction)
        assert env_config.episode_fn.episode_fn_config == {}
        assert env_config.cut_episode_len == custom_cut_episode_len

    def test_functionalities_custom_episode_fn(self):
        """
        Test functionalities with a custom EpisodeFunction
        """
        env_config = EnvConfig()
        custom_episode_fn = EpisodeFunction({"custom_key": "custom_value"})
        env_config.functionalities(episode_fn=custom_episode_fn)
        assert env_config.episode_fn == custom_episode_fn
        assert env_config.episode_fn.episode_fn_config == {"custom_key": "custom_value"}

    def test_functionalities_custom_episode_fn_2(self):
        """
        Test that functionalities method correctly sets a custom episode_fn
        """
        # Arrange
        env_config = EnvConfig()
        custom_episode_fn = EpisodeFunction({"test_key": "test_value"})

        # Act
        env_config.functionalities(episode_fn=custom_episode_fn)

        # Assert
        assert env_config.episode_fn == custom_episode_fn
        assert env_config.episode_fn.episode_fn_config == {"test_key": "test_value"}
        assert env_config.cut_episode_len == 0

    def test_functionalities_default_values(self):
        """
        Test that functionalities method uses default values when not provided
        """
        # Arrange
        env_config = EnvConfig()

        # Act
        env_config.functionalities()

        # Assert
        assert isinstance(env_config.episode_fn, EpisodeFunction)
        assert env_config.episode_fn.episode_fn_config == {}
        assert env_config.cut_episode_len == 0

    def test_functionalities_empty_input(self):
        """
        Test functionalities with empty input
        """
        env_config = EnvConfig()
        env_config.functionalities()
        assert env_config.episode_fn == EpisodeFunction({})
        assert env_config.cut_episode_len == 0

    def test_functionalities_float_cut_episode_len(self):
        """
        Test functionalities with float value for cut_episode_len
        """
        env_config = EnvConfig()
        with pytest.raises(TypeError):
            env_config.functionalities(cut_episode_len=1.5)

    def test_functionalities_incorrect_type_cut_episode_len(self):
        """
        Test functionalities with incorrect type for cut_episode_len
        """
        env_config = EnvConfig()
        with pytest.raises(TypeError):
            env_config.functionalities(cut_episode_len="invalid")

    def test_functionalities_invalid_episode_fn(self):
        """
        Test functionalities with invalid episode_fn
        """
        env_config = EnvConfig()
        with pytest.raises(TypeError):
            env_config.functionalities(episode_fn="invalid")

    def test_functionalities_large_cut_episode_len(self):
        """
        Test functionalities with a very large cut_episode_len
        """
        env_config = EnvConfig()
        large_value = 10**10
        env_config.functionalities(cut_episode_len=large_value)
        assert env_config.cut_episode_len == large_value

    def test_functionalities_negative_cut_episode_len(self):
        """
        Test functionalities with negative cut_episode_len
        """
        env_config = EnvConfig()
        env_config.functionalities(cut_episode_len=-1)
        assert env_config.cut_episode_len == -1

    def test_generals_default_values(self):
        """
        Test generals method default values.
        """
        env_config = EnvConfig()
        env_config.generals(epjson_path="path/to/epjson", epw_path="path/to/epw", output_path="path/to/output")
        assert env_config.ep_terminal_output is True
        assert env_config.timeout == 10.0

    def test_generals_default_values_2(self):
        """
        Test that the generals method uses default values when not provided
        """
        env_config = EnvConfig()

        epjson_path = "/path/to/model.epjson"
        epw_path = "/path/to/weather.epw"
        output_path = "/path/to/output"

        env_config.generals(
            epjson_path=epjson_path,
            epw_path=epw_path,
            output_path=output_path
        )

        assert env_config.epjson_path == epjson_path
        assert env_config.epw_path == epw_path
        assert env_config.output_path == output_path
        assert env_config.ep_terminal_output is True
        assert env_config.timeout == 10.0

    def test_generals_empty_paths(self):
        """
        Test generals method with empty path strings.
        """
        env_config = EnvConfig()
        env_config.generals(epjson_path="", epw_path="", output_path="")
        assert env_config.epjson_path == ""
        assert env_config.epw_path == ""
        assert env_config.output_path == ""

    def test_generals_invalid_ep_terminal_output(self):
        """
        Test generals method with invalid ep_terminal_output.
        """
        env_config = EnvConfig()
        with pytest.raises(TypeError):
            env_config.generals(ep_terminal_output="True")

    def test_generals_invalid_input(self):
        """
        Test generals method with invalid input.
        """
        env_config = EnvConfig()
        with pytest.raises(TypeError):
            env_config.generals(epjson_path=123)  # Invalid type for epjson_path
        with pytest.raises(TypeError):
            env_config.generals(epw_path=True)  # Invalid type for epw_path
        with pytest.raises(TypeError):
            env_config.generals(output_path=[])

    def test_generals_invalid_timeout(self):
        """
        Test generals method with invalid timeout values.
        """
        env_config = EnvConfig()
        with pytest.raises(ValueError):
            env_config.generals(timeout=-1.0)  # Timeout should be positive
        with pytest.raises(TypeError):
            env_config.generals(timeout="10")

    def test_generals_not_implemented_paths(self):
        """
        Test generals method with NotImplemented paths.
        """
        env_config = EnvConfig()
        env_config.generals()
        assert env_config.epjson_path is NotImplemented
        assert env_config.epw_path is NotImplemented
        assert env_config.output_path is NotImplemented

    def test_generals_partial_update(self):
        """
        Test that the generals method correctly updates only provided parameters
        """
        env_config = EnvConfig()

        # Set initial values
        env_config.generals(
            epjson_path="/initial/model.epjson",
            epw_path="/initial/weather.epw",
            output_path="/initial/output",
            ep_terminal_output=True,
            timeout=10.0
        )

        # Update only some parameters
        new_epjson_path = "/new/model.epjson"
        new_timeout = 20.0

        env_config.generals(
            epjson_path=new_epjson_path,
            timeout=new_timeout
        )

        assert env_config.epjson_path == new_epjson_path
        assert env_config.epw_path == "/initial/weather.epw"
        assert env_config.output_path == "/initial/output"
        assert env_config.ep_terminal_output is True
        assert env_config.timeout == new_timeout

    def test_generals_set_all_parameters(self):
        """
        Test that the generals method correctly sets all parameters when provided
        """
        env_config = EnvConfig()
        
        epjson_path = "/path/to/model.epjson"
        epw_path = "/path/to/weather.epw"
        output_path = "/path/to/output"
        ep_terminal_output = False
        timeout = 15.0

        env_config.generals(
            epjson_path=epjson_path,
            epw_path=epw_path,
            output_path=output_path,
            ep_terminal_output=ep_terminal_output,
            timeout=timeout
        )

        assert env_config.epjson_path == epjson_path
        assert env_config.epw_path == epw_path
        assert env_config.output_path == output_path
        assert env_config.ep_terminal_output == ep_terminal_output
        assert env_config.timeout == timeout

    def test_init_default_values(self):
        """
        Test that EnvConfig initializes with correct default values.
        """
        env_config = EnvConfig()

        # Check general configurations
        assert env_config.epjson_path == NotImplemented
        assert env_config.epw_path == NotImplemented
        assert env_config.output_path == NotImplemented
        assert env_config.ep_terminal_output is True
        assert env_config.timeout == 10.0

        # Check agents configuration
        assert env_config.agents_config == NotImplemented

        # Check observations configuration
        assert env_config.observation_fn == NotImplemented
        assert env_config.observation_fn_config == {}
        assert env_config.variables_env == []
        assert env_config.variables_thz == []
        assert env_config.variables_obj == {}
        assert env_config.meters == {}
        assert env_config.static_variables == {}

        # Check simulation parameters
        assert isinstance(env_config.simulation_parameters, dict)
        assert all(value == False for value in env_config.simulation_parameters.values())

        # Check zone simulation parameters
        assert isinstance(env_config.zone_simulation_parameters, dict)
        assert all(value == False for value in env_config.zone_simulation_parameters.values())

        # Check other observation-related configurations
        assert env_config.infos_variables == NotImplemented
        assert env_config.no_observable_variables == NotImplemented
        assert env_config.use_actuator_state == False
        assert env_config.use_agent_indicator == True
        assert env_config.use_thermal_zone_indicator == False
        assert env_config.use_agent_type == False
        assert env_config.use_building_properties == False
        assert env_config.building_properties == NotImplemented
        assert env_config.use_one_day_weather_prediction == False
        assert env_config.prediction_hours == 24

        # Check prediction variables
        assert isinstance(env_config.prediction_variables, dict)
        assert all(value == False for value in env_config.prediction_variables.values())

        # Check actions configuration
        assert env_config.action_fn == NotImplemented
        assert env_config.action_fn_config == {}

        # Check rewards configuration
        assert env_config.reward_fn == NotImplemented

        # Check functionalities configuration
        assert env_config.cut_episode_len == 0
        assert isinstance(env_config.episode_fn, EpisodeFunction)
        assert env_config.episode_fn.episode_fn_config == {}

    def test_init_with_invalid_action_fn(self):
        """
        Test initialization with an invalid action_fn.
        """
        with pytest.raises(NotImplementedError):
            EnvConfig().actions(action_fn=NotImplemented)

    def test_init_with_invalid_agents_config(self):
        """
        Test initialization with an invalid agents_config.
        """
        with pytest.raises(NotImplementedError):
            EnvConfig().agents(agents_config=NotImplemented)

    def test_init_with_invalid_cut_episode_len(self):
        """
        Test initialization with an invalid cut_episode_len value.
        """
        with pytest.raises(ValueError):
            EnvConfig().functionalities(cut_episode_len=-1)

    def test_init_with_invalid_epjson_path(self):
        """
        Test initialization with an invalid epjson_path.
        """
        with pytest.raises(NotImplementedError):
            EnvConfig().generals(epjson_path="invalid_path.epjson")

    def test_init_with_invalid_epw_path(self):
        """
        Test initialization with an invalid epw_path.
        """
        with pytest.raises(NotImplementedError):
            EnvConfig().generals(epw_path="invalid_path.epw")

    def test_init_with_invalid_observation_fn(self):
        """
        Test initialization with an invalid observation_fn.
        """
        with pytest.raises(NotImplementedError):
            EnvConfig().observations(observation_fn=NotImplemented)

    def test_init_with_invalid_prediction_hours(self):
        """
        Test initialization with an invalid prediction_hours value.
        """
        with pytest.raises(ValueError):
            EnvConfig().observations(use_one_day_weather_prediction=True, prediction_hours=25)

    def test_init_with_invalid_reward_fn(self):
        """
        Test initialization with an invalid reward_fn.
        """
        with pytest.raises(NotImplementedError):
            EnvConfig().rewards(reward_fn=NotImplemented)

    def test_init_with_invalid_timeout(self):
        """
        Test initialization with an invalid timeout value.
        """
        with pytest.raises(ValueError):
            EnvConfig().generals(timeout=-1.0)

    def test_observations_2(self):
        """
        Test observations method with various invalid inputs and edge cases
        """
        env_config = EnvConfig()

        # Test with NotImplemented observation_fn
        with pytest.raises(NotImplementedError):
            env_config.observations(observation_fn=NotImplemented)

        # Test with use_building_properties=True but building_properties=NotImplemented
        with pytest.raises(NotImplementedError):
            env_config.observations(
                observation_fn=ObservationFunction({}),
                use_building_properties=True,
                building_properties=NotImplemented
            )

        # Test with invalid prediction_hours
        with pytest.raises(ValueError):
            env_config.observations(
                observation_fn=ObservationFunction({}),
                use_one_day_weather_prediction=True,
                prediction_hours=25
            )

        # Test with use_one_day_weather_prediction=True and invalid prediction_variables
        with pytest.raises(ValueError):
            env_config.observations(
                observation_fn=ObservationFunction({}),
                use_one_day_weather_prediction=True,
                prediction_variables={'invalid_key': True}
            )

        # Test with invalid simulation_parameters
        with pytest.raises(ValueError):
            env_config.observations(
                observation_fn=ObservationFunction({}),
                simulation_parameters={'invalid_key': True}
            )

        # Test with invalid zone_simulation_parameters
        with pytest.raises(ValueError):
            env_config.observations(
                observation_fn=ObservationFunction({}),
                zone_simulation_parameters={'invalid_key': True}
            )

        # Test with NotImplemented infos_variables
        with pytest.raises(NotImplementedError):
            env_config.observations(
                observation_fn=ObservationFunction({}),
                infos_variables=NotImplemented
            )

        # Test with valid inputs
        env_config.observations(
            observation_fn=ObservationFunction({}),
            observation_fn_config={},
            variables_env=['Temperature'],
            variables_thz=['Humidity'],
            variables_obj={'Zone1': {'Temp': 'ZoneTemp'}},
            meters=['Electricity'],
            static_variables=['Volume'],
            simulation_parameters={'hour': True},
            zone_simulation_parameters={'zone_time_step': True},
            infos_variables={'variables_env': ['Temperature']},
            no_observable_variables={'variables_thz': ['Humidity']},
            use_actuator_state=True,
            use_agent_indicator=False,
            use_thermal_zone_indicator=True,
            use_agent_type=True,
            use_building_properties=True,
            building_properties={'Zone1': {'area': 100.0}},
            use_one_day_weather_prediction=True,
            prediction_hours=12,
            prediction_variables={'outdoor_dry_bulb': True}
        )

        assert env_config.observation_fn is not None
        assert env_config.use_actuator_state is True
        assert env_config.use_agent_indicator == False
        assert env_config.use_thermal_zone_indicator == True
        assert env_config.use_agent_type == True
        assert env_config.use_building_properties == True
        assert env_config.building_properties == {'Zone1': {'area': 100.0}}
        assert env_config.use_one_day_weather_prediction == True
        assert env_config.prediction_hours == 12
        assert env_config.prediction_variables == {'outdoor_dry_bulb': True}

    def test_observations_3_raises_errors(self):
        """
        Test that observations method raises appropriate errors for invalid inputs.
        """
        env_config = EnvConfig()

        # Test observation_fn == NotImplemented
        with pytest.raises(NotImplementedError, match="observation_function must be defined."):
            env_config.observations(observation_fn=NotImplemented)

        # Test use_building_properties True but building_properties NotImplemented
        with pytest.raises(NotImplementedError, match="building_properties must be defined if use_building_properties is True."):
            env_config.observations(
                observation_fn=ObservationFunction({}),
                use_building_properties=True,
                building_properties=NotImplemented
            )

        # Test invalid prediction_hours
        with pytest.raises(ValueError, match="The variable 'prediction_hours' must be between 1 and 24."):
            env_config.observations(
                observation_fn=ObservationFunction({}),
                use_one_day_weather_prediction=True,
                prediction_hours=25
            )

        # Test invalid prediction_variables
        with pytest.raises(ValueError, match="The key 'invalid_key' is not admissible in the prediction_variables."):
            env_config.observations(
                observation_fn=ObservationFunction({}),
                use_one_day_weather_prediction=True,
                prediction_variables={'invalid_key': True}
            )

        # Test invalid simulation_parameters
        with pytest.raises(ValueError, match="The key 'invalid_key' is not admissible in the simulation_parameters."):
            env_config.observations(
                observation_fn=ObservationFunction({}),
                simulation_parameters={'invalid_key': True}
            )

        # Test invalid zone_simulation_parameters
        with pytest.raises(ValueError, match="The key 'invalid_key' is not admissible in the zone_simulation_parameters."):
            env_config.observations(
                observation_fn=ObservationFunction({}),
                zone_simulation_parameters={'invalid_key': True}
            )

        # Test infos_variables == NotImplemented
        with pytest.raises(NotImplementedError, match="infos_variables must be defined."):
            env_config.observations(
                observation_fn=ObservationFunction({}),
                infos_variables=NotImplemented
            )

    def test_observations_4_raises_multiple_exceptions(self):
        """
        Test that observations method raises multiple exceptions when given invalid inputs.
        """
        env_config = EnvConfig()
        
        with pytest.raises(NotImplementedError) as exc_info:
            env_config.observations(
                observation_fn=NotImplemented,
                use_building_properties=True,
                building_properties=NotImplemented,
                use_one_day_weather_prediction=True,
                prediction_hours=25,
                prediction_variables={'invalid_key': True},
                infos_variables=NotImplemented
            )
        
        assert "observation_function must be defined." in str(exc_info.value)
        
        # Create a dummy ObservationFunction to bypass the first exception
        dummy_observation_fn = ObservationFunction({})
        
        with pytest.raises(NotImplementedError) as exc_info:
            env_config.observations(
                observation_fn=dummy_observation_fn,
                use_building_properties=True,
                building_properties=NotImplemented,
                use_one_day_weather_prediction=True,
                prediction_hours=25,
                prediction_variables={'invalid_key': True},
                infos_variables=NotImplemented
            )
        
        assert "building_properties must be defined if use_building_properties is True." in str(exc_info.value)
        
        # Provide valid building_properties to bypass the second exception
        valid_building_properties = {'zone1': {'area': 100.0}}
        
        with pytest.raises(ValueError) as exc_info:
            env_config.observations(
                observation_fn=dummy_observation_fn,
                use_building_properties=True,
                building_properties=valid_building_properties,
                use_one_day_weather_prediction=True,
                prediction_hours=25,
                prediction_variables={'invalid_key': True},
                infos_variables=NotImplemented
            )
        
        assert "The variable 'prediction_hours' must be between 1 and 24." in str(exc_info.value)
        
        # Provide valid prediction_hours to bypass the third exception
        with pytest.raises(ValueError) as exc_info:
            env_config.observations(
                observation_fn=dummy_observation_fn,
                use_building_properties=True,
                building_properties=valid_building_properties,
                use_one_day_weather_prediction=True,
                prediction_hours=24,
                prediction_variables={'invalid_key': True},
                infos_variables=NotImplemented
            )
        
        assert "The key 'invalid_key' is not admissible in the prediction_variables." in str(exc_info.value)
        
        # Provide valid prediction_variables to bypass the fourth exception
        with pytest.raises(NotImplementedError) as exc_info:
            env_config.observations(
                observation_fn=dummy_observation_fn,
                use_building_properties=True,
                building_properties=valid_building_properties,
                use_one_day_weather_prediction=True,
                prediction_hours=24,
                prediction_variables={'outdoor_dry_bulb': True},
                infos_variables=NotImplemented
            )
        
        assert "infos_variables must be defined." in str(exc_info.value)

    def test_observations_5_raises_multiple_errors(self):
        """
        Test that observations method raises multiple errors when invalid inputs are provided.
        """
        env_config = EnvConfig()
        
        with pytest.raises(NotImplementedError) as excinfo:
            env_config.observations(
                observation_fn=NotImplemented,
                use_building_properties=True,
                building_properties=NotImplemented,
                prediction_hours=0,
                use_one_day_weather_prediction=True,
                prediction_variables={'invalid_key': True},
                infos_variables=NotImplemented
            )
        
        assert "observation_function must be defined." in str(excinfo.value)
        
        # Create a mock ObservationFunction to bypass the first error
        mock_observation_fn = ObservationFunction({})
        
        with pytest.raises(NotImplementedError) as excinfo:
            env_config.observations(
                observation_fn=mock_observation_fn,
                use_building_properties=True,
                building_properties=NotImplemented,
                prediction_hours=0,
                use_one_day_weather_prediction=True,
                prediction_variables={'invalid_key': True},
                infos_variables=NotImplemented
            )
        
        assert "building_properties must be defined if use_building_properties is True." in str(excinfo.value)
        
        with pytest.raises(ValueError) as excinfo:
            env_config.observations(
                observation_fn=mock_observation_fn,
                use_building_properties=False,
                prediction_hours=0,
                use_one_day_weather_prediction=True,
                prediction_variables={'invalid_key': True},
                infos_variables=NotImplemented
            )
        
        assert "The variable 'prediction_hours' must be between 1 and 24." in str(excinfo.value)
        
        with pytest.raises(ValueError) as excinfo:
            env_config.observations(
                observation_fn=mock_observation_fn,
                use_building_properties=False,
                prediction_hours=24,
                use_one_day_weather_prediction=True,
                prediction_variables={'invalid_key': True},
                infos_variables=NotImplemented
            )
        
        assert "The key 'invalid_key' is not admissible in the prediction_variables." in str(excinfo.value)
        
        with pytest.raises(NotImplementedError) as excinfo:
            env_config.observations(
                observation_fn=mock_observation_fn,
                use_building_properties=False,
                prediction_hours=24,
                use_one_day_weather_prediction=False,
                infos_variables=NotImplemented
            )
        
        assert "infos_variables must be defined." in str(excinfo.value)

    def test_observations_6_raises_multiple_errors(self):
        """
        Test that observations method raises multiple errors when invalid parameters are provided.
        """
        env_config = EnvConfig()

        with pytest.raises(NotImplementedError) as exc_info:
            env_config.observations(
                observation_fn=NotImplemented,
                use_building_properties=True,
                building_properties=NotImplemented,
                prediction_hours=0,
                use_one_day_weather_prediction=False,
                simulation_parameters=False,
                zone_simulation_parameters=False,
                infos_variables=NotImplemented
            )

        assert "observation_function must be defined." in str(exc_info.value)

        with pytest.raises(NotImplementedError) as exc_info:
            env_config.observations(
                observation_fn=ObservationFunction({}),
                use_building_properties=True,
                building_properties=NotImplemented,
                prediction_hours=0,
                use_one_day_weather_prediction=False,
                simulation_parameters=False,
                zone_simulation_parameters=False,
                infos_variables=NotImplemented
            )

        assert "building_properties must be defined if use_building_properties is True." in str(exc_info.value)

        with pytest.raises(ValueError) as exc_info:
            env_config.observations(
                observation_fn=ObservationFunction({}),
                use_building_properties=False,
                prediction_hours=0,
                use_one_day_weather_prediction=False,
                simulation_parameters=False,
                zone_simulation_parameters=False,
                infos_variables={}
            )

        assert "The variable 'prediction_hours' must be between 1 and 24." in str(exc_info.value)

        with pytest.raises(NotImplementedError) as exc_info:
            env_config.observations(
                observation_fn=ObservationFunction({}),
                use_building_properties=False,
                prediction_hours=24,
                use_one_day_weather_prediction=False,
                simulation_parameters=False,
                zone_simulation_parameters=False,
                infos_variables=NotImplemented
            )

        assert "infos_variables must be defined. The variables defined here are used in the reward function." in str(exc_info.value)

    def test_observations_7_raises_multiple_errors(self):
        """
        Test that observations method raises multiple errors when given invalid inputs.
        """
        env_config = EnvConfig()
        
        with pytest.raises(NotImplementedError) as exc_info:
            env_config.observations(
                observation_fn=NotImplemented,
                use_building_properties=True,
                building_properties=NotImplemented,
                prediction_hours=25,
                use_one_day_weather_prediction=True,
                prediction_variables={'invalid_key': True},
                simulation_parameters=False,
                zone_simulation_parameters=False,
                infos_variables={'valid_key': []}
            )
        
        assert "observation_function must be defined." in str(exc_info.value)
        
        # Test building_properties error
        with pytest.raises(NotImplementedError) as exc_info:
            env_config.observations(
                observation_fn=ObservationFunction({}),
                use_building_properties=True,
                building_properties=NotImplemented
            )
        
        assert "building_properties must be defined if use_building_properties is True." in str(exc_info.value)
        
        # Test prediction_hours error
        with pytest.raises(ValueError) as exc_info:
            env_config.observations(
                observation_fn=ObservationFunction({}),
                prediction_hours=25
            )
        
        assert "The variable 'prediction_hours' must be between 1 and 24." in str(exc_info.value)
        
        # Test prediction_variables error
        with pytest.raises(ValueError) as exc_info:
            env_config.observations(
                observation_fn=ObservationFunction({}),
                use_one_day_weather_prediction=True,
                prediction_variables={'invalid_key': True}
            )
        
        assert "The key 'invalid_key' is not admissible in the prediction_variables." in str(exc_info.value)

        # Verify that no errors are raised for valid inputs
        env_config.observations(
            observation_fn=ObservationFunction({}),
            simulation_parameters=False,
            zone_simulation_parameters=False,
            infos_variables={'valid_key': []}
        )
        
        assert env_config.observation_fn is not None
        assert env_config.simulation_parameters == {}
        assert env_config.zone_simulation_parameters == {}
        assert env_config.infos_variables == {'valid_key': []}

    def test_observations_empty_observation_fn(self):
        """
        Test that NotImplementedError is raised when observation_fn is not provided.
        """
        env_config = EnvConfig()
        with pytest.raises(NotImplementedError, match="observation_function must be defined."):
            env_config.observations()

    def test_observations_invalid_prediction_hours(self):
        """
        Test that ValueError is raised when prediction_hours is outside the valid range.
        """
        env_config = EnvConfig()
        with pytest.raises(ValueError, match="The variable 'prediction_hours' must be between 1 and 24."):
            env_config.observations(
                observation_fn=ObservationFunction({}),
                use_one_day_weather_prediction=True,
                prediction_hours=0
            )

    def test_observations_invalid_prediction_variables(self):
        """
        Test that ValueError is raised when invalid prediction_variables are provided.
        """
        env_config = EnvConfig()
        with pytest.raises(ValueError, match="The key 'invalid_key' is not admissible in the prediction_variables."):
            env_config.observations(
                observation_fn=ObservationFunction({}),
                use_one_day_weather_prediction=True,
                prediction_variables={'invalid_key': True}
            )

    def test_observations_invalid_simulation_parameters(self):
        """
        Test that ValueError is raised when invalid simulation_parameters are provided.
        """
        env_config = EnvConfig()
        with pytest.raises(ValueError, match="The key 'invalid_param' is not admissible in the simulation_parameters."):
            env_config.observations(
                observation_fn=ObservationFunction({}),
                simulation_parameters={'invalid_param': True}
            )

    def test_observations_invalid_zone_simulation_parameters(self):
        """
        Test that ValueError is raised when invalid zone_simulation_parameters are provided.
        """
        env_config = EnvConfig()
        with pytest.raises(ValueError, match="The key 'invalid_zone_param' is not admissible in the zone_simulation_parameters."):
            env_config.observations(
                observation_fn=ObservationFunction({}),
                zone_simulation_parameters={'invalid_zone_param': True}
            )

    def test_observations_missing_building_properties(self):
        """
        Test that NotImplementedError is raised when use_building_properties is True but building_properties is not defined.
        """
        env_config = EnvConfig()
        with pytest.raises(NotImplementedError, match="building_properties must be defined if use_building_properties is True."):
            env_config.observations(
                observation_fn=ObservationFunction({}),
                use_building_properties=True
            )

    def test_observations_missing_infos_variables(self):
        """
        Test that NotImplementedError is raised when infos_variables is not provided.
        """
        env_config = EnvConfig()
        with pytest.raises(NotImplementedError, match="infos_variables must be defined."):
            env_config.observations(
                observation_fn=ObservationFunction({}),
                infos_variables=NotImplemented
            )

    def test_observations_raises_error_for_invalid_prediction_hours(self):
        """
        Test that observations method raises ValueError when prediction_hours is invalid.
        """
        env_config = EnvConfig()
        dummy_observation_fn = ObservationFunction({})

        with pytest.raises(ValueError, match="The variable 'prediction_hours' must be between 1 and 24."):
            env_config.observations(
                observation_fn=dummy_observation_fn,
                use_one_day_weather_prediction=True,
                prediction_hours=25
            )

    def test_observations_raises_error_for_invalid_prediction_variables(self):
        """
        Test that observations method raises ValueError when invalid prediction variables are provided.
        """
        env_config = EnvConfig()
        dummy_observation_fn = ObservationFunction({})

        with pytest.raises(ValueError, match="The key 'invalid_key' is not admissible in the prediction_variables."):
            env_config.observations(
                observation_fn=dummy_observation_fn,
                use_one_day_weather_prediction=True,
                prediction_variables={'invalid_key': True}
            )

    def test_observations_raises_error_for_missing_building_properties(self):
        """
        Test that observations method raises NotImplementedError when use_building_properties is True but building_properties is not provided.
        """
        env_config = EnvConfig()
        dummy_observation_fn = ObservationFunction({})

        with pytest.raises(NotImplementedError, match="building_properties must be defined if use_building_properties is True."):
            env_config.observations(
                observation_fn=dummy_observation_fn,
                use_building_properties=True
            )

    def test_observations_raises_error_for_missing_infos_variables(self):
        """
        Test that observations method raises NotImplementedError when infos_variables is not provided.
        """
        env_config = EnvConfig()
        dummy_observation_fn = ObservationFunction({})

        with pytest.raises(NotImplementedError, match="infos_variables must be defined."):
            env_config.observations(
                observation_fn=dummy_observation_fn
            )

    def test_observations_raises_notimplementederror(self):
        """
        Test that observations method raises NotImplementedError when observation_fn is not provided.
        """
        env_config = EnvConfig()
        
        with pytest.raises(NotImplementedError, match="observation_function must be defined."):
            env_config.observations()

    def test_observations_updates_simulation_parameters(self):
        """
        Test that observations method correctly updates simulation parameters.
        """
        env_config = EnvConfig()
        dummy_observation_fn = ObservationFunction({})

        env_config.observations(
            observation_fn=dummy_observation_fn,
            simulation_parameters={'year': True},
            infos_variables={}
        )

        assert env_config.simulation_parameters['year'] is True

    def test_observations_updates_zone_simulation_parameters(self):
        """
        Test that observations method correctly updates zone simulation parameters.
        """
        env_config = EnvConfig()
        dummy_observation_fn = ObservationFunction({})

        env_config.observations(
            observation_fn=dummy_observation_fn,
            zone_simulation_parameters={'zone_time_step': True},
            infos_variables={}
        )

        assert env_config.zone_simulation_parameters['zone_time_step'] is True

    def test_rewards_raises_error_for_not_implemented(self):
        """
        Test that the rewards method raises a NotImplementedError when reward_fn is NotImplemented.
        """
        env_config = EnvConfig()

        with pytest.raises(NotImplementedError, match="reward_fn must be defined."):
            env_config.rewards(reward_fn=NotImplemented)

    def test_rewards_raises_error_when_reward_fn_not_implemented(self):
        """
        Test that rewards method raises NotImplementedError when reward_fn is NotImplemented.
        """
        env_config = EnvConfig()
        
        with pytest.raises(NotImplementedError) as excinfo:
            env_config.rewards(reward_fn=NotImplemented)
        
        assert str(excinfo.value) == "reward_fn must be defined."

    def test_rewards_sets_reward_fn_when_valid(self):
        """
        Test that rewards method sets the reward_fn when a valid RewardFunction is provided.
        """
        env_config = EnvConfig()
        mock_reward_fn = RewardFunction()
        
        env_config.rewards(reward_fn=mock_reward_fn)
        
        assert env_config.reward_fn == mock_reward_fn

    def test_rewards_valid_reward_function(self):
        """
        Test that the rewards method correctly sets the reward function when a valid RewardFunction is provided.
        """
        env_config = EnvConfig()

        class DummyRewardFunction(RewardFunction):
            def __call__(self, env, infos):
                return 0

        dummy_reward_fn = DummyRewardFunction()

        env_config.rewards(reward_fn=dummy_reward_fn)

        assert env_config.reward_fn == dummy_reward_fn

    def test_rewards_with_empty_reward_function(self):
        """
        Test rewards method with an empty RewardFunction
        """
        env_config = EnvConfig()
        empty_reward_function = RewardFunction({})
        env_config.rewards(reward_fn=empty_reward_function)
        assert env_config.reward_fn == empty_reward_function

    def test_rewards_with_incorrect_type(self):
        """
        Test rewards method with incorrect type (int instead of RewardFunction)
        """
        env_config = EnvConfig()
        with pytest.raises(AttributeError):
            env_config.rewards(reward_fn=42)

    def test_rewards_with_invalid_input(self):
        """
        Test rewards method with invalid input (not a RewardFunction)
        """
        env_config = EnvConfig()
        with pytest.raises(AttributeError):
            env_config.rewards(reward_fn="not_a_reward_function")

    def test_rewards_with_no_input(self):
        """
        Test rewards method with no input (empty input)
        """
        env_config = EnvConfig()
        with pytest.raises(NotImplementedError, match="reward_fn must be defined."):
            env_config.rewards()

    def test_rewards_with_none_input(self):
        """
        Test rewards method with None input
        """
        env_config = EnvConfig()
        with pytest.raises(AttributeError):
            env_config.rewards(reward_fn=None)

    def test_rewards_with_notimplemented(self):
        """
        Test rewards method with NotImplemented input
        """
        env_config = EnvConfig()
        with pytest.raises(NotImplementedError, match="reward_fn must be defined."):
            env_config.rewards(reward_fn=NotImplemented)