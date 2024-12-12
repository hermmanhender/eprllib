from eprllib.Env.MultiAgent.EnvUtils import EP_API_add_path
from eprllib.Env.MultiAgent.EnvUtils import actuators_to_agents
from eprllib.Env.MultiAgent.EnvUtils import continuous_action_space
from eprllib.Env.MultiAgent.EnvUtils import discrete_action_space
from gymnasium.spaces import Box
from gymnasium.spaces import Discrete
from src.eprllib.Env.MultiAgent.EnvUtils import actuators_to_agents
from typing import Dict, List
from typing import Dict, List, Tuple
from unittest.mock import patch
import numpy as np
import pytest
import sys

class TestEnvutils:

    def test_EP_API_add_path_2(self):
        """
        Test EP_API_add_path when path is None, os_platform is Linux, and new_path is not in sys.path
        """
        with patch('sys.platform', 'linux'), \
             patch('sys.path', ['some_existing_path']), \
             patch('builtins.print') as mock_print:
            
            # Call the function
            EP_API_add_path(version="23-2-0")

            # Assert that the new path was added to sys.path
            assert "/usr/local/EnergyPlus-23-2-0" in sys.path
            assert sys.path.index("/usr/local/EnergyPlus-23-2-0") == 0

            # Assert that the print statement was called
            mock_print.assert_called_once_with("EnergyPlus API path added: /usr/local/EnergyPlus-23-2-0")

    def test_EP_API_add_path_custom_path(self):
        """
        Test EP_API_add_path when a custom path is provided and not already in sys.path.
        """
        custom_path = "/custom/energyplus/path"
        
        # Mocking sys.path to ensure the custom path is not already present
        with patch.object(sys, 'path', ['/some/other/path']):
            with patch('builtins.print') as mock_print:
                EP_API_add_path(path=custom_path)
                
                # Assert that the custom path was added to sys.path
                assert custom_path == sys.path[0]
                
                # Assert that the correct message was printed
                mock_print.assert_called_once_with(f"EnergyPlus API path added: {custom_path}")

    def test_EP_API_add_path_custom_path_already_in_sys_path(self):
        """
        Test EP_API_add_path when a custom path is provided and it's already in sys.path.
        """
        # Setup
        custom_path = "/custom/energyplus/path"
        sys.path.insert(0, custom_path)
        original_sys_path = sys.path.copy()

        # Execute
        EP_API_add_path(path=custom_path)

        # Assert
        assert sys.path == original_sys_path
        assert custom_path in sys.path
        assert sys.path.index(custom_path) == 0

        # Cleanup
        sys.path.remove(custom_path)

    def test_EP_API_add_path_duplicate_addition(self):
        """
        Test EP_API_add_path to ensure it doesn't add duplicate paths.
        """
        initial_path_count = len(sys.path)
        EP_API_add_path()
        first_addition_count = len(sys.path)
        EP_API_add_path()
        second_addition_count = len(sys.path)
        assert first_addition_count == initial_path_count + 1
        assert second_addition_count == first_addition_count

    def test_EP_API_add_path_empty_version(self):
        """
        Test EP_API_add_path with an empty version string.
        """
        with pytest.raises(ValueError):
            EP_API_add_path(version="")

    def test_EP_API_add_path_incorrect_type(self):
        """
        Test EP_API_add_path with incorrect type for version.
        """
        with pytest.raises(TypeError):
            EP_API_add_path(version=123)

    def test_EP_API_add_path_invalid_path_type(self):
        """
        Test EP_API_add_path with an invalid path type.
        """
        with pytest.raises(TypeError):
            EP_API_add_path(path=123)

    def test_EP_API_add_path_invalid_version(self):
        """
        Test EP_API_add_path with an invalid version format.
        """
        with pytest.raises(ValueError):
            EP_API_add_path(version="invalid_version")

    def test_EP_API_add_path_nonexistent_path(self):
        """
        Test EP_API_add_path with a nonexistent path.
        """
        non_existent_path = "/path/does/not/exist"
        EP_API_add_path(path=non_existent_path)
        assert non_existent_path not in sys.path

    def test_EP_API_add_path_unsupported_platform(self, monkeypatch):
        """
        Test EP_API_add_path with an unsupported platform.
        """
        monkeypatch.setattr(sys, 'platform', 'unsupported_platform')
        with pytest.raises(ValueError):
            EP_API_add_path()

    def test_actuators_to_agents_2(self):
        """
        Test actuators_to_agents with multiple agents in the same thermal zone.
        """
        agent_config = {
            "agent1": ["Component1", "Control1", "Variable1", "Zone1", "Type1"],
            "agent2": ["Component2", "Control2", "Variable2", "Zone1", "Type2"],
            "agent3": ["Component3", "Control3", "Variable3", "Zone1", "Type3"]
        }

        agent_ids, thermal_zone_ids, agents_actuators, agents_thermal_zones, agents_types = actuators_to_agents(agent_config)

        assert agent_ids == ["agent1", "agent2", "agent3"]
        assert thermal_zone_ids == ["Zone1"]
        assert agents_actuators == {
            "agent1": ("Component1", "Control1", "Variable1"),
            "agent2": ("Component2", "Control2", "Variable2"),
            "agent3": ("Component3", "Control3", "Variable3")
        }
        assert agents_thermal_zones == {
            "agent1": "Zone1",
            "agent2": "Zone1",
            "agent3": "Zone1"
        }
        assert agents_types == {
            "agent1": "Type1",
            "agent2": "Type2",
            "agent3": "Type3"
        }

    def test_actuators_to_agents_duplicate_thermal_zones(self):
        """
        Test actuators_to_agents with duplicate thermal zones to ensure they are handled correctly.
        """
        input_dict = {
            "agent1": ["actuator1", "type1", "key1", "zone1", "type1"],
            "agent2": ["actuator2", "type2", "key2", "zone1", "type2"]
        }
        agent_ids, thermal_zone_ids, _, _, _ = actuators_to_agents(input_dict)
        assert len(thermal_zone_ids) == 1, "Duplicate thermal zones should be removed"

    def test_actuators_to_agents_empty_input(self):
        """
        Test actuators_to_agents with an empty input dictionary.
        """
        with pytest.raises(ValueError, match="Input dictionary cannot be empty"):
            actuators_to_agents({})

    def test_actuators_to_agents_empty_string_values(self):
        """
        Test actuators_to_agents with empty string values in the input dictionary.
        """
        invalid_input = {
            "agent1": ["", "", "", "", ""],
            "agent2": ["actuator2", "type2", "key2", "zone2", "type2"]
        }
        with pytest.raises(ValueError, match="Input values cannot be empty strings"):
            actuators_to_agents(invalid_input)

    def test_actuators_to_agents_incorrect_value_format(self):
        """
        Test actuators_to_agents with incorrect value format in the input dictionary.
        """
        invalid_input = {
            "agent1": ["actuator1", "type1", "key1", "zone1"],  # Missing one element
            "agent2": ["actuator2", "type2", "key2", "zone2", "type2", "extra"]  # Extra element
        }
        with pytest.raises(AssertionError):
            actuators_to_agents(invalid_input)

    def test_actuators_to_agents_invalid_input_type(self):
        """
        Test actuators_to_agents with an invalid input type.
        """
        with pytest.raises(TypeError, match="Input must be a dictionary"):
            actuators_to_agents("invalid input")

    def test_actuators_to_agents_multiple_zones(self):
        """
        Test actuators_to_agents with multiple agents in different thermal zones.
        """
        agent_config = {
            "agent1": ["Component1", "Control1", "Variable1", "Zone1", "Type1"],
            "agent2": ["Component2", "Control2", "Variable2", "Zone2", "Type2"],
            "agent3": ["Component3", "Control3", "Variable3", "Zone1", "Type3"]
        }

        agent_ids, thermal_zone_ids, agents_actuators, agents_thermal_zones, agents_types = actuators_to_agents(agent_config)

        assert agent_ids == ["agent1", "agent2", "agent3"]
        assert set(thermal_zone_ids) == {"Zone1", "Zone2"}
        assert agents_actuators == {
            "agent1": ("Component1", "Control1", "Variable1"),
            "agent2": ("Component2", "Control2", "Variable2"),
            "agent3": ("Component3", "Control3", "Variable3")
        }
        assert agents_thermal_zones == {
            "agent1": "Zone1",
            "agent2": "Zone2",
            "agent3": "Zone1"
        }
        assert agents_types == {
            "agent1": "Type1",
            "agent2": "Type2",
            "agent3": "Type3"
        }

    def test_actuators_to_agents_non_list_values(self):
        """
        Test actuators_to_agents with non-list values in the input dictionary.
        """
        invalid_input = {
            "agent1": "invalid value",
            "agent2": ["actuator2", "type2", "key2", "zone2", "type2"]
        }
        with pytest.raises(TypeError, match="Dictionary values must be lists"):
            actuators_to_agents(invalid_input)

    def test_actuators_to_agents_non_string_keys(self):
        """
        Test actuators_to_agents with non-string keys in the input dictionary.
        """
        invalid_input = {
            1: ["actuator1", "type1", "key1", "zone1", "type1"],
            "agent2": ["actuator2", "type2", "key2", "zone2", "type2"]
        }
        with pytest.raises(TypeError, match="Dictionary keys must be strings"):
            actuators_to_agents(invalid_input)

    def test_actuators_to_agents_unique_thermal_zones(self):
        """
        Test actuators_to_agents function with unique thermal zones for each agent.
        """
        # Prepare test input
        agent_config = {
            "agent1": ["component1", "control_type1", "actuator_key1", "zone1", "type1"],
            "agent2": ["component2", "control_type2", "actuator_key2", "zone2", "type2"],
            "agent3": ["component3", "control_type3", "actuator_key3", "zone3", "type3"]
        }

        # Call the function
        agent_ids, thermal_zone_ids, agents_actuators, agents_thermal_zones, agents_types = actuators_to_agents(agent_config)

        # Assertions
        assert agent_ids == ["agent1", "agent2", "agent3"]
        assert thermal_zone_ids == ["zone1", "zone2", "zone3"]
        assert agents_actuators == {
            "agent1": ("component1", "control_type1", "actuator_key1"),
            "agent2": ("component2", "control_type2", "actuator_key2"),
            "agent3": ("component3", "control_type3", "actuator_key3")
        }
        assert agents_thermal_zones == {
            "agent1": "zone1",
            "agent2": "zone2",
            "agent3": "zone3"
        }
        assert agents_types == {
            "agent1": "type1",
            "agent2": "type2",
            "agent3": "type3"
        }

    def test_continuous_action_space_bounds(self):
        """
        Test that the returned Box has the correct lower and upper bounds.
        """
        action_space = continuous_action_space()
        assert np.all(action_space.low == 0.0)
        assert np.all(action_space.high == 1.0)

    def test_continuous_action_space_dtype(self):
        """
        Test that the returned Box has the correct data type.
        """
        action_space = continuous_action_space()
        assert action_space.dtype == np.float32

    def test_continuous_action_space_no_parameters(self):
        """
        Test that the function works correctly without any parameters.
        """
        try:
            continuous_action_space()
        except TypeError:
            pytest.fail("continuous_action_space() raised TypeError unexpectedly!")

    def test_continuous_action_space_return_type(self):
        """
        Test that continuous_action_space returns a Box object.
        """
        action_space = continuous_action_space()
        assert isinstance(action_space, Box)

    def test_continuous_action_space_returns_correct_box(self):
        """
        Test that continuous_action_space returns a Box with correct parameters.
        """
        action_space = continuous_action_space()

        assert isinstance(action_space, Box)
        assert action_space.low == 0.0
        assert action_space.high == 1.0
        assert action_space.shape == (1,)
        assert action_space.dtype == np.float32

    def test_continuous_action_space_shape(self):
        """
        Test that the returned Box has the correct shape.
        """
        action_space = continuous_action_space()
        assert action_space.shape == (1,)

    def test_continuous_action_space_with_parameters(self):
        """
        Test that the function raises a TypeError when parameters are provided.
        """
        with pytest.raises(TypeError):
            continuous_action_space(0, 1)

    def test_discrete_action_space_custom(self):
        """
        Test discrete_action_space with custom parameter
        """
        n = 5
        action_space = discrete_action_space(n)
        assert isinstance(action_space, Discrete)
        assert action_space.n == n

    def test_discrete_action_space_default(self):
        """
        Test discrete_action_space with default parameter
        """
        action_space = discrete_action_space()
        assert isinstance(action_space, Discrete)
        assert action_space.n == 2

    def test_discrete_action_space_negative(self):
        """
        Test discrete_action_space with negative parameter
        """
        with pytest.raises(ValueError):
            discrete_action_space(-1)

    def test_discrete_action_space_with_float_input(self):
        """
        Test discrete_action_space with a float input, which is an incorrect type.
        """
        with pytest.raises(TypeError):
            discrete_action_space(2.5)

    def test_discrete_action_space_with_large_input(self):
        """
        Test discrete_action_space with a very large input to check for potential overflow issues.
        """
        result = discrete_action_space(1000000)
        assert isinstance(result, Discrete)
        assert result.n == 1000000

    def test_discrete_action_space_with_negative_input(self):
        """
        Test discrete_action_space with a negative input, which is outside accepted bounds.
        """
        with pytest.raises(ValueError):
            discrete_action_space(-1)

    def test_discrete_action_space_with_none_input(self):
        """
        Test discrete_action_space with None input, which is an invalid input.
        """
        with pytest.raises(TypeError):
            discrete_action_space(None)

    def test_discrete_action_space_with_string_input(self):
        """
        Test discrete_action_space with a string input, which is an incorrect type.
        """
        with pytest.raises(TypeError):
            discrete_action_space("2")

    def test_discrete_action_space_with_zero(self):
        """
        Test discrete_action_space with n=0, which is an invalid input.
        """
        with pytest.raises(ValueError):
            discrete_action_space(0)

    def test_discrete_action_space_zero(self):
        """
        Test discrete_action_space with zero as parameter
        """
        with pytest.raises(ValueError):
            discrete_action_space(0)