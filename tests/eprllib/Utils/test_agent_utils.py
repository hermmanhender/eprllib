from eprllib.Utils.agent_utils import config_validation
from typing import Tuple
import pytest

class TestAgentUtils:

    def test_config_validation_2(self):
        """
        Tests config_validation function when a key is present, the expected type is a tuple,
        but the value is not a tuple, has incorrect number of elements, or contains an item
        of incorrect type.
        """
        config = {
            'test_key': [1, 'string']
        }
        required_keys = {
            'test_key': Tuple[int, int, str]
        }

        with pytest.raises(TypeError) as exc_info:
            config_validation(config, required_keys)

        assert str(exc_info.value) == "The key 'test_key' must be a tuple, but got a list"

        # Change config to have a tuple with incorrect number of elements
        config['test_key'] = (1, 'string')

        with pytest.raises(ValueError) as exc_info:
            config_validation(config, required_keys)

        assert str(exc_info.value) == "The key 'test_key' must be a tuple with 3 elements, but has 2"

        # Change config to have a tuple with correct number of elements but incorrect type
        config['test_key'] = (1, 2, 2)

        with pytest.raises(TypeError) as exc_info:
            config_validation(config, required_keys)

        assert str(exc_info.value) == "The element 2 in key 'test_key' must be a str, but got a int"

    def test_config_validation_incorrect_tuple_length(self):
        """
        Test config_validation when a tuple value in the config has an incorrect length.
        This should raise a ValueError.
        """
        config = {"key1": (1, 2, 3)}
        required_keys = {"key1": Tuple[int, int]}
        with pytest.raises(ValueError) as context:
            config_validation(config, required_keys)
            
        assert str(context.value) == "The key 'key1' must be a tuple with 2 elements, but has 3"

    def test_config_validation_incorrect_tuple_type(self):
        """
        Test config_validation when a tuple value in the config has an incorrect type.
        This should raise a TypeError.
        """
        config = {"key1": (1, "two")}
        required_keys = {"key1": Tuple[int, int]}
        with pytest.raises(TypeError) as context:
            config_validation(config, required_keys)
        
        assert str(context.value) == "The element 1 in key 'key1' must be a int, but got a str"

    def test_config_validation_incorrect_type(self):
        """
        Test config_validation when a value in the config has an incorrect type.
        This should raise a TypeError.
        """
        config = {"key1": "value1", "key2": "not_an_int"}
        required_keys = {"key1": str, "key2": int}
        with pytest.raises(TypeError) as context:
            config_validation(config, required_keys)
        
        assert str(context.value) == "The key 'key2' expects the type int, but got str"

    def test_config_validation_missing_key(self):
        """
        Test config_validation when a required key is missing from the config.
        This test covers the path where a key is not in the config,
        and the code raises a ValueError.
        """
        config = {"key1": "value1"}
        required_keys = {"key1": str, "key2": int}

        with pytest.raises(ValueError) as excinfo:
            config_validation(config, required_keys)

        assert "The following key is missing: 'key2'" in str(excinfo.value)

    def test_config_validation_missing_key_2(self):
        """
        Test config_validation when a required key is missing from the config.
        This should raise a ValueError.
        """
        config = {"key1": "value1"}
        required_keys = {"key1": str, "key2": int}
        with pytest.raises(ValueError) as context:
            config_validation(config, required_keys)
        
        assert str(context.value) == "The following key is missing: 'key2'"

    def test_config_validation_missing_key_and_tuple_mismatch(self):
        """
        Test config_validation when a key is missing, and a tuple value doesn't match the expected type.

        This test covers the following scenarios:
        1. A required key is missing from the config
        2. A value is a tuple when expected
        3. The tuple's length doesn't match the expected length
        4. The tuple's elements have the correct types
        """
        config = {
            'existing_key': 'value',
            'tuple_key': (1, 'string', True)
        }
        required_keys = {
            'missing_key': str,
            'tuple_key': Tuple[int, str, bool, float]
        }

        with pytest.raises(ValueError) as excinfo:
            config_validation(config, required_keys)
        assert "The following key is missing: 'missing_key'" in str(excinfo.value)

        # Update config to include the missing key
        config['missing_key'] = 'value'

        with pytest.raises(ValueError) as excinfo:
            config_validation(config, required_keys)
        assert "The key 'tuple_key' must be a tuple with 4 elements, but has 3" in str(excinfo.value)

    def test_config_validation_missing_key_and_tuple_type_mismatch(self):
        """
        Test config_validation with missing key, tuple type mismatch, and incorrect tuple element type.

        This test covers the following scenarios:
        1. A required key is missing from the config.
        2. A value that should be a tuple is not a tuple.
        3. A tuple value has an incorrect number of elements.
        4. An element in a tuple has an incorrect type.
        """
        config = {
            'existing_key': 'value',
            'tuple_key': (1, 'string'),
            'wrong_tuple_key': [1, 2, 3],
            'wrong_tuple_length': (1, 2),
            'wrong_tuple_type': (1, 2, 'string')
        }
        required_keys = {
            'missing_key': str,
            'tuple_key': Tuple[int, str],
            'wrong_tuple_key': Tuple[int, int],
            'wrong_tuple_length': Tuple[int, int, int],
            'wrong_tuple_type': Tuple[int, int, int]
        }

        with pytest.raises(ValueError, match="The following key is missing: 'missing_key'"):
            config_validation(config, required_keys)

        config['missing_key'] = 'value'
        with pytest.raises(TypeError, match="The key 'wrong_tuple_key' must be a tuple, but got a list"):
            config_validation(config, required_keys)

        config['wrong_tuple_key'] = (1, 2)
        with pytest.raises(ValueError, match="The key 'wrong_tuple_length' must be a tuple with 3 elements, but has 2"):
            config_validation(config, required_keys)

        config['wrong_tuple_length'] = (1, 2, 3)
        with pytest.raises(TypeError, match="The element 2 in key 'wrong_tuple_type' must be a int, but got a str"):
            config_validation(config, required_keys)

    def test_config_validation_missing_key_and_wrong_type(self):
        """
        Test config_validation when a required key is missing and another key has an incorrect type.

        This test covers the scenario where:
        1. A required key is not present in the config dictionary.
        2. Another key is present but its value has an incorrect type.

        The test expects ValueError to be raised for the missing key.
        """
        config = {
            'existing_key': 'string_value'
        }
        required_keys = {
            'missing_key': int,
            'existing_key': int
        }

        with pytest.raises(ValueError) as excinfo:
            config_validation(config, required_keys)

        assert "The following key is missing: 'missing_key'" in str(excinfo.value)

    def test_config_validation_non_tuple_for_tuple_type(self):
        """
        Test config_validation when a non-tuple value is provided for a tuple type.
        This should raise a TypeError.
        """
        config = {"key1": "not_a_tuple"}
        required_keys = {"key1": Tuple[int, int]}
        with pytest.raises(TypeError) as context:
            config_validation(config, required_keys)
        
        assert str(context.value) == "The key 'key1' must be a tuple, but got a str"

    def test_config_validation_tuple_type_mismatch(self):
        """
        Test config_validation when a tuple value contains an element of incorrect type.

        This test covers the following path:
        - The key exists in the config
        - The expected type is a tuple
        - The value is a tuple
        - The tuple length matches the expected length
        - One element in the tuple has an incorrect type
        """
        config = {
            'test_key': (1, 'string', 3.14)
        }
        required_keys = {
            'test_key': Tuple[int, str, int]
        }

        with pytest.raises(TypeError) as context:
            config_validation(config, required_keys)

        assert str(context.value) == "The element 2 in key 'test_key' must be a int, but got a float"

    def test_config_validation_tuple_type_mismatch_2(self):
        """
        Test config_validation when a key is missing, the expected type is a tuple,
        the value is a tuple but with incorrect length, and contains an item of incorrect type.
        """
        config = {
            'key1': (1, 'string', 3.14)
        }
        required_keys = {
            'key1': Tuple[int, str, float],
            'missing_key': str
        }

        with pytest.raises(ValueError) as excinfo:
            config_validation(config, required_keys)
        assert "The following key is missing: 'missing_key'" in str(excinfo.value)

        config['missing_key'] = 'value'
        config['key1'] = (1, 'string', '3.14', 4)  # Incorrect length and type

        with pytest.raises(ValueError) as excinfo:
            config_validation(config, required_keys)
        assert "The key 'key1' must be a tuple with 3 elements, but has 4" in str(excinfo.value)

        config['key1'] = (1, 'string', '3.14')  # Correct length but incorrect type for last element

        with pytest.raises(TypeError) as excinfo:
            config_validation(config, required_keys)
        assert "The element 2 in key 'key1' must be a float, but got a str" in str(excinfo.value)
