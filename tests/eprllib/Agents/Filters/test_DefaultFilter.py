from eprllib.Agents.Filters.DefaultFilter import DefaultFilter
from typing import Any, Dict
import numpy as np
import pytest

class TestDefaultfilter:

    def test___init___1(self):
        """
        Test the initialization of DefaultFilter with a basic filter_fn_config.

        This test verifies that the DefaultFilter can be instantiated with a simple
        configuration dictionary and that it correctly initializes its superclass.
        """
        filter_fn_config: Dict[str, Any] = {"test_key": "test_value"}
        default_filter = DefaultFilter("agent_name",filter_fn_config)
        assert isinstance(default_filter, DefaultFilter)
        assert default_filter.filter_fn_config == filter_fn_config

    def test_get_filtered_obs_1(self):
        """
        Test that get_filtered_obs returns a numpy array of float64 values
        containing the values from the agent_states dictionary.
        """
        # Create a DefaultFilter instance
        filter_fn_config = {}
        default_filter = DefaultFilter("agent_name",filter_fn_config)

        # Prepare test inputs
        env_config = {}
        agent_states = {"state1": 1.0, "state2": 2.0, "state3": 3.0}

        # Call the method under test
        result = default_filter.get_filtered_obs(env_config, agent_states)

        # Assert the result is a numpy array of float64 values
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float64

        # Assert the result contains the correct values
        expected = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        np.testing.assert_array_equal(result, expected)

    # def test_get_filtered_obs_with_empty_agent_states(self):
    #     """
    #     Test the get_filtered_obs method with an empty agent_states dictionary.
    #     This tests the edge case of having no agent states to filter.
    #     """
    #     filter_fn_config = {}
    #     default_filter = DefaultFilter(filter_fn_config)
    #     env_config = {}
    #     agent_states = {}

    #     with pytest.raises(ValueError) as excinfo:
    #         default_filter.get_filtered_obs(env_config, agent_states)
            
    #     assert str(excinfo.value) == "DefaultFilter: The agent_states dictionary is empty"


    # def test_get_filtered_obs_with_non_numeric_values(self):
    #     """
    #     Test the get_filtered_obs method with non-numeric values in agent_states.
    #     This tests the edge case of handling non-numeric data types.
    #     """
    #     filter_fn_config = {}
    #     default_filter = DefaultFilter(filter_fn_config)
    #     env_config = {}
    #     agent_states = {"state1": "string", "state2": True, "state3": None}

    #     with pytest.raises(ValueError) as excinfo:
    #         default_filter.get_filtered_obs(env_config, agent_states)
        
    #     assert str(excinfo.value) == "DefaultFilter: All values in agent_states must be numeric"


    def test_init_with_empty_dict(self):
        """
        Test initializing DefaultFilter with an empty dictionary.
        This is a valid edge case as the method accepts any dictionary without validation.
        """
        empty_config: Dict[str, Any] = {}
        filter_instance = DefaultFilter("agent_name",empty_config)
        assert isinstance(filter_instance, DefaultFilter)
        assert filter_instance.filter_fn_config == empty_config
