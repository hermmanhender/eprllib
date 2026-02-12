from eprllib.Agents.Filters.FullySharedParametersFilter import FullySharedParametersFilter
from typing import Any, Dict
import numpy as np
import pytest

class TestFullysharedparametersfilter:

    def test___init___1(self):
        """
        Test the initialization of FullySharedParametersFilter.

        This test verifies that the FullySharedParametersFilter is correctly initialized
        with the provided filter_fn_config and that the agent_name attribute is set to None.
        """
        filter_fn_config: Dict[str, Any] = {}
        filter_instance = FullySharedParametersFilter(filter_fn_config)

        assert isinstance(filter_instance, FullySharedParametersFilter)
        assert filter_instance.agent_name is None

    def test_get_filtered_obs_1(self):
        """
        Test that get_filtered_obs correctly filters out actuator states when self.agent_name is None.

        This test verifies that:
        1. The method correctly identifies the agent name from the agent_states dictionary.
        2. Actuator states are removed from the observation.
        3. The returned observation is a numpy array of float64 values.
        4. The returned observation contains only the non-actuator state values.
        """
        # Setup
        filter_fn_config = {}
        env_config = {
            "agents_config": {
                "agent1": {
                    "action": {
                        "actuators": [
                            ["actuator1", "type1", "subtype1"],
                            ["actuator2", "type2", "subtype2"]
                        ]
                    }
                }
            }
        }
        agent_states = {
            "agent1: state1": 1.0,
            "agent1: state2": 2.0,
            "agent1: actuator1: type1: subtype1": 3.0,
            "agent1: actuator2: type2: subtype2": 4.0
        }

        # Create filter instance
        filter_instance = FullySharedParametersFilter(filter_fn_config)

        # Call the method under test
        result = filter_instance.get_filtered_obs(env_config, agent_states)

        # Assertions
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float64
        np.testing.assert_array_equal(result, np.array([1.0, 2.0], dtype=np.float64))
        assert filter_instance.agent_name == "agent1"

    def test_get_filtered_obs_2(self):
        """
        Test the get_filtered_obs method when self.agent_name is not None.
        This test verifies that the method correctly filters out actuator states
        and returns a numpy array of the remaining agent states.
        """
        filter_fn_config = {}
        filter_instance = FullySharedParametersFilter(filter_fn_config)
        filter_instance.agent_name = "agent1"

        env_config = {
            "agents_config": {
                "agent1": {
                    "action": {
                        "actuators": [
                            ["actuator1", "type1", "subtype1"],
                            ["actuator2", "type2", "subtype2"]
                        ]
                    }
                }
            }
        }

        agent_states = {
            "agent1: state1": 1.0,
            "agent1: state2": 2.0,
            "agent1: actuator1: type1: subtype1": 3.0,
            "agent1: actuator2: type2: subtype2": 4.0,
            "agent1: state3": 5.0
        }

        result = filter_instance.get_filtered_obs(env_config, agent_states)

        expected_result = np.array([1.0, 2.0, 5.0], dtype='float64')
        np.testing.assert_array_equal(result, expected_result)

    def test_get_filtered_obs_empty_agent_states(self):
        """
        Test the get_filtered_obs method with empty agent_states.
        This tests the edge case where no agent states are provided, which should result in an empty numpy array.
        """
        filter_fn_config = {}
        filter_instance = FullySharedParametersFilter(filter_fn_config)

        env_config = {"agents_config": {"agent1": {"action": {"actuators": []}}}}
        agent_states = {}

        with pytest.raises(ValueError):
            filter_instance.get_filtered_obs(env_config, agent_states)
        

    def test_get_filtered_obs_missing_agent_config(self):
        """
        Test the get_filtered_obs method when the agent configuration is missing from the env_config.
        This tests the edge case where the expected agent configuration is not present in the environment config.
        """
        filter_fn_config = {}
        filter_instance = FullySharedParametersFilter(filter_fn_config)

        env_config = {"agents_config": {}}
        agent_states = {"agent1: state1": 1.0, "agent1: state2": 2.0}

        with pytest.raises(KeyError):
            filter_instance.get_filtered_obs(env_config, agent_states)

    def test_get_filtered_obs_no_actuators(self):
        """
        Test the get_filtered_obs method when there are no actuators configured for the agent.
        This tests the edge case where the agent has no actuators, so no filtering should occur.
        """
        filter_fn_config = {}
        filter_instance = FullySharedParametersFilter(filter_fn_config)

        env_config = {"agents_config": {"agent1": {"action": {"actuators": []}}}}
        agent_states = {"agent1: state1": 1.0, "agent1: state2": 2.0}

        result = filter_instance.get_filtered_obs(env_config, agent_states)
        expected = np.array([1.0, 2.0], dtype=np.float64)
        np.testing.assert_array_equal(result, expected)

    def test_init_with_empty_dict(self):
        """
        Test initialization of FullySharedParametersFilter with an empty dictionary.
        This is to verify that the method handles the edge case of an empty input correctly.
        """
        filter_fn_config = {}
        filter_instance = FullySharedParametersFilter(filter_fn_config)
        assert filter_instance.agent_name is None
        assert isinstance(filter_instance, FullySharedParametersFilter)
