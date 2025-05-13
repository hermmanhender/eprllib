import pytest
import numpy as np
from eprllib.Agents.Filters.BaseFilter import BaseFilter
from eprllib.Agents.Filters.DefaultFilter import DefaultFilter
from eprllib.Agents.Filters.FullySharedParametersFilter import FullySharedParametersFilter

class TestBaseFilter:
    def test_base_filter_is_abstract(self):
        """Test that BaseFilter cannot be instantiated directly"""
        with pytest.raises(TypeError):
            BaseFilter()

    def test_base_filter_requires_filter_method(self):
        """Test that subclasses must implement filter method"""
        class InvalidFilter(BaseFilter):
            pass
        
        with pytest.raises(TypeError):
            InvalidFilter()

class TestDefaultFilter:
    def test_default_filter_initialization(self):
        """Test DefaultFilter initialization"""
        filter = DefaultFilter()
        assert isinstance(filter, DefaultFilter)
        assert isinstance(filter, BaseFilter)

    def test_default_filter_returns_numpy_array(self):
        """Test that DefaultFilter returns observation as numpy array"""
        filter = DefaultFilter({})
        env_config = {"some_config": "value"}
        agent_states = {"temp": 20.0, "humidity": 50.0}
        
        filtered_obs = filter.get_filtered_obs(env_config, agent_states)
        
        assert isinstance(filtered_obs, np.ndarray)
        assert filtered_obs.dtype == np.float32
        assert np.array_equal(filtered_obs, np.array([20.0, 50.0], dtype=np.float32))

class TestFullySharedParametersFilter:
    def test_fully_shared_parameters_filter_initialization(self):
        """Test FullySharedParametersFilter initialization"""
        filter = FullySharedParametersFilter({})
        assert isinstance(filter, FullySharedParametersFilter)
        assert isinstance(filter, BaseFilter)

    def test_fully_shared_parameters_filter_with_empty_observation(self):
        """Test FullySharedParametersFilter with empty observation"""
        filter = FullySharedParametersFilter({})
        env_config = {
            "agents_config": {
                "agent1": {
                    "action": {
                        "actuators": []
                    }
                }
            }
        }
        agent_states = {"agent1:temperature": 25.0}
        filtered_obs = filter.get_filtered_obs(env_config, agent_states)
        assert isinstance(filtered_obs, np.ndarray)
        assert filtered_obs.dtype == np.float32
        assert np.array_equal(filtered_obs, np.array([25.0], dtype=np.float32))

    def test_fully_shared_parameters_filter_with_actuator_state(self):
        """Test filtering with actuator states that should be removed"""
        filter = FullySharedParametersFilter({})
        env_config = {
            "agents_config": {
                "agent1": {
                    "action": {
                        "actuators": [
                            ["zone1", "hvac", "heating"]
                        ]
                    }
                }
            }
        }
        agent_states = {
            "agent1:temperature": 25.0,
            "agent1:zone1_hvac_heating": 1.0
        }
        filtered_obs = filter.get_filtered_obs(env_config, agent_states)
        assert isinstance(filtered_obs, np.ndarray)
        assert filtered_obs.dtype == np.float32
        expected = np.array([25.0], dtype=np.float32)
        assert np.array_equal(filtered_obs, expected)

    def test_fully_shared_parameters_filter_with_multiple_actuators(self):
        """Test filtering with multiple actuator states that should be removed"""
        filter = FullySharedParametersFilter({})
        env_config = {
            "agents_config": {
                "agent1": {
                    "action": {
                        "actuators": [
                            ["zone1", "hvac", "heating"],
                            ["zone1", "hvac", "cooling"]
                        ]
                    }
                }
            }
        }
        agent_states = {
            "agent1:temperature": 25.0,
            "agent1:humidity": 50.0,
            "agent1:zone1_hvac_heating": 1.0,
            "agent1:zone1_hvac_cooling": 0.0
        }
        filtered_obs = filter.get_filtered_obs(env_config, agent_states)
        assert isinstance(filtered_obs, np.ndarray)
        assert filtered_obs.dtype == np.float32
        expected = np.array([50.0, 25.0], dtype=np.float32)
        assert np.array_equal(filtered_obs, expected)

    def test_fully_shared_parameters_filter_auto_detect_agent(self):
        """Test that filter correctly auto-detects agent name from state keys"""
        filter = FullySharedParametersFilter({})
        env_config = {
            "agents_config": {
                "agent1": {
                    "action": {
                        "actuators": [
                            ["zone1", "hvac", "heating"]
                        ]
                    }
                }
            }
        }
        agent_states = {
            "agent1:temperature": 25.0,
            "agent1:zone1_hvac_heating": 1.0
        }
        filtered_obs = filter.get_filtered_obs(env_config, agent_states)
        assert isinstance(filtered_obs, np.ndarray)
        expected = np.array([25.0], dtype=np.float32)
        assert np.array_equal(filtered_obs, expected)

    def test_fully_shared_parameters_filter_missing_actuator(self):
        """Test filtering when an actuator state is missing from observations"""
        filter = FullySharedParametersFilter({})
        env_config = {
            "agents_config": {
                "agent1": {
                    "action": {
                        "actuators": [
                            ["zone1", "hvac", "heating"],
                            ["zone1", "hvac", "cooling"]
                        ]
                    }
                }
            }
        }
        # Missing zone1_hvac_cooling actuator state
        agent_states = {
            "agent1:temperature": 25.0,
            "agent1:zone1_hvac_heating": 1.0
        }
        filtered_obs = filter.get_filtered_obs(env_config, agent_states)
        assert isinstance(filtered_obs, np.ndarray)
        expected = np.array([25.0], dtype=np.float32)
        assert np.array_equal(filtered_obs, expected)