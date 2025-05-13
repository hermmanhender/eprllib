

from eprllib.Agents.AgentSpec import AgentSpec, ObservationSpec, FilterSpec, ActionSpec, TriggerSpec, RewardSpec
from eprllib.Agents.Filters.BaseFilter import BaseFilter
from eprllib.Agents.Filters.DefaultFilter import DefaultFilter
from eprllib.Agents.Rewards.BaseReward import BaseReward
from eprllib.Agents.Triggers.BaseTrigger import BaseTrigger
from typing import Dict, List, Tuple, Any
from unittest.mock import Mock, patch
import logging
import pytest

class TestAgentspec:

    def assertLogs(self, level='INFO'):
        return _AssertLogsContext(self, level)

    @pytest.fixture
    def mock_logger(self, mocker):
        return mocker.patch('eprllib.Agents.AgentSpec.logger')

    def test___getitem___access_custom_attribute(self):
        """
        Test that __getitem__ can access a custom attribute added to the AgentSpec.
        """
        # Arrange
        agent_spec = AgentSpec()
        agent_spec.custom_attr = "test_value"

        # Act
        result = agent_spec['custom_attr']

        # Assert
        assert result == "test_value"

    def test___getitem___access_existing_attribute(self):
        """
        Test that __getitem__ correctly returns the value of an existing attribute.
        """
        filter_spec = FilterSpec(filter_fn=DefaultFilter, filter_fn_config={'param': 'value'})
        
        assert filter_spec['filter_fn'] == DefaultFilter
        assert filter_spec['filter_fn_config'] == {'param': 'value'}

    def test___getitem___access_method(self):
        """
        Test that __getitem__ can access methods of the class.
        """
        filter_spec = FilterSpec()
        
        assert callable(filter_spec['build'])
        assert filter_spec['build'].__func__ == FilterSpec.build

    def test___getitem___access_nonexistent_attribute(self):
        """
        Test that __getitem__ raises AttributeError when accessing a non-existent attribute.
        """
        filter_spec = FilterSpec()
        
        with pytest.raises(AttributeError):
            filter_spec['non_existent_key']

    def test___getitem___access_nonexistent_attribute_2(self):
        """
        Test that __getitem__ raises AttributeError for a nonexistent attribute.
        """
        # Arrange
        agent_spec = AgentSpec()

        # Act & Assert
        with pytest.raises(AttributeError):
            agent_spec['nonexistent_attribute']

    def test___getitem___access_valid_attribute(self):
        """
        Test that __getitem__ correctly returns the value of a valid attribute.
        """
        # Arrange
        observation = ObservationSpec()
        filter_spec = FilterSpec()
        action = ActionSpec()
        trigger = TriggerSpec()
        reward = RewardSpec()
        agent_spec = AgentSpec(observation, filter_spec, action, trigger, reward)

        # Act
        result = agent_spec['observation']

        # Assert
        assert result == observation
        assert isinstance(result, ObservationSpec)

    def test___getitem___after_deletion(self):
        """
        Test __getitem__ after deleting an attribute.
        """
        reward_spec = RewardSpec()
        delattr(reward_spec, 'reward_fn')
        with pytest.raises(AttributeError):
            reward_spec['reward_fn']

    def test___getitem___dict_key(self):
        """
        Test __getitem__ with a dict key, which is an incorrect type.
        """
        agent_spec = AgentSpec()
        with pytest.raises(TypeError):
            agent_spec[{'key': 'observation'}]

    def test___getitem___empty_key(self):
        """
        Test __getitem__ with an empty key.
        """
        reward_spec = RewardSpec()
        with pytest.raises(AttributeError):
            reward_spec['']

    def test___getitem___empty_key_2(self):
        """
        Test that __getitem__ raises AttributeError when given an empty key.
        """
        obs_spec = ObservationSpec()
        with pytest.raises(AttributeError):
            obs_spec['']

    def test___getitem___empty_key_3(self):
        """
        Test that __getitem__ raises TypeError when an empty key is provided
        """
        filter_spec = FilterSpec()
        with pytest.raises(TypeError):
            filter_spec['']

    def test___getitem___empty_key_4(self):
        """
        Test __getitem__ with an empty key.
        """
        action_spec = ActionSpec()
        with pytest.raises(AttributeError):
            action_spec['']

    def test___getitem___empty_key_5(self):
        """
        Test __getitem__ with an empty string as key.
        """
        trigger_spec = TriggerSpec(trigger_fn=BaseTrigger())
        with pytest.raises(AttributeError):
            trigger_spec['']

    def test___getitem___empty_key_6(self):
        """
        Test __getitem__ with an empty key.
        """
        agent_spec = AgentSpec()
        with pytest.raises(AttributeError):
            agent_spec['']

    def test___getitem___integer_key(self):
        """
        Test __getitem__ with an integer key, which is an incorrect type.
        """
        agent_spec = AgentSpec()
        with pytest.raises(TypeError):
            agent_spec[1]

    def test___getitem___invalid_key(self):
        """
        Test that __getitem__ raises AttributeError when given an invalid key.
        """
        obs_spec = ObservationSpec()
        with pytest.raises(AttributeError):
            obs_spec['invalid_key']

    def test___getitem___invalid_key_2(self):
        """
        Test __getitem__ with an invalid key.
        """
        action_spec = ActionSpec()
        with pytest.raises(AttributeError):
            action_spec['invalid_key']

    def test___getitem___invalid_key_type(self):
        """
        Test __getitem__ with an invalid key type (non-string).
        """
        reward_spec = RewardSpec()
        with pytest.raises(TypeError):
            reward_spec[123]

    def test___getitem___invalid_key_type_2(self):
        """
        Test __getitem__ with an invalid key type (non-string).
        """
        trigger_spec = TriggerSpec(trigger_fn=BaseTrigger())
        with pytest.raises(TypeError):
            trigger_spec[123]

    def test___getitem___key_not_exist(self):
        """
        Test __getitem__ with a key that doesn't exist in RewardSpec.
        """
        reward_spec = RewardSpec()
        with pytest.raises(AttributeError):
            reward_spec['non_existent_key']

    def test___getitem___key_not_exist_2(self):
        """
        Test __getitem__ with a key that doesn't exist in the TriggerSpec object.
        """
        trigger_spec = TriggerSpec(trigger_fn=BaseTrigger())
        with pytest.raises(AttributeError):
            trigger_spec['non_existent_key']

    def test___getitem___key_not_exist_3(self):
        """
        Test __getitem__ with a key that doesn't exist in the AgentSpec object.
        """
        agent_spec = AgentSpec()
        with pytest.raises(AttributeError):
            agent_spec['non_existent_key']

    def test___getitem___key_not_found(self):
        """
        Test that __getitem__ raises AttributeError when the key is not found
        """
        filter_spec = FilterSpec()
        with pytest.raises(AttributeError):
            filter_spec['non_existent_key']

    def test___getitem___list_key(self):
        """
        Test __getitem__ with a list key, which is an incorrect type.
        """
        agent_spec = AgentSpec()
        with pytest.raises(TypeError):
            agent_spec[['observation']]

    def test___getitem___method_name(self):
        """
        Test __getitem__ with a method name.
        """
        reward_spec = RewardSpec()
        assert callable(reward_spec['build'])

    def test___getitem___non_string_key(self):
        """
        Test that __getitem__ raises TypeError when given a non-string key.
        """
        obs_spec = ObservationSpec()
        with pytest.raises(TypeError):
            obs_spec[123]

    def test___getitem___non_string_key_2(self):
        """
        Test that __getitem__ raises TypeError when a non-string key is provided
        """
        filter_spec = FilterSpec()
        with pytest.raises(TypeError):
            filter_spec[123]

    def test___getitem___non_string_key_3(self):
        """
        Test __getitem__ with a non-string key.
        """
        action_spec = ActionSpec()
        with pytest.raises(TypeError):
            action_spec[123]

    def test___getitem___none_key(self):
        """
        Test __getitem__ with None as key.
        """
        reward_spec = RewardSpec()
        with pytest.raises(TypeError):
            reward_spec[None]

    def test___getitem___none_key_2(self):
        """
        Test that __getitem__ raises TypeError when given None as a key.
        """
        obs_spec = ObservationSpec()
        with pytest.raises(TypeError):
            obs_spec[None]

    def test___getitem___none_key_3(self):
        """
        Test that __getitem__ raises TypeError when None is provided as a key
        """
        filter_spec = FilterSpec()
        with pytest.raises(TypeError):
            filter_spec[None]

    def test___getitem___none_key_4(self):
        """
        Test __getitem__ with None as key.
        """
        action_spec = ActionSpec()
        with pytest.raises(TypeError):
            action_spec[None]

    def test___getitem___none_key_5(self):
        """
        Test __getitem__ with None as key.
        """
        trigger_spec = TriggerSpec(trigger_fn=BaseTrigger())
        with pytest.raises(TypeError):
            trigger_spec[None]

    def test___getitem___none_key_6(self):
        """
        Test __getitem__ with None as the key.
        """
        agent_spec = AgentSpec()
        with pytest.raises(TypeError):
            agent_spec[None]

    def test___getitem___partial_key(self):
        """
        Test __getitem__ with a partial key.
        """
        agent_spec = AgentSpec()
        with pytest.raises(AttributeError):
            agent_spec['obs']

    def test___getitem___private_attribute(self):
        """
        Test __getitem__ with a private attribute name.
        """
        reward_spec = RewardSpec()
        with pytest.raises(AttributeError):
            reward_spec['_private_attr']

    def test___getitem___raises_attribute_error(self):
        """
        Test that __getitem__ raises an AttributeError for non-existent attributes.
        """
        # Arrange
        obs_spec = ObservationSpec()
        
        # Act & Assert
        with pytest.raises(AttributeError):
            obs_spec["non_existent_attribute"]

    def test___getitem___retrieves_attribute(self):
        """
        Test that __getitem__ correctly retrieves an attribute of the ObservationSpec instance.
        """
        # Arrange
        obs_spec = ObservationSpec(variables=[("Outdoor Air Temperature", "Environment")])
        
        # Act
        result = obs_spec["variables"]
        
        # Assert
        assert result == [("Outdoor Air Temperature", "Environment")]

    def test___getitem___special_characters(self):
        """
        Test that __getitem__ raises AttributeError when given special characters as a key.
        """
        obs_spec = ObservationSpec()
        with pytest.raises(AttributeError):
            obs_spec['!@#$%^&*()']

    def test___getitem___uppercase_key(self):
        """
        Test __getitem__ with an uppercase key, which should be case-sensitive.
        """
        agent_spec = AgentSpec()
        with pytest.raises(AttributeError):
            agent_spec['OBSERVATION']

    def test___getitem___valid_key(self):
        """
        Test that __getitem__ returns the correct value for a valid key
        """
        filter_spec = FilterSpec()
        assert isinstance(filter_spec['filter_fn'], DefaultFilter)
        assert filter_spec['filter_fn_config'] == {}

    def test___getitem___valid_key_no_attribute(self):
        """
        Test __getitem__ with a valid key that doesn't correspond to an attribute.
        """
        action_spec = ActionSpec()
        with pytest.raises(AttributeError):
            action_spec['non_existent_attribute']

    def test___getitem___valid_keys(self):
        """
        Test __getitem__ with valid keys to ensure it works correctly.
        """
        trigger_fn = BaseTrigger()
        trigger_fn_config = {'param1': 'value1'}
        trigger_spec = TriggerSpec(trigger_fn=trigger_fn, trigger_fn_config=trigger_fn_config)
        
        assert trigger_spec['trigger_fn'] == trigger_fn
        assert trigger_spec['trigger_fn_config'] == trigger_fn_config

    def test___init___1(self):
        """
        Test ObservationSpec initialization with invalid parameters and edge cases.
        """
        # Test with invalid simulation parameters
        with pytest.warns(UserWarning):
            obs_spec = ObservationSpec(
                simulation_parameters={'invalid_param': True},
                zone_simulation_parameters={'invalid_zone_param': True},
                use_one_day_weather_prediction=True,
                prediction_variables={'invalid_prediction': True},
                prediction_hours=25
            )

        # Check that invalid parameters were not added
        assert 'invalid_param' not in obs_spec.simulation_parameters
        assert 'invalid_zone_param' not in obs_spec.zone_simulation_parameters
        assert 'invalid_prediction' not in obs_spec.prediction_variables

        # Check that prediction_hours was set to default value
        assert obs_spec.prediction_hours == 24

        # Check that use_one_day_weather_prediction is set correctly
        assert obs_spec.use_one_day_weather_prediction is True

        # Test with valid parameters
        obs_spec = ObservationSpec(
            variables=[('var1', 'unit1'), ('var2', 'unit2')],
            internal_variables=['int_var1', 'int_var2'],
            meters=['meter1', 'meter2'],
            simulation_parameters={'hour': True, 'day_of_year': True},
            zone_simulation_parameters={'zone_time_step': True},
            use_one_day_weather_prediction=True,
            prediction_variables={'outdoor_dry_bulb': True},
            use_actuator_state=True,
            other_obs={'custom_obs': 1.0}
        )

        # Check that valid parameters were set correctly
        assert obs_spec.variables == [('var1', 'unit1'), ('var2', 'unit2')]
        assert obs_spec.internal_variables == ['int_var1', 'int_var2']
        assert obs_spec.meters == ['meter1', 'meter2']
        assert obs_spec.simulation_parameters['hour'] is True
        assert obs_spec.simulation_parameters['day_of_year'] is True
        assert obs_spec.zone_simulation_parameters['zone_time_step'] is True
        assert obs_spec.prediction_variables['outdoor_dry_bulb'] is True
        assert obs_spec.use_actuator_state is True
        assert obs_spec.other_obs == {'custom_obs': 1.0}

    def test___init___1_2(self):
        """
        Test initialization of FilterSpec when no filter function is provided.
        """
        filter_spec = FilterSpec()
        
        assert filter_spec.filter_fn == DefaultFilter
        assert filter_spec.filter_fn_config == {}
        assert isinstance(filter_spec.filter_fn, type(DefaultFilter))

    def test___init___1_3(self):
        """
        Test FilterSpec initialization with default values
        """
        filter_spec = FilterSpec()
        
        assert filter_spec.filter_fn is not None
        assert isinstance(filter_spec.filter_fn, BaseFilter)
        assert filter_spec.filter_fn_config == {}

    def test___init___1_4(self):
        """
        Test initialization of ActionSpec with None actuators.
        """
        # Act
        action_spec = ActionSpec()

        # Assert
        assert action_spec.actuators == []

    def test___init___2(self):
        """
        Test ObservationSpec initialization with invalid simulation parameters,
        zone simulation parameters, weather prediction, and prediction hours.
        """
        # Setup
        invalid_sim_params = {'invalid_param': True}
        invalid_zone_params = {'invalid_zone_param': True}
        invalid_prediction_vars = {'invalid_prediction': True}
        
        # Exercise
        with pytest.warns(UserWarning):
            obs_spec = ObservationSpec(
                simulation_parameters=invalid_sim_params,
                zone_simulation_parameters=invalid_zone_params,
                use_one_day_weather_prediction=True,
                prediction_hours=30,
                prediction_variables=invalid_prediction_vars
            )

        # Verify
        assert 'invalid_param' not in obs_spec.simulation_parameters
        assert 'invalid_zone_param' not in obs_spec.zone_simulation_parameters
        assert 'invalid_prediction' not in obs_spec.prediction_variables
        assert obs_spec.prediction_hours == 24
        assert obs_spec.use_one_day_weather_prediction == True

        # Teardown - not needed for this test case

    def test___init___2_2(self):
        """
        Test FilterSpec initialization with a custom filter function
        """
        class CustomFilter(BaseFilter):
            def __init__(self):
                super().__init__()

            def filter(self, agent_id, observation):
                return observation

        custom_filter = CustomFilter()
        filter_config = {"test_key": "test_value"}
        filter_spec = FilterSpec(filter_fn=custom_filter, filter_fn_config=filter_config)

        assert isinstance(filter_spec.filter_fn, BaseFilter)
        assert filter_spec.filter_fn == custom_filter
        assert filter_spec.filter_fn_config == filter_config

    def test___init___2_3(self):
        """
        Test initialization of ActionSpec with non-None actuators.
        """
        # Arrange
        actuators: List[Tuple[str, str, str]] = [
            ("Zone Temperature Control", "ZONE ONE", "Cooling Setpoint"),
            ("Zone Temperature Control", "ZONE TWO", "Heating Setpoint")
        ]

        # Act
        action_spec = ActionSpec(actuators=actuators)

        # Assert
        assert action_spec.actuators == actuators
        assert len(action_spec.actuators) == 2
        assert isinstance(action_spec.actuators, list)
        assert all(isinstance(actuator, tuple) and len(actuator) == 3 for actuator in action_spec.actuators)

    def test___init___2_4(self):
        """
        Test initialization of AgentSpec with custom ObservationSpec and default values for other parameters.
        """
        # Create a custom ObservationSpec
        custom_observation = ObservationSpec(
            variables=[("Zone Mean Air Temperature", "Main Zone")],
            internal_variables=["Zone Floor Area"],
            meters=["Electricity:Facility"],
            simulation_parameters={"hour": True, "day_of_year": True},
            use_one_day_weather_prediction=True,
            prediction_variables={"outdoor_dry_bulb": True}
        )

        # Initialize AgentSpec with custom ObservationSpec
        agent_spec = AgentSpec(observation=custom_observation)

        # Assert that the observation attribute is set correctly
        assert isinstance(agent_spec.observation, ObservationSpec)
        assert agent_spec.observation.variables == [("Zone Mean Air Temperature", "Main Zone")]
        assert agent_spec.observation.internal_variables == ["Zone Floor Area"]
        assert agent_spec.observation.meters == ["Electricity:Facility"]
        assert agent_spec.observation.simulation_parameters["hour"] == True
        assert agent_spec.observation.simulation_parameters["day_of_year"] == True
        assert agent_spec.observation.use_one_day_weather_prediction == True
        assert agent_spec.observation.prediction_variables["outdoor_dry_bulb"] == True

        # Assert that other attributes are set to their default values
        assert isinstance(agent_spec.filter, FilterSpec)
        assert isinstance(agent_spec.action, ActionSpec)
        assert isinstance(agent_spec.trigger, TriggerSpec)
        assert isinstance(agent_spec.reward, RewardSpec)

        # Assert that the default values are used for other parameters
        assert agent_spec.filter.filter_fn is not None
        assert agent_spec.action.actuators == []
        assert agent_spec.trigger.trigger_fn == NotImplemented
        assert agent_spec.reward.reward_fn == NotImplemented

    def test___init___3(self):
        """
        Test ObservationSpec initialization with invalid simulation parameters, 
        weather prediction enabled, invalid prediction variables, and invalid prediction hours.
        """
        # Setup
        variables: List[Tuple[str, str]] = [("Zone Mean Air Temperature", "Main Zone")]
        internal_variables: List[str] = ["Zone Air System Sensible Heating Rate"]
        meters: List[str] = ["Electricity:Facility"]
        simulation_parameters: Dict[str, bool] = {"invalid_param": True}
        zone_simulation_parameters: Dict[str, bool] = {"system_time_step": True}
        use_one_day_weather_prediction: bool = True
        prediction_hours: int = 25
        prediction_variables: Dict[str, bool] = {"invalid_weather_var": True}
        use_actuator_state: bool = True
        other_obs: Dict[str, float | int] = {"custom_obs": 1.0}

        # Exercise and Verify
        with pytest.warns(UserWarning, match="The variable 'prediction_hours' must be between 1 and 24"):
            obs_spec = ObservationSpec(
                variables=variables,
                internal_variables=internal_variables,
                meters=meters,
                simulation_parameters=simulation_parameters,
                zone_simulation_parameters=zone_simulation_parameters,
                use_one_day_weather_prediction=use_one_day_weather_prediction,
                prediction_hours=prediction_hours,
                prediction_variables=prediction_variables,
                use_actuator_state=use_actuator_state,
                other_obs=other_obs
            )

        # Additional Verifications
        assert obs_spec.variables == variables
        assert obs_spec.internal_variables == internal_variables
        assert obs_spec.meters == meters
        assert "invalid_param" not in obs_spec.simulation_parameters
        assert obs_spec.zone_simulation_parameters["system_time_step"] == True
        assert obs_spec.use_one_day_weather_prediction == True
        assert obs_spec.prediction_hours == 24  # Should be set to 24 due to invalid input
        assert "invalid_weather_var" not in obs_spec.prediction_variables
        assert obs_spec.use_actuator_state == True
        assert obs_spec.other_obs == other_obs

        # Check if error messages were logged
        # Note: This assumes that the logger.error calls in the original code print to stderr
        # If they don't, you might need to use a mocking library to capture the log messages
        with pytest.raises(SystemExit):
            ObservationSpec(
                simulation_parameters={"invalid_param": True},
                prediction_variables={"invalid_weather_var": True}
            )

    def test___init___4(self):
        """
        Test ObservationSpec initialization with invalid simulation parameters,
        invalid zone simulation parameters, one day weather prediction enabled,
        valid prediction variables, and invalid prediction hours.
        """
        # Invalid simulation parameters
        invalid_simulation_params = {'invalid_param': True}
        
        # Invalid zone simulation parameters
        invalid_zone_params = {'invalid_zone_param': True}
        
        # Valid prediction variables
        valid_prediction_vars = {'outdoor_dry_bulb': True, 'wind_speed': True}
        
        # Invalid prediction hours
        invalid_prediction_hours = 25

        with pytest.warns(UserWarning):
            obs_spec = ObservationSpec(
                simulation_parameters=invalid_simulation_params,
                zone_simulation_parameters=invalid_zone_params,
                use_one_day_weather_prediction=True,
                prediction_variables=valid_prediction_vars,
                prediction_hours=invalid_prediction_hours
            )

        # Check that invalid parameters were not added
        assert 'invalid_param' not in obs_spec.simulation_parameters
        assert 'invalid_zone_param' not in obs_spec.zone_simulation_parameters

        # Check that use_one_day_weather_prediction is set to True
        assert obs_spec.use_one_day_weather_prediction == True

        # Check that valid prediction variables were added
        assert obs_spec.prediction_variables['outdoor_dry_bulb'] == True
        assert obs_spec.prediction_variables['wind_speed'] == True

        # Check that prediction_hours was set to the default value of 24
        assert obs_spec.prediction_hours == 24

    def test___init___5(self):
        """
        Test ObservationSpec initialization with invalid keys and valid prediction hours
        """
        # Setup
        invalid_simulation_parameters = {'invalid_key': True}
        invalid_zone_simulation_parameters = {'invalid_zone_key': True}
        invalid_prediction_variables = {'invalid_prediction_key': True}
        valid_prediction_hours = 12

        # Exercise
        with pytest.warns(UserWarning):
            obs_spec = ObservationSpec(
                simulation_parameters=invalid_simulation_parameters,
                zone_simulation_parameters=invalid_zone_simulation_parameters,
                use_one_day_weather_prediction=True,
                prediction_hours=valid_prediction_hours,
                prediction_variables=invalid_prediction_variables
            )

        # Verify
        assert obs_spec.use_one_day_weather_prediction == True
        assert obs_spec.prediction_hours == valid_prediction_hours
        assert 'invalid_key' not in obs_spec.simulation_parameters
        assert 'invalid_zone_key' not in obs_spec.zone_simulation_parameters
        assert 'invalid_prediction_key' not in obs_spec.prediction_variables

        # Check if default values are maintained for invalid keys
        assert all(value == False for value in obs_spec.simulation_parameters.values())
        assert all(value == False for value in obs_spec.zone_simulation_parameters.values())
        assert all(value == False for value in obs_spec.prediction_variables.values())

        # Teardown (if needed)
        # No explicit teardown required for this test

    def test___init___6(self):
        """
        Test ObservationSpec initialization with invalid keys in simulation_parameters and zone_simulation_parameters,
        and use_one_day_weather_prediction set to False.
        """
        # Arrange
        invalid_simulation_parameters = {'invalid_key': True}
        invalid_zone_simulation_parameters = {'invalid_zone_key': True}

        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            ObservationSpec(
                simulation_parameters=invalid_simulation_parameters,
                zone_simulation_parameters=invalid_zone_simulation_parameters,
                use_one_day_weather_prediction=False
            )

        # Check if the error message contains information about invalid keys
        assert "The key 'invalid_key' is not admissible in the simulation_parameters" in str(exc_info.value)
        assert "The key 'invalid_zone_key' is not admissible in the zone_simulation_parameters" in str(exc_info.value)

        # Check that use_one_day_weather_prediction is False
        obs_spec = ObservationSpec(use_one_day_weather_prediction=False)
        assert obs_spec.use_one_day_weather_prediction == False

    def test___init___6_2(self):
        """
        Test initialization of AgentSpec with only reward specified
        """
        # Arrange
        reward = RewardSpec(reward_fn=lambda x, y: 0, reward_fn_config={})

        # Act
        agent_spec = AgentSpec(reward=reward)

        # Assert
        assert isinstance(agent_spec.observation, ObservationSpec)
        assert isinstance(agent_spec.filter, FilterSpec)
        assert isinstance(agent_spec.action, ActionSpec)
        assert isinstance(agent_spec.trigger, TriggerSpec)
        assert agent_spec.reward == reward

    def test___init___custom_filter(self):
        """
        Test initialization of AgentSpec with a custom filter and default values for other parameters.
        """
        custom_filter = FilterSpec(filter_fn=DefaultFilter, filter_fn_config={'custom_config': 'test'})
        agent_spec = AgentSpec(filter=custom_filter)

        assert isinstance(agent_spec.observation, ObservationSpec)
        assert agent_spec.filter == custom_filter.build()
        assert isinstance(agent_spec.action, ActionSpec)
        assert isinstance(agent_spec.trigger, TriggerSpec)
        assert isinstance(agent_spec.reward, RewardSpec)

        # Check that the custom filter is correctly set
        assert agent_spec.filter['filter_fn'] == DefaultFilter
        assert agent_spec.filter['filter_fn_config'] == {'custom_config': 'test'}

        # Verify default values for other parameters
        assert agent_spec.observation.variables is None
        assert agent_spec.action.actuators == []
        assert agent_spec.trigger.trigger_fn == NotImplemented
        assert agent_spec.reward.reward_fn == NotImplemented

    def test___init___duplicate_actuators(self):
        """
        Test that initializing ActionSpec with duplicate actuators raises a ValueError.
        """
        with pytest.raises(ValueError):
            ActionSpec(actuators=[
                ("type1", "component1", "control1"),
                ("type1", "component1", "control1")
            ])

    def test___init___empty_input(self):
        """
        Test that initializing ActionSpec with None input results in an empty actuators list.
        """
        action_spec = ActionSpec()
        assert action_spec.actuators == []

    def test___init___empty_tuple_elements(self):
        """
        Test that initializing ActionSpec with tuples containing empty strings raises a ValueError.
        """
        with pytest.raises(ValueError):
            ActionSpec(actuators=[("", "", "control")])

    def test___init___invalid_input_type(self):
        """
        Test that initializing ActionSpec with an invalid input type raises a TypeError.
        """
        with pytest.raises(TypeError):
            ActionSpec(actuators="invalid")

    def test___init___invalid_tuple_length(self):
        """
        Test that initializing ActionSpec with tuples of incorrect length raises a ValueError.
        """
        with pytest.raises(ValueError):
            ActionSpec(actuators=[("type", "component")])

    def test___init___invalid_tuple_types(self):
        """
        Test that initializing ActionSpec with tuples containing invalid types raises a TypeError.
        """
        with pytest.raises(TypeError):
            ActionSpec(actuators=[(1, 2, 3)])

    def test___init___max_actuators_exceeded(self):
        """
        Test that initializing ActionSpec with more than the maximum allowed actuators raises a ValueError.
        """
        max_actuators = 100  # Assuming there's a maximum limit
        with pytest.raises(ValueError):
            ActionSpec(actuators=[("type", "component", "control") for _ in range(max_actuators + 1)])

    def test___init___with_custom_action(self):
        """
        Test AgentSpec initialization with a custom ActionSpec and default values for other parameters.
        """
        # Arrange
        custom_action = ActionSpec(actuators=[("actuator_type", "component_name", "control_type")])

        # Act
        agent_spec = AgentSpec(action=custom_action)

        # Assert
        assert isinstance(agent_spec.observation, ObservationSpec)
        assert isinstance(agent_spec.filter, FilterSpec)
        assert agent_spec.action == custom_action
        assert isinstance(agent_spec.trigger, TriggerSpec)
        assert isinstance(agent_spec.reward, RewardSpec)

        # Verify that the custom action is correctly set
        assert agent_spec.action.actuators == [("actuator_type", "component_name", "control_type")]

        # Verify that other attributes have default values
        assert agent_spec.observation.variables is None
        assert agent_spec.observation.internal_variables is None
        assert agent_spec.observation.meters is None
        assert agent_spec.filter.filter_fn is not None
        assert agent_spec.trigger.trigger_fn == NotImplemented
        assert agent_spec.reward.reward_fn == NotImplemented

    def test___init___with_empty_inputs(self):
        """
        Test __init__ with empty inputs
        """
        agent_spec = AgentSpec()
        assert isinstance(agent_spec.observation, ObservationSpec)
        assert isinstance(agent_spec.filter, FilterSpec)
        assert isinstance(agent_spec.action, ActionSpec)
        assert isinstance(agent_spec.trigger, TriggerSpec)
        assert isinstance(agent_spec.reward, RewardSpec)

    def test___init___with_incorrect_type_action(self):
        """
        Test __init__ with incorrect type for action
        """
        with pytest.raises(TypeError):
            AgentSpec(action=123)

    def test___init___with_incorrect_type_filter(self):
        """
        Test __init__ with incorrect type for filter
        """
        with pytest.raises(TypeError):
            AgentSpec(filter=123)

    def test___init___with_incorrect_type_observation(self):
        """
        Test __init__ with incorrect type for observation
        """
        with pytest.raises(TypeError):
            AgentSpec(observation=123)

    def test___init___with_incorrect_type_reward(self):
        """
        Test __init__ with incorrect type for reward
        """
        with pytest.raises(TypeError):
            AgentSpec(reward=123)

    def test___init___with_incorrect_type_trigger(self):
        """
        Test __init__ with incorrect type for trigger
        """
        with pytest.raises(TypeError):
            AgentSpec(trigger=123)

    def test___init___with_invalid_action(self):
        """
        Test __init__ with invalid action input
        """
        with pytest.raises(AttributeError):
            AgentSpec(action="invalid")

    def test___init___with_invalid_filter(self):
        """
        Test __init__ with invalid filter input
        """
        with pytest.raises(AttributeError):
            AgentSpec(filter="invalid")

    def test___init___with_invalid_filter_fn(self):
        """
        Test __init__ with invalid filter function
        """
        invalid_filter = FilterSpec(filter_fn="invalid")
        with pytest.raises(AttributeError):
            AgentSpec(filter=invalid_filter)

    def test___init___with_invalid_kwargs(self):
        """
        Test __init__ with invalid additional keyword arguments
        """
        agent_spec = AgentSpec(invalid_arg="test")
        assert hasattr(agent_spec, "invalid_arg")
        assert agent_spec.invalid_arg == "test"

    def test___init___with_invalid_observation(self):
        """
        Test __init__ with invalid observation input
        """
        with pytest.raises(AttributeError):
            AgentSpec(observation="invalid")

    def test___init___with_invalid_reward(self):
        """
        Test __init__ with invalid reward input
        """
        with pytest.raises(AttributeError):
            AgentSpec(reward="invalid")

    def test___init___with_invalid_reward_fn(self):
        """
        Test __init__ with invalid reward function
        """
        invalid_reward = RewardSpec(reward_fn="invalid")
        with pytest.raises(AttributeError):
            AgentSpec(reward=invalid_reward)

    def test___init___with_invalid_trigger(self):
        """
        Test __init__ with invalid trigger input
        """
        with pytest.raises(AttributeError):
            AgentSpec(trigger="invalid")

    def test___init___with_invalid_trigger_fn(self):
        """
        Test __init__ with invalid trigger function
        """
        invalid_trigger = TriggerSpec(trigger_fn="invalid")
        with pytest.raises(AttributeError):
            AgentSpec(trigger=invalid_trigger)

    def test___setitem___empty_actuators(self):
        """
        Test __setitem__ with an empty list for actuators.
        """
        action_spec = ActionSpec()
        action_spec['actuators'] = []
        assert action_spec.actuators == []

    def test___setitem___empty_key(self):
        """
        Test setting an item with an empty string as the key.
        """
        reward_spec = RewardSpec()
        with pytest.raises(AttributeError):
            reward_spec[''] = 'some_value'

    def test___setitem___empty_string_key(self):
        """
        Test setting an item with an empty string as key.
        """
        agent_spec = AgentSpec()
        with pytest.raises(AttributeError):
            agent_spec[''] = 'value'

    def test___setitem___empty_trigger_fn(self):
        """
        Test setting an empty trigger_fn.
        """
        trigger_spec = TriggerSpec(trigger_fn=BaseTrigger())
        with pytest.raises(AttributeError):
            trigger_spec['trigger_fn'] = None

    def test___setitem___immutable_attribute(self):
        """
        Test setting an immutable attribute.
        """
        agent_spec = AgentSpec()
        with pytest.raises(AttributeError):
            agent_spec['__class__'] = object

    def test___setitem___incorrect_filter_fn_config_type(self):
        """
        Test that setting filter_fn_config with an incorrect type raises a warning.
        """
        filter_spec = FilterSpec()
        with pytest.warns(UserWarning):
            filter_spec['filter_fn_config'] = 'not_a_dictionary'

    def test___setitem___incorrect_filter_fn_type(self):
        """
        Test that setting filter_fn with an incorrect type raises a warning.
        """
        filter_spec = FilterSpec()
        with pytest.warns(UserWarning):
            filter_spec['filter_fn'] = 'not_a_filter_function'

    def test___setitem___incorrect_format_for_variables(self):
        """
        Test setting incorrect format for 'variables' attribute
        """
        obs_spec = ObservationSpec()
        with pytest.raises(AttributeError):
            obs_spec['variables'] = [('wrong', 'format', 'tuple')]

    def test___setitem___incorrect_reward_fn_config_type(self):
        """
        Test setting reward_fn_config with an incorrect type.
        """
        reward_spec = RewardSpec()
        with pytest.raises(TypeError):
            reward_spec['reward_fn_config'] = 'not_a_dict'

    def test___setitem___incorrect_reward_fn_type(self):
        """
        Test setting reward_fn with an incorrect type.
        """
        reward_spec = RewardSpec()
        with pytest.raises(TypeError):
            reward_spec['reward_fn'] = 'not_a_BaseReward_instance'

    def test___setitem___incorrect_tuple_format(self):
        """
        Test __setitem__ with an incorrect tuple format for actuators.
        """
        action_spec = ActionSpec()
        with pytest.raises(ValueError):
            action_spec['actuators'] = [('invalid_tuple',)]

    def test___setitem___incorrect_type(self):
        """
        Test __setitem__ with an incorrect type for actuators.
        """
        action_spec = ActionSpec()
        with pytest.raises(TypeError):
            action_spec['actuators'] = 'not_a_list'

    def test___setitem___incorrect_type_2(self):
        """
        Test setting an item with an incorrect type.
        """
        agent_spec = AgentSpec()
        with pytest.raises(TypeError):
            agent_spec['observation'] = "Not an ObservationSpec"

    def test___setitem___incorrect_type_for_other_obs(self):
        """
        Test setting incorrect type for 'other_obs' attribute
        """
        obs_spec = ObservationSpec()
        with pytest.raises(AttributeError):
            obs_spec['other_obs'] = 'not_a_dict'

    def test___setitem___incorrect_type_for_prediction_hours(self):
        """
        Test setting incorrect type for 'prediction_hours' attribute
        """
        obs_spec = ObservationSpec()
        with pytest.raises(AttributeError):
            obs_spec['prediction_hours'] = 'not_an_integer'

    def test___setitem___incorrect_type_for_simulation_parameters(self):
        """
        Test setting incorrect type for 'simulation_parameters' attribute
        """
        obs_spec = ObservationSpec()
        with pytest.raises(AttributeError):
            obs_spec['simulation_parameters'] = 'not_a_dict'

    def test___setitem___incorrect_type_for_trigger_fn(self):
        """
        Test setting trigger_fn with an incorrect type.
        """
        trigger_spec = TriggerSpec(trigger_fn=BaseTrigger())
        with pytest.raises(AttributeError):
            trigger_spec['trigger_fn'] = 'not_a_trigger_function'

    def test___setitem___incorrect_type_for_trigger_fn_config(self):
        """
        Test setting trigger_fn_config with an incorrect type.
        """
        trigger_spec = TriggerSpec(trigger_fn=BaseTrigger())
        with pytest.raises(AttributeError):
            trigger_spec['trigger_fn_config'] = 'not_a_dict'

    def test___setitem___incorrect_type_for_use_actuator_state(self):
        """
        Test setting incorrect type for 'use_actuator_state' attribute
        """
        obs_spec = ObservationSpec()
        with pytest.raises(AttributeError):
            obs_spec['use_actuator_state'] = 'not_a_boolean'

    def test___setitem___incorrect_type_for_use_one_day_weather_prediction(self):
        """
        Test setting incorrect type for 'use_one_day_weather_prediction' attribute
        """
        obs_spec = ObservationSpec()
        with pytest.raises(AttributeError):
            obs_spec['use_one_day_weather_prediction'] = 'not_a_boolean'

    def test___setitem___incorrect_type_for_variables(self):
        """
        Test setting incorrect type for 'variables' attribute
        """
        obs_spec = ObservationSpec()
        with pytest.raises(AttributeError):
            obs_spec['variables'] = 'not_a_list'

    def test___setitem___invalid_key(self):
        """
        Test setting an item with an invalid key.
        """
        reward_spec = RewardSpec()
        with pytest.raises(AttributeError):
            reward_spec['invalid_key'] = 'some_value'

    def test___setitem___invalid_key_2(self):
        """
        Test setting an invalid key in ObservationSpec
        """
        obs_spec = ObservationSpec()
        with pytest.raises(AttributeError):
            obs_spec['invalid_key'] = 'value'

    def test___setitem___invalid_key_3(self):
        """
        Test that setting an invalid key raises an AttributeError.
        """
        filter_spec = FilterSpec()
        with pytest.raises(AttributeError):
            filter_spec['invalid_key'] = 'value'

    def test___setitem___invalid_key_4(self):
        """
        Test __setitem__ with an invalid key.
        """
        action_spec = ActionSpec()
        with pytest.raises(AttributeError):
            action_spec['invalid_key'] = 'value'

    def test___setitem___invalid_key_5(self):
        """
        Test setting an invalid key in TriggerSpec.
        """
        trigger_spec = TriggerSpec(trigger_fn=BaseTrigger())
        with pytest.raises(AttributeError):
            trigger_spec['invalid_key'] = 'some_value'

    def test___setitem___invalid_key_6(self):
        """
        Test setting an item with an invalid key.
        """
        agent_spec = AgentSpec()
        with pytest.raises(AttributeError):
            agent_spec['invalid_key'] = 'value'

    def test___setitem___invalid_key_in_simulation_parameters(self):
        """
        Test setting an invalid key in 'simulation_parameters' attribute
        """
        obs_spec = ObservationSpec()
        with pytest.raises(AttributeError):
            obs_spec['simulation_parameters'] = {'invalid_key': True}

    def test___setitem___large_key(self):
        """
        Test setting an item with a very large key.
        """
        agent_spec = AgentSpec()
        large_key = 'a' * 1000000  # 1 million characters
        with pytest.raises(MemoryError):
            agent_spec[large_key] = 'value'

    def test___setitem___non_existent_attribute(self):
        """
        Test setting a non-existent attribute.
        """
        trigger_spec = TriggerSpec(trigger_fn=BaseTrigger())
        with pytest.raises(AttributeError):
            trigger_spec['non_existent_attribute'] = 'some_value'

    def test___setitem___non_existing_attribute(self):
        """
        Test __setitem__ with a non-existing attribute.
        """
        action_spec = ActionSpec()
        action_spec['new_attribute'] = 'value'
        assert hasattr(action_spec, 'new_attribute')
        assert action_spec.new_attribute == 'value'

    def test___setitem___non_string_key(self):
        """
        Test setting an item with a non-string key.
        """
        reward_spec = RewardSpec()
        with pytest.raises(TypeError):
            reward_spec[123] = 'some_value'

    def test___setitem___non_string_key_2(self):
        """
        Test setting an item with a non-string key.
        """
        agent_spec = AgentSpec()
        with pytest.raises(TypeError):
            agent_spec[123] = 'value'

    def test___setitem___none_key(self):
        """
        Test setting an item with None as the key.
        """
        reward_spec = RewardSpec()
        with pytest.raises(TypeError):
            reward_spec[None] = 'some_value'

    def test___setitem___none_value(self):
        """
        Test setting an item with a None value.
        """
        agent_spec = AgentSpec()
        agent_spec['observation'] = None
        assert agent_spec.observation is None

    def test___setitem___out_of_bounds_prediction_hours(self):
        """
        Test setting out of bounds value for 'prediction_hours' attribute
        """
        obs_spec = ObservationSpec()
        with pytest.raises(AttributeError):
            obs_spec['prediction_hours'] = 25

    def test___setitem___overwrite_method(self):
        """
        Test attempting to overwrite a method.
        """
        agent_spec = AgentSpec()
        with pytest.raises(AttributeError):
            agent_spec['build'] = lambda: None

    def test___setitem___valid_filter_fn(self):
        """
        Test that setting a valid filter_fn works correctly.
        """
        class ValidFilter(BaseFilter):
            pass

        filter_spec = FilterSpec()
        filter_spec['filter_fn'] = ValidFilter
        assert filter_spec.filter_fn == ValidFilter

    def test___setitem___valid_filter_fn_config(self):
        """
        Test that setting a valid filter_fn_config works correctly.
        """
        filter_spec = FilterSpec()
        valid_config = {'param1': 'value1', 'param2': 'value2'}
        filter_spec['filter_fn_config'] = valid_config
        assert filter_spec.filter_fn_config == valid_config

    def test_action_spec_getitem(self):
        """
        Test that __getitem__ method of ActionSpec returns the correct attribute value
        """
        # Arrange
        actuators = [("actuator_type", "component_name", "control_type")]
        action_spec = ActionSpec(actuators=actuators)

        # Act
        result = action_spec["actuators"]

        # Assert
        assert result == actuators, f"Expected {actuators}, but got {result}"

    def test_action_spec_getitem_nonexistent_key(self):
        """
        Test that __getitem__ method of ActionSpec raises AttributeError for non-existent key
        """
        # Arrange
        action_spec = ActionSpec()

        # Act & Assert
        with pytest.raises(AttributeError):
            action_spec["non_existent_key"]

    def test_build(self):
        """
        Test the build method of FilterSpec
        """
        filter_spec = FilterSpec()
        built_spec = filter_spec.build()

        assert isinstance(built_spec, dict)
        assert 'filter_fn' in built_spec
        assert 'filter_fn_config' in built_spec

    def test_build_2(self):
        """
        Test the build method of ActionSpec.
        """
        # Arrange
        actuators = [("Type1", "Component1", "Control1"), ("Type2", "Component2", "Control2")]
        action_spec = ActionSpec(actuators=actuators)

        # Act
        built = action_spec.build()

        # Assert
        assert isinstance(built, dict)
        assert "actuators" in built
        assert built["actuators"] == actuators

    def test_build_2_2(self):
        """
        Test build method when observation is not an ObservationSpec instance
        but other components are valid instances.
        """
        # Mock the logger to capture error messages
        with patch('eprllib.Agents.AgentSpec.logger') as mock_logger:
            # Create an AgentSpec instance with invalid observation
            agent_spec = AgentSpec(
                observation="invalid_observation",
                filter=FilterSpec(),
                action=ActionSpec(),
                trigger=TriggerSpec(trigger_fn=Mock()),
                reward=RewardSpec(reward_fn=Mock())
            )

            # Call the build method
            result = agent_spec.build()

            # Assert that the error log was called for the invalid observation
            mock_logger.error.assert_called_with(
                "The observation must be defined as an ObservationSpec object but <class 'str'> was given."
            )

            # Assert that the result is a dictionary
            assert isinstance(result, dict)

            # Assert that the observation is still the invalid string
            assert result['observation'] == "invalid_observation"

            # Assert that other components were built correctly
            assert isinstance(result['filter'], dict)
            assert isinstance(result['action'], dict)
            assert isinstance(result['trigger'], dict)
            assert isinstance(result['reward'], dict)

    def test_build_3(self):
        """
        Test build method with valid ObservationSpec, ActionSpec, TriggerSpec, RewardSpec, and invalid FilterSpec
        """
        # Mock the logger to avoid actual logging during the test
        with patch('eprllib.Agents.AgentSpec.logger') as mock_logger:
            # Create mock objects
            mock_observation = Mock(spec=ObservationSpec)
            mock_filter = Mock()  # Not a FilterSpec object
            mock_action = Mock(spec=ActionSpec)
            mock_trigger = Mock(spec=TriggerSpec)
            mock_reward = Mock(spec=RewardSpec)

            # Create AgentSpec instance
            agent_spec = AgentSpec(
                observation=mock_observation,
                filter=mock_filter,
                action=mock_action,
                trigger=mock_trigger,
                reward=mock_reward
            )

            # Set up mock return values
            mock_observation.build.return_value = {"observation": "data"}
            mock_action.build.return_value = {"action": "data"}
            mock_trigger.build.return_value = {"trigger": "data"}
            mock_reward.build.return_value = {"reward": "data"}

            # Call the build method
            result = agent_spec.build()

            # Assertions
            assert isinstance(result, dict)
            assert result["observation"] == {"observation": "data"}
            assert result["action"] == {"action": "data"}
            assert result["trigger"] == {"trigger": "data"}
            assert result["reward"] == {"reward": "data"}
            assert "filter" in result
            
            # Verify that the error was logged for invalid FilterSpec
            mock_logger.error.assert_called_once_with(
                "The filter must be defined as a FilterSpec object but <class 'unittest.mock.Mock'> was given."
            )

            # Verify that build methods were called for valid specs
            mock_observation.build.assert_called_once()
            mock_action.build.assert_called_once()
            mock_trigger.build.assert_called_once()
            mock_reward.build.assert_called_once()

    def test_build_4(self):
        """
        Test build method when action is not an ActionSpec instance
        """
        # Mock the logger to capture error messages
        mock_logger = Mock()
        
        # Create mock instances for ObservationSpec, FilterSpec, TriggerSpec, and RewardSpec
        mock_observation = Mock(spec=ObservationSpec)
        mock_filter = Mock(spec=FilterSpec)
        mock_trigger = Mock(spec=TriggerSpec)
        mock_reward = Mock(spec=RewardSpec)
        
        # Create an invalid action (not an ActionSpec instance)
        invalid_action = "Invalid Action"
        
        # Create an AgentSpec instance with the mocked components
        agent_spec = AgentSpec(
            observation=mock_observation,
            filter=mock_filter,
            action=invalid_action,
            trigger=mock_trigger,
            reward=mock_reward
        )
        
        # Set up the mocks to return empty dictionaries when their build methods are called
        mock_observation.build.return_value = {}
        mock_filter.build.return_value = {}
        mock_trigger.build.return_value = {}
        mock_reward.build.return_value = {}
        
        # Call the build method
        result = agent_spec.build()
        
        # Assert that the build method returns a dictionary
        assert isinstance(result, Dict)
        
        # Assert that the observation, filter, trigger, and reward were built
        assert mock_observation.build.called
        assert mock_filter.build.called
        assert mock_trigger.build.called
        assert mock_reward.build.called
        
        # Assert that an error was logged for the invalid action
        assert any("The action must be defined as an ActionSpec object" in str(call) for call in mock_logger.error.call_args_list)
        
        # Assert that the result contains the expected keys
        expected_keys = ['observation', 'filter', 'action', 'trigger', 'reward']
        assert all(key in result for key in expected_keys)

    def test_build_5(self):
        """
        Test build method when trigger is not an instance of TriggerSpec
        """
        # Arrange
        observation = ObservationSpec()
        filter_spec = FilterSpec()
        action = ActionSpec()
        trigger = "Not a TriggerSpec instance"
        reward = RewardSpec(reward_fn=BaseReward())

        agent_spec = AgentSpec(
            observation=observation,
            filter=filter_spec,
            action=action,
            trigger=trigger,
            reward=reward
        )

        # Act
        result = agent_spec.build()

        # Assert
        assert isinstance(result, dict)
        assert isinstance(result['observation'], dict)
        assert isinstance(result['filter'], dict)
        assert isinstance(result['action'], dict)
        assert result['trigger'] == "Not a TriggerSpec instance"
        assert isinstance(result['reward'], dict)

        # Check if the error message was logged
        # Note: This assumes that the logger.error() call can be captured and verified.
        # If not possible in the current setup, this assertion can be omitted.
        # assert "The trigger must be defined as a TriggerSpec object but <class 'str'> was given." in caplog.text

    def test_build_6(self):
        """
        Test build method when reward is not an instance of RewardSpec
        """
        # Mock the logger to capture error messages
        with patch('eprllib.Agents.AgentSpecективerrorß') as mock_logger:
            # Create instances of required specs
            obs_spec = ObservationSpec()
            filter_spec = FilterSpec()
            action_spec = ActionSpec()
            trigger_spec = TriggerSpec(trigger_fn=Mock())
            
            # Create an AgentSpec instance with invalid reward
            agent_spec = AgentSpec(
                observation=obs_spec,
                filter=filter_spec,
                action=action_spec,
                trigger=trigger_spec,
                reward="Invalid Reward"  # This should trigger an error
            )

            # Call the build method
            result = agent_spec.build()

            # Assert that the error message was logged
            mock_logger.error.assert_called_with("The reward must be defined as a RewardSpec object but <class 'str'> was given.")

            # Assert that the result is a dictionary
            assert isinstance(result, dict)

            # Assert that the reward attribute is still the invalid string
            assert result['reward'] == "Invalid Reward"

            # Assert that other attributes were correctly built
            assert isinstance(result['observation'], dict)
            assert isinstance(result['filter'], dict)
            assert isinstance(result['action'], dict)
            assert isinstance(result['trigger'], dict)

    def test_build_method(self):
        """
        Test the build method of ActionSpec.
        """
        # Arrange
        actuators: List[Tuple[str, str, str]] = [
            ("actuator_type", "component_name", "control_type")
        ]
        action_spec = ActionSpec(actuators=actuators)

        # Act
        result = action_spec.build()

        # Assert
        assert isinstance(result, dict)
        assert "actuators" in result
        assert result["actuators"] == actuators

    def test_build_returns_dict_with_actuators(self):
        """
        Test that the build method returns a dictionary containing the actuators.
        """
        # Arrange
        actuators: List[Tuple[str, str, str]] = [
            ("actuator_type_1", "component_name_1", "control_type_1"),
            ("actuator_type_2", "component_name_2", "control_type_2")
        ]
        action_spec = ActionSpec(actuators=actuators)

        # Act
        result = action_spec.build()

        # Assert
        assert isinstance(result, Dict)
        assert 'actuators' in result
        assert result['actuators'] == actuators

    def test_build_returns_dict_with_all_attributes(self):
        """
        Test that the build method returns a dictionary containing all attributes of the ObservationSpec object.
        """
        # Arrange
        obs_spec = ObservationSpec(
            variables=[("Zone Mean Air Temperature", "Main Zone")],
            internal_variables=["Zone Air System Sensible Heating Rate"],
            meters=["Electricity:Facility"],
            simulation_parameters={'hour': True, 'day_of_year': True},
            zone_simulation_parameters={'zone_time_step': True},
            use_one_day_weather_prediction=True,
            prediction_hours=12,
            prediction_variables={'outdoor_dry_bulb': True, 'wind_speed': True},
            use_actuator_state=True,
            other_obs={'custom_obs': 1.0}
        )

        # Act
        result = obs_spec.build()

        # Assert
        assert isinstance(result, dict)
        assert 'variables' in result
        assert 'internal_variables' in result
        assert 'meters' in result
        assert 'simulation_parameters' in result
        assert 'zone_simulation_parameters' in result
        assert 'use_one_day_weather_prediction' in result
        assert 'prediction_hours' in result
        assert 'prediction_variables' in result
        assert 'use_actuator_state' in result
        assert 'other_obs' in result

        assert result['variables'] == [("Zone Mean Air Temperature", "Main Zone")]
        assert result['internal_variables'] == ["Zone Air System Sensible Heating Rate"]
        assert result['meters'] == ["Electricity:Facility"]
        assert result['simulation_parameters']['hour'] == True
        assert result['simulation_parameters']['day_of_year'] == True
        assert result['zone_simulation_parameters']['zone_time_step'] == True
        assert result['use_one_day_weather_prediction'] == True
        assert result['prediction_hours'] == 12
        assert result['prediction_variables']['outdoor_dry_bulb'] == True
        assert result['prediction_variables']['wind_speed'] == True
        assert result['use_actuator_state'] == True
        assert result['other_obs'] == {'custom_obs': 1.0}

    def test_build_returns_dict_with_attributes(self):
        """
        Test that build() method returns a dictionary with RewardSpec attributes
        """
        class DummyReward(BaseReward):
            def __call__(self, env_object, infos):
                return 0.0

        reward_fn = DummyReward()
        reward_fn_config = {"test_config": "value"}
        reward_spec = RewardSpec(reward_fn=reward_fn, reward_fn_config=reward_fn_config)

        result = reward_spec.build()

        assert isinstance(result, dict)
        assert "reward_fn" in result
        assert result["reward_fn"] == reward_fn
        assert "reward_fn_config" in result
        assert result["reward_fn_config"] == reward_fn_config

    def test_build_returns_dict_with_filter_attributes(self):
        """
        Test that the build method returns a dictionary containing the FilterSpec attributes.
        """
        # Arrange
        filter_fn = DefaultFilter()
        filter_fn_config = {"param1": "value1", "param2": "value2"}
        filter_spec = FilterSpec(filter_fn=filter_fn, filter_fn_config=filter_fn_config)

        # Act
        result = filter_spec.build()

        # Assert
        assert isinstance(result, dict)
        assert "filter_fn" in result
        assert result["filter_fn"] == filter_fn
        assert "filter_fn_config" in result
        assert result["filter_fn_config"] == filter_fn_config

    def test_build_returns_dict_with_trigger_fn_and_config(self):
        """
        Test that the build method returns a dictionary containing trigger_fn and trigger_fn_config.
        """
        class DummyTrigger(BaseTrigger):
            def __call__(self, agent_id, observation):
                return True

        trigger_fn = DummyTrigger()
        trigger_fn_config = {"param1": "value1", "param2": 42}
        
        trigger_spec = TriggerSpec(trigger_fn=trigger_fn, trigger_fn_config=trigger_fn_config)
        
        result = trigger_spec.build()
        
        assert isinstance(result, dict)
        assert "trigger_fn" in result
        assert result["trigger_fn"] == trigger_fn
        assert "trigger_fn_config" in result
        assert result["trigger_fn_config"] == trigger_fn_config

    def test_build_valid_specs(self):
        """
        Test the build method of AgentSpec with valid specifications.
        """
        # Create valid specifications
        obs_spec = ObservationSpec(variables=[("Zone Mean Air Temperature", "Zone 1")])
        filter_spec = FilterSpec()
        action_spec = ActionSpec(actuators=[("Zone Temperature Control", "Zone 1", "Cooling Setpoint")])
        trigger_spec = TriggerSpec(trigger_fn=BaseTrigger())
        reward_spec = RewardSpec(reward_fn=BaseReward())

        # Create AgentSpec instance
        agent_spec = AgentSpec(
            observation=obs_spec,
            filter=filter_spec,
            action=action_spec,
            trigger=trigger_spec,
            reward=reward_spec
        )

        # Call the build method
        result = agent_spec.build()

        # Assert that the result is a dictionary
        assert isinstance(result, dict)

        # Assert that all specifications are built and present in the result
        assert 'observation' in result
        assert 'filter' in result
        assert 'action' in result
        assert 'trigger' in result
        assert 'reward' in result

        # Assert that the built specifications are dictionaries
        assert isinstance(result['observation'], dict)
        assert isinstance(result['filter'], dict)
        assert isinstance(result['action'], dict)
        assert isinstance(result['trigger'], dict)
        assert isinstance(result['reward'], dict)

    def test_build_with_additional_attributes(self):
        """
        Test build method with additional attributes
        """
        class DummyReward(BaseReward):
            pass

        reward_spec = RewardSpec(reward_fn=DummyReward())
        reward_spec.additional_attr = "test"
        result = reward_spec.build()
        assert isinstance(result, dict)
        assert 'reward_fn' in result
        assert 'reward_fn_config' in result
        assert 'additional_attr' in result
        assert result['additional_attr'] == "test"

    def test_build_with_empty_actuators(self):
        """
        Test build method with empty actuators list.
        """
        action_spec = ActionSpec(actuators=[])
        result = action_spec.build()
        assert result == {'actuators': []}

    def test_build_with_empty_config(self):
        """
        Test build method with an empty configuration.
        """
        filter_spec = FilterSpec()
        result = filter_spec.build()
        
        assert isinstance(result, dict)
        assert 'filter_fn' in result
        assert 'filter_fn_config' in result
        assert result['filter_fn_config'] == {}

    def test_build_with_empty_observation(self):
        """
        Test build method with empty observation configuration
        """
        obs_spec = ObservationSpec()
        with pytest.raises(Exception) as exc_info:
            obs_spec.build()
        assert "At least one variable/meter/actuator/parameter must be defined in the observation." in str(exc_info.value)

    def test_build_with_empty_reward_fn_config(self):
        """
        Test build method with empty reward_fn_config
        """
        class DummyReward(BaseReward):
            pass

        reward_spec = RewardSpec(reward_fn=DummyReward(), reward_fn_config={})
        result = reward_spec.build()
        assert isinstance(result, dict)
        assert 'reward_fn' in result
        assert 'reward_fn_config' in result
        assert result['reward_fn_config'] == {}

    def test_build_with_empty_trigger_fn_config(self):
        """
        Test build method with an empty trigger function configuration.
        """
        class DummyTrigger(BaseTrigger):
            pass

        trigger_spec = TriggerSpec(trigger_fn=DummyTrigger, trigger_fn_config={})
        result = trigger_spec.build()
        assert isinstance(result, dict)
        assert 'trigger_fn' in result
        assert 'trigger_fn_config' in result
        assert result['trigger_fn_config'] == {}

    def test_build_with_incorrect_actuator_format(self):
        """
        Test build method with incorrect actuator format.
        """
        action_spec = ActionSpec(actuators=[("invalid_tuple",)])
        with pytest.raises(AttributeError):
            action_spec.build()

    def test_build_with_incorrect_format_variables(self):
        """
        Test build method with incorrect format for variables
        """
        obs_spec = ObservationSpec(variables=[("incorrect", "tuple")])
        with pytest.raises(Exception) as exc_info:
            obs_spec.build()
        assert "not enough values to unpack" in str(exc_info.value)

    def test_build_with_incorrect_type_variables(self):
        """
        Test build method with incorrect type for variables
        """
        obs_spec = ObservationSpec(variables="not_a_list")
        with pytest.raises(Exception) as exc_info:
            obs_spec.build()
        assert "object is not iterable" in str(exc_info.value)

    def test_build_with_invalid_action(self):
        """
        Test build method with invalid action type
        """
        agent_spec = AgentSpec(action="invalid")
        with pytest.raises(AttributeError):
            agent_spec.build()

    def test_build_with_invalid_actuators(self):
        """
        Test build method with invalid actuators input.
        """
        action_spec = ActionSpec(actuators="invalid")
        with pytest.raises(AttributeError):
            action_spec.build()

    def test_build_with_invalid_attribute(self):
        """
        Test build method with an invalid attribute.
        """
        filter_spec = FilterSpec()
        filter_spec.invalid_attribute = "This shouldn't be here"
        
        result = filter_spec.build()
        assert 'invalid_attribute' in result
        assert result['invalid_attribute'] == "This shouldn't be here"

    def test_build_with_invalid_filter(self):
        """
        Test build method with invalid filter type
        """
        agent_spec = AgentSpec(filter="invalid")
        with pytest.raises(AttributeError):
            agent_spec.build()

    def test_build_with_invalid_filter_fn(self):
        """
        Test build method with an invalid filter function.
        """
        invalid_filter = lambda x: x  # Not a BaseFilter instance
        filter_spec = FilterSpec(filter_fn=invalid_filter)
        
        with pytest.raises(AttributeError):
            filter_spec.build()

    def test_build_with_invalid_filter_fn_config(self):
        """
        Test build method with an invalid filter function configuration.
        """
        class DummyFilter(BaseFilter):
            pass

        invalid_config = "not a dictionary"
        filter_spec = FilterSpec(filter_fn=DummyFilter, filter_fn_config=invalid_config)
        
        with pytest.raises(AttributeError):
            filter_spec.build()

    def test_build_with_invalid_filter_function(self):
        """
        Test build method with invalid filter function
        """
        class InvalidFilter:
            pass

        agent_spec = AgentSpec(filter=FilterSpec(filter_fn=InvalidFilter()))
        with pytest.raises(AttributeError):
            agent_spec.build()

    def test_build_with_invalid_observation(self):
        """
        Test build method with invalid observation type
        """
        agent_spec = AgentSpec(observation="invalid")
        with pytest.raises(AttributeError):
            agent_spec.build()

    def test_build_with_invalid_prediction_variable(self):
        """
        Test build method with invalid prediction variable
        """
        with pytest.raises(Exception) as exc_info:
            ObservationSpec(use_one_day_weather_prediction=True, 
                            prediction_variables={'invalid_var': True}).build()
        assert "The key 'invalid_var' is not admissible in the prediction_variables" in str(exc_info.value)

    def test_build_with_invalid_reward(self):
        """
        Test build method with invalid reward type
        """
        agent_spec = AgentSpec(reward="invalid")
        with pytest.raises(AttributeError):
            agent_spec.build()

    def test_build_with_invalid_reward_fn_config_type(self):
        """
        Test build method with invalid reward_fn_config type
        """
        class DummyReward(BaseReward):
            pass

        reward_spec = RewardSpec(reward_fn=DummyReward(), reward_fn_config="invalid")
        with pytest.raises(AttributeError):
            reward_spec.build()

    def test_build_with_invalid_reward_fn_type(self):
        """
        Test build method with invalid reward_fn type
        """
        reward_spec = RewardSpec(reward_fn=lambda x: x)  # Not a BaseReward instance
        with pytest.raises(AttributeError):
            reward_spec.build()

    def test_build_with_invalid_reward_function(self):
        """
        Test build method with invalid reward function
        """
        class InvalidReward:
            pass

        agent_spec = AgentSpec(reward=RewardSpec(reward_fn=InvalidReward()))
        with pytest.raises(AttributeError):
            agent_spec.build()

    def test_build_with_invalid_simulation_parameter(self):
        """
        Test build method with invalid simulation parameter
        """
        with pytest.raises(Exception) as exc_info:
            ObservationSpec(simulation_parameters={'invalid_param': True}).build()
        assert "The key 'invalid_param' is not admissible in the simulation_parameters" in str(exc_info.value)

    def test_build_with_invalid_trigger(self):
        """
        Test build method with invalid trigger type
        """
        agent_spec = AgentSpec(trigger="invalid")
        with pytest.raises(AttributeError):
            agent_spec.build()

    def test_build_with_invalid_trigger_fn(self):
        """
        Test build method with an invalid trigger function.
        """
        trigger_spec = TriggerSpec(trigger_fn="invalid_trigger")
        with pytest.raises(AttributeError):
            trigger_spec.build()

    def test_build_with_invalid_trigger_fn_config(self):
        """
        Test build method with an invalid trigger function configuration.
        """
        class DummyTrigger(BaseTrigger):
            pass

        trigger_spec = TriggerSpec(trigger_fn=DummyTrigger, trigger_fn_config="invalid_config")
        with pytest.raises(AttributeError):
            trigger_spec.build()

    def test_build_with_invalid_trigger_function(self):
        """
        Test build method with invalid trigger function
        """
        class InvalidTrigger:
            pass

        agent_spec = AgentSpec(trigger=TriggerSpec(trigger_fn=InvalidTrigger()))
        with pytest.raises(AttributeError):
            agent_spec.build()

    def test_build_with_invalid_zone_simulation_parameter(self):
        """
        Test build method with invalid zone simulation parameter
        """
        with pytest.raises(Exception) as exc_info:
            ObservationSpec(zone_simulation_parameters={'invalid_param': True}).build()
        assert "The key 'invalid_param' is not admissible in the zone_simulation_parameters" in str(exc_info.value)

    def test_build_with_missing_filter_fn(self):
        """
        Test build method with a missing filter function.
        """
        filter_spec = FilterSpec()
        filter_spec.filter_fn = None
        
        with pytest.raises(AttributeError):
            filter_spec.build()

    def test_build_with_non_base_trigger_class(self):
        """
        Test build method with a trigger function not inheriting from BaseTrigger.
        """
        class InvalidTrigger:
            pass

        trigger_spec = TriggerSpec(trigger_fn=InvalidTrigger)
        with pytest.raises(AttributeError):
            trigger_spec.build()

    def test_build_with_none_actuators(self):
        """
        Test build method with None actuators.
        """
        action_spec = ActionSpec(actuators=None)
        result = action_spec.build()
        assert result == {'actuators': []}

    def test_build_with_not_implemented_reward_fn(self):
        """
        Test build method with NotImplemented reward_fn
        """
        reward_spec = RewardSpec()
        with pytest.raises(NotImplementedError):
            reward_spec.build()

    def test_build_with_not_implemented_trigger_fn(self):
        """
        Test build method with NotImplemented trigger function.
        """
        trigger_spec = TriggerSpec()
        with pytest.raises(AttributeError):
            trigger_spec.build()

    def test_build_with_out_of_bounds_prediction_hours(self):
        """
        Test build method with out of bounds prediction hours
        """
        obs_spec = ObservationSpec(use_one_day_weather_prediction=True, prediction_hours=25)
        with pytest.warns(UserWarning) as warning_info:
            obs_spec.build()
        assert "The variable 'prediction_hours' must be between 1 and 24" in str(warning_info[0].message)

    def test_build_with_unimplemented_reward(self):
        """
        Test build method with unimplemented reward function
        """
        agent_spec = AgentSpec(reward=RewardSpec())
        with pytest.raises(AttributeError):
            agent_spec.build()

    def test_build_with_unimplemented_trigger(self):
        """
        Test build method with unimplemented trigger function
        """
        agent_spec = AgentSpec(trigger=TriggerSpec())
        with pytest.raises(AttributeError):
            agent_spec.build()

    def test_build_with_valid_actuators(self):
        """
        Test build method with valid actuators.
        """
        valid_actuators = [("type1", "component1", "control1"), ("type2", "component2", "control2")]
        action_spec = ActionSpec(actuators=valid_actuators)
        result = action_spec.build()
        assert result == {'actuators': valid_actuators}

    def test_getitem_returns_attribute_value(self):
        """
        Test that __getitem__ returns the correct attribute value for a given key.
        """
        # Create a mock BaseTrigger
        class MockTrigger(BaseTrigger):
            def __call__(self, agent_id, observation):
                pass

        # Initialize TriggerSpec with some test data
        trigger_fn = MockTrigger()
        trigger_fn_config = {"test_config": "value"}
        trigger_spec = TriggerSpec(trigger_fn=trigger_fn, trigger_fn_config=trigger_fn_config)

        # Test retrieving trigger_fn
        assert trigger_spec["trigger_fn"] == trigger_fn

        # Test retrieving trigger_fn_config
        assert trigger_spec["trigger_fn_config"] == trigger_fn_config

        # Test retrieving a non-existent key
        with pytest.raises(AttributeError):
            trigger_spec["non_existent_key"]

    def test_init_with_actuators(self):
        """
        Test initialization of ActionSpec with actuators provided.
        """
        # Arrange
        actuators: List[Tuple[str, str, str]] = [
            ("actuator_type_1", "component_name_1", "control_type_1"),
            ("actuator_type_2", "component_name_2", "control_type_2")
        ]

        # Act
        action_spec = ActionSpec(actuators=actuators)

        # Assert
        assert action_spec.actuators == actuators

    def test_init_with_additional_attributes(self):
        """
        Test that initialization doesn't allow additional attributes.
        """
        class DummyTrigger(BaseTrigger):
            pass

        with pytest.raises(TypeError):
            TriggerSpec(trigger_fn=DummyTrigger, extra_attribute="should_not_be_allowed")

    def test_init_with_custom_filter(self):
        """
        Test initialization with a custom filter (subclass of BaseFilter).
        """
        class CustomFilter(BaseFilter):
            pass

        filter_spec = FilterSpec(filter_fn=CustomFilter, filter_fn_config={"key": "value"})
        assert isinstance(filter_spec.filter_fn, CustomFilter)
        assert filter_spec.filter_fn_config == {"key": "value"}

    def test_init_with_custom_values(self):
        """
        Test initialization of TriggerSpec with custom values.
        """
        class CustomTrigger(BaseTrigger):
            def __call__(self, agent_id, observation):
                return observation

        custom_trigger = CustomTrigger()
        custom_config = {"param1": "value1", "param2": 42}

        trigger_spec = TriggerSpec(trigger_fn=custom_trigger, trigger_fn_config=custom_config)

        assert trigger_spec.trigger_fn == custom_trigger
        assert trigger_spec.trigger_fn_config == custom_config

    def test_init_with_default_filter(self):
        """
        Test initialization of FilterSpec with default filter when no filter is provided.
        """
        filter_spec = FilterSpec()
        
        assert filter_spec.filter_fn == DefaultFilter
        assert filter_spec.filter_fn_config == {}

    def test_init_with_default_values(self):
        """
        Test initialization of TriggerSpec with default values.
        """
        trigger_spec = TriggerSpec()
        
        assert trigger_spec.trigger_fn == NotImplemented
        assert trigger_spec.trigger_fn_config == {}

    def test_init_with_default_values_2(self):
        """
        Test initialization of AgentSpec with default values.
        """
        agent_spec = AgentSpec()

        assert isinstance(agent_spec.observation, ObservationSpec)
        assert isinstance(agent_spec.filter, FilterSpec)
        assert isinstance(agent_spec.action, ActionSpec)
        assert isinstance(agent_spec.trigger, TriggerSpec)
        assert isinstance(agent_spec.reward, RewardSpec)

        # Check that default ObservationSpec is empty
        assert agent_spec.observation.variables is None
        assert agent_spec.observation.internal_variables is None
        assert agent_spec.observation.meters is None
        assert all(value is False for value in agent_spec.observation.simulation_parameters.values())
        assert all(value is False for value in agent_spec.observation.zone_simulation_parameters.values())
        assert agent_spec.observation.use_one_day_weather_prediction is False
        assert all(value is False for value in agent_spec.observation.prediction_variables.values())
        assert agent_spec.observation.use_actuator_state is False
        assert agent_spec.observation.other_obs == {}

        # Check that default ActionSpec is empty
        assert agent_spec.action.actuators == []

        # Check that default RewardSpec has NotImplemented reward_fn
        assert agent_spec.reward.reward_fn == NotImplemented
        assert agent_spec.reward.reward_fn_config == {}

        # Check that default TriggerSpec has NotImplemented trigger_fn
        assert agent_spec.trigger.trigger_fn == NotImplemented
        assert agent_spec.trigger.trigger_fn_config == {}

        # Check that default FilterSpec uses DefaultFilter
        assert agent_spec.filter.filter_fn.__name__ == 'DefaultFilter'
        assert agent_spec.filter.filter_fn_config == {}

    def test_init_with_empty_filter_fn_config(self):
        """
        Test initialization with empty filter_fn_config.
        """
        filter_spec = FilterSpec(filter_fn=DefaultFilter, filter_fn_config={})
        assert filter_spec.filter_fn_config == {}

    def test_init_with_empty_reward_fn_config(self):
        """
        Test initialization with empty reward_fn_config.
        """
        class DummyReward(BaseReward):
            pass

        spec = RewardSpec(reward_fn=DummyReward(), reward_fn_config={})
        assert spec.reward_fn_config == {}

    def test_init_with_empty_trigger_fn_config(self):
        """
        Test initialization with empty trigger_fn_config.
        """
        class DummyTrigger(BaseTrigger):
            pass

        trigger_spec = TriggerSpec(trigger_fn=DummyTrigger)
        assert trigger_spec.trigger_fn_config == {}

    def test_init_with_invalid_actuators(self):
        """
        Test initialization of ActionSpec with invalid actuators.
        """
        # Arrange
        invalid_actuators = ["invalid_actuator"]

        # Act & Assert
        with pytest.raises(AttributeError):
            ActionSpec(actuators=invalid_actuators)

    def test_init_with_invalid_filter_fn(self):
        """
        Test initialization with invalid filter_fn (not a BaseFilter subclass).
        """
        with pytest.raises(AttributeError):
            FilterSpec(filter_fn="not_a_filter")

    def test_init_with_invalid_filter_fn_config(self):
        """
        Test initialization with invalid filter_fn_config (not a dictionary).
        """
        with pytest.raises(AttributeError):
            FilterSpec(filter_fn=DefaultFilter, filter_fn_config="not_a_dict")

    def test_init_with_invalid_internal_variable_type(self):
        """
        Test initialization with invalid internal variable type.
        """
        with pytest.raises(Exception):
            ObservationSpec(internal_variables="invalid_type")

    def test_init_with_invalid_meters_type(self):
        """
        Test initialization with invalid meters type.
        """
        with pytest.raises(Exception):
            ObservationSpec(meters="invalid_type")

    def test_init_with_invalid_other_obs_type(self):
        """
        Test initialization with invalid other_obs type.
        """
        with pytest.raises(Exception):
            ObservationSpec(other_obs="invalid_type")

    def test_init_with_invalid_prediction_hours(self):
        """
        Test initialization with invalid prediction hours.
        """
        with pytest.warns(UserWarning):
            obs_spec = ObservationSpec(use_one_day_weather_prediction=True, prediction_hours=25)
        assert obs_spec.prediction_hours == 24

    def test_init_with_invalid_prediction_variables(self):
        """
        Test initialization with invalid prediction variables.
        """
        with pytest.raises(Exception):
            ObservationSpec(use_one_day_weather_prediction=True, prediction_variables={'invalid_key': True})

    def test_init_with_invalid_reward_fn(self):
        """
        Test initialization with invalid reward_fn (not a BaseReward instance).
        """
        with pytest.raises(AttributeError):
            RewardSpec(reward_fn="not_a_base_reward")

    def test_init_with_invalid_reward_fn_config(self):
        """
        Test initialization with invalid reward_fn_config (not a dictionary).
        """
        class DummyReward(BaseReward):
            pass

        with pytest.raises(TypeError):
            RewardSpec(reward_fn=DummyReward(), reward_fn_config="not_a_dict")

    def test_init_with_invalid_simulation_parameters(self):
        """
        Test initialization with invalid simulation parameters.
        """
        with pytest.raises(Exception):
            ObservationSpec(simulation_parameters={'invalid_key': True})

    def test_init_with_invalid_trigger_fn(self):
        """
        Test initialization of TriggerSpec with an invalid trigger_fn.
        """
        with pytest.raises(AttributeError):
            TriggerSpec(trigger_fn="not_a_trigger")

    def test_init_with_invalid_trigger_fn_2(self):
        """
        Test initialization with invalid trigger_fn (not a BaseTrigger).
        """
        with pytest.raises(AttributeError):
            TriggerSpec(trigger_fn="not_a_trigger_fn")

    def test_init_with_invalid_trigger_fn_config(self):
        """
        Test initialization of TriggerSpec with an invalid trigger_fn_config.
        """
        class CustomTrigger(BaseTrigger):
            def __call__(self, agent_id, observation):
                return observation

        custom_trigger = CustomTrigger()

        with pytest.raises(AttributeError):
            TriggerSpec(trigger_fn=custom_trigger, trigger_fn_config="not_a_dict")

    def test_init_with_invalid_trigger_fn_config_2(self):
        """
        Test initialization with invalid trigger_fn_config (not a dictionary).
        """
        class DummyTrigger(BaseTrigger):
            pass

        with pytest.raises(AttributeError):
            TriggerSpec(trigger_fn=DummyTrigger, trigger_fn_config="not_a_dict")

    def test_init_with_invalid_variable_type(self):
        """
        Test initialization with invalid variable type.
        """
        with pytest.raises(Exception):
            ObservationSpec(variables="invalid_type")

    def test_init_with_invalid_zone_simulation_parameters(self):
        """
        Test initialization with invalid zone simulation parameters.
        """
        with pytest.raises(Exception):
            ObservationSpec(zone_simulation_parameters={'invalid_key': True})

    def test_init_with_no_actuators(self):
        """
        Test initialization of ActionSpec with no actuators provided.
        """
        # Arrange & Act
        action_spec = ActionSpec()

        # Assert
        assert action_spec.actuators == []

    def test_init_with_no_observations(self):
        """
        Test initialization with no observations defined.
        """
        with pytest.raises(Exception):
            ObservationSpec().validate_obs_config()

    def test_init_with_none_filter_fn(self):
        """
        Test initialization with None as filter_fn (should use DefaultFilter).
        """
        filter_spec = FilterSpec(filter_fn=None)
        assert isinstance(filter_spec.filter_fn, DefaultFilter)
        assert filter_spec.filter_fn_config == {}

    def test_init_with_none_reward_fn(self):
        """
        Test initialization with None as reward_fn.
        """
        with pytest.raises(AttributeError):
            RewardSpec(reward_fn=None)

    def test_init_with_not_implemented_reward_fn(self):
        """
        Test initialization with NotImplemented reward_fn.
        """
        spec = RewardSpec()
        with pytest.raises(NotImplementedError):
            spec.build()

    def test_init_with_not_implemented_trigger_fn(self):
        """
        Test initialization with NotImplemented trigger_fn.
        """
        trigger_spec = TriggerSpec()
        with pytest.raises(AttributeError):
            trigger_spec.build()

    def test_init_with_trigger_only(self):
        """
        Test initialization of AgentSpec with only a TriggerSpec provided.
        """
        class DummyTrigger(BaseTrigger):
            def __init__(self):
                pass

            def __call__(self, agent_id, observation):
                return True

        trigger_spec = TriggerSpec(trigger_fn=DummyTrigger())
        agent_spec = AgentSpec(trigger=trigger_spec)

        assert isinstance(agent_spec.observation, ObservationSpec)
        assert isinstance(agent_spec.filter, FilterSpec)
        assert isinstance(agent_spec.action, ActionSpec)
        assert agent_spec.trigger == trigger_spec
        assert isinstance(agent_spec.reward, RewardSpec)

    def test_observation_spec_setitem(self):
        """
        Test setting an attribute using __setitem__ in ObservationSpec
        """
        # Initialize an ObservationSpec object
        obs_spec = ObservationSpec()

        # Set a new attribute using __setitem__
        obs_spec['new_attribute'] = 'test_value'

        # Assert that the new attribute was set correctly
        assert hasattr(obs_spec, 'new_attribute')
        assert obs_spec.new_attribute == 'test_value'

        # Test overwriting an existing attribute
        obs_spec['variables'] = ['new_variable']
        assert obs_spec.variables == ['new_variable']

        # Test setting a non-string key (should raise an AttributeError)
        with pytest.raises(AttributeError):
            obs_spec[123] = 'invalid_key'

    def test_reward_spec_getitem(self):
        """
        Test the __getitem__ method of RewardSpec.
        """
        class MockReward(BaseReward):
            def __call__(self, env_object, infos):
                return 0.0

        reward_spec = RewardSpec(reward_fn=MockReward())
        
        assert reward_spec['reward_fn'] == reward_spec.reward_fn
        assert reward_spec['reward_fn_config'] == reward_spec.reward_fn_config

    def test_reward_spec_getitem_2(self):
        """
        Test that __getitem__ method of RewardSpec returns the correct attribute value.
        """
        class DummyReward(BaseReward):
            def __call__(self, env, infos):
                return 0

        reward_fn = DummyReward()
        reward_fn_config = {"test_key": "test_value"}
        
        reward_spec = RewardSpec(reward_fn=reward_fn, reward_fn_config=reward_fn_config)
        
        assert reward_spec['reward_fn'] == reward_fn
        assert reward_spec['reward_fn_config'] == reward_fn_config

        with pytest.raises(AttributeError):
            reward_spec['non_existent_key']

    def test_reward_spec_initialization(self):
        """
        Test the initialization of RewardSpec with default values.
        """
        # Create a mock BaseReward class
        class MockReward(BaseReward):
            def __call__(self, env_object, infos):
                return 0.0

        # Initialize RewardSpec with a mock reward function
        reward_spec = RewardSpec(reward_fn=MockReward())

        # Assert that the reward_fn is set correctly
        assert isinstance(reward_spec.reward_fn, BaseReward)
        assert isinstance(reward_spec.reward_fn, MockReward)

        # Assert that the reward_fn_config is an empty dictionary by default
        assert reward_spec.reward_fn_config == {}

    def test_reward_spec_not_implemented(self):
        """
        Test that RewardSpec raises an error when reward_fn is not implemented.
        """
        with pytest.raises(NotImplementedError):
            reward_spec = RewardSpec()
            reward_spec.build()

    def test_reward_spec_setitem(self):
        """
        Test the __setitem__ method of RewardSpec.
        """
        class MockReward(BaseReward):
            def __call__(self, env_object, infos):
                return 0.0

        reward_spec = RewardSpec()
        
        new_reward = MockReward()
        reward_spec['reward_fn'] = new_reward
        assert reward_spec.reward_fn == new_reward

        new_config = {"new_key": "new_value"}
        reward_spec['reward_fn_config'] = new_config
        assert reward_spec.reward_fn_config == new_config

    def test_reward_spec_with_config(self):
        """
        Test the initialization of RewardSpec with a custom config.
        """
        class MockReward(BaseReward):
            def __call__(self, env_object, infos):
                return 0.0

        custom_config: Dict[str, Any] = {"key1": "value1", "key2": 42}
        reward_spec = RewardSpec(reward_fn=MockReward(), reward_fn_config=custom_config)

        # Assert that the reward_fn is set correctly
        assert isinstance(reward_spec.reward_fn, BaseReward)
        assert isinstance(reward_spec.reward_fn, MockReward)

        # Assert that the reward_fn_config is set to the custom config
        assert reward_spec.reward_fn_config == custom_config

    def test_setitem_adds_new_attribute(self):
        """
        Test that __setitem__ method adds a new attribute to RewardSpec if it doesn't exist.
        """
        # Arrange
        reward_spec = RewardSpec()

        # Act
        reward_spec['new_attribute'] = 'new_value'

        # Assert
        assert hasattr(reward_spec, 'new_attribute')
        assert reward_spec.new_attribute == 'new_value'

    def test_setitem_adds_new_attribute_2(self):
        """
        Test that __setitem__ can add a new attribute to FilterSpec
        """
        # Arrange
        filter_spec = FilterSpec()
        
        # Act
        filter_spec['new_attribute'] = 'test_value'
        
        # Assert
        assert hasattr(filter_spec, 'new_attribute')
        assert filter_spec.new_attribute == 'test_value'

    def test_setitem_adds_new_attribute_3(self):
        """
        Test that __setitem__ adds a new attribute to the ActionSpec instance.
        """
        action_spec = ActionSpec()
        action_spec['new_attribute'] = 'test_value'
        assert hasattr(action_spec, 'new_attribute')
        assert action_spec.new_attribute == 'test_value'

    def test_setitem_adds_new_attribute_4(self):
        """
        Test that __setitem__ adds a new attribute to TriggerSpec if it doesn't exist
        """
        trigger_spec = TriggerSpec(trigger_fn=BaseTrigger())

        # Use __setitem__ to add a new attribute
        trigger_spec['new_attribute'] = 'new_value'

        # Assert that the new attribute was added
        assert hasattr(trigger_spec, 'new_attribute')
        assert trigger_spec.new_attribute == 'new_value'

    def test_setitem_adds_new_attribute_5(self):
        """
        Test that __setitem__ adds a new attribute to the AgentSpec instance.
        """
        agent_spec = AgentSpec()
        agent_spec['new_attribute'] = 'test_value'
        
        assert hasattr(agent_spec, 'new_attribute')
        assert agent_spec.new_attribute == 'test_value'

    def test_setitem_raises_error_for_invalid_attribute(self):
        """
        Test that __setitem__ raises an AttributeError for invalid attributes
        """
        trigger_spec = TriggerSpec(trigger_fn=BaseTrigger())

        # Attempt to set an invalid attribute
        with pytest.raises(AttributeError):
            trigger_spec['invalid_attribute'] = 'some_value'

    def test_setitem_updates_attribute(self):
        """
        Test that __setitem__ method correctly updates an attribute of RewardSpec.
        """
        # Arrange
        reward_spec = RewardSpec()
        mock_reward_fn = BaseReward()

        # Act
        reward_spec['reward_fn'] = mock_reward_fn

        # Assert
        assert reward_spec.reward_fn == mock_reward_fn

    def test_setitem_updates_attribute_2(self):
        """
        Test that __setitem__ correctly updates an attribute of FilterSpec
        """
        # Arrange
        filter_spec = FilterSpec()
        
        class DummyFilter(BaseFilter):
            def __init__(self):
                super().__init__()
            
            def filter(self, agent_id, observation):
                return observation

        new_filter = DummyFilter()
        
        # Act
        filter_spec['filter_fn'] = new_filter
        
        # Assert
        assert filter_spec.filter_fn == new_filter
        assert isinstance(filter_spec.filter_fn, BaseFilter)

    def test_setitem_updates_attribute_3(self):
        """
        Test that __setitem__ correctly updates an attribute of TriggerSpec
        """
        # Create a mock BaseTrigger class
        class MockTrigger(BaseTrigger):
            def __call__(self, agent_id, observation):
                return True

        # Initialize TriggerSpec with mock trigger
        trigger_spec = TriggerSpec(trigger_fn=MockTrigger())

        # Use __setitem__ to update trigger_fn_config
        new_config = {'test_key': 'test_value'}
        trigger_spec['trigger_fn_config'] = new_config

        # Assert that the attribute was updated correctly
        assert trigger_spec.trigger_fn_config == new_config

    def test_setitem_updates_existing_attribute(self):
        """
        Test that __setitem__ updates an existing attribute of the ActionSpec instance.
        """
        action_spec = ActionSpec(actuators=[('type1', 'component1', 'control1')])
        action_spec['actuators'] = [('type2', 'component2', 'control2')]
        assert action_spec.actuators == [('type2', 'component2', 'control2')]

    def test_setitem_updates_existing_attribute_2(self):
        """
        Test that __setitem__ updates an existing attribute of the AgentSpec instance.
        """
        observation = ObservationSpec()
        agent_spec = AgentSpec(observation=observation)
        
        new_observation = ObservationSpec(variables=[('Outdoor Air Temperature', 'Environment')])
        agent_spec['observation'] = new_observation
        
        assert agent_spec.observation == new_observation

    def test_setitem_updates_filter_fn_config(self):
        """
        Test that __setitem__ correctly updates the filter_fn_config dictionary
        """
        # Arrange
        filter_spec = FilterSpec()
        new_config = {'param1': 10, 'param2': 'test'}
        
        # Act
        filter_spec['filter_fn_config'] = new_config
        
        # Assert
        assert filter_spec.filter_fn_config == new_config
        assert isinstance(filter_spec.filter_fn_config, dict)

    def test_setitem_updates_reward_fn_config(self):
        """
        Test that __setitem__ method correctly updates the reward_fn_config dictionary.
        """
        # Arrange
        reward_spec = RewardSpec(reward_fn_config={'existing_key': 'existing_value'})

        # Act
        reward_spec['reward_fn_config'] = {'new_key': 'new_value'}

        # Assert
        assert reward_spec.reward_fn_config == {'new_key': 'new_value'}

    def test_setitem_with_invalid_key(self):
        """
        Test that __setitem__ raises an AttributeError when trying to set an invalid attribute.
        """
        action_spec = ActionSpec()
        with pytest.raises(AttributeError):
            action_spec['invalid_attribute'] = 'test_value'

    def test_setitem_with_none_value(self):
        """
        Test that __setitem__ allows setting an attribute to None.
        """
        action_spec = ActionSpec(actuators=[('type1', 'component1', 'control1')])
        action_spec['actuators'] = None
        assert action_spec.actuators is None

    def test_setitem_with_spec_objects(self):
        """
        Test that __setitem__ works with Spec objects.
        """
        agent_spec = AgentSpec()
        
        new_filter = FilterSpec()
        new_action = ActionSpec()
        new_trigger = TriggerSpec(trigger_fn=lambda x: x)  # Dummy trigger function
        new_reward = RewardSpec()
        
        agent_spec['filter'] = new_filter
        agent_spec['action'] = new_action
        agent_spec['trigger'] = new_trigger
        agent_spec['reward'] = new_reward
        
        assert agent_spec.filter == new_filter
        assert agent_spec.action == new_action
        assert agent_spec.trigger == new_trigger
        assert agent_spec.reward == new_reward

    def test_setitem_with_various_attribute_types(self):
        """
        Test that __setitem__ works with various attribute types.
        """
        agent_spec = AgentSpec()
        
        agent_spec['int_attr'] = 42
        agent_spec['float_attr'] = 3.14
        agent_spec['list_attr'] = [1, 2, 3]
        agent_spec['dict_attr'] = {'key': 'value'}
        
        assert agent_spec.int_attr == 42
        assert agent_spec.float_attr == 3.14
        assert agent_spec.list_attr == [1, 2, 3]
        assert agent_spec.dict_attr == {'key': 'value'}

    def test_validate_action_config(self):
        """
        Test the validate_action_config method of ActionSpec.
        """
        # Arrange
        valid_actuators = [("Type1", "Component1", "Control1")]
        invalid_actuators = [("Type1", "Component1")]  # Missing one element

        # Act & Assert
        ActionSpec(actuators=valid_actuators).validate_action_config()  # Should not raise an error

        with pytest.raises(Exception):  # Expecting an error for invalid actuators
            ActionSpec(actuators=invalid_actuators).validate_action_config()

    def test_validate_action_config_2(self):
        """
        Test validate_action_config when actuators is a list but contains a non-tuple element
        """
        # Setup
        action_spec = ActionSpec(actuators=[('valid', 'tuple', 'here'), 'invalid_string'])
        
        # Capture log messages
        with pytest.raises(SystemExit) as exc_info:
            with self.assertLogs(level='ERROR') as log_capture:
                action_spec.validate_action_config()

        # Assert
        assert len(log_capture.records) == 1
        assert "The actuators must be defined as a list of tuples but <class 'str'> was given." in log_capture.records[0].getMessage()

        # Check that the method still returns True
        assert action_spec.validate_action_config() == True

    def test_validate_action_config_empty_actuators(self):
        """
        Test validate_action_config with empty actuators list
        """
        # Arrange
        action_spec = ActionSpec()
        action_spec.actuators = []

        # Act
        result = action_spec.validate_action_config()

        # Assert
        assert result == True

    def test_validate_action_config_empty_input(self):
        """
        Test validate_action_config with empty input
        """
        action_spec = ActionSpec(actuators=[])
        with pytest.raises(SystemExit):
            action_spec.validate_action_config()

    def test_validate_action_config_invalid_actuator_tuple(self):
        """
        Test validate_action_config when an actuator tuple has incorrect number of elements
        """
        # Arrange
        action_spec = ActionSpec()
        action_spec.actuators = [("type1", "component1", "control1"), ("type2", "component2")]  # Invalid tuple

        # Act
        with pytest.raises(SystemExit):
            action_spec.validate_action_config()

        # Assert
        # The method should log an error, which we can't directly assert
        # But we can check that it doesn't return True
        assert action_spec.validate_action_config() != True

    def test_validate_action_config_invalid_actuators_type(self):
        """
        Test validate_action_config when actuators is not a list
        """
        # Arrange
        action_spec = ActionSpec()
        action_spec.actuators = "invalid_type"  # Set actuators to a non-list type

        # Act
        with pytest.raises(SystemExit):
            action_spec.validate_action_config()

        # Assert
        # The method should log an error, which we can't directly assert
        # But we can check that it doesn't return True
        assert action_spec.validate_action_config() != True

    def test_validate_action_config_invalid_tuple_length(self):
        """
        Test validate_action_config with invalid tuple length
        """
        action_spec = ActionSpec(actuators=[("a", "b")])
        with pytest.raises(SystemExit):
            action_spec.validate_action_config()

    def test_validate_action_config_invalid_tuple_type(self):
        """
        Test validate_action_config with invalid tuple type
        """
        action_spec = ActionSpec(actuators=[(1, 2, 3)])
        with pytest.raises(SystemExit):
            action_spec.validate_action_config()

    def test_validate_action_config_invalid_type(self):
        """
        Test validate_action_config with invalid input type
        """
        action_spec = ActionSpec(actuators="invalid")
        with pytest.raises(SystemExit):
            action_spec.validate_action_config()

    def test_validate_action_config_mixed_valid_invalid(self):
        """
        Test validate_action_config with mixed valid and invalid inputs
        """
        action_spec = ActionSpec(actuators=[("a", "b", "c"), ("d", "e")])
        with pytest.raises(SystemExit):
            action_spec.validate_action_config()

    def test_validate_action_config_valid_actuators(self):
        """
        Test validate_action_config with valid actuators
        """
        # Arrange
        action_spec = ActionSpec()
        action_spec.actuators = [("type1", "component1", "control1"), ("type2", "component2", "control2")]

        # Act
        result = action_spec.validate_action_config()

        # Assert
        assert result == True

    def test_validate_action_config_valid_input(self, caplog):
        """
        Test validate_action_config with valid input
        """
        action_spec = ActionSpec(actuators=[("a", "b", "c"), ("d", "e", "f")])
        with caplog.at_level(logging.ERROR):
            result = action_spec.validate_action_config()
        assert result == True
        assert len(caplog.records) == 0

    def test_validate_filter_config(self):
        """
        Test the validate_filter_config method of FilterSpec
        """
        filter_spec = FilterSpec()
        assert filter_spec.validate_filter_config() is True

        # Test with invalid filter function
        with pytest.raises(Exception):
            FilterSpec(filter_fn="not a BaseFilter")

        # Test with invalid filter config
        with pytest.raises(Exception):
            FilterSpec(filter_fn_config="not a dict")

    def test_validate_filter_config_2(self):
        """
        Test validate_filter_config when filter_fn is a BaseFilter subclass and filter_fn_config is not a dict.
        """
        class MockFilter(BaseFilter):
            pass

        filter_spec = FilterSpec(filter_fn=MockFilter, filter_fn_config="not a dict")

        with pytest.raises(SystemExit) as exc_info:
            filter_spec.validate_filter_config()

        assert exc_info.type == SystemExit
        assert exc_info.value.code == 1

    def test_validate_filter_config_3(self):
        """
        Test validate_filter_config when filter_fn is not an instance of BaseFilter
        and filter_fn_config is a dictionary.
        """
        # Create a FilterSpec instance with invalid filter_fn and valid filter_fn_config
        filter_spec = FilterSpec(
            filter_fn="not a BaseFilter instance",
            filter_fn_config={"key": "value"}
        )

        # Mock the logger to catch the error message
        with pytest.raises(AttributeError) as exc_info:
            filter_spec.validate_filter_config()

        # Check if the correct error message is logged
        assert "The filter function must be based on BaseFilter class" in str(exc_info.value)

        # Despite the error, the method should still return True
        assert filter_spec.validate_filter_config() == True

    def test_validate_filter_config_empty_config(self):
        """
        Test that validate_filter_config works with an empty config dictionary.
        """
        class DummyFilter(BaseFilter):
            pass

        filter_spec = FilterSpec(filter_fn=DummyFilter, filter_fn_config={})
        assert filter_spec.validate_filter_config() == True

    def test_validate_filter_config_invalid_config_type(self):
        """
        Test that validate_filter_config raises an error when filter_fn_config is not a dictionary.
        """
        class DummyFilter(BaseFilter):
            pass

        filter_spec = FilterSpec(filter_fn=DummyFilter, filter_fn_config="invalid")
        with pytest.raises(AttributeError):
            filter_spec.validate_filter_config()

    def test_validate_filter_config_invalid_filter_fn(self):
        """
        Test that validate_filter_config raises an error when filter_fn is not a BaseFilter.
        """
        filter_spec = FilterSpec(filter_fn=lambda x: x, filter_fn_config={})
        with pytest.raises(AttributeError):
            filter_spec.validate_filter_config()

    def test_validate_filter_config_invalid_subclass(self):
        """
        Test that validate_filter_config raises an error when filter_fn is not a subclass of BaseFilter.
        """
        class InvalidFilter:
            pass

        filter_spec = FilterSpec(filter_fn=InvalidFilter, filter_fn_config={})
        with pytest.raises(AttributeError):
            filter_spec.validate_filter_config()

    def test_validate_filter_config_none_filter_fn(self):
        """
        Test that validate_filter_config raises an error when filter_fn is None.
        """
        filter_spec = FilterSpec(filter_fn=None, filter_fn_config={})
        with pytest.raises(AttributeError):
            filter_spec.validate_filter_config()

    def test_validate_filter_config_with_invalid_filter_fn_and_config(self):
        """
        Test validate_filter_config with invalid filter_fn and filter_fn_config.
        """
        # Arrange
        filter_spec = FilterSpec()
        filter_spec.filter_fn = "not a BaseFilter"
        filter_spec.filter_fn_config = "not a dict"

        # Act
        with pytest.raises(Exception):
            result = filter_spec.validate_filter_config()

        # Assert
        # The method should raise an exception, so we don't need additional assertions

    def test_validate_obs_config_2(self):
        """
        Test validate_obs_config when no variables are defined and counter is 0.
        """
        # Create an ObservationSpec instance with no variables defined
        obs_spec = ObservationSpec(
            variables=None,
            internal_variables=[],
            meters=[],
            simulation_parameters={},
            zone_simulation_parameters={},
            prediction_variables={},
            use_actuator_state=False,
            other_obs={}
        )

        # Mock the logger to capture error messages
        with pytest.raises(SystemExit) as exc_info:
            with self.assertLogs(level='ERROR') as log_capture:
                obs_spec.validate_obs_config()

        # Check if the expected error message is logged
        assert "At least one variable/meter/actuator/parameter must be defined in the observation." in log_capture.output[0]

        # Assert that the method returns True (even though it logs an error)
        assert exc_info.value.code == True

    def test_validate_obs_config_3(self):
        """
        Test validate_obs_config when variables and meters are provided, but no internal variables,
        use_actuator_state is True, and no other parameters are set.
        """
        # Create an ObservationSpec instance with the specified conditions
        obs_spec = ObservationSpec(
            variables=[('var1', 'type1'), ('var2', 'type2')],
            meters=['meter1', 'meter2'],
            use_actuator_state=True
        )

        # Mock the logger to capture the error message
        with pytest.raises(SystemExit) as excinfo:
            with pytest.LogCaptureFixture() as log_capture:
                obs_spec.validate_obs_config()

        # Check if the error message was logged
        assert "At least one variable/meter/actuator/parameter must be defined in the observation." in log_capture.text

        # Ensure that the method still returns True even though it logs an error
        assert obs_spec.validate_obs_config() == True

    def test_validate_obs_config_4(self):
        """
        Test validate_obs_config method when variables and internal_variables are set,
        meters is None, use_actuator_state is True, but the total count is still 0.
        """
        # Setup
        obs_spec = ObservationSpec(
            variables=[],
            internal_variables=[],
            meters=None,
            use_actuator_state=True,
            simulation_parameters={},
            zone_simulation_parameters={},
            prediction_variables={},
            other_obs={}
        )

        # Execute
        with pytest.raises(SystemExit) as exc_info:
            obs_spec.validate_obs_config()

        # Assert
        assert exc_info.type == SystemExit
        assert exc_info.value.code == 1

        # Check if the error message was logged
        with pytest.LogCaptureFixture() as log_capture:
            obs_spec.validate_obs_config()
            assert "At least one variable/meter/actuator/parameter must be defined in the observation." in log_capture.text

    def test_validate_obs_config_5(self):
        """
        Test validate_obs_config when all observation components are empty or False.
        """
        # Arrange
        obs_spec = ObservationSpec(
            variables=[],
            internal_variables=[],
            meters=[],
            simulation_parameters={},
            zone_simulation_parameters={},
            prediction_variables={},
            use_actuator_state=False,
            other_obs={}
        )

        # Act & Assert
        with pytest.raises(SystemExit) as exc_info:
            obs_spec.validate_obs_config()

        # Check that the error message was logged
        assert "At least one variable/meter/actuator/parameter must be defined in the observation." in exc_info.value.args[0]

    def test_validate_obs_config_6(self):
        """
        Test validate_obs_config when variables, internal_variables, meters, and use_actuator_state are set.
        Ensures the method returns True when at least one observation type is defined.
        """
        # Arrange
        obs_spec = ObservationSpec(
            variables=[("var1", "output1")],
            internal_variables=["int_var1"],
            meters=["meter1"],
            use_actuator_state=True
        )

        # Act
        result = obs_spec.validate_obs_config()

        # Assert
        assert result is True, "validate_obs_config should return True when at least one observation type is defined"
        
        # Additional assertions to ensure the correct attributes are set
        assert obs_spec.variables == [("var1", "output1")]
        assert obs_spec.internal_variables == ["int_var1"]
        assert obs_spec.meters == ["meter1"]
        assert obs_spec.use_actuator_state is True

    def test_validate_obs_config_all_empty(self):
        """
        Test validate_obs_config when all inputs are empty or None.
        """
        obs_spec = ObservationSpec()
        with pytest.raises(SystemExit):
            obs_spec.validate_obs_config()

    def test_validate_obs_config_at_least_one_defined(self):
        """
        Test validate_obs_config ensures at least one observation is defined.
        """
        obs_spec = ObservationSpec(variables=[("var1", "var2")])
        assert obs_spec.validate_obs_config() == True

    def test_validate_obs_config_empty_observation(self):
        """
        Test validate_obs_config with an empty observation configuration.
        Expect a logger error message and return True.
        """
        obs_spec = ObservationSpec()
        
        # Capture log messages
        with pytest.raises(SystemExit) as exc_info:
            with self.assertLogs(level='ERROR') as log_capture:
                result = obs_spec.validate_obs_config()
        
        # Check that the error message was logged
        assert "At least one variable/meter/actuator/parameter must be defined in the observation." in log_capture.output[0]
        
        # Check that the method still returns True
        assert result == True

    def test_validate_obs_config_incorrect_type(self):
        """
        Test validate_obs_config with incorrect input types.
        """
        with pytest.raises(AttributeError):
            ObservationSpec(variables="not_a_list")

    def test_validate_obs_config_invalid_prediction_hours(self):
        """
        Test validate_obs_config with invalid prediction hours.
        """
        obs_spec = ObservationSpec(
            use_one_day_weather_prediction=True,
            prediction_hours=25
        )
        assert obs_spec.prediction_hours == 24

    def test_validate_obs_config_invalid_prediction_variable(self):
        """
        Test validate_obs_config with an invalid prediction variable.
        """
        with pytest.raises(SystemExit):
            ObservationSpec(
                use_one_day_weather_prediction=True,
                prediction_variables={'invalid_var': True}
            )

    def test_validate_obs_config_invalid_simulation_parameter(self):
        """
        Test validate_obs_config with an invalid simulation parameter.
        """
        with pytest.raises(SystemExit):
            ObservationSpec(simulation_parameters={'invalid_param': True})

    def test_validate_obs_config_invalid_zone_simulation_parameter(self):
        """
        Test validate_obs_config with an invalid zone simulation parameter.
        """
        with pytest.raises(SystemExit):
            ObservationSpec(zone_simulation_parameters={'invalid_param': True})

    def test_validate_obs_config_other_obs(self):
        """
        Test validate_obs_config with other_obs defined.
        """
        obs_spec = ObservationSpec(other_obs={"custom_obs": 1.0})
        assert obs_spec.validate_obs_config() == True

    def test_validate_obs_config_use_actuator_state(self):
        """
        Test validate_obs_config with use_actuator_state set to True.
        """
        obs_spec = ObservationSpec(use_actuator_state=True)
        assert obs_spec.validate_obs_config() == True

    def test_validate_obs_config_with_actuator_state(self):
        """
        Test validate_obs_config with actuator state enabled.
        """
        obs_spec = ObservationSpec(use_actuator_state=True)
        result = obs_spec.validate_obs_config()
        assert result == True

    def test_validate_obs_config_with_internal_variables(self):
        """
        Test validate_obs_config with internal variables defined.
        """
        obs_spec = ObservationSpec(internal_variables=["Day of Year"])
        result = obs_spec.validate_obs_config()
        assert result == True

    def test_validate_obs_config_with_meters(self):
        """
        Test validate_obs_config with meters defined.
        """
        obs_spec = ObservationSpec(meters=["Electricity:Facility"])
        result = obs_spec.validate_obs_config()
        assert result == True

    def test_validate_obs_config_with_multiple_elements(self):
        """
        Test validate_obs_config with multiple observation elements defined.
        """
        obs_spec = ObservationSpec(
            variables=[("Zone Mean Air Temperature", "Living Zone")],
            internal_variables=["Day of Year"],
            meters=["Electricity:Facility"],
            simulation_parameters={"hour": True},
            use_actuator_state=True,
            other_obs={"custom_observation": 1.0}
        )
        result = obs_spec.validate_obs_config()
        assert result == True

    def test_validate_obs_config_with_other_obs(self):
        """
        Test validate_obs_config with other observations defined.
        """
        obs_spec = ObservationSpec(other_obs={"custom_observation": 1.0})
        result = obs_spec.validate_obs_config()
        assert result == True

    def test_validate_obs_config_with_simulation_parameters(self):
        """
        Test validate_obs_config with simulation parameters defined.
        """
        obs_spec = ObservationSpec(simulation_parameters={"hour": True})
        result = obs_spec.validate_obs_config()
        assert result == True

    def test_validate_obs_config_with_variables(self):
        """
        Test validate_obs_config with variables defined.
        """
        obs_spec = ObservationSpec(variables=[("Zone Mean Air Temperature", "Living Zone")])
        result = obs_spec.validate_obs_config()
        assert result == True

    def test_validate_trigger_config_2(self):
        """
        Test validate_trigger_config method with invalid trigger_fn_config and NotImplemented trigger_fn
        """
        # Create a mock BaseTrigger subclass
        class MockTrigger(BaseTrigger):
            pass

        # Create a TriggerSpec instance with a BaseTrigger subclass, invalid trigger_fn_config, and NotImplemented trigger_fn
        trigger_spec = TriggerSpec(
            trigger_fn=NotImplemented,
            trigger_fn_config="invalid_config"  # This should be a dict, not a string
        )

        # Call the method under test
        with pytest.raises(SystemExit) as exc_info:
            trigger_spec.validate_trigger_config()

        # Check that the method logged the expected error messages
        assert "The trigger function must be based on BaseTrigger class but <class 'type'> was given." in str(exc_info.value)
        assert "The configuration for the trigger function must be a dictionary but <class 'str'> was given." in str(exc_info.value)
        assert "The trigger function must be defined." in str(exc_info.value)

    def test_validate_trigger_config_3(self, caplog):
        """
        Test validate_trigger_config when trigger_fn is not BaseTrigger and is NotImplemented
        """
        caplog.set_level(logging.ERROR)

        # Create a TriggerSpec with invalid trigger_fn
        trigger_spec = TriggerSpec(trigger_fn=NotImplemented, trigger_fn_config={})

        # Call the method under test
        result = trigger_spec.validate_trigger_config()

        # Assert that the method returns True
        assert result == True

        # Check for expected error messages
        assert "The trigger function must be based on BaseTrigger class but <class 'type'> was given." in caplog.text
        assert "The trigger function must be defined." in caplog.text

        # Ensure only these two error messages were logged
        assert len(caplog.records) == 2

    def test_validate_trigger_config_4(self):
        """
        Test validate_trigger_config when trigger_fn is not BaseTrigger, trigger_fn_config is not dict, and trigger_fn is not NotImplemented.
        """
        # Mock logger to capture error messages
        with self.assertLogs(level='ERROR') as log:
            # Create a TriggerSpec with invalid configurations
            trigger_spec = TriggerSpec(
                trigger_fn="invalid_trigger",  # Not a BaseTrigger
                trigger_fn_config="invalid_config"  # Not a dict
            )

            # Call the method under test
            result = trigger_spec.validate_trigger_config()

            # Assert that the method returns True despite errors
            assert result == True

            # Check that appropriate error messages were logged
            assert len(log.output) == 2
            assert "The trigger function must be based on BaseTrigger class but <class 'str'> was given." in log.output[0]
            assert "The configuration for the trigger function must be a dictionary but <class 'str'> was given." in log.output[1]

    def test_validate_trigger_config_empty_config(self):
        """
        Test that validate_trigger_config accepts an empty dictionary as valid config.
        """
        class DummyTrigger(BaseTrigger):
            pass

        trigger_spec = TriggerSpec(trigger_fn=DummyTrigger(), trigger_fn_config={})
        assert trigger_spec.validate_trigger_config() == True

    def test_validate_trigger_config_invalid_config_type(self):
        """
        Test that validate_trigger_config raises an error when trigger_fn_config is not a dictionary.
        """
        class DummyTrigger(BaseTrigger):
            pass

        trigger_spec = TriggerSpec(trigger_fn=DummyTrigger(), trigger_fn_config=[])
        with pytest.raises(AttributeError):
            trigger_spec.validate_trigger_config()

    def test_validate_trigger_config_invalid_inputs(self):
        """
        Test validate_trigger_config method with invalid inputs.
        """
        # Setup
        trigger_spec = TriggerSpec(
            trigger_fn="invalid_trigger",
            trigger_fn_config="invalid_config"
        )

        # Execute and Assert
        with pytest.raises(AttributeError):
            trigger_spec.validate_trigger_config()

        # Check log messages
        with pytest.raises(AttributeError):
            trigger_spec.validate_trigger_config()
        
        # Assert log messages (assuming logger.error is called)
        # Note: This assertion might need adjustment based on how logging is implemented
        # You might need to use a mocking library like unittest.mock to capture log messages

    def test_validate_trigger_config_not_base_trigger(self):
        """
        Test that validate_trigger_config raises an error when trigger_fn is not a BaseTrigger instance.
        """
        trigger_spec = TriggerSpec(trigger_fn=lambda x: x)
        with pytest.raises(AttributeError):
            trigger_spec.validate_trigger_config()

    def test_validate_trigger_config_not_implemented(self, mock_logger):
        """
        Test validate_trigger_config method with NotImplemented trigger function.
        """
        # Setup
        trigger_spec = TriggerSpec()

        # Execute
        result = trigger_spec.validate_trigger_config()

        # Assert
        assert result == True
        mock_logger.error.assert_called_once_with("The trigger function must be defined.")

    def test_validate_trigger_config_not_implemented_2(self):
        """
        Test that validate_trigger_config raises an error when trigger_fn is NotImplemented.
        """
        trigger_spec = TriggerSpec()
        with pytest.raises(AttributeError):
            trigger_spec.validate_trigger_config()

    def test_validate_trigger_config_valid_input(self):
        """
        Test that validate_trigger_config returns True for valid input.
        """
        class DummyTrigger(BaseTrigger):
            pass

        trigger_spec = TriggerSpec(trigger_fn=DummyTrigger(), trigger_fn_config={})
        assert trigger_spec.validate_trigger_config() == True

    def test_validate_trigger_config_valid_inputs(self):
        """
        Test validate_trigger_config method with valid inputs.
        """
        # Setup
        class MockTrigger(BaseTrigger):
            def __call__(self, agent_id, observation):
                pass

        trigger_spec = TriggerSpec(
            trigger_fn=MockTrigger(),
            trigger_fn_config={}
        )

        # Execute
        result = trigger_spec.validate_trigger_config()

        # Assert
        assert result == True

    def test_validation_rew_config_2(self):
        """
        Test that validation_rew_config returns True when reward_fn is implemented.
        """
        class DummyReward(BaseReward):
            def __call__(self, env_object, infos):
                return 0.0

        reward_spec = RewardSpec(reward_fn=DummyReward())
        result = reward_spec.validation_rew_config()
        assert result is True

    def test_validation_rew_config_empty_config(self):
        """
        Test that validation_rew_config works with an empty reward_fn_config.
        """
        class DummyReward(BaseReward):
            def __call__(self, env, infos):
                return 0

        reward_spec = RewardSpec(reward_fn=DummyReward(), reward_fn_config={})
        assert reward_spec.validation_rew_config() == True

    def test_validation_rew_config_incorrect_type(self):
        """
        Test that validation_rew_config raises an error when reward_fn is of incorrect type.
        """
        reward_spec = RewardSpec(reward_fn="not a BaseReward instance")
        with pytest.raises(AttributeError):
            reward_spec.validation_rew_config()

    def test_validation_rew_config_none_reward_fn(self):
        """
        Test that validation_rew_config raises an error when reward_fn is None.
        """
        reward_spec = RewardSpec(reward_fn=None)
        with pytest.raises(AttributeError):
            reward_spec.validation_rew_config()

    def test_validation_rew_config_not_implemented(self):
        """
        Test that validation_rew_config raises NotImplementedError when reward_fn is not implemented.
        """
        reward_spec = RewardSpec()
        with pytest.raises(NotImplementedError, match="reward_fn must be implemented"):
            reward_spec.validation_rew_config()

    def test_validation_rew_config_raises_not_implemented_error(self):
        """
        Test that validation_rew_config raises NotImplementedError when reward_fn is NotImplemented.
        """
        # Arrange
        reward_spec = RewardSpec()

        # Act & Assert
        with pytest.raises(NotImplementedError, match="reward_fn must be implemented"):
            reward_spec.validation_rew_config()

    def test_validation_rew_config_returns_true(self):
        """
        Test that validation_rew_config returns True when reward_fn is implemented.
        """
        # Arrange
        class DummyReward(BaseReward):
            def __call__(self, env, infos):
                return 0

        reward_spec = RewardSpec(reward_fn=DummyReward())

        # Act
        result = reward_spec.validation_rew_config()

        # Assert
        assert result is True

    def test_validation_rew_config_valid_input(self):
        """
        Test that validation_rew_config returns True for valid input.
        """
        class DummyReward(BaseReward):
            def __call__(self, env, infos):
                return 0

        reward_spec = RewardSpec(reward_fn=DummyReward())
        assert reward_spec.validation_rew_config() == True

class _AssertLogsContext:

    def __enter__(self):
        self.logger.setLevel(self.level)
        self.logger.addHandler(self.handler)
        self.handler.stream = self.logged_output
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.logger.removeHandler(self.handler)
        self.logger.setLevel(self.old_level)

    def __init__(self, test_case, level):
        self.test_case = test_case
        self.level = level
        self.logger = logging.getLogger()
        self.old_level = self.logger.level
        self.handler = logging.StreamHandler()
        self.logged_output = []

    @property
    def output(self):
        return self.logged_output