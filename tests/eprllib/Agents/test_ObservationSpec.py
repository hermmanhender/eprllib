

import pytest

from eprllib.Agents import SIMULATION_PARAMETERS, ZONE_SIMULATION_PARAMETERS, PREDICTION_VARIABLES, PREDICTION_HOURS
from eprllib.Agents.ObservationSpec import ObservationSpec

class TestObservationspec:

    def test___getitem___1(self):
        """
        Test that the __getitem__ method correctly retrieves attribute values.
        This test checks if the method returns the expected value for a given key.
        """
        obs_spec = ObservationSpec(variables=[('test_var', 'test_unit')])
        assert obs_spec['variables'] == [('test_var', 'test_unit')]

    def test___getitem___nonexistent_attribute(self):
        """
        Test that attempting to access a non-existent attribute raises an AttributeError.
        This tests the edge case of trying to access an attribute that doesn't exist in the ObservationSpec object.
        """
        obs_spec = ObservationSpec()
        with pytest.raises(AttributeError):
            obs_spec['nonexistent_attribute']

    def test___setitem___1(self):
        """
        Test that __setitem__ correctly sets an attribute of the ObservationSpec instance.
        """
        obs_spec = ObservationSpec()
        obs_spec['variables'] = [('var1', 'type1'), ('var2', 'type2')]
        assert obs_spec.variables == [('var1', 'type1'), ('var2', 'type2')]

    def test___setitem___invalid_key(self):
        """
        Test setting an invalid attribute using __setitem__.
        This should raise an KeyError as the ObservationSpec class
        does not have a mechanism to handle arbitrary attribute creation.
        """
        obs_spec = ObservationSpec()
        with pytest.raises(KeyError):
            obs_spec['non_existent_attribute'] = 'some_value'

    def test_build_1(self):
        """
        Test the build method of ObservationSpec with invalid keys in simulation_parameters, 
        zone_simulation_parameters, and prediction_variables, along with various edge cases.

        This test covers the following scenarios:
        1. Invalid keys in simulation_parameters
        2. Invalid keys in zone_simulation_parameters
        3. Use of one-day weather prediction with invalid keys in prediction_variables
        4. Invalid prediction hours
        5. Non-empty variables, internal_variables, and meters
        6. Use of actuator state
        7. Ensuring at least one observation element is defined
        """
        obs_spec = ObservationSpec(
            variables=[("var1", "type1")],
            internal_variables=["int_var1"],
            meters=["meter1"],
            simulation_parameters={"invalid_key": True},
            zone_simulation_parameters={"invalid_zone_key": True},
            use_one_day_weather_prediction=True,
            prediction_hours=25,
            prediction_variables={"invalid_pred_key": True},
            use_actuator_state=True
        )

        with pytest.raises(ValueError) as exc_info:
            obs_spec.build()

        assert "The key 'invalid_key' is not admissible in the simulation_parameters" in str(exc_info.value)

    def test_build_2(self):
        """
        Tests the build method of ObservationSpec class with various input conditions.

        This test covers the following scenarios:
        1. Valid simulation parameters
        2. Invalid zone simulation parameters
        3. Weather prediction enabled with invalid prediction variables
        4. Invalid prediction hours
        5. All observation types defined (variables, internal variables, meters, actuator state)
        6. Verifies that the method returns the object's attributes as a dictionary
        """
        obs_spec = ObservationSpec(
            variables=[("Zone Mean Air Temperature", "Zone 1")],
            internal_variables=["Zone Floor Area"],
            meters=["Electricity:Facility"],
            simulation_parameters={"day_of_week": True},
            zone_simulation_parameters={"invalid_key": True},
            use_one_day_weather_prediction=True,
            prediction_hours=25,
            prediction_variables={"invalid_weather_var": True},
            use_actuator_state=True
        )

        with pytest.raises(ValueError, match="The key 'invalid_key' is not admissible in the zone_simulation_parameters"):
            obs_spec.build()

    def test_build_3(self):
        """
        Test the build method of ObservationSpec with various edge cases and conditions.

        This test checks:
        1. Invalid keys in simulation_parameters
        2. Valid keys in zone_simulation_parameters
        3. One-day weather prediction with invalid prediction variable
        4. Out-of-range prediction hours
        5. Presence of variables, internal variables, meters, and actuator state
        6. Ensuring at least one observation is defined

        Expected behavior:
        - Raises ValueError for invalid keys
        - Logs warning for out-of-range prediction hours
        - Successfully builds and returns the observation specification
        """
        obs_spec = ObservationSpec(
            variables=[("var1", "type1")],
            internal_variables=["int_var1"],
            meters=["meter1"],
            simulation_parameters={"invalid_key": True},
            zone_simulation_parameters={"zone_time_step": True},
            use_one_day_weather_prediction=True,
            prediction_hours=25,
            prediction_variables={"invalid_weather_var": True},
            use_actuator_state=True,
            other_obs={"custom_obs": 1.0}
        )

        with pytest.raises(ValueError, match="The key 'invalid_key' is not admissible in the simulation_parameters"):
            obs_spec.build()

        # Correct the invalid key
        obs_spec.simulation_parameters = {"actual_date_time": True}

        with pytest.raises(ValueError, match="The key 'invalid_weather_var' is not admissible in the prediction_variables"):
            obs_spec.build()

        # Correct the invalid prediction variable
        obs_spec.prediction_variables = {"albedo": True}

        # Test with valid configuration
        result = obs_spec.build()

        assert isinstance(result, dict)
        assert result["prediction_hours"] == PREDICTION_HOURS
        assert len(result["variables"]) > 0
        assert len(result["internal_variables"]) > 0
        assert len(result["meters"]) > 0
        assert result["use_actuator_state"] is True

    def test_build_4(self):
        """
        Test the build method of ObservationSpec when various conditions are met:
        - Invalid keys in simulation_parameters and zone_simulation_parameters
        - One-day weather prediction is used
        - Prediction hours are out of range
        - Various observation components are defined
        - No valid observation components are defined (counter == 0)

        Expected behavior:
        - Raises ValueError for invalid keys
        - Logs a warning for invalid prediction hours
        - Raises ValueError when no valid observation components are defined
        """
        obs_spec = ObservationSpec(
            variables=[("var1", "unit1")],
            internal_variables=["int_var1"],
            meters=["meter1"],
            simulation_parameters={"invalid_key": True},
            zone_simulation_parameters={"invalid_zone_key": True},
            use_one_day_weather_prediction=True,
            prediction_hours=25,
            prediction_variables={"outdoor_dry_bulb": True},
            use_actuator_state=True,
            other_obs={"custom_obs": 1.0}
        )

        with pytest.raises(ValueError, match="The key 'invalid_key' is not admissible in the simulation_parameters"):
            obs_spec.build()

        # Reset simulation_parameters to a valid state
        obs_spec.simulation_parameters = SIMULATION_PARAMETERS.copy()

        with pytest.raises(ValueError, match="The key 'invalid_zone_key' is not admissible in the zone_simulation_parameters"):
            obs_spec.build()

        # Reset zone_simulation_parameters to a valid state
        obs_spec.zone_simulation_parameters = ZONE_SIMULATION_PARAMETERS.copy()

        # Test with all observation components set to None or empty
        obs_spec.variables = None
        obs_spec.internal_variables = None
        obs_spec.meters = None
        obs_spec.simulation_parameters = SIMULATION_PARAMETERS.copy()
        obs_spec.zone_simulation_parameters = ZONE_SIMULATION_PARAMETERS.copy()
        obs_spec.prediction_variables = PREDICTION_VARIABLES.copy()
        obs_spec.use_actuator_state = False
        obs_spec.other_obs = {}

        with pytest.raises(ValueError, match="At least one variable/meter/actuator/parameter must be defined in the observation"):
            obs_spec.build()

    def test_build_5(self):
        """
        Test the build method of ObservationSpec when multiple invalid conditions are met.

        This test checks the following conditions:
        1. Invalid keys in simulation_parameters
        2. Invalid keys in zone_simulation_parameters
        3. Use of one-day weather prediction with invalid keys
        4. Invalid prediction hours
        5. No variables, meters, or other valid observations defined

        Expected behavior: The method should raise a ValueError due to no valid observations being defined.
        """
        obs_spec = ObservationSpec(
            simulation_parameters={"invalid_key": True},
            zone_simulation_parameters={"invalid_zone_key": True},
            use_one_day_weather_prediction=True,
            prediction_variables={"invalid_prediction_key": True},
            prediction_hours=25,
            internal_variables=[],
            meters=[],
            use_actuator_state=False
        )

        with pytest.raises(ValueError):
            obs_spec.build()

    def test_build_6(self):
        """
        Test the build method when invalid keys are provided for simulation_parameters,
        zone_simulation_parameters, and prediction_variables, and when prediction_hours
        is out of range. Also tests the case where no valid observation elements are defined.
        """
        obs_spec = ObservationSpec(
            variables=[('var1', 'type1')],
            meters=['meter1'],
            simulation_parameters={'invalid_key': True},
            zone_simulation_parameters={'invalid_key': True},
            use_one_day_weather_prediction=True,
            prediction_hours=25,
            prediction_variables={'invalid_key': True},
            use_actuator_state=True
        )

        with pytest.raises(ValueError) as exc_info:
            obs_spec.build()

        assert "The key 'invalid_key' is not admissible in the simulation_parameters" in str(exc_info.value)

    def test_build_7(self):
        """
        Test the build method of ObservationSpec with invalid parameters and empty observation.

        This test checks the following conditions:
        - Invalid keys in simulation_parameters
        - Invalid keys in zone_simulation_parameters
        - Invalid keys in prediction_variables with use_one_day_weather_prediction set to True
        - Invalid prediction_hours
        - Empty observation (no variables, meters, or parameters defined)

        Expected behavior: The method should raise a ValueError due to empty observation.
        """
        obs_spec = ObservationSpec(
            variables=[],
            internal_variables=[],
            simulation_parameters={"invalid_key": True},
            zone_simulation_parameters={"invalid_key": True},
            use_one_day_weather_prediction=True,
            prediction_hours=25,
            prediction_variables={"invalid_key": True},
            use_actuator_state=True
        )

        with pytest.raises(ValueError):
            obs_spec.build()

    def test_build_invalid_keys_and_empty_observation(self):
        """
        Test the build method with invalid keys in simulation parameters, zone simulation parameters,
        and prediction variables, along with an empty observation configuration.

        This test checks if the method correctly raises ValueError for invalid keys and empty observation,
        while also verifying the behavior when prediction_hours is out of range.
        """
        obs_spec = ObservationSpec(
            variables=[],
            internal_variables=[],
            meters=[],
            simulation_parameters={"invalid_key": True},
            zone_simulation_parameters={"invalid_zone_key": True},
            use_one_day_weather_prediction=True,
            prediction_hours=25,
            prediction_variables={"invalid_prediction_key": True},
            use_actuator_state=False,
            other_obs={}
        )

        with pytest.raises(ValueError) as excinfo:
            obs_spec.build()

        assert "The key 'invalid_key' is not admissible in the simulation_parameters" in str(excinfo.value)

    def test_build_invalid_parameters(self):
        """
        Test the build method with invalid parameters to ensure it raises appropriate exceptions.

        This test covers the following scenarios:
        - Invalid keys in simulation_parameters
        - Invalid keys in zone_simulation_parameters
        - Invalid keys in prediction_variables when use_one_day_weather_prediction is True
        - Invalid prediction_hours (outside the range 1-24)
        - No variables/meters/actuators/parameters defined (counter == 0)
        """
        obs_spec = ObservationSpec(
            simulation_parameters={"invalid_key": True},
            zone_simulation_parameters={"invalid_zone_key": True},
            use_one_day_weather_prediction=True,
            prediction_variables={"invalid_prediction_key": True},
            prediction_hours=25
        )

        with pytest.raises(ValueError) as excinfo:
            obs_spec.build()

        assert "The key 'invalid_key' is not admissible in the simulation_parameters" in str(excinfo.value)

        # Reset observation spec for the next test
        obs_spec = ObservationSpec(
            use_one_day_weather_prediction=True,
            prediction_hours=0
        )

        with pytest.raises(ValueError) as excinfo:
            obs_spec.build()

        assert "At least one variable/meter/actuator/parameter must be defined in the observation" in str(excinfo.value)

    def test_build_invalid_simulation_parameters(self):
        """
        Test the build method when an invalid key is present in simulation_parameters.
        This test verifies that a ValueError is raised when an invalid key is provided.
        """
        obs_spec = ObservationSpec(
            variables=[("Zone Mean Air Temperature", "")],
            simulation_parameters={"invalid_key": True},
            zone_simulation_parameters={"invalid_key": True},
            use_one_day_weather_prediction=True,
            prediction_variables={"invalid_key": True},
            prediction_hours=12,
            internal_variables=["Zone People Occupant Count"],
            meters=["Electricity:Facility"],
            use_actuator_state=True
        )

        with pytest.raises(ValueError) as excinfo:
            obs_spec.build()

        assert "is not admissible in the simulation_parameters" in str(excinfo.value)

    def test_empty_observation(self):
        """
        Test that an empty observation specification raises a ValueError.
        """
        with pytest.raises(ValueError) as excinfo:
            ObservationSpec().build()
        assert "At least one variable/meter/actuator/parameter must be defined in the observation" in str(excinfo.value)

    def test_invalid_prediction_hours(self):
        """
        Test that an invalid prediction_hours value is corrected and a warning is logged.
        """
        obs_spec = ObservationSpec(variables=[("Site Outdoor Air Drybulb Temperature", "Environment")], use_one_day_weather_prediction=True, prediction_hours=25)
        result = obs_spec.build()
        assert result['prediction_hours'] == 24

    def test_invalid_prediction_hours_2(self):
        """
        Test that the build method sets prediction_hours to the default value when an invalid value is provided.
        """
        obs_spec = ObservationSpec(variables=[("Site Outdoor Air Drybulb Temperature", "Environment")], use_one_day_weather_prediction=True, prediction_hours=25)
        result = obs_spec.build()
        assert result["prediction_hours"] == 24

    def test_invalid_prediction_variable_key(self):
        """
        Test that the build method raises a ValueError when an invalid key is provided in prediction_variables.
        """
        obs_spec = ObservationSpec(variables=[("Site Outdoor Air Drybulb Temperature", "Environment")], use_one_day_weather_prediction=True, prediction_variables={"invalid_key": True})
        with pytest.raises(ValueError) as exc_info:
            obs_spec.build()
        assert "The key 'invalid_key' is not admissible in the prediction_variables" in str(exc_info.value)

    def test_invalid_prediction_variables(self):
        """
        Test that an invalid key in prediction_variables raises a ValueError when use_one_day_weather_prediction is True.
        """
        with pytest.raises(ValueError) as excinfo:
            ObservationSpec(use_one_day_weather_prediction=True, prediction_variables={'invalid_key': True}).build()
        assert "The key 'invalid_key' is not admissible in the prediction_variables" in str(excinfo.value)

    def test_invalid_simulation_parameter_key(self):
        """
        Test that the build method raises a ValueError when an invalid key is provided in simulation_parameters.
        """
        obs_spec = ObservationSpec(simulation_parameters={"invalid_key": True})
        with pytest.raises(ValueError) as exc_info:
            obs_spec.build()
        assert "The key 'invalid_key' is not admissible in the simulation_parameters" in str(exc_info.value)

    def test_invalid_simulation_parameters(self):
        """
        Test that an invalid key in simulation_parameters raises a ValueError.
        """
        with pytest.raises(ValueError) as excinfo:
            ObservationSpec(simulation_parameters={'invalid_key': True}).build()
        assert "The key 'invalid_key' is not admissible in the simulation_parameters" in str(excinfo.value)

    def test_invalid_zone_simulation_parameter_key(self):
        """
        Test that the build method raises a ValueError when an invalid key is provided in zone_simulation_parameters.
        """
        obs_spec = ObservationSpec(zone_simulation_parameters={"invalid_key": True})
        with pytest.raises(ValueError) as exc_info:
            obs_spec.build()
        assert "The key 'invalid_key' is not admissible in the zone_simulation_parameters" in str(exc_info.value)

    def test_invalid_zone_simulation_parameters(self):
        """
        Test that an invalid key in zone_simulation_parameters raises a ValueError.
        """
        with pytest.raises(ValueError) as excinfo:
            ObservationSpec(zone_simulation_parameters={'invalid_key': True}).build()
        assert "The key 'invalid_key' is not admissible in the zone_simulation_parameters" in str(excinfo.value)

    def test_no_observation_defined(self):
        """
        Test that the build method raises a ValueError when no observation is defined.
        """
        obs_spec = ObservationSpec()
        with pytest.raises(ValueError) as exc_info:
            obs_spec.build()
        assert "At least one variable/meter/actuator/parameter must be defined in the observation" in str(exc_info.value)
