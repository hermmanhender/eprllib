from eprllib.Agents.ActionSpec import ActionSpec
import io
import pytest
import sys

class TestActionspec:

    def test___getitem___1(self):
        """
        Test the __getitem__ method of ActionSpec class.

        This test verifies that the __getitem__ method correctly retrieves
        the value of an existing attribute using the key.
        """
        # Create an instance of ActionSpec with some actuators
        action_spec = ActionSpec(actuators=[('type1', 'component1', 'control1')])

        # Test retrieving the 'actuators' attribute
        assert action_spec['actuators'] == [('type1', 'component1', 'control1')]

    def test___getitem___invalid_key(self):
        """
        Test __getitem__ method with an invalid key.
        This tests the edge case where a non-existent attribute is accessed,
        which should raise an AttributeError according to the method's implementation.
        """
        action_spec = ActionSpec()
        with pytest.raises(AttributeError):
            _ = action_spec["non_existent_key"]

    def test___init___1(self):
        """
        Test the __init__ method of ActionSpec when no actuators are provided.

        This test verifies that:
        1. The ActionSpec object is created successfully without actuators.
        2. The actuators attribute is initialized as an empty list.
        3. A message is printed indicating no actuators were provided.
        """
        # Redirect stdout to capture print output
        captured_output = io.StringIO()
        sys.stdout = captured_output

        # Create ActionSpec object with no actuators
        action_spec = ActionSpec()

        # Reset stdout
        sys.stdout = sys.__stdout__

        # Assert that actuators is an empty list
        assert action_spec.actuators == []

        # Assert that the correct message was printed
        assert captured_output.getvalue().strip() == "No actuators provided."

    def test___init___2(self):
        """
        Test the initialization of ActionSpec with non-None actuators.

        This test verifies that when actuators are provided during initialization,
        they are correctly assigned to the ActionSpec instance.
        """
        actuators = [("type1", "component1", "control1"), ("type2", "component2", "control2")]
        action_spec = ActionSpec(actuators=actuators)
        assert action_spec.actuators == actuators

    def test___init___invalid_actuator_length(self):
        """
        Test the __init__ method with an invalid actuator tuple length.
        This test verifies that a ValueError is raised when an actuator tuple
        does not contain exactly 3 elements.
        """
        with pytest.raises(ValueError) as excinfo:
            ActionSpec(actuators=[("invalid", "tuple")]).build()
        assert "The actuators must be defined as a list of tuples of 3 elements" in str(excinfo.value)

    def test___init___invalid_actuator_type(self):
        """
        Test the __init__ method with an invalid actuator type.
        This test verifies that a ValueError is raised when the actuators
        argument is not a list of tuples.
        """
        with pytest.raises(ValueError) as excinfo:
            ActionSpec(actuators=["invalid_actuator"]).build()
        assert "The actuators must be defined as a list of tuples" in str(excinfo.value)

    def test___setitem___2(self):
        """
        Test that __setitem__ successfully sets a valid key-value pair.

        This test verifies that when a valid key is provided to __setitem__,
        the corresponding attribute is set to the given value without raising
        any exceptions.
        """
        action_spec = ActionSpec()
        action_spec['actuators'] = [('type1', 'component1', 'control1')]
        assert action_spec.actuators == [('type1', 'component1', 'control1')]

    def test___setitem___invalid_key(self):
        """
        Test that __setitem__ raises a KeyError when an invalid key is provided.
        This tests the explicit error handling in the method for invalid keys.
        """
        action_spec = ActionSpec()
        with pytest.raises(KeyError, match="Invalid key: invalid_key"):
            action_spec["invalid_key"] = "some_value"

    def test___setitem___invalid_key_2(self):
        """
        Tests the __setitem__ method of ActionSpec when an invalid key is provided.
        This test verifies that a KeyError is raised when attempting to set an item
        with a key that is not present in the object's __dict__.
        """
        action_spec = ActionSpec()
        with pytest.raises(KeyError) as exc_info:
            action_spec["invalid_key"] = "some_value"
        assert str(exc_info.value) == "'Invalid key: invalid_key.'"

    # def test_build_1(self):
    #     """
    #     Test that the build method raises a ValueError when actuators is not a list.
    #     """
    #     action_spec = ActionSpec(actuators="not a list")
    #     with pytest.raises(ValueError) as context:
    #         action_spec.build()

    #     assert "The actuators must be defined as a list of tuples but <class 'str'> was given." == str(context.value)

    # def test_build_2(self):
    #     """
    #     Test that build() raises ValueError when actuators contain a non-tuple element.

    #     This test verifies that the build() method raises a ValueError when the
    #     actuators list contains an element that is not a tuple. It covers the path
    #     where isinstance(self.actuators, list) is True, but isinstance(actuator, tuple)
    #     is False for at least one actuator.
    #     """
    #     action_spec = ActionSpec(actuators=[('a', 'b', 'c'), 'not_a_tuple'])
    #     with pytest.raises(ValueError, match="The actuators must be defined as a list of tuples but <class 'str'> was given."):
    #         action_spec.build()

    # def test_build_invalid_actuator_element_type(self):
    #     """
    #     Test the build method with an invalid actuator element type (not a tuple).
    #     This should raise a ValueError.
    #     """
    #     action_spec = ActionSpec(actuators=[["invalid"]])
    #     with pytest.raises(ValueError) as exc_info:
    #         action_spec.build()
    #     assert str(exc_info.value) == "The actuators must be defined as a list of tuples but <class 'list'> was given."

    def test_build_invalid_actuator_tuple_length(self):
        """
        Test the build method with an invalid actuator tuple length (not 3 elements).
        This should raise a ValueError.
        """
        action_spec = ActionSpec(actuators=[("invalid", "tuple")])
        with pytest.raises(ValueError) as exc_info:
            action_spec.build()
        assert str(exc_info.value) == "The actuators must be defined as a list of tuples of 3 elements but 2 was given."

    # def test_build_invalid_actuator_type(self):
    #     """
    #     Test the build method with an invalid actuator type (not a list).
    #     This should raise a ValueError.
    #     """
    #     action_spec = ActionSpec(actuators="invalid")
    #     with pytest.raises(ValueError) as exc_info:
    #         action_spec.build()
    #     assert str(exc_info.value) == "The actuators must be defined as a list of tuples but <class 'str'> was given."
