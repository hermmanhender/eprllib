from eprllib.Agents.ActionMappers.BaseActionMapper import BaseActionMapper
from eprllib.Agents.ActionMappers.ActionMapperSpec import ActionMapperSpec
import pytest

class TestActionMapperSpec:

    def test___getitem___access_valid_attribute(self):
        """
        Test that __getitem__ method correctly returns the value of a valid attribute.
        """
        class DummyActionMapper(BaseActionMapper):
            pass

        action_mapper_spec = ActionMapperSpec(action_mapper=DummyActionMapper, action_mapper_config={'key': 'value'})

        assert action_mapper_spec['action_mapper'] == DummyActionMapper
        assert action_mapper_spec['action_mapper_config'] == {'key': 'value'}

    def test___getitem___non_existent_attribute(self):
        """
        Test the __getitem__ method with a non-existent attribute.
        This should raise an AttributeError as the method directly uses getattr().
        """
        action_mapper_spec = ActionMapperSpec()
        with pytest.raises(AttributeError):
            action_mapper_spec['non_existent_key']

    # def test___init___default_values(self):
    #     """
    #     Test the __init__ method of ActionMapperSpec with default values.

    #     This test verifies that the ActionMapperSpec can be instantiated with default values
    #     and that the attributes are set correctly.
    #     """
    #     action_mapper_spec = ActionMapperSpec()
    #     assert action_mapper_spec.action_mapper == NotImplemented
    #     assert action_mapper_spec.action_mapper_config == {}

    # def test___init___invalid_action_mapper(self):
    #     """
    #     Test that initializing ActionMapperSpec with a action_mapper that is not a subclass of BaseActionMapper
    #     raises a ValueError.
    #     """
    #     with pytest.raises(ValueError) as exc_info:
    #         ActionMapperSpec(action_mapper=str).build()
    #     assert "The action_mapper function must be based on BaseActionMapper class" in str(exc_info.value)

    # def test___init___invalid_action_mapper_config(self):
    #     """
    #     Test that initializing ActionMapperSpec with a action_mapper_config that is not a dictionary
    #     raises a ValueError.
    #     """
    #     class DummyActionMapper(BaseActionMapper):
    #         pass

    #     with pytest.raises(ValueError) as exc_info:
    #         ActionMapperSpec(action_mapper=DummyActionMapper, action_mapper_config=[]).build()
    #     assert "The configuration for the action_mapper function must be a dictionary" in str(exc_info.value)

    # def test___init___not_implemented_action_mapper(self):
    #     """
    #     Test that initializing ActionMapperSpec with NotImplemented as action_mapper
    #     raises a NotImplementedError when build() is called.
    #     """
    #     action_mapper_spec = ActionMapperSpec(action_mapper=NotImplemented)
    #     with pytest.raises(ValueError) as exc_info:
    #         action_mapper_spec.build()
    #     assert "The action_mapper function must be defined" in str(exc_info.value)

    def test___setitem___invalid_key(self):
        """
        Test that __setitem__ raises a KeyError when an invalid key is provided.
        """
        action_mapper_spec = ActionMapperSpec()
        with pytest.raises(KeyError) as excinfo:
            action_mapper_spec['invalid_key'] = 'some_value'
        assert str(excinfo.value) == "'ActionMapperSpec: Invalid key: invalid_key.'"

    def test___setitem___invalid_key_2(self):
        """
        Test that __setitem__ raises a KeyError when an invalid key is provided.

        This test verifies that when attempting to set an item with a key that
        is not in the valid keys of the ActionMapperSpec instance, a KeyError is raised
        with an appropriate error message.
        """
        action_mapper_spec = ActionMapperSpec()
        invalid_key = "invalid_key"

        with pytest.raises(KeyError) as excinfo:
            action_mapper_spec[invalid_key] = "some_value"

        assert str(excinfo.value) == f"'ActionMapperSpec: Invalid key: {invalid_key}.'"

    def test___setitem___valid_key(self):
        """
        Test that __setitem__ successfully sets a value for a valid key.
        This test covers the path where the key is in valid_keys.
        """
        action_mapper_spec = ActionMapperSpec(action_mapper=BaseActionMapper, action_mapper_config={})

        # Set a value for an existing attribute
        action_mapper_spec['action_mapper'] = None

        assert action_mapper_spec.action_mapper is None

    # def test_build_2(self):
    #     """
    #     Test the build method when action_mapper is not NotImplemented,
    #     but is not a subclass of BaseActionMapper, and action_mapper_config is not a dict.
    #     Expects a ValueError to be raised.
    #     """
    #     # Create a ActionMapperSpec instance with invalid action_mapper and action_mapper_config
    #     action_mapper_spec = ActionMapperSpec(action_mapper=str, action_mapper_config="invalid_config")

    #     # Assert that ValueError is raised when calling build()
    #     with pytest.raises(ValueError) as excinfo:
    #         action_mapper_spec.build()

    #     # Check that the error message indicates the action_mapper is not based on BaseActionMapper
    #     assert "The action_mapper function must be based on BaseActionMapper class" in str(excinfo.value)

    # def test_build_3_invalid_action_mapper_config(self):
    #     """
    #     Test that build() raises a ValueError when action_mapper is NotImplemented
    #     and action_mapper_config is not a dictionary.
    #     """
    #     action_mapper_spec = ActionMapperSpec(action_mapper=NotImplemented, action_mapper_config="invalid_config")

    #     with pytest.raises(ValueError) as context:
    #         action_mapper_spec.build()

    #     assert str(context.value) == "The action_mapper function must be defined."

    # def test_build_4(self):
    #     """
    #     Test the build method when action_mapper is NotImplemented.

    #     This test verifies that the build method raises a ValueError
    #     when the action_mapper is set to NotImplemented, which violates
    #     the requirement that action_mapper must be defined.
    #     """
    #     action_mapper_spec = ActionMapperSpec(action_mapper=NotImplemented, action_mapper_config={})

    #     with pytest.raises(ValueError) as exc_info:
    #         action_mapper_spec.build()

    #     assert str(exc_info.value) == "The action_mapper function must be defined."

    # def test_build_invalid_action_mapper(self):
    #     """
    #     Test the build method when action_mapper is NotImplemented.

    #     This test verifies that the build method raises a ValueError
    #     when the action_mapper is set to NotImplemented.
    #     """
    #     action_mapper_spec = ActionMapperSpec(action_mapper=NotImplemented, action_mapper_config={})

    #     with pytest.raises(ValueError) as context:
    #         action_mapper_spec.build()

    #     assert str(context.value) == "The action_mapper function must be defined."

    # def test_build_with_non_BaseActionMapper_function(self):
    #     """
    #     Test the build method when the action_mapper function is not a subclass of BaseActionMapper.
    #     This should raise a ValueError.
    #     """
    #     class NonBaseActionMapper:
    #         pass

    #     action_mapper_spec = ActionMapperSpec(action_mapper=NonBaseActionMapper)
    #     with pytest.raises(ValueError) as context:
    #         action_mapper_spec.build()
        
    #     assert str(context.value) == f"The action_mapper function must be based on BaseActionMapper class but {type(NonBaseActionMapper)} was given."

    # def test_build_with_non_dict_config(self):
    #     """
    #     Test the build method when the action_mapper function config is not a dictionary.
    #     This should raise a ValueError.
    #     """
    #     class MockActionMapper(BaseActionMapper):
    #         pass

    #     action_mapper_spec = ActionMapperSpec(action_mapper=MockActionMapper, action_mapper_config="not a dict")
    #     with pytest.raises(ValueError) as context:
    #         action_mapper_spec.build()
    #     assert str(context.value) == f"The configuration for the action_mapper function must be a dictionary but {type(action_mapper_spec.action_mapper_config)} was given."

    def test_build_with_undefined_action_mapper_function(self):
        """
        Test the build method when the ActionMapper class is not defined (NotImplemented).
        This should raise a ValueError.
        """
        action_mapper_spec = ActionMapperSpec()
        with pytest.raises(ValueError) as context:
            action_mapper_spec.build()
        assert str(context.value) == "ActionMapperSpec: The ActionMapper class must be defined."
