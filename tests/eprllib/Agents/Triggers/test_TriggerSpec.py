from eprllib.Agents.Triggers.BaseTrigger import BaseTrigger
from eprllib.Agents.Triggers.TriggerSpec import TriggerSpec
import pytest

class TestTriggerspec:

    def test___getitem___access_valid_attribute(self):
        """
        Test that __getitem__ method correctly returns the value of a valid attribute.
        """
        class DummyTrigger(BaseTrigger):
            pass

        trigger_spec = TriggerSpec(trigger_fn=DummyTrigger, trigger_fn_config={'key': 'value'})

        assert trigger_spec['trigger_fn'] == DummyTrigger
        assert trigger_spec['trigger_fn_config'] == {'key': 'value'}

    def test___getitem___non_existent_attribute(self):
        """
        Test the __getitem__ method with a non-existent attribute.
        This should raise an AttributeError as the method directly uses getattr().
        """
        trigger_spec = TriggerSpec()
        with pytest.raises(AttributeError):
            trigger_spec['non_existent_key']

    # def test___init___default_values(self):
    #     """
    #     Test the __init__ method of TriggerSpec with default values.

    #     This test verifies that the TriggerSpec can be instantiated with default values
    #     and that the attributes are set correctly.
    #     """
    #     trigger_spec = TriggerSpec()
    #     assert trigger_spec.trigger_fn == NotImplemented
    #     assert trigger_spec.trigger_fn_config == {}

    # def test___init___invalid_trigger_fn(self):
    #     """
    #     Test that initializing TriggerSpec with a trigger_fn that is not a subclass of BaseTrigger
    #     raises a ValueError.
    #     """
    #     with pytest.raises(ValueError) as exc_info:
    #         TriggerSpec(trigger_fn=str).build()
    #     assert "The trigger function must be based on BaseTrigger class" in str(exc_info.value)

    # def test___init___invalid_trigger_fn_config(self):
    #     """
    #     Test that initializing TriggerSpec with a trigger_fn_config that is not a dictionary
    #     raises a ValueError.
    #     """
    #     class DummyTrigger(BaseTrigger):
    #         pass

    #     with pytest.raises(ValueError) as exc_info:
    #         TriggerSpec(trigger_fn=DummyTrigger, trigger_fn_config=[]).build()
    #     assert "The configuration for the trigger function must be a dictionary" in str(exc_info.value)

    # def test___init___not_implemented_trigger_fn(self):
    #     """
    #     Test that initializing TriggerSpec with NotImplemented as trigger_fn
    #     raises a NotImplementedError when build() is called.
    #     """
    #     trigger_spec = TriggerSpec(trigger_fn=NotImplemented)
    #     with pytest.raises(ValueError) as exc_info:
    #         trigger_spec.build()
    #     assert "The trigger function must be defined" in str(exc_info.value)

    def test___setitem___invalid_key(self):
        """
        Test that __setitem__ raises a KeyError when an invalid key is provided.
        """
        trigger_spec = TriggerSpec()
        with pytest.raises(KeyError) as excinfo:
            trigger_spec['invalid_key'] = 'some_value'
        assert str(excinfo.value) == "'TriggerSpec: Invalid key: invalid_key.'"

    def test___setitem___invalid_key_2(self):
        """
        Test that __setitem__ raises a KeyError when an invalid key is provided.

        This test verifies that when attempting to set an item with a key that
        is not in the valid keys of the TriggerSpec instance, a KeyError is raised
        with an appropriate error message.
        """
        trigger_spec = TriggerSpec()
        invalid_key = "invalid_key"

        with pytest.raises(KeyError) as excinfo:
            trigger_spec[invalid_key] = "some_value"

        assert str(excinfo.value) == f"'TriggerSpec: Invalid key: {invalid_key}.'"

    def test___setitem___valid_key(self):
        """
        Test that __setitem__ successfully sets a value for a valid key.
        This test covers the path where the key is in valid_keys.
        """
        trigger_spec = TriggerSpec(trigger_fn=BaseTrigger, trigger_fn_config={})

        # Set a value for an existing attribute
        trigger_spec['trigger_fn'] = None

        assert trigger_spec.trigger_fn is None

    # def test_build_2(self):
    #     """
    #     Test the build method when trigger_fn is not NotImplemented,
    #     but is not a subclass of BaseTrigger, and trigger_fn_config is not a dict.
    #     Expects a ValueError to be raised.
    #     """
    #     # Create a TriggerSpec instance with invalid trigger_fn and trigger_fn_config
    #     trigger_spec = TriggerSpec(trigger_fn=str, trigger_fn_config="invalid_config")

    #     # Assert that ValueError is raised when calling build()
    #     with pytest.raises(ValueError) as excinfo:
    #         trigger_spec.build()

    #     # Check that the error message indicates the trigger_fn is not based on BaseTrigger
    #     assert "The trigger function must be based on BaseTrigger class" in str(excinfo.value)

    # def test_build_3_invalid_trigger_fn_config(self):
    #     """
    #     Test that build() raises a ValueError when trigger_fn is NotImplemented
    #     and trigger_fn_config is not a dictionary.
    #     """
    #     trigger_spec = TriggerSpec(trigger_fn=NotImplemented, trigger_fn_config="invalid_config")

    #     with pytest.raises(ValueError) as context:
    #         trigger_spec.build()

    #     assert str(context.value) == "The trigger function must be defined."

    # def test_build_4(self):
    #     """
    #     Test the build method when trigger_fn is NotImplemented.

    #     This test verifies that the build method raises a ValueError
    #     when the trigger_fn is set to NotImplemented, which violates
    #     the requirement that trigger_fn must be defined.
    #     """
    #     trigger_spec = TriggerSpec(trigger_fn=NotImplemented, trigger_fn_config={})

    #     with pytest.raises(ValueError) as exc_info:
    #         trigger_spec.build()

    #     assert str(exc_info.value) == "The trigger function must be defined."

    # def test_build_invalid_trigger_fn(self):
    #     """
    #     Test the build method when trigger_fn is NotImplemented.

    #     This test verifies that the build method raises a ValueError
    #     when the trigger_fn is set to NotImplemented.
    #     """
    #     trigger_spec = TriggerSpec(trigger_fn=NotImplemented, trigger_fn_config={})

    #     with pytest.raises(ValueError) as context:
    #         trigger_spec.build()

    #     assert str(context.value) == "The trigger function must be defined."

    # def test_build_with_non_basetrigger_function(self):
    #     """
    #     Test the build method when the trigger function is not a subclass of BaseTrigger.
    #     This should raise a ValueError.
    #     """
    #     class NonBaseTrigger:
    #         pass

    #     trigger_spec = TriggerSpec(trigger_fn=NonBaseTrigger)
    #     with pytest.raises(ValueError) as context:
    #         trigger_spec.build()
        
    #     assert str(context.value) == f"The trigger function must be based on BaseTrigger class but {type(NonBaseTrigger)} was given."

    # def test_build_with_non_dict_config(self):
    #     """
    #     Test the build method when the trigger function config is not a dictionary.
    #     This should raise a ValueError.
    #     """
    #     class MockTrigger(BaseTrigger):
    #         pass

    #     trigger_spec = TriggerSpec(trigger_fn=MockTrigger, trigger_fn_config="not a dict")
    #     with pytest.raises(ValueError) as context:
    #         trigger_spec.build()
    #     assert str(context.value) == f"The configuration for the trigger function must be a dictionary but {type(trigger_spec.trigger_fn_config)} was given."

    def test_build_with_undefined_trigger_function(self):
        """
        Test the build method when the trigger function is not defined (NotImplemented).
        This should raise a ValueError.
        """
        trigger_spec = TriggerSpec()
        with pytest.raises(ValueError) as context:
            trigger_spec.build()
        assert str(context.value) == "TriggerSpec: The trigger function must be defined."
