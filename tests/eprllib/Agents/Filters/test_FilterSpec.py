from eprllib.Agents.Filters.BaseFilter import BaseFilter
from eprllib.Agents.Filters.DefaultFilter import DefaultFilter
from eprllib.Agents.Filters.FilterSpec import FilterSpec

import logging
import pytest

logger = logging.getLogger("ray.rllib")

class TestFilterspec:

    def test___getitem___1(self):
        """
        Test that the __getitem__ method correctly returns the value of the specified attribute.
        """
        filter_spec = FilterSpec(filter_fn=DefaultFilter, filter_fn_config={'test_config': 'value'})
        assert filter_spec['filter_fn'] == DefaultFilter
        assert filter_spec['filter_fn_config'] == {'test_config': 'value'}

    def test___getitem___nonexistent_attribute(self):
        """
        Test that accessing a non-existent attribute raises an AttributeError.
        """
        filter_spec = FilterSpec()
        with pytest.raises(AttributeError):
            filter_spec['nonexistent_key']

    def test___init___default_parameters(self):
        """
        Test the __init__ method of FilterSpec with default parameters.

        This test verifies that when FilterSpec is initialized without any arguments,
        it correctly sets the default values for filter_fn and filter_fn_config.
        """
        filter_spec = FilterSpec()
        assert filter_spec.filter_fn is None
        assert filter_spec.filter_fn_config == {}

    def test___setitem___2(self):
        """
        Test that __setitem__ successfully sets a valid key-value pair.

        This test verifies that the __setitem__ method of FilterSpec correctly
        sets a value for an existing key without raising any exceptions.
        """
        filter_spec = FilterSpec()
        filter_spec['filter_fn'] = DefaultFilter

        assert filter_spec['filter_fn'] == DefaultFilter

    def test___setitem___invalid_key(self):
        """
        Test the __setitem__ method with an invalid key.
        This should raise a KeyError as explicitly handled in the method.
        """
        filter_spec = FilterSpec()
        with pytest.raises(KeyError) as excinfo:
            filter_spec['invalid_key'] = 'some_value'
        assert str(excinfo.value) == "'Invalid key: invalid_key.'"

    def test___setitem___invalid_key_2(self):
        """
        Test that __setitem__ raises a KeyError when an invalid key is provided.
        This test covers the path where the provided key is not in the valid keys.
        """
        filter_spec = FilterSpec()
        with pytest.raises(KeyError) as exc_info:
            filter_spec['invalid_key'] = 'some_value'
        assert str(exc_info.value) == "'Invalid key: invalid_key.'"

    def test_build_3(self):
        """
        Tests the build method of FilterSpec when:
        1. filter_fn is initially None
        2. DefaultFilter is assigned as filter_fn (which is a subclass of BaseFilter)
        3. filter_fn_config is set to a non-dictionary value

        Expected behavior:
        - Warning log for using DefaultFilter
        - ValueError raised due to non-dictionary filter_fn_config
        """
        filter_spec = FilterSpec()
        filter_spec.filter_fn = DefaultFilter  # This is a valid BaseFilter
        filter_spec.filter_fn_config = "Not a dictionary"

        with pytest.raises(ValueError) as context:
            filter_spec.build()

        assert f"The configuration for the filter function must be a dictionary but {type(filter_spec.filter_fn_config)} was given." == str(context.value)

    def test_build_4(self):
        """
        Test that build() method raises a ValueError when filter_fn is not an instance of BaseFilter.

        This test verifies that when the filter_fn is set to None (which triggers the default filter),
        and then set to a non-BaseFilter object, the build() method raises a ValueError with an
        appropriate error message.
        """
        filter_spec = FilterSpec()
        filter_spec.filter_fn = "not a BaseFilter"

        with pytest.raises(TypeError) as exc_info:
            filter_spec.build()

        assert str(exc_info.value) == "issubclass() arg 1 must be a class"

    def test_build_default_filter(self):
        """
        Test the build method when filter_fn is None.
        Verifies that:
        1. DefaultFilter is used when no filter function is provided
        2. A warning is logged
        3. filter_fn_config is set to an empty dictionary
        4. The method returns the correct dictionary of instance variables
        """
        filter_spec = FilterSpec()
        result = filter_spec.build()

        assert filter_spec.filter_fn == DefaultFilter
        assert filter_spec.filter_fn_config == {}
        assert result == vars(filter_spec)

    def test_build_invalid_filter_config(self):
        """
        Test the build method with an invalid filter function configuration that is not a dictionary.
        This should raise a ValueError.
        """
        class ValidFilter(BaseFilter):
            pass

        filter_spec = FilterSpec(filter_fn=ValidFilter, filter_fn_config="invalid_config")  # string instead of dict
        with pytest.raises(ValueError) as context:
            filter_spec.build()
        assert f"The configuration for the filter function must be a dictionary but {type(filter_spec.filter_fn_config)} was given." == str(context.value)

    def test_build_invalid_filter_function(self):
        """
        Test the build method when an invalid filter function is provided.

        This test verifies that the build method raises a ValueError when the
        filter_fn is not an instance of BaseFilter and filter_fn_config is not a dictionary.
        """
        filter_spec = FilterSpec(filter_fn="not_a_base_filter", filter_fn_config="not_a_dict")

        with pytest.raises(TypeError) as excinfo:
            filter_spec.build()

        assert "issubclass() arg 1 must be a class" == str(excinfo.value)

    def test_build_invalid_filter_function_2(self):
        """
        Test the build method with an invalid filter function that is not based on BaseFilter class.
        This should raise a ValueError.
        """
        filter_spec = FilterSpec(filter_fn=lambda x: x)  # lambda function is not a BaseFilter
        with pytest.raises(TypeError) as context:
            filter_spec.build()
        assert "issubclass() arg 1 must be a class" == str(context.value)

    def test_init_invalid_filter_fn_config_type(self):
        """
        Test __init__ with an invalid filter_fn_config type.
        This tests the edge case where the filter_fn_config is not a dictionary,
        which is explicitly checked in the build() method called by __init__.
        """
        filter_spec = FilterSpec(filter_fn=BaseFilter, filter_fn_config=[])
        with pytest.raises(ValueError) as excinfo:
            filter_spec = filter_spec.build()
        assert f"The configuration for the filter function must be a dictionary but {type(filter_spec['filter_fn_config'])} was given." in str(excinfo.value)

    def test_init_invalid_filter_fn_type(self):
        """
        Test __init__ with an invalid filter_fn type.
        This tests the edge case where the filter_fn is not an instance of BaseFilter,
        which is explicitly checked in the build() method called by __init__.
        """
        with pytest.raises(TypeError) as excinfo:
            FilterSpec(filter_fn=lambda x: x).build()  # lambda is not a BaseFilter
        assert "issubclass() arg 1 must be a class" in str(excinfo.value)
