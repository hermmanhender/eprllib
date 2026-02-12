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
        assert str(excinfo.value) == "'FilterSpec: Invalid key: invalid_key.'"

    def test___setitem___invalid_key_2(self):
        """
        Test that __setitem__ raises a KeyError when an invalid key is provided.
        This test covers the path where the provided key is not in the valid keys.
        """
        filter_spec = FilterSpec()
        with pytest.raises(KeyError) as exc_info:
            filter_spec['invalid_key'] = 'some_value'
        assert str(exc_info.value) == "'FilterSpec: Invalid key: invalid_key.'"

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
