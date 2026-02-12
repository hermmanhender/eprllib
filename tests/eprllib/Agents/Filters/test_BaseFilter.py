from eprllib.Agents.Filters.BaseFilter import BaseFilter
import pytest

class TestBasefilter:

    def test___init___1(self):
        """
        Test the initialization of BaseFilter with a valid filter_fn_config.

        This test verifies that the BaseFilter constructor correctly initializes
        the filter_fn_config attribute with the provided configuration dictionary.
        """
        filter_fn_config = {"key": "value"}
        base_filter = BaseFilter(filter_fn_config)
        assert base_filter.filter_fn_config == filter_fn_config

    def test_get_filtered_obs_1(self):
        """
        Test that get_filtered_obs raises NotImplementedError in BaseFilter.

        This test verifies that calling get_filtered_obs on a BaseFilter instance
        raises a NotImplementedError with the expected message, as the method
        should be implemented in subclasses.
        """
        base_filter = BaseFilter({})
        env_config = {}
        agent_states = {}

        with pytest.raises(NotImplementedError) as excinfo:
            base_filter.get_filtered_obs(env_config, agent_states)

        assert str(excinfo.value) == "BaseFilter: This method should be implemented in a subclass."

    def test_get_filtered_obs_not_implemented(self):
        """
        Test that calling get_filtered_obs on BaseFilter raises NotImplementedError.
        This is the only edge case explicitly handled in the focal method's implementation.
        """
        base_filter = BaseFilter({})
        with pytest.raises(NotImplementedError) as excinfo:
            base_filter.get_filtered_obs({}, {})
        assert str(excinfo.value) == "BaseFilter: This method should be implemented in a subclass."


    def test_init_with_empty_dict(self):
        """
        Test initialization of BaseFilter with an empty dictionary.
        This tests the edge case where the filter_fn_config is an empty dictionary,
        which is a valid input but may lead to unexpected behavior in subclasses.
        """
        filter_instance = BaseFilter({})
        assert filter_instance.filter_fn_config == {}

    # def test_init_with_none_input(self):
    #     """
    #     Test initialization of BaseFilter with None as input.
    #     This tests the edge case where the filter_fn_config is None, which is not
    #     a valid dictionary input as required by the method signature.
    #     """
    #     with pytest.raises(TypeError) as excinfo:
    #         BaseFilter(None)
    #     assert str(excinfo.value) == "filter_fn_config must be a dictionary"