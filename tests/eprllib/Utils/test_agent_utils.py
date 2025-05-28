from eprllib.Utils.agent_utils import get_agent_name
import pytest

class TestAgentUtils:

    def test_get_agent_name_1(self):
        """
        Tests the get_agent_name function when the input is a list.
        Verifies that the function correctly extracts the agent name
        from the first element of the list by splitting on ':'.
        """
        state = ["agent1:variable1:value1", "agent1:variable2:value2"]
        result = get_agent_name(state)
        assert result == "agent1"

    def test_get_agent_name_2(self):
        """
        Test the get_agent_name function when the input is a dictionary.

        This test verifies that the function correctly extracts the agent name
        from the first key of the input dictionary by splitting on ':' and
        returning the first part.
        """

        # Test input
        test_state = {"agent1: var1": 1, "agent1: var2": 2}

        # Expected output
        expected_name = "agent1"

        # Call the function and check the result
        result = get_agent_name(test_state)
        assert result == expected_name, f"Expected {expected_name}, but got {result}"

    def test_get_agent_name_3(self):
        """
        Test get_agent_name function with an input that is neither a list nor a dict.

        This test checks the behavior of get_agent_name when provided with an input
        that doesn't match the expected types (Dict[str, Any] or List). It verifies
        that the function raises a TypeError in such cases.
        """
        # Arrange
        invalid_input = "invalid_input"

        # Act & Assert
        with pytest.raises(TypeError):
            get_agent_name(invalid_input)

    def test_get_agent_name_empty_dict(self):
        """
        Test get_agent_name with an empty dictionary input.
        This tests the edge case of an empty dictionary, which would cause an IndexError when accessing the first key.
        """
        with pytest.raises(IndexError):
            get_agent_name({})

    def test_get_agent_name_empty_list(self):
        """
        Test get_agent_name with an empty list input.
        This tests the edge case of an empty list, which would cause an IndexError when accessing the first element.
        """
        with pytest.raises(IndexError):
            get_agent_name([])

    def test_get_agent_name_empty_list(self):
        """
        Test get_agent_name with an empty list input.
        This tests the edge case of an empty list, which would cause an IndexError when accessing the first element.
        """
        with pytest.raises(ValueError):
            get_agent_name([])
            
    def test_get_agent_name_empty_dict(self):
        """
        Test get_agent_name with an empty dictionary input.
        This tests the edge case of an empty dictionary, which would cause an IndexError when accessing the first key.
        """
        with pytest.raises(ValueError):
            get_agent_name({})