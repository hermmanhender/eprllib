from eprllib.Agents.Rewards.BaseReward import BaseReward
from eprllib.Agents.Rewards.RewardSpec import RewardSpec
import pytest

class TestRewardspec:

    def test___getitem___1(self):
        """
        Test the __getitem__ method of RewardSpec class.

        This test verifies that the __getitem__ method correctly retrieves
        attribute values using the key provided.
        """
        # Create a mock BaseReward subclass
        class MockReward(BaseReward):
            pass

        # Create a RewardSpec instance with some test data
        reward_spec = RewardSpec(reward_fn=MockReward, reward_fn_config={"test_key": "test_value"})

        # Test __getitem__ for existing attributes
        assert reward_spec["reward_fn"] == MockReward
        assert reward_spec["reward_fn_config"] == {"test_key": "test_value"}

        # Test __getitem__ for non-existent key
        with pytest.raises(AttributeError):
            reward_spec["non_existent_key"]

    def test___getitem___nonexistent_attribute(self):
        """
        Test __getitem__ method with a nonexistent attribute.
        This should raise an AttributeError as the method directly uses getattr().
        """
        reward_spec = RewardSpec()
        with pytest.raises(AttributeError):
            reward_spec['nonexistent_key']

    # def test___init___default_values(self):
    #     """
    #     Test the __init__ method of RewardSpec with default values.

    #     This test verifies that the RewardSpec object is correctly initialized
    #     with default values when no arguments are provided.
    #     """
    #     reward_spec = RewardSpec()
    #     assert reward_spec.reward_fn == NotImplemented
    #     assert reward_spec.reward_fn_config == {}

    def test___setitem___2(self):
        """
        Test setting a valid key in RewardSpec using __setitem__.
        This test verifies that setting an existing attribute using the __setitem__ method
        works correctly without raising any exceptions.
        """
        reward_spec = RewardSpec(reward_fn=BaseReward, reward_fn_config={})

        # Set a valid key
        reward_spec['reward_fn'] = BaseReward

        # Verify that the attribute was set correctly
        assert reward_spec.reward_fn == BaseReward

    def test___setitem___invalid_key(self):
        """
        Test the __setitem__ method with an invalid key that is not in the object's __dict__.
        This should raise a KeyError with an appropriate error message.
        """
        reward_spec = RewardSpec()
        with pytest.raises(KeyError) as excinfo:
            reward_spec['invalid_key'] = 'some_value'
        assert str(excinfo.value) == "'RewardSpec: Invalid key: invalid_key.'"

    def test___setitem___invalid_key_2(self):
        """
        Test the __setitem__ method of RewardSpec when an invalid key is provided.

        This test verifies that a KeyError is raised when attempting to set
        a value for a key that is not in the valid keys of the RewardSpec object.
        """
        reward_spec = RewardSpec()

        with pytest.raises(KeyError) as excinfo:
            reward_spec['invalid_key'] = 'some_value'

        assert str(excinfo.value) == "'RewardSpec: Invalid key: invalid_key.'"

    # def test_build_3(self):
    #     """
    #     Test the build method when reward_fn is a subclass of BaseReward but reward_fn_config is not a dictionary.

    #     This test verifies that the build method raises a TypeError when the reward_fn_config
    #     is not a dictionary, even if the reward_fn is a valid subclass of BaseReward.
    #     """
    #     class ValidReward(BaseReward):
    #         pass

    #     reward_spec = RewardSpec(reward_fn=ValidReward, reward_fn_config="invalid_config")

    #     with pytest.raises(TypeError) as exc_info:
    #         reward_spec.build()

    #     assert str(exc_info.value) == "The configuration for the reward function must be a dictionary but <class 'str'> was given."

    # def test_build_4(self):
    #     """
    #     Test the build method when reward_fn is NotImplemented and not a subclass of BaseReward.

    #     This test verifies that the build method raises a NotImplementedError when the reward_fn
    #     is set to NotImplemented, and does not reach the subsequent checks for subclass and
    #     configuration type.
    #     """
    #     reward_spec = RewardSpec(reward_fn=NotImplemented, reward_fn_config={})

    #     with pytest.raises(NotImplementedError) as excinfo:
    #         reward_spec.build()

    #     assert str(excinfo.value) == "No reward function provided."

    # def test_build_invalid_reward_config(self):
    #     """
    #     Test the build method when an invalid reward function configuration (not a dictionary) is provided.
    #     This should raise a TypeError.
    #     """
    #     class ValidReward(BaseReward):
    #         pass

    #     reward_spec = RewardSpec(reward_fn=ValidReward, reward_fn_config="invalid_config")
    #     with pytest.raises(TypeError) as excinfo:
    #         reward_spec.build()
    #     assert str(excinfo.value) == f"The configuration for the reward function must be a dictionary but <class 'str'> was given."

    # def test_build_invalid_reward_fn_and_config(self):
    #     """
    #     Test the build method when reward_fn is not a subclass of BaseReward
    #     and reward_fn_config is not a dictionary.

    #     This test covers the following path constraints:
    #     - self.reward_fn is not NotImplemented
    #     - self.reward_fn is not a subclass of BaseReward
    #     - self.reward_fn_config is not a dictionary

    #     Expected behavior: 
    #     - Raises a TypeError due to invalid reward_fn
    #     - The error about reward_fn_config is not reached due to the previous error
    #     """
    #     reward_spec = RewardSpec(reward_fn=str, reward_fn_config="invalid_config")

    #     with pytest.raises(TypeError) as excinfo:
    #         reward_spec.build()

    #     assert "The reward function must be based on BaseReward class" in str(excinfo.value)

    # def test_build_invalid_reward_function(self):
    #     """
    #     Test the build method when an invalid reward function (not a subclass of BaseReward) is provided.
    #     This should raise a TypeError.
    #     """
    #     class InvalidReward:
    #         pass

    #     reward_spec = RewardSpec(reward_fn=InvalidReward)
    #     with pytest.raises(TypeError) as excinfo:
    #         reward_spec.build()
    #     assert str(excinfo.value) == f"The reward function must be based on BaseReward class but <class 'type'> was given."

    def test_build_no_reward_function(self):
        """
        Test the build method when no reward function is provided.
        This should raise a NotImplementedError.
        """
        reward_spec = RewardSpec()
        with pytest.raises(NotImplementedError) as excinfo:
            reward_spec.build()
        assert str(excinfo.value) == "RewardSpec: No reward function provided."

    def test_build_raises_notimplementederror(self):
        """
        Test that build() raises NotImplementedError when reward_fn is NotImplemented.
        """
        reward_spec = RewardSpec()
        with pytest.raises(NotImplementedError, match="RewardSpec: No reward function provided."):
            reward_spec.build()

    # def test_init_with_non_basereward_subclass(self):
    #     """
    #     Test that initializing RewardSpec with a reward_fn that is not a subclass
    #     of BaseReward raises a TypeError when build() is called.
    #     """
    #     class InvalidReward:
    #         pass

    #     reward_spec = RewardSpec(reward_fn=InvalidReward)
    #     with pytest.raises(TypeError, match="The reward function must be based on BaseReward class"):
    #         reward_spec.build()

    # def test_init_with_non_dict_reward_fn_config(self):
    #     """
    #     Test that initializing RewardSpec with a non-dict reward_fn_config
    #     raises a TypeError when build() is called.
    #     """
    #     class ValidReward(BaseReward):
    #         pass

    #     reward_spec = RewardSpec(reward_fn=ValidReward, reward_fn_config="invalid_config")
    #     with pytest.raises(TypeError, match="The configuration for the reward function must be a dictionary"):
    #         reward_spec.build()

    # def test_init_with_not_implemented_reward_fn(self):
    #     """
    #     Test that initializing RewardSpec with NotImplemented as reward_fn
    #     raises a NotImplementedError when build() is called.
    #     """
    #     reward_spec = RewardSpec(reward_fn=NotImplemented)
    #     with pytest.raises(NotImplementedError, match="No reward function provided."):
    #         reward_spec.build()
