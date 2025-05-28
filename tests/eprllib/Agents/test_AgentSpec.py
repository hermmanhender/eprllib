from eprllib.Agents.AgentSpec import RewardSpec
from eprllib.Agents.Rewards.BaseReward import BaseReward
import pytest

class TestAgentspec:

    def test___getitem___1(self):
        """
        Test that __getitem__ method of RewardSpec correctly retrieves attribute values.
        """
        reward_fn = BaseReward
        reward_fn_config = {"test_key": "test_value"}
        reward_spec = RewardSpec(reward_fn=reward_fn, reward_fn_config=reward_fn_config)

        assert reward_spec["reward_fn"] == reward_fn
        assert reward_spec["reward_fn_config"] == reward_fn_config

    def test___getitem___non_existent_attribute(self):
        """
        Test that accessing a non-existent attribute via __getitem__ raises an AttributeError.
        """
        reward_spec = RewardSpec(BaseReward,{})
        with pytest.raises(AttributeError):
            reward_spec['non_existent_attribute']

    def test___init___1(self):
        """
        Test the initialization of RewardSpec with default parameters.
        """
        with pytest.raises(NotImplementedError) as e:
            RewardSpec()
        assert str(e.value) == "reward_fn must be implemented"

    def test___setitem___1(self):
        """
        Test that the __setitem__ method correctly sets a new attribute on the RewardSpec object.
        """
        reward_spec = RewardSpec(BaseReward,{})
        reward_spec['new_attribute'] = 'test_value'
        assert hasattr(reward_spec, 'new_attribute')
        assert reward_spec.new_attribute == 'test_value'

    def test_build_1(self):
        """
        Test that the build method of RewardSpec returns a dictionary containing
        the object's attributes when a valid reward function is provided.
        """
        class DummyReward(BaseReward):
            def __call__(self, env_object, infos):
                return 0.0

        reward_fn = DummyReward
        reward_fn_config = {"key": "value"}
        reward_spec = RewardSpec(reward_fn=reward_fn, reward_fn_config=reward_fn_config)

        result = reward_spec.build()

        assert isinstance(result, dict)
        assert result['reward_fn'] == reward_fn
        assert result['reward_fn_config'] == reward_fn_config

    def test_reward_spec_init_with_invalid_reward_fn(self):
        """
        Test initializing RewardSpec with an invalid reward_fn that is not a BaseReward instance.
        This should raise a ValueError as per the validation in the RewardSpec.validation_rew_config method.
        """
        with pytest.raises(TypeError) as e:
            RewardSpec(reward_fn="invalid_reward_fn", reward_fn_config={})
        assert str(e.value) == "reward_fn must be a class that inherits from BaseReward"

    def test_reward_spec_init_with_not_implemented_reward_fn(self):
        """
        Test initializing RewardSpec with NotImplemented as the reward_fn.
        This should raise a NotImplementedError as per the validation in the RewardSpec.validation_rew_config method.
        """
        with pytest.raises(NotImplementedError) as e:
            RewardSpec(reward_fn=NotImplemented)
            
        assert str(e.value) == "reward_fn must be implemented"

    def test_validation_rew_config_2(self):
        """
        Test that validation_rew_config returns True when reward_fn is implemented.
        """
        class TestReward(BaseReward):
            def __call__(self, env, infos):
                return 0.0

        reward_spec = RewardSpec(reward_fn=TestReward, reward_fn_config={})
        assert reward_spec.validation_rew_config() == True

