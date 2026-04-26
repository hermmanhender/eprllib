from typing import Dict, Any
from ray.rllib.algorithms.ppo.ppo import PPOConfig
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
from eprllib.Environment.MultiAgentEnvironment import MultiAgentEnvironment

def policy_map_fn(agent_id, episode, **kwargs): # type: ignore
    return "single_policy"

def ppo_config(
    env_config: Dict[str, Any],
    ) -> Dict[str, Any]:

    # Config the RLlib algorithm.
    algo = PPOConfig()
    algo.framework(
        framework = "torch",
    )
    algo.learners( # type: ignore
        num_learners = 0,
    )
    algo.environment( # type: ignore
        env = MultiAgentEnvironment,
        env_config = env_config,
        clip_actions = True,
    )
    algo.fault_tolerance(
        restart_failed_env_runners = True,
    )
    algo.multi_agent( # type: ignore
        policies = {
            'single_policy': PolicySpec(),
        },
        policy_mapping_fn = policy_map_fn, # type: ignore
        count_steps_by = "env_steps",
    )
    algo.reporting(
        min_sample_timesteps_per_iteration = 50,
    )
    algo.checkpointing(
        export_native_model_files = True,
    )
    algo.debugging( # type: ignore
        log_level = "INFO",
        seed = 1,
    )
    algo.resources(
        num_gpus = 0,
    )
    algo.rl_module( # type: ignore
        model_config=DefaultModelConfig(
            fcnet_hiddens=[32, 32],
            fcnet_activation="relu",
        )
    )
    algo.api_stack(
        enable_rl_module_and_learner=True,
        enable_env_runner_and_connector_v2=True,
    )
    algo.training( # type: ignore
        # === General Algo Configs ===
        gamma = 0.98,
        lr = 1e-4,
        train_batch_size_per_learner = 249,
        minibatch_size = 83,
        num_epochs = 2,
        
        # === PPO Configs ===
        use_critic = True,
        use_gae = True,
        lambda_ = 0.95,
        use_kl_loss = True,
        kl_coeff = 0.2,
        kl_target = 0.7,
        shuffle_batch_per_epoch = True,
        vf_loss_coeff = 0.25,
        entropy_coeff = 0.01,
        clip_param = 0.2,
        vf_clip_param = 0.2,
    )
    algo.env_runners( # type: ignore
        num_env_runners = 0,
        num_envs_per_env_runner = 1,
        sample_timeout_s = 3000000,
        rollout_fragment_length = "auto",
        batch_mode = "complete_episodes",
        explore = True,
    )
    
    return algo.to_dict() # type: ignore
