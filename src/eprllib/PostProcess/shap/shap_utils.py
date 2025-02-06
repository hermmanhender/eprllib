"""
SHAP Utilities to analyse policies
===================================


"""
import pandas as pd
import torch
import shap
from typing import Dict, Any
from ray.rllib.policy.policy import Policy
from eprllib.Environment.BaseEnvironment import BaseEnvironment

def generate_experience(
    policy: Policy, 
    env: BaseEnvironment, 
    num_episodes=5):
    first_episode = True
    done = {"__all__": False}
    for _ in range(num_episodes):
        obs, info = env.reset()
        _agent_ids = [agent for agent in obs.keys()]
        
        if first_episode:
            all_observations = {agent: [] for agent in _agent_ids}
            all_actions = {agent: [] for agent in _agent_ids}
            all_rewards = {agent: [] for agent in _agent_ids}
            all_dones = {agent: [] for agent in _agent_ids}
            all_infos = {agent: [] for agent in _agent_ids}
            first_episode = False
        
        action: Dict[str,Any] = {agent: 0 for agent in _agent_ids}
        done["__all__"] = False
        episode_observations = {agent: [] for agent in _agent_ids}
        episode_actions = {agent: [] for agent in _agent_ids}
        episode_rewards = {agent: [] for agent in _agent_ids}
        episode_dones = {agent: [] for agent in _agent_ids}
        episode_infos = {agent: [] for agent in _agent_ids}

        while not done["__all__"]:
            for agent in _agent_ids:
                action[agent], _, _ = policy.compute_single_action(obs[agent], explore=False) # No exploration
               
            new_obs, reward, done, truncateds, new_info = env.step(action)

            for agent in _agent_ids:
                episode_observations[agent].append(obs[agent])
                episode_actions[agent].append(action[agent])
                episode_rewards[agent].append(reward[agent])
                episode_dones[agent].append(done["__all__"])
                episode_infos[agent].append(info[agent])

            obs = new_obs
            info = new_info

        for agent in _agent_ids:
            all_observations[agent].extend(episode_observations[agent])
            all_actions[agent].extend(episode_actions[agent])
            all_rewards[agent].extend(episode_rewards[agent])
            all_dones[agent].extend(episode_dones[agent])
            all_infos[agent].extend(episode_infos[agent])

    df = pd.DataFrame({
        agent:{
            "observations": all_observations[agent],
            "actions": all_actions[agent],
            "rewards": all_rewards[agent],
            "dones": all_dones[agent],
            "infos": all_infos[agent]
        } for agent in _agent_ids
    })
    return df

# Model prediction function (adapted for RLlib)
def model_predict(
    data,
    policy: Policy
    ):
    data_tensor = torch.tensor(data, dtype=torch.float32)
    with torch.no_grad():
        predictions = policy.compute_actions(obs_batch=data_tensor.numpy(),exploration=False)[0]
    return predictions

def EPExplainer(
    model_predict,
    data,
    feature_names,
    sample_size:int = None
):
    # apply sample data if given
    if sample_size is not None:
        data = shap.sample(data, sample_size)
    
    shap.KernelExplainer(model_predict, data, feature_names)
    