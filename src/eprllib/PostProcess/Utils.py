

from typing import List, Dict, Any
from eprllib.Environment.Environment import Environment
from ray.rllib.policy.policy import Policy
import pandas as pd

def generate_experience(
    env: Environment,
    env_config: Dict[str,Any],
    agent_list: List[str],
    policies: Dict[str,Policy],
    rnn_use: Dict[str, bool],
    num_episodes: int = 1
) -> pd.DataFrame:
    """Function to generate experience and store in a DataFrame

    Args:
        env (Environment): The environment to use.
        env_config (Dict[str,Any]): The configuration for the environment.
        agent_list (List[str]): List of agents in the environment.
        policies (Dict[str,Policy]): Dictionary of policies for each agent.
        rnn_use (Dict[str, bool]): Dictionary indicating if RNN is used for each agent.
        num_episodes (int, optional): Number of episodes to generate. Defaults to 1.
    Raises:
        NotImplementedError: If the environment is not supported.

    Returns:
        pd.DataFrame: DataFrame containing the experience of all episodes for each agent.
    """
    env = env(env_config)
    # Experience storage.
    all_observations = {agent: [] for agent in agent_list}
    all_actions = {agent: [] for agent in agent_list}
    all_rewards = {agent: [] for agent in agent_list}
    all_dones = {agent: [] for agent in agent_list}
    all_infos = {agent: [] for agent in agent_list}
    if rnn_use:
        all_states = {agent: [] for agent in agent_list}
    
    for _ in range(num_episodes):
        
        # Episode experience storage.
        episode_observations = {agent: [] for agent in agent_list}
        episode_actions = {agent: [] for agent in agent_list}
        episode_rewards = {agent: [] for agent in agent_list}
        episode_dones = {agent: [] for agent in agent_list}
        episode_infos = {agent: [] for agent in agent_list}
        if rnn_use:
            episode_states = {agent: [] for agent in agent_list}
            state = {agent: policies[agent].get_initial_state() for agent in agent_list}

        # Reset the environment and get the initial observation and infos.
        obs, info = env.reset()

        # List the agents in the environment for the first timestep.
        _agent_ids = list(obs.keys())

        # Reward for timestep 0.
        reward = {agent: 0 for agent in _agent_ids}
        # Control variable to finish the episode.
        done = {"__all__": False}
        
        # Store the experience for each agent.
        for agent in _agent_ids:
            episode_observations[agent].append(obs[agent])
            episode_rewards[agent].append(reward[agent])
            episode_dones[agent].append(done["__all__"])
            episode_infos[agent].append(info[agent])
            if rnn_use:
                episode_states[agent].append(state[agent])
        
        # Loop to generate the experience.
        while not done["__all__"]:
            # Action dictionary.
            action: Dict[str,Any] = {agent: 0 for agent in _agent_ids}
            if not rnn_use:
                # Get the action for each agent in the timestep.
                for agent in _agent_ids:
                    action[agent], _, _ = policies[agent].compute_single_action(obs[agent], explore=False) # No exploration
                    episode_actions[agent].append(action[agent])
            else:
                # Get the action for each agent in the timestep.
                for agent in _agent_ids:
                    action[agent], state[agent], _ = policies[agent].compute_single_action(obs[agent], state=state[agent], explore=False) # No exploration
                    episode_actions[agent].append(action[agent])
            # Execute the action in the environment.   
            new_obs, new_reward, done, truncateds, new_info = env.step(action)

            # Update the observation and info for the next timestep.
            obs = new_obs
            info = new_info
            reward = new_reward
            # List the agents in the environment for the next timestep.
            _agent_ids = list(obs.keys())
            
            # Store the experience for each agent.
            for agent in _agent_ids:
                episode_observations[agent].append(obs[agent])
                episode_rewards[agent].append(reward[agent])
                episode_dones[agent].append(done["__all__"])
                episode_infos[agent].append(info[agent])
                if rnn_use:
                    episode_states[agent].append(state[agent])
            

        # Store the episode experience.
        for agent in agent_list:
            all_observations[agent].extend(episode_observations[agent])
            all_actions[agent].extend(episode_actions[agent])
            all_rewards[agent].extend(episode_rewards[agent])
            all_dones[agent].extend(episode_dones[agent])
            all_infos[agent].extend(episode_infos[agent])
            if rnn_use:
                all_states[agent].extend(episode_states[agent])

    # Create the DataFrame with the experience of all episodes.
    if not rnn_use:
        return pd.DataFrame({
            agent:{
                "observations": all_observations[agent],
                "actions": all_actions[agent],
                "rewards": all_rewards[agent],
                "dones": all_dones[agent],
                "infos": all_infos[agent]
            }
            for agent
            in agent_list
        })
    else:
        return pd.DataFrame({
            agent:{
                "observations": all_observations[agent],
                "actions": all_actions[agent],
                "rewards": all_rewards[agent],
                "dones": all_dones[agent],
                "infos": all_infos[agent],
                "states": all_states[agent]
            }
            for agent
            in agent_list
        })
        