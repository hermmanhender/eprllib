"""
RUN DRL CONTROLS
=================

This script execute the conventional controls in the evaluation scenario.
"""
from typing import Dict, Any, List
from ray.rllib.policy.policy import Policy
from pathlib import Path
from ray.rllib.core.rl_module.rl_module import RLModule
from eprllib.Environment.Environment import Environment
import torch
import numpy as np

def generate_experience(
    env: Environment,
    env_config: Dict[str, Any],
    policies: Dict[str, Policy],
    num_episodes: int = 1
) -> Dict[str, Any]:
    """
    Function to generate experience and store it in a structured dictionary.

    Args:
        env (Environment): The environment class to instantiate.
        env_config (Dict[str, Any]): Configuration for the environment.
        policies (Dict[str, Policy]): A dictionary mapping agent IDs to their policies.
        num_episodes (int, optional): The number of episodes to simulate. Defaults to 1.

    Returns:
        Dict[str, Any]: A dictionary containing the generated experience
                        structured as:
                        {
                            'experiment': {
                                'episode_1': {
                                    'timestep_1': {
                                        'agent_name_1': {
                                            'observation': value,
                                            'action': value,
                                            'reward': value,
                                            'done': value,
                                            'info': value,
                                            'state': value (if rnn_use[agent] is True)
                                        },
                                        'agent_name_2': { ... }
                                    },
                                    'timestep_2': { ... }
                                },
                                'episode_2': { ... }
                            }
                        }
    """
    # Instantiate the environment
    env_instance: Environment = env(env_config)
    
    # Main dictionary to store all experiment data
    experiment_data: Dict[str, Any] = {'experiment': {}}
    
    for episode_idx in range(1, num_episodes + 1):
        
        # Dictionary to store data for the current episode
        episode_data: Dict[str, Any] = {}
        timestep_counter = 0
        
        # Initialize RNN states for agents that use them
        agent_list = policies.keys()
        current_states = {
            agent: policies[agent].get_initial_state()
            for agent in agent_list if policies[agent].is_recurrent()# Check rnn_use for each agent
        }
        
        # Reset the environment and get the initial observation and infos.
        obs, info = env_instance.reset()

        # List the agents in the environment for the first timestep.
        _agent_ids: List[str] = list(obs.keys())

        # Reward for timestep 0 (initial state, no action taken yet).
        reward = {agent: 0 for agent in _agent_ids}
        # Control variable to finish the episode.
        done = {"__all__": False}
        truncateds = {agent: False for agent in _agent_ids} # Initialize truncateds for consistency

        # --- Store the experience for the initial timestep (timestep_0 or timestep_1 depending on convention) ---
        timestep_counter += 1
        timestep_data: Dict[str, Any] = {}
        for agent in _agent_ids:
            agent_experience: Dict[str, Any] = {
                'observation': obs[agent],
                'action': None, # No action taken yet at timestep 0
                'reward': reward[agent],
                'done': done["__all__"],
                'truncated': truncateds[agent], # Add truncateds
                'info': info[agent]
            }
            if policies[agent].is_recurrent(): # Check rnn_use for the specific agent
                agent_experience['state'] = current_states.get(agent) # Use .get() in case agent not in current_states
            timestep_data[agent] = agent_experience
        episode_data[f'timestep_{timestep_counter}'] = timestep_data
        
        # Loop to generate the experience for subsequent timesteps.
        while not done["__all__"]:
            # Action dictionary for the current timestep.
            action: Dict[str, Any] = {}
            new_states = {}
            
            # Get the action for each agent in the timestep.
            for agent in _agent_ids:
                if policies[agent].is_recurrent(): # Check rnn_use for the specific agent
                    # Compute action with RNN state
                    agent_action, agent_state, _ = policies[agent].compute_single_action(
                        obs[agent], state=current_states.get(agent), explore=False
                    )
                    action[agent] = agent_action
                    new_states[agent] = agent_state
                    
                    # Update current_states for the next iteration if RNN is used by any agent
                    current_states[agent] = new_states[agent]
                    
                else:
                    # Compute action without RNN state
                    agent_action, _, _ = policies[agent].compute_single_action(
                        obs[agent], explore=False
                    )
                    action[agent] = agent_action

                # assert type(action[agent]) is float|int, f"Action must be a float or int, but got {type(action[agent])} for agent {agent} at timestep {timestep_counter}."
            # Execute the action in the environment.
            new_obs, new_reward, done, truncateds, new_info = env_instance.step(action)

            # Update the observation, reward, done, truncateds, and info for the next timestep.
            obs = new_obs
            info = new_info
            reward = new_reward
            
            # List the agents in the environment for the next timestep.
            # This handles cases where agents might leave the environment.
            _agent_ids = list(obs.keys())
            
            # --- Store the experience for the current timestep ---
            timestep_counter += 1
            timestep_data = {}
            for agent in _agent_ids:
                agent_experience = {
                    'observation': obs[agent],
                    'action': action.get(agent), # Action taken to reach this state
                    'reward': reward[agent],
                    'done': done["__all__"],
                    'truncated': truncateds["__all__"],
                    'info': info[agent]
                }
                if policies[agent].is_recurrent(): # Check rnn_use for the specific agent
                    # Store the state *after* computing the action for this timestep
                    agent_experience['state'] = current_states.get(agent) # Use .get()
                timestep_data[agent] = agent_experience
            episode_data[f'timestep_{timestep_counter}'] = timestep_data

        # Add the completed episode data to the experiment data
        experiment_data['experiment'][f'episode_{episode_idx}'] = episode_data

    return experiment_data

def generate_experience_V2(
    env: Environment,
    env_config: Dict[str, Any],
    rl_module_checpoints: Dict[str, str],
    policy_names: Dict[str, str],
    num_episodes: int = 1
) -> Dict[str, Any]:
    """
    Function to generate experience and store it in a structured dictionary.

    Args:
        env (Environment): The environment class to instantiate.
        env_config (Dict[str, Any]): Configuration for the environment.
        policies (Dict[str, Policy]): A dictionary mapping agent IDs to their policies.
        num_episodes (int, optional): The number of episodes to simulate. Defaults to 1.

    Returns:
        Dict[str, Any]: A dictionary containing the generated experience
                        structured as:
                        {
                            'experiment': {
                                'episode_1': {
                                    'timestep_1': {
                                        'agent_name_1': {
                                            'observation': value,
                                            'action': value,
                                            'reward': value,
                                            'done': value,
                                            'info': value,
                                            'state': value (if rnn_use[agent] is True)
                                        },
                                        'agent_name_2': { ... }
                                    },
                                    'timestep_2': { ... }
                                },
                                'episode_2': { ... }
                            }
                        }
    """
    # Instantiate the environment
    env_instance: Environment = env(env_config)
    
    # Main dictionary to store all experiment data
    experiment_data: Dict[str, Any] = {'experiment': {}}
    
    agent_list: List[str] = list(rl_module_checpoints.keys())
    rl_module_dict: Dict[str, RLModule] = {}
    for agent in agent_list:
        rl_module_dict[agent] = RLModule.from_checkpoint(Path(rl_module_checpoints[agent]) / "learner_group" / "learner" / "rl_module" / policy_names[agent])

        # assert type(rl_module_dict[agent]) == RLModule, f"assert rl_module_dict[{agent}] is not RLModule, is {type(rl_module_dict[agent])}."
    
    for episode_idx in range(1, num_episodes + 1):
        
        # Dictionary to store data for the current episode
        episode_data: Dict[str, Any] = {}
        timestep_counter = 0
        
        # Initialize RNN states for agents that use them
        current_states: Dict[str, Any]= {
            agent: rl_module_dict[agent].get_initial_state()
            for agent in agent_list if rl_module_dict[agent].is_stateful()# Check rnn_use for each agent
        }
        
        # Reset the environment and get the initial observation and infos.
        obs, info = env_instance.reset()

        # List the agents in the environment for the first timestep.
        _agent_ids: List[str] = list(obs.keys())

        # Reward for timestep 0 (initial state, no action taken yet).
        reward = {agent: 0 for agent in _agent_ids}
        # Control variable to finish the episode.
        done = {"__all__": False}
        truncateds = {agent: False for agent in _agent_ids} # Initialize truncateds for consistency

        # --- Store the experience for the initial timestep (timestep_0 or timestep_1 depending on convention) ---
        timestep_counter += 1
        timestep_data: Dict[str, Any] = {}
        for agent in _agent_ids:
            agent_experience: Dict[str, Any] = {
                'observation': obs[agent],
                'action': None, # No action taken yet at timestep 0
                'reward': reward[agent],
                'done': done["__all__"],
                'truncated': truncateds[agent], # Add truncateds
                'info': info[agent]
            }
            if rl_module_dict[agent].is_stateful(): # Check rnn_use for the specific agent
                agent_experience['state'] = current_states.get(agent) # Use .get() in case agent not in current_states
            timestep_data[agent] = agent_experience
        episode_data[f'timestep_{timestep_counter}'] = timestep_data
        
        # Loop to generate the experience for subsequent timesteps.
        while not done["__all__"]:
            # Action dictionary for the current timestep.
            action: Dict[str, Any] = {agent: None for agent in _agent_ids}
            obs_batch: Dict[str, Any] = {agent: None for agent in _agent_ids}
            new_states: Dict[str, Any]= {agent: None for agent in _agent_ids}
            
            # Get the action for each agent in the timestep.
            for agent in _agent_ids:
                # Compute the next action from a batch (B=1) of observations.
                obs_batch[agent] = torch.from_numpy(obs[agent]).unsqueeze(0)  # add batch B=1 dimension
                
                if rl_module_dict[agent].is_stateful(): # Check rnn_use for the specific agent
                    # Compute action with RNN state
                    model_outputs = rl_module_dict[agent].forward_inference({"obs": obs_batch[agent]})
                    # Extract the action distribution parameters from the output and dissolve batch dim.
                    action_dist_params = model_outputs["action_dist_inputs"][0].numpy()
                    
                    # We have continuous actions -> take the mean (max likelihood).
                    # greedy_action = np.clip(
                    #     action_dist_params[0:1],  # 0=mean, 1=log(stddev), [0:1]=use mean, but keep shape=(1,)
                    #     a_min=env_instance.action_space[agent].low[0],
                    #     a_max=env_instance.action_space[agent].high[0],
                    # )
                    
                    # For discrete actions, you should take the argmax over the logits:
                    greedy_action = np.argmax(action_dist_params)
                    action[agent] = greedy_action
                    # new_states[agent] = agent_state
                    
                    # Update current_states for the next iteration if RNN is used by any agent
                    current_states[agent] = new_states[agent]
                    
                else:
                    # Compute action without RNN state
                    model_outputs  = rl_module_dict[agent].forward_inference({"obs": obs_batch[agent]})
                    # Extract the action distribution parameters from the output and dissolve batch dim.
                    action_dist_params = model_outputs["action_dist_inputs"][0].numpy()
                    # We have continuous actions -> take the mean (max likelihood).
                    # greedy_action = np.clip(
                    #     action_dist_params[0:1],  # 0=mean, 1=log(stddev), [0:1]=use mean, but keep shape=(1,)
                    #     a_min=env_instance.action_space[agent].low[0],
                    #     a_max=env_instance.action_space[agent].high[0],
                    # )
                    # For discrete actions, you should take the argmax over the logits:
                    greedy_action = np.argmax(action_dist_params)
                    action[agent] = greedy_action

                # assert type(action[agent]) is float|int, f"Action must be a float or int, but got {type(action[agent])} for agent {agent} at timestep {timestep_counter}."
            # Execute the action in the environment.
            new_obs, new_reward, done, truncateds, new_info = env_instance.step(action)

            # Update the observation, reward, done, truncateds, and info for the next timestep.
            obs = new_obs
            info = new_info
            reward = new_reward
            
            # List the agents in the environment for the next timestep.
            # This handles cases where agents might leave the environment.
            _agent_ids = list(obs.keys())
            
            # --- Store the experience for the current timestep ---
            timestep_counter += 1
            timestep_data = {}
            for agent in _agent_ids:
                agent_experience = {
                    'observation': obs[agent],
                    'action': action.get(agent), # Action taken to reach this state
                    'reward': reward[agent],
                    'done': done["__all__"],
                    'truncated': truncateds["__all__"],
                    'info': info[agent]
                }
                if rl_module_dict[agent].is_stateful(): # Check rnn_use for the specific agent
                    # Store the state *after* computing the action for this timestep
                    agent_experience['state'] = current_states.get(agent) # Use .get()
                timestep_data[agent] = agent_experience
            episode_data[f'timestep_{timestep_counter}'] = timestep_data

        # Add the completed episode data to the experiment data
        experiment_data['experiment'][f'episode_{episode_idx}'] = episode_data

    return experiment_data
