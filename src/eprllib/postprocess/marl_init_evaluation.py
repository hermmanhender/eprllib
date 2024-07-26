"""# RUN DRL CONTROLS

This script execute the conventional controls in the evaluation scenario.
"""
import os
from ray.rllib.policy.policy import Policy
from eprllib.env.multiagent.marl_ep_gym_env import EnergyPlusEnv_v0
import numpy as np
import pandas as pd
from typing import Dict, Any

def init_drl_evaluation(
    env_config: Dict[str, Any],
    checkpoint_path: str,
    name: str,
    use_RNN: bool = True,
    lstm_cell_size: int = 256,
) -> float:
    """This method restore a DRL Policy from `checkpoint_path` for `EnergyPlusEnv_v0` with 
    `env_config` configuration and save the results of an evaluation episode in 
    `env_config['output']/name` file.

    Args:
        env_config (dict): Environment configuration
        checkpoint_path (str): Path to the checkpointing produced during training.
        name (str): file name where the results will be save.
    
    Return:
        float: The acumulated reward in the episode.
    """
    # Use the `from_checkpoint` utility of the Policy class:
    policy = Policy.from_checkpoint(checkpoint_path)
    # se inicia el entorno con la configuración especificada
    env = EnergyPlusEnv_v0(env_config)
    _agent_ids = env.get_agent_ids()
    # create the output folder if it doesn't exist
    if not os.path.exists(env_config['output']):
        os.makedirs(env_config['output'])
    # open the file in the write mode
    # data = open(env_config['output']+'/'+name+'.csv', 'w')
    # define a DataFrame to save the simulation data
    

    terminated = {}
    terminated["__all__"] = False # variable de control de lazo (es verdadera cuando termina un episodio)
    episode_reward = 0
    # se obtiene la observaión inicial del entorno para el episodio
    obs_dict, infos = env.reset()
    
    if use_RNN:
        # range(2) b/c h- and c-states of the LSTM.
        state = [np.zeros([lstm_cell_size], np.float32) for _ in range(2)]
    
    # Create an empty DataFrame to store the data
    obs_keys = env.energyplus_runner.obs_keys
    infos_keys = env.energyplus_runner.infos_keys
    
    obs_df = pd.DataFrame(
        columns=['agent_id']+['timestep']+obs_keys+['Action']+['Reward']+['Terminated']+['Truncated']+infos_keys
    )
    timestep = 0
    while not terminated["__all__"]: # se ejecuta un paso de tiempo hasta terminar el episodio
        # se calculan las acciones convencionales de cada elemento
        actions_dict = {}
        for agent in _agent_ids:
            if use_RNN:
                action, state, _ = policy['shared_policy'].compute_single_action(obs_dict[agent], state)
            else:
                action, _, _ = policy['shared_policy'].compute_single_action(obs_dict[agent])
            actions_dict[agent] = action
        
        # Get the values of the variables for a timestep
        obs_dict, reward, terminated, truncated, infos = env.step(actions_dict)
        # Sum the rewards for all agents
        episode_reward += sum(reward.values())
        
        for agent in _agent_ids:
            new_index = len(obs_df)
            new_row = [agent, timestep] + list(obs_dict[agent]) + [actions_dict[agent], reward[agent], terminated["__all__"], truncated["__all__"]] + [value for value in infos[agent].values()]
            obs_df.loc[new_index] = new_row
            
        timestep += 1
        if timestep % 1000 == 0:
            print(f"Step {timestep}")
    
    # save the list of rows as JSON
    with open(env_config['output'] + '/' + name + '.csv', 'w') as f:
        obs_df.to_csv(f, index=False)
    
    return episode_reward
