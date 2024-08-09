"""
Converting conventional experiences to batch format
===================================================

This script execute the conventional controls and generate experiences in batch format.
"""
from tempfile import TemporaryDirectory
from eprllib.Env.MultiAgent.EnergyPlusEnvironment import EnergyPlusEnv_v0
from eprllib.Agents.ConventionalAgent import ConventionalAgent

import gymnasium as gym
import numpy as np
import os

import ray._private.utils

from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.evaluation.collectors.sample_collector import SampleCollector
from ray.rllib.offline.json_writer import JsonWriter

def PolicyMap():
    return 'conventional_policy'

if __name__ == "__main__":
    collector = SampleCollector(
        policy_map = PolicyMap,
        clip_rewards = False,
        callbacks = None,
        multiple_episodes_in_batch = True,
        rollout_fragment_length = 256,
        count_steps_by = "env_steps"
    )  # or MultiAgentSampleBatchBuilder
    writer = JsonWriter(
        os.path.join(ray._private.utils.get_user_temp_dir(), "demo-out")
    )

    # Se debe especificar el path según la PC utilizada
    path = 'C:/Users/grhen/Documents'

    # Controles de la simulación
    ep_terminal_output = True # Esta variable indica si se imprimen o no las salidas de la simulación de EnergyPlus
    name = 'VN_P1_0.5_RB'

    policy_config = { # configuracion del control convencional
        'SP_temp': 22, #es el valor de temperatura de confort
        'dT_up': 2.5, #es el límite superior para el rango de confort
        'dT_dn': 2.5, #es el límite inferior para el rango de confort
    }
    env_config={ 
        'weather_folder': 'C:/Users/grhen/Documents/GitHub/natural_ventilation_EP_RLlib/epw/GEF',
        'output': TemporaryDirectory("output","DQN_",'C:/Users/grhen/Documents/Resultados_RLforEP').name,
        'epjson_folderpath': 'C:/Users/grhen/Documents/GitHub/natural_ventilation_EP_RLlib/epjson',
        'epjson_output_folder': 'C:/Users/grhen/Documents/models',
        # Configure the directories for the experiment.
        'ep_terminal_output': False,
        # For dubugging is better to print in the terminal the outputs of the EnergyPlus simulation process.
        'beta': 0.5,
        # This parameter is used to balance between energy and comfort of the inhabitatns. A
        # value equal to 0 give a no importance to comfort and a value equal to 1 give no importance 
        # to energy consume. Mathematically is the reward: 
        # r = - beta*normaliced_energy - (1-beta)*normalized_comfort
        # The range of this value goes from 0.0 to 1.0.,
        'is_test': False,
        # For evaluation process 'is_test=True' and for trainig False.
        'test_init_day': 1,
        'action_space': gym.spaces.Discrete(4),
        # action space for simple agent case
        'observation_space': gym.spaces.Box(float("-inf"), float("inf"), (1465,)),
        # observation space for simple agent case
        
        # BUILDING CONFIGURATION
        'building_name': 'prot_1',
        'volumen': 131.6565,
        'window_area_relation_north': 0,
        'window_area_relation_west': 0,
        'window_area_relation_south': 0.0115243076,
        'window_area_relation_east': 0.0276970753,
        'episode_len': 365,
        'rotation': 0,
    }
    # se importan las políticas convencionales para la configuracion especificada
    policy = ConventionalAgent(policy_config)
    # se inicia el entorno con la configuración especificada
    env = EnergyPlusEnv_v0(env_config)

    # RLlib uses preprocessors to implement transforms such as one-hot encoding
    # and flattening of tuple and dict observations. For CartPole a no-op
    # preprocessor is used, but this may be relevant for more complex envs.
    prep = get_preprocessor(env.observation_space)(env.observation_space)
    print("The preprocessor is", prep)
            
    for eps_id in range(100):
        obs, info = env.reset()
        
        collector.add_init_obs(
            episode=eps_id,
            agent_id = 0,
            policy_id="conventional_policy",
            t=-1,
            init_obs=obs,
            init_infos=info,
        )
        
        prev_action = np.zeros_like(env.action_space.sample())
        prev_reward = 0
        terminated = truncated = False
        t = 0
        while not terminated and not truncated:
            To = obs[0]
            Ti = obs[1]
            action_w1 = obs[8]
            action_w2 = obs[9]
            
            action_1 = policy.window_opening(Ti, To, action_w1)
            action_2 = policy.window_opening(Ti, To, action_w2)
            action = ConventionalAgent.compute_single_action(action_1, action_2)
            
            new_obs, rew, terminated, truncated, info = env.step(action)
            
            collector.add_action_reward_next_obs(
                episode_id = eps_id,
                agent_id = 0,
                env_id = 0,
                policy_id = 'conventional_policy',
                agent_done = False if not terminated and not truncated else True,
                values = {
                    "action": action,
                    "obs": obs,
                    "reward": rew,
                    "terminated": terminated,
                    "truncated": truncated,
                    "info": info
                }
            )
            
            obs = new_obs
            prev_action = action
            prev_reward = rew
            t += 1
        writer.write(collector)