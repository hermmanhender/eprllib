"""# RUN DRL CONTROLS

This script execute the conventional controls in the evaluation scenario.
"""
import sys
sys.path.insert(0, 'C:/Users/grhen/Documents/GitHub/natural_ventilation_EP_RLlib')
import csv
from ray.rllib.policy.policy import Policy
from env.VENT_ep_gym_env import EnergyPlusEnv_v0


def init_drl_evaluation(
    env_config: dict,
    checkpoint_path: str,
    name: str
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
    Example:
    ```
    env_config={ 
        'weather_folder': 'C:/Users/grhen/Documents/GitHub/natural_ventilation_EP_RLlib/epw/GEF',
        'output': TemporaryDirectory("output","DQN_",'C:/Users/grhen/Documents/Resultados_RLforEP').name,
        'epjson_folderpath': 'C:/Users/grhen/Documents/GitHub/natural_ventilation_EP_RLlib/epjson',
        'epjson_output_folder': 'C:/Users/grhen/Documents/models',
        'ep_terminal_output': False,
        'beta': 0.5,
        'is_test': True,
        'test_init_day': 1,
        'action_space': gym.spaces.Discrete(4),
        'observation_space': gym.spaces.Box(float("-inf"), float("inf"), (1465,)),
        'building_name': 'prot_1',
        'volumen': 131.6565,
        'window_area_relation_north': 0,
        'window_area_relation_west': 0,
        'window_area_relation_south': 0.0115243076,
        'window_area_relation_east': 0.0276970753,
        'episode_len': 365,
        'rotation': 0,
    }
    checkpoint_path = 'C:/Users/grhen/ray_results/VN_P1_0.5_DQN/1024p2x512_dueT1x512_douT_DQN_cb9bc_00000/checkpoint_000064'
    name = 'VN_P1_0.5_DQN'
    
    episode_reward = init_drl_evaluation(
        env_config=env_config,
        checkpoint_path=checkpoint_path,
        name=name
    )
    
    print(f"Episode reward is: {episode_reward}.")
    ```
    """
    # Use the `from_checkpoint` utility of the Policy class:
    policy = Policy.from_checkpoint(checkpoint_path)
    # se inicia el entorno con la configuración especificada
    env = EnergyPlusEnv_v0(env_config)

    # open the file in the write mode
    data = open(env_config['output']+'/'+name+'.csv', 'w')
    # create the csv writer
    writer = csv.writer(data)

    terminated = False # variable de control de lazo (es verdadera cuando termina un episodio)
    episode_reward = 0
    # se obtiene la observaión inicial del entorno para el episodio
    obs, info = env.reset()
    while not terminated: # se ejecuta un paso de tiempo hasta terminar el episodio
        # se calculan las acciones convencionales de cada elemento
        central_actions, _, _ = policy['default_policy'].compute_single_action(obs)
        
        # se ejecuta un paso de tiempo
        obs, reward, terminated, truncated, infos = env.step(central_actions)
        # se guardan los datos
        # write a row to the csv file
        row = []
        
        obs_list = obs.tolist()
        for _ in range(len(obs_list)):
            row.append(obs_list[_])
        
        row.append(reward)
        row.append(terminated)
        row.append(truncated)
        
        info_list = list(infos.values())
        for _ in range(len(info_list)):
            row.append(info_list[_])
        
        writer.writerow(row)
        episode_reward += reward
    # close the file
    data.close()
    
    return episode_reward

if __name__ == '__main__':
    
    import gymnasium as gym
    import os

    name = 'natural_drl_control'
    
    # Controles de la simulación
    env_config={
        'weather_folder': 'C:/Users/grhen/Documents/GitHub/natural_ventilation_EP_RLlib/epw/GEF',
        'output': 'C:/Users/grhen/Documents/Resultados_RLforEP/'+name,
        'epjson_folderpath': 'C:/Users/grhen/Documents/GitHub/natural_ventilation_EP_RLlib/epjson',
        'epjson_output_folder': 'C:/Users/grhen/Documents/models',
        # Configure the directories for the experiment.
        'ep_terminal_output': True,
        # For dubugging is better to print in the terminal the outputs of the EnergyPlus simulation process.
        'is_test': True,
        # For evaluation process 'is_test=True' and for trainig False.
        'test_init_day': 1,
        'action_space': gym.spaces.Discrete(4),
        # action space for simple agent case
        'observation_space': gym.spaces.Box(float("-inf"), float("inf"), (303,)),
        # observation space for simple agent case
        
        # BUILDING CONFIGURATION
        'building_name': 'prot_1(natural)',
    }

    # se importan las políticas convencionales para la configuracion especificada
    checkpoint_path = 'C:/Users/grhen/ray_results/20240306_VN_prot_1_natural_DQN/3x512_dueT1x512_douT_DQN_4353d_00000/checkpoint_000085'
    
    try:
        os.makedirs(env_config['output'])
    except OSError:
        pass
    
    episode_reward = init_drl_evaluation(
        env_config,
        checkpoint_path,
        name
    )
    
    print(f"Episode reward is: {episode_reward}.")