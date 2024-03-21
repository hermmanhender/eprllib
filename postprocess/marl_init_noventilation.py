"""# RUN CONVENTIONAL CONTROLS

This script execute the conventional controls in the evaluation scenario.
"""
import sys
sys.path.insert(0, 'C:/Users/grhen/Documents/GitHub/natural_ventilation_EP_RLlib')
import csv
from env.marl_ep_gym_env import EnergyPlusEnv_v0
from agents.conventional import Conventional


def init_rb_evaluation(
    env_config: dict,
    policy_config: dict,
    name: str
) -> float:
    """This method execute RB Natural Ventilation Policy with `policy_config` configuration from
    `checkpoint_path` for `EnergyPlusEnv_v0` with `env_config` configuration and save the results 
    of an evaluation episode in `env_config['output']/name` file.

    Args:
        env_config (dict): Environment configuration
        policy_config (str): Configuration for the conventional policy.
        name (str): file name where the results will be save.
    
    Return:
        float: The acumulated reward in the episode.
    Example:
    ```
    env_config={ 
        'weather_folder': 'C:/Users/grhen/Documents/GitHub/natural_ventilation_EP_RLlib/epw/GEF',
        'output': TemporaryDirectory("output","RB_",'C:/Users/grhen/Documents/Resultados_RLforEP').name,
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
    
    policy_config = { # configuracion del control convencional
        'SP_temp': 22, #es el valor de temperatura de confort
        'dT_up': 2.5, #es el límite superior para el rango de confort
        'dT_dn': 2.5, #es el límite inferior para el rango de confort
    }
    
    name = 'VN_P1_0.5_RB'
    
    episode_reward = init_rb_evaluation(
        env_config=env_config,
        policy_config=policy_config,
        name=name
    )
    
    print(f"Episode reward is: {episode_reward}.")
    ```
    """
    # se inicia el entorno con la configuración especificada
    env = EnergyPlusEnv_v0(env_config)
    _agents_id = env.get_agent_ids()
    _agents_id_list = list(_agents_id)

    # open the file in the write mode
    data = open(env_config['output']+'/'+name+'.csv', 'w')
    # create the csv writer
    writer = csv.writer(data)
    terminated = {}
    terminated["__all__"] = False # variable de control de lazo (es verdadera cuando termina un episodio)
    episode_reward = 0
    # se obtiene la observaión inicial del entorno para el episodio
    obs_dict, infos = env.reset()
    while not terminated["__all__"]: # se ejecuta un paso de tiempo hasta terminar el episodio
        # se calculan las acciones convencionales de cada elemento
        
        actions_dict = {
            'window_opening_1': 0,
            'window_opening_2': 0
        }
        
        # se ejecuta un paso de tiempo
        obs_dict, reward, terminated, truncated, infos = env.step(actions_dict)
        # se guardan los datos
        # write a row to the csv file
        row = []
        
        obs_list = obs_dict[_agents_id_list[0]].tolist()
        for _ in range(len(obs_list)):
            row.append(obs_list[_])
        
        action_list = list(actions_dict.values())
        for _ in range(len(action_list)):
            row.append(action_list[_])
        
        row.append(reward[_agents_id_list[0]])
        row.append(terminated["__all__"])
        row.append(truncated["__all__"])
        
        info_list = list(infos[_agents_id_list[0]].values())
        for _ in range(len(info_list)):
            row.append(info_list[_])
        
        writer.writerow(row)
        episode_reward += reward[_agents_id_list[0]]
    # close the file
    data.close()

    return episode_reward

if __name__ == '__main__':
    
    import gymnasium as gym
    import os
    from tools import rewards
    
    name = 'prot3_novent'
    
    env_config={ 
        'weather_folder': 'C:/Users/grhen/Documents/GitHub/natural_ventilation_EP_RLlib/epw/GEF',
        'output': 'C:/Users/grhen/Documents/Resultados_RLforEP/'+ name,
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
        'observation_space': gym.spaces.Box(float("-inf"), float("inf"), (307,)),
        # observation space for simple agent case
        
        # BUILDING CONFIGURATION
        'building_name': 'prot_3_ceiling',
        'reward_function': rewards.reward_function,
    }
    
    policy_config = { # configuracion del control convencional
        'SP_temp': 22, #es el valor de temperatura de confort
        'dT_up': 2.5, #es el límite superior para el rango de confort
        'dT_dn': 2.5, #es el límite inferior para el rango de confort
    }
    try:
        os.makedirs(env_config['output'])
    except OSError:
        pass
    
    episode_reward = init_rb_evaluation(env_config, policy_config, name)
    print(f"Episode reward is: {episode_reward}.")