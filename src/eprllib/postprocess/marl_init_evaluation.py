"""# RUN DRL CONTROLS

This script execute the conventional controls in the evaluation scenario.
"""
import sys
sys.path.insert(0, 'C:/Users/grhen/Documents/GitHub/natural_ventilation_EP_RLlib')
import csv
from ray.rllib.policy.policy import Policy
from eprllib.env.multiagent.marl_ep_gym_env import EnergyPlusEnv_v0


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
        actions_dict = {}
        for agent in _agents_id:
            action, _, _ = policy['shared_policy'].compute_single_action(obs_dict[agent])
            actions_dict[agent] = action
        
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
    from eprllib.tools import rewards

    name = 'prot3_drl_5'
    
    # Controles de la simulación
    env_config={
        # === ENERGYPLUS OPTIONS === #
        'epjson': "C:/Users/grhen/Documents/GitHub/eprllib_experiments/natural_ventilation/files/prot_3_ceiling.epJSON",
        "epw": "C:/Users/grhen/Documents/GitHub/eprllib_experiments/natural_ventilation/files/GEF_Lujan_de_cuyo-hour-H4.epw",
        'output': 'C:/Users/grhen/Documents/Resultados_RLforEP/'+ name,
        'ep_terminal_output': True,

        # === EXPERIMENT OPTIONS === #
        'is_test': True,
        
        # === ENVIRONMENT OPTIONS === #
        'action_space': gym.spaces.Discrete(2),
        'reward_function': rewards.reward_function_T3,
        "ep_variables":{
            "To": ("Site Outdoor Air Drybulb Temperature", "Environment"),
            "Ti": ("Zone Mean Air Temperature", "Thermal Zone: Living"),
            "v": ("Site Wind Speed", "Environment"),
            "d": ("Site Wind Direction", "Environment"),
            "RHo": ("Site Outdoor Air Relative Humidity", "Environment"),
            "RHi": ("Zone Air Relative Humidity", "Thermal Zone: Living"),
            "pres": ("Site Outdoor Air Barometric Pressure", "Environment"),
            "occupancy": ("Zone People Occupant Count", "Thermal Zone: Living"),
            "ppd": ("Zone Thermal Comfort Fanger Model PPD", "Living Occupancy")
        },
        "ep_meters": {
            "electricity": "Electricity:Zone:THERMAL ZONE: LIVING",
            "gas": "NaturalGas:Zone:THERMAL ZONE: LIVING",
        },
        "ep_actuators": {
            "opening_window_1": ("AirFlow Network Window/Door Opening", "Venting Opening Factor", "living_NW_window"),
            "opening_window_2": ("AirFlow Network Window/Door Opening", "Venting Opening Factor", "living_E_window"),
        },
        'time_variables': [
            'hour',
            'day_of_year',
            'day_of_the_week',
            ],
        'weather_variables': [
            'is_raining',
            'sun_is_up',
            "today_weather_beam_solar_at_time",
            ],
        "infos_variables": ["ppd", "occupancy", "Ti"],
        "no_observable_variables": ["ppd"],
        
        # === OPTIONAL === #
        "timeout": 10,
        "T_confort": 23.5,
        "weather_prob_days": 2
    }

    # se importan las políticas convencionales para la configuracion especificada
    # prot3_drl_1 - Caso de 3x256_dueT1x256_douT_DQN y train_batch_size=512
    # checkpoint_path = 'C:/Users/grhen/ray_results/20240503164618_VN_marl_DQN/3x256_dueT1x256_douT_DQN_e6d20_00000/checkpoint_000047'
    # prot3_drl_2 - Caso de 256+2x512+256_dueT1x256_douT_DQN y train_batch_size=1024
    # checkpoint_path = 'C:/Users/grhen/ray_results/20240504023229_VN_marl_DQN/3x256_dueT1x256_douT_DQN_ca4f7_00000/checkpoint_000045'
    # prot3_drl_3 - Caso de 3x256_dueT1x256_douT_DQN y train_batch_size=512, r=-ppd
    # checkpoint_path = 'C:/Users/grhen/ray_results/20240504182238_VN_marl_DQN/3x256_dueT1x256_douT_DQN_8637b_00000/checkpoint_000046'
    # prot3_drl_4 - train_batch_size=10000
    checkpoint_path = 'C:/Users/grhen/ray_results/20240506100817_VN_marl_DQN/3x256_dueT1x256_douT_DQN_cbd3d_00000/checkpoint_000072'
    
    # prot3_drl_5 - PPO
    checkpoint_path = 'C:/Users/grhen/ray_results/20240510100839_VN_marl_PPO/3x256_dueT1x256_douT_PPO_82336_00000/checkpoint_000004'
    
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