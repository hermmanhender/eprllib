"""# RUN DRL CONTROLS

This script execute the conventional controls in the evaluation scenario.
"""
import gymnasium as gym
import csv
from ray.rllib.policy.policy import Policy
from ray.rllib.utils.framework import try_import_tf
from env.VENT_ep_gym_env import EnergyPlusEnv_v0
from tempfile import TemporaryDirectory

tf1, tf, _ = try_import_tf()
tf1.enable_eager_execution()

# Se debe especificar el path según la PC utilizada
path = 'C:/Users/grhen/Documents'

# Controles de la simulación
tune_runner = True
restore = False
algorithm = 'DQN'

ep_terminal_output = True # Esta variable indica si se imprimen o no las salidas de la simulación de EnergyPlus
name = 'VN_P1_0.5_DQN'

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
    'is_test': True,
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
checkpoint_path = 'C:/Users/grhen/ray_results/VN_P1_0.5_DQN/1024p2x512_dueT1x512_douT_DQN_cb9bc_00000/checkpoint_000064'

# Use the `from_checkpoint` utility of the Policy class:
policy = Policy.from_checkpoint(checkpoint_path)
# se inicia el entorno con la configuración especificada
env = EnergyPlusEnv_v0(env_config)

# open the file in the write mode
data = open(path+'/ray_results/proposed_test/'+name+'.csv', 'w')
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
    for _ in range(25):
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
print(f"Episode reward is: {episode_reward}.")