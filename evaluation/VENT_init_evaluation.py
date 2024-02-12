"""# RUN DRL CONTROLS

This script execute the conventional controls in the evaluation scenario.
"""

import numpy as np
import csv
from ray.rllib.policy.policy import Policy
from ray.rllib.utils.framework import try_import_tf

from tempfile import TemporaryDirectory
from VENT_ep_gym_env import EnergyPlusEnv_v0

tf1, tf, _ = try_import_tf()
tf1.enable_eager_execution()

# Se debe especificar el path según la PC utilizada
#path = "/home/german/Documentos"
#path = 'E:/Usuario/Cliope/Documents'
path = 'C:/Users/grhen/Documents'

# Controles de la simulación
tune_runner = True
restore = False
algorithm = 'PPO'
centralized_action_space = np.loadtxt(
        path+'/GitHub/EP_RLlib/centralized_action_space.csv',
        delimiter=',',
        skiprows=1,
        dtype=int
        )
ep_terminal_output = True # Esta variable indica si se imprimen o no las salidas de la simulación de EnergyPlus
name = 'PPO_test_4'

env_config = {
    'sys_path': path,
    'ep_terminal_output': ep_terminal_output,
    'csv': False,
    'output': TemporaryDirectory("output","DQN_",path+'/Resultados_RLforEP',"E:/tmp").name,
    'epw': path+'/GitHub/EP_RLlib/EP_Wheater_Configuration/GEF/GEF_Lujan_de_cuyo-hour-H1.epw',
    'idf': path+'/GitHub/EP_RLlib/EP_IDF_Configuration/Prototipo_1.epJSON',
    'idf_folderpath': path+"/GitHub/EP_RLlib/EP_IDF_Configuration",
    'idf_output_folder': path+"/models",
    'climatic_stads': path+'/GitHub/EP_RLlib/EP_Wheater_Configuration',
    'beta': 0.5, # Parámetro para ajustar las preferencias del usuario (valor entre 0 y 1)
    'E_max': 2.5/6, # in epJSON file: maximum_total_cooling_capacity/1000 / number_of_timesteps_per_hour
    'latitud':0,
    'longitud':0,
    'altitud':0,
    'separate_state_space': True,
    'one_hot_state_encoding': True,
    'episode': -1,
    'is_test': True,
    'centralized_action_space':centralized_action_space,
    'longitud_episodio': 31,
}
# se importan las políticas convencionales para la configuracion especificada
checkpoint_path = "C:/Users/grhen/ray_results/VN_P1_S_0.5/PPO_4ee8f_00174_ASHA/checkpoint_000003"
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
    for _ in range(len(obs_list)):
        row.append(obs_list[_])
    
    row.append(reward)
    row.append(terminated)
    row.append(truncated)
    
    info_list = list(infos.values())
    for _ in range(len(info_list)):
        row.append(info_list[_])
    
    writer.writerow(row)
    #pd.DataFrame([obs, reward, terminated, truncated, infos]).to_csv(path+'/ray_results/'+name+'.csv', mode="a", index=False, header=False)
    episode_reward += reward
# close the file
data.close()
print(f"Episode reward is: {episode_reward}.")