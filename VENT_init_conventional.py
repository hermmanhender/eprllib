"""# RUN CONVENTIONAL CONTROLS

This script execute the conventional controls in the evaluation scenario.
"""
import csv
from tempfile import TemporaryDirectory
from env.VENT_ep_gym_env import EnergyPlusEnv_v0
from agents.conventional import Conventional
import gymnasium as gym

def inverse_transforming_central_action(agent_actions:dict):
    """This method transforms a dictionary of agent actions into a centralized action.

    Args:
        agent_actions (dict): Dictionary of agent actions in the format {agent_id: action}.

    Returns:
        int: Centralized action.
    """
    agents_ids = [
            0, #"agent_sp"
            1, #"agent_b1"
            2, #"agent_b2"
            3, #"agent_w1"
            4  #"agent_w2"
            ]
    central_action = 0
    for agent_id in agents_ids:
        if agent_id in agent_actions:
            action = agent_actions.get(agent_id, 0)
            central_action = (central_action << 1) | int(action)

    return central_action

# Se debe especificar el path según la PC utilizada
path = 'C:/Users/grhen/Documents'

# Controles de la simulación
ep_terminal_output = True # Esta variable indica si se imprimen o no las salidas de la simulación de EnergyPlus
name = 'conventional_test_0.9'

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
policy = Conventional(policy_config)
# se inicia el entorno con la configuración especificada
env = EnergyPlusEnv_v0(env_config)

# open the file in the write mode
data = open(path+'/ray_results/conventional_test/'+name+'.csv', 'w')
# create the csv writer
writer = csv.writer(data)
terminated = False # variable de control de lazo (es verdadera cuando termina un episodio)
episode_reward = 0
# se obtiene la observaión inicial del entorno para el episodio
obs, info = env.reset()
while not terminated: # se ejecuta un paso de tiempo hasta terminar el episodio
    # se calculan las acciones convencionales de cada elemento
    To = obs[0]
    Ti = obs[1]
    action_w1 = obs[8]
    action_w2 = obs[9]
    
    action_1 = policy.window_opening(Ti, To, action_w1)
    action_2 = policy.window_opening(Ti, To, action_w2)
    action_space = [
        [0,0],
        [0,1],
        [1,0],
        [1,1]
    ]
    index = 0
    for a in action_space:
        if a == [action_1,action_2]:
            actions = index
            break
        else:
            index =+ 1
    
    # se ejecuta un paso de tiempo
    obs, reward, terminated, truncated, infos = env.step(actions)
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
    #pd.DataFrame((obs, reward, terminated, truncated, infos)).to_csv(path+'/ray_results/'+name+'.csv', mode="a", index=False, header=False)
    episode_reward += reward
# close the file
data.close()
print(f"Episode reward is: {episode_reward}.")