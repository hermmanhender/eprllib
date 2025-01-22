"""# RUN DRL CONTROLS

This script execute the conventional controls in the evaluation scenario.
"""
import os
from ray.rllib.policy.policy import Policy
from eprllib.Env.MultiAgent.EnergyPlusEnvironment import EnergyPlusEnv_v0
from eprllib.ActionFunctions.ActionFunctions import ActionFunction
import numpy as np
import pandas as pd
import threading
from queue import Queue, Empty
from typing import Dict, Any, Optional

class drl_evaluation:
    
    def __init__(
        self,
        env_config: Dict[str, Any],
        checkpoint_path: str,
        name: str,
        use_RNN: bool = True,
        lstm_cell_size: int = 256,
    ) -> None:
        
        self.env_config = env_config
        self.name = name
        self.use_RNN = use_RNN
        self.lstm_cell_size = lstm_cell_size
        self.policy = Policy.from_checkpoint(checkpoint_path)
        print(f"Checkpoint path restore: {checkpoint_path}")
        self.env = EnergyPlusEnv_v0(env_config)
        self.agents = self.env.agents
        self.terminated = False
        self.data_queue: Optional[Queue] = None
        self.data_processing: Optional[step_processing] = None
        self.timestep = 0
        
        
        self.action_fn: ActionFunction = self.env_config['action_fn']
        
        if not os.path.exists(env_config['output_path']):
            os.makedirs(env_config['output_path'])
    
    def start_simulation(self) -> None:
        self.data_queue = Queue()
        self.data_processing = step_processing(
            self.data_queue,
            f"{self.env_config['output_path']}/{self.name}.csv",
        )
        self.data_processing.run()
        # se obtiene la observaión inicial del entorno para el episodio
        obs_dict, infos = self.env.reset()
        
        if self.use_RNN:
            # range(2) b/c h- and c-states of the LSTM.
            state = [np.zeros([self.lstm_cell_size], np.float32) for _ in range(2)]
        
        # Create an empty DataFrame to store the data
        obs_keys = self.env.energyplus_runner.obs_keys
        infos_keys = self.env.energyplus_runner.infos_keys
        
        data = ['agent_id']+['timestep']+obs_keys+['Action']+['Reward']+['Terminated']+['Truncated']+infos_keys
        # coloca los datos en una cola
        self.data_queue.put(data)
        
        while not self.terminated: # se ejecuta un paso de tiempo hasta terminar el episodio
            # se calculan las acciones convencionales de cada elemento
            actions_dict = {agent: 0 for agent in self.agents}
            for agent in self.agents:
                if self.use_RNN:
                    action, state, _ = self.policy['shared_policy'].compute_single_action(obs_dict[agent], state)
                else:
                    action, _, _ = self.policy['shared_policy'].compute_single_action(obs_dict[agent])
                actions_dict[agent] = action
            
            # Get the values of the variables for a timestep
            obs_dict, reward, terminated, truncated, infos = self.env.step(actions_dict)
            
            # The action is transformer inside the step method, but here is transformed to save the correct value
            actions_dict = self.action_fn.transform_action(actions_dict)
                
            for agent in self.agents:
                data = [agent, self.timestep] + list(obs_dict[agent]) + [actions_dict[agent], reward[agent], terminated["__all__"], truncated["__all__"]] + [value for value in infos[agent].values()]
                # coloca los datos en una cola
                self.data_queue.put(data)
            self.timestep += 1
            self.terminated = terminated["__all__"]


class step_processing:
    def __init__(
        self, 
        data_queue: Queue,
        output_path: str,
    ) -> None:
        
        self.data_queue = data_queue
        self.output_path = output_path
    
    def save_data(self) -> None:
        # Función que consume los datos de la cola y los agrega al DataFrame
        data_df = pd.DataFrame()
        data_saved_len = 0
        while True:
            try:
                datos = self.data_queue.get(timeout=100)
                data_df = pd.concat([data_df, pd.DataFrame([datos])], ignore_index=True)
                # Guarda el DataFrame periódicamente o al final del episodio
                if len(data_df) >= 1000:
                    data_saved_len += len(data_df)
                    print(f"Saving {len(data_df)} and the total amount of timestep saved are: {data_saved_len}.")
                    with open(self.output_path, 'a') as f:
                        data_df.to_csv(f, index=False, header=False)
                    data_df = pd.DataFrame()
            except (Empty):
                if len(data_df) != 0:
                    data_saved_len += len(data_df)
                    print(f"Saving {len(data_df)} and the total amount of timestep saved are: {data_saved_len}.")
                    with open(self.output_path, 'a') as f:
                        data_df.to_csv(f, index=False, header=False)
                break
        # join the Thread back to the main thread, otherwise the program will close
        self.stop()
        
    def run(self) -> None:
        # Inicia un hilo para guardar los datos
        self.thread = threading.Thread(target=self.save_data)
        self.thread.start()
        
    def stop(self) -> None:
        # Detiene el hilo
        print("Stopping data processing.")
        self.thread.join()
        