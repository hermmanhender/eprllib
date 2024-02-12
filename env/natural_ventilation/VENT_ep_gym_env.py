"""# ENERGYPLUS RLLIB ENVIRONMENT

"""

import gymnasium as gym
import logging

from queue import Empty, Full, Queue
from typing import Any, Dict, Optional
from VENT_ep_runner import EnergyPlusRunner

logger = logging.getLogger(__name__)


class EnergyPlusEnv_v0(gym.Env):
    """_summary_

    Args:
        gym (_type_): _description_
    """
    def __init__(
        self,
        env_config: Dict[str, Any]
        ):
        super().__init__()
        # configuración del entorno
        self.env_config = env_config
        self.episode = -1
        self.timestep = 0
        
        # Agents
        self._agent_ids = [
            0, #"agent_sp"
            1, #"agent_b1"
            2, #"agent_b2"
            3, #"agent_w1"
            4  #"agent_w2"
            ]
        
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(float("-inf"), float("inf"), (49,))
        self.last_obs = {}
        self.energyplus_runner: Optional[EnergyPlusRunner] = None
        # queue definitions
        self.obs_queue: Optional[Queue] = None # comunicación de observaciones
        self.act_queue: Optional[Queue] = None # comunicación de acciones
        self.cooling_queue: Optional[Queue] = None # comunicación de métrica cooling
        self.heating_queue: Optional[Queue] = None # comunicación de métrica heating 
        self.pmv_queue: Optional[Queue] = None # comunicación de variable PMV
        self.PPD_queue: Optional[Queue] = None

    def reset(
        self, *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ):
        # contador de episodio se incrementa en 1
        self.episode += 1
        self.env_config['episode'] = self.episode
        self.episode_module = self.episode%10
        
        #self.last_obs = self.observation_space.sample()
        
        if self.energyplus_runner is not None and self.energyplus_runner.simulation_complete:
            self.energyplus_runner.stop()
        
        # observations and actions queues for flow control
        # queues have a default max size of 1
        # as only 1 E+ timestep is processed at a time
        self.obs_queue = Queue(maxsize=1)
        self.act_queue = Queue(maxsize=1)
        self.cooling_queue = Queue(maxsize=1)
        self.heating_queue = Queue(maxsize=1)
        self.pmv_queue = Queue(maxsize=1)
        self.PPD_queue = Queue(maxsize=1)

        # Se inicia EnergyPlusRunner.
        self.energyplus_runner = EnergyPlusRunner(
            episode=self.episode,
            env_config=self.env_config,
            agents_ids=self._agent_ids,
            obs_queue=self.obs_queue,
            act_queue=self.act_queue,
            cooling_queue=self.cooling_queue,
            heating_queue=self.heating_queue,
            pmv_queue = self.pmv_queue,
            PPD_queue = self.PPD_queue
        )
        # En este punto el hilo se divide en dos
        self.energyplus_runner.start()
        
        # Una vez iniciado se consulta por la primer observación realizada
        # durante la simulación.
        self.energyplus_runner.obs_event.wait()
        obs = self.obs_queue.get()
        #print(f"Observation geted in first timestep of episode {self.episode}.")
        self.energyplus_runner.cooling_event.wait()
        ec = self.cooling_queue.get()
        self.energyplus_runner.heating_event.wait()
        eh = self.heating_queue.get()
        self.energyplus_runner.pmv_event.wait()
        pmv = self.pmv_queue.get()
        self.energyplus_runner.PPD_event.wait()
        ppd = self.PPD_queue.get()
        self.last_obs = obs
        
        return obs, {}

    def step(self, action):
        # Se incrementa el conteo de los pasos de tiempo
        self.timestep += 1
        done = False
        
        # timeout is set to 2s to handle end of simulation cases, which happens async
        # and materializes by worker thread waiting on this queue (EnergyPlus callback
        # not consuming yet/anymore)
        # timeout value can be increased if E+ timestep takes longer
        timeout = 40
        
        # simulation_complete is likely to happen after last env step()
        # is called, hence leading to waiting on queue for a timeout
        if self.energyplus_runner.simulation_complete:
            # check for simulation errors
            if self.energyplus_runner.failed():
                raise Exception("Faulty episode")
            done = True
            obs = self.last_obs
        else:
            # enqueue action (received by EnergyPlus through dedicated callback)
            # then wait to get next observation.
            try:
                #print(f"Puting actionin timestep {self.timestep}.")
                self.act_queue.put(action,timeout=timeout)
                self.energyplus_runner.act_event.set()
                self.energyplus_runner.obs_event.wait(timeout=timeout)
                obs = self.obs_queue.get(timeout=timeout)
                #print(f"Observation geted in timestep {self.timestep}.")
                self.last_obs = obs
            except (Full, Empty):
                done = True
                obs = self.last_obs
        
        if self.energyplus_runner.failed():
            raise Exception("Faulty episode")
        # compute reward
        reward, energy, comfort, ppd = self._compute_reward(obs[46], obs[47])
        # se generan los diccionarios de retorno
        terminated = done
        truncated = False
        #infos = {"done": done,"energy": energy, "pmv": comfort, "PPD": ppd}
        infos = {
            'terminated': done,
            'energy': energy,
            'comfort': comfort,
            'ppd': ppd,
        }
        if terminated or truncated:
            print(f"Terminated: {terminated}.\nTruncated: {truncated}.")
        return obs, reward, terminated, truncated, infos

    def close(self):
        if self.energyplus_runner is not None:
            self.energyplus_runner.stop()
    
    def render(self, mode="human"):
        pass

    def _compute_reward(self, beta, E_max) -> float:
        """_summary_

        Args:
            beta (_type_): _description_
            E_max (_type_): _description_

        Raises:
            Exception: _description_
            Exception: _description_
            Exception: _description_
            Exception: _description_

        Returns:
            float: _description_
        """
        self.energyplus_runner.cooling_event.wait(10)
        if self.energyplus_runner.failed():
            raise Exception("Faulty episode")
        ec = self.cooling_queue.get()
        self.energyplus_runner.heating_event.wait(10)
        if self.energyplus_runner.failed():
            raise Exception("Faulty episode")
        eh = self.heating_queue.get()
        self.energyplus_runner.pmv_event.wait(10)
        if self.energyplus_runner.failed():
            raise Exception("Faulty episode")
        pmv = self.pmv_queue.get()
        self.energyplus_runner.PPD_event.wait(10)
        if self.energyplus_runner.failed():
            raise Exception("Faulty episode")
        ppd = self.PPD_queue.get()
        e_C = (abs(ec))/(3600000)
        e_H = (abs(eh))/(3600000)
        
        reward = (-beta*(e_H + e_C)/(E_max) - (1-beta)*(ppd/100)) #/(int(self.env_config['longitud_episodio'])*24*6-1)
        return reward, e_C+e_H, pmv, ppd