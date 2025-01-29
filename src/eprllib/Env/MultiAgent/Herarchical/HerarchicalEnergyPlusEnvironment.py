"""
Multi-Agent Herarchical Environment for EnergyPlus in RLlib
============================================================

This script define the environment of EnergyPlus implemented in RLlib. To works 
need to define the EnergyPlus Runner.
"""

from queue import Queue
from typing import Any, Dict, Optional
from eprllib.Env.MultiAgent.Herarchical.HerarchicalEnergyPlusRunner import HerarchicalEnergyPlusRunner
from eprllib.Utils.annotations import override
from eprllib.Env.MultiAgent.EnergyPlusEnvironment import EnergyPlusEnv_v0

class HerarchicalEnergyPlusEnv(EnergyPlusEnv_v0):
    def __init__(
        self,
        env_config: Dict[str,Any]
    ) -> None:
        super.__init__(env_config)
        
    @override(EnergyPlusEnv_v0)
    def reset(
        self, *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ):
        # Increment the counting of episodes in 1.
        self.episode += 1
        # saving the episode in the env_config to use across functions.
        self.env_config['episode'] = self.episode
        # stablish the timestep counting in zero.
        self.timestep = 0
        # Condition of truncated episode
        if not self.truncateds:
            # Condition implemented to restart a new epsiode when simulation is completed and 
            # EnergyPlus Runner is already inicialized.
            if self.energyplus_runner is not None:
                self.close()
            # Define the queues for flow control between MDP and EnergyPlus threads in a max size 
            # of 1 because EnergyPlus timestep will be processed at a time.
            self.obs_queue = Queue(maxsize=1)
            self.act_queue = Queue(maxsize=1)
            self.infos_queue = Queue(maxsize=1)
            
            # episode_config_fn: Function that take the env_config as argument and upgrade the value
            # of env_config['epjson'] (str). Buid-in function allocated in tools.ep_episode_config
            self.env_config = self.episode_fn.get_episode_config(self.env_config)
            self.agents = self.episode_fn.get_episode_agents(self.env_config, self.possible_agents)
            
            for agent in self.agents:
                self.reward_fn.update({agent: self.env_config["agents_config"][agent]["reward"]['reward_fn'](self.env_config["agents_config"][agent]["reward"]["reward_fn_config"])})
            
            # Start EnergyPlusRunner whith the following configuration.
            self.energyplus_runner = HerarchicalEnergyPlusRunner(
                episode = self.episode,
                env_config = self.env_config,
                obs_queue = self.obs_queue,
                act_queue = self.act_queue,
                infos_queue = self.infos_queue,
                agents = self.agents,
                observation_fn = self.observation_fn,
                action_fn = self.action_fn
            )
            # Divide the thread in two in this point.
            self.energyplus_runner.start()
            # Wait untill an observation and an infos are made, and get the values.
            self.energyplus_runner.obs_event.wait()
            self.last_obs = self.obs_queue.get()
            self.energyplus_runner.infos_event.wait()
            self.last_infos = self.infos_queue.get()
        
        # Asign the obs and infos to the environment.
        obs = self.last_obs
        infos = self.last_infos
        
        self.terminateds = False
        self.truncateds = False
            
        return obs, infos
    