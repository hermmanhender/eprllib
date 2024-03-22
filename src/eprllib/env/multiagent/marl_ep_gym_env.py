"""# ENERGYPLUS RLLIB ENVIRONMENT

This script define the environment of EnergyPlus implemented in RLlib. To works need to define the
EnergyPlus Runner.
"""
from ray.rllib.env.multi_agent_env import MultiAgentEnv
# Used to define the environment base and the size of action and observation spaces.
from queue import Empty, Full, Queue
# Used to separate the execution in two threads and comunicate EnergyPlus with this environment.
from typing import Any, Dict, Optional
# To specify the types of variables espected.
from eprllib.env.multiagent.marl_ep_runner import EnergyPlusRunner
# The EnergyPlus Runner.

class EnergyPlusEnv_v0(MultiAgentEnv):
    def __init__(
        self,
        env_config: Dict[str, Any]
        ):
        # super init of the base class gym.Env.
        super().__init__()
        # asigning the configuration of the environment.
        self.env_config = env_config
        # asignation of the agents ids for the environment.
        self._agent_ids = env_config['agent_ids']
        # asignation of environment spaces.
        self.action_space = self.env_config['action_space']
        self.observation_space = self.env_config['observation_space']
        
        # EnergyPlus Runner class.
        self.energyplus_runner: Optional[EnergyPlusRunner] = None
        # queues for communication between MDP and EnergyPlus.
        self.obs_queue: Optional[Queue] = None
        self.act_queue: Optional[Queue] = None
        self.infos_queue: Optional[Queue] = None
        
        # ===CONTROLS=== #
        # variable for the registry of the episode number.
        self.episode = -1
        # dict to save the last observation and infos in the environment.
        self.last_obs = {}
        self.last_infos = {}
        for agent in env_config['agent_ids']:
            self.last_obs[agent] = []
            self.last_infos[agent] = []
            

    def reset(
        self, *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ):
        # Increment the counting of episodes in 1.
        self.episode += 1
        # stablish the timestep counting in zero.
        self.timestep = 0

        # Condition implemented to restart a new epsiode when simulation is completed and 
        # EnergyPlus Runner is already inicialized.
        if self.energyplus_runner is not None and self.energyplus_runner.simulation_complete:
            self.energyplus_runner.stop()
        
        # Define the queues for flow control between MDP and EnergyPlus threads in a max size 
        # of 1 because EnergyPlus timestep will be processed at a time.
        self.obs_queue = Queue(maxsize=1)
        self.act_queue = Queue(maxsize=1)
        self.infos_queue = Queue(maxsize=1)
        
        # Start EnergyPlusRunner whith the following configuration.
        self.energyplus_runner = EnergyPlusRunner(
            episode=self.episode,
            env_config=self.env_config,
            obs_queue=self.obs_queue,
            act_queue=self.act_queue,
            infos_queue=self.infos_queue
        )
        
        # Divide the thread in two in this point.
        self.energyplus_runner.start()
        # Wait untill an observation and an infos are made, and get the values.
        self.energyplus_runner.obs_event.wait()
        obs = self.obs_queue.get()
        self.energyplus_runner.infos_event.wait()
        infos = self.infos_queue.get()
        
        # Save the observation as a last observation.
        self.last_obs = obs
        self.last_infos = infos
            
        return obs, infos

    def step(self, action):
        # increment the timestep in 1.
        self.timestep += 1
        # ===CONTROLS=== #
        # terminated variable is used to determine the end of a episode. Is stablished as False until the
        # environment present a terminal state.
        terminated = {}
        terminateds = False
        truncated = {}
        truncateds = False
        
        # timeout is set to 5s to handle the time of calculation of EnergyPlus simulation.
        # timeout value can be increased if EnergyPlus timestep takes longer.
        timeout = self.env_config.get("timeout", 5)
        
        # simulation_complete is likely to happen after last env step()
        # is called, hence leading to waiting on queue for a timeout.
        if self.energyplus_runner.simulation_complete:
            # check for simulation errors.
            if self.energyplus_runner.failed():
                raise Exception("Faulty episode")
            
            # if the simulation is complete, the episode is ended.
            terminateds = True
            # we use the last observation as a observation for the timestep.
            obs = self.last_obs
            infos = self.last_infos

        # if the simulation is not complete, enqueue action (received by EnergyPlus through 
        # dedicated callback) and then wait to get next observation.
        else:
            try:
                # Send the action to the EnergyPlus Runner flow.
                self.act_queue.put(action,timeout=timeout)
                self.energyplus_runner.act_event.set()
                # Get the return observation and infos after the action is applied.
                self.energyplus_runner.obs_event.wait(timeout=timeout)
                obs = self.obs_queue.get(timeout=timeout)
                self.energyplus_runner.infos_event.wait(timeout=timeout)
                infos = self.infos_queue.get(timeout=timeout)
                # Upgrade last observation and infos dicts.
                self.last_obs = obs
                self.last_infos = infos

            except (Full, Empty):
                # Set the terminated variable into True to finish the episode.
                terminateds = True
                # We use the last observation as a observation for the timestep.
                obs = self.last_obs
                infos = self.last_infos
        
        # Raise an exception if the episode is faulty.
        if self.energyplus_runner.failed():
            truncateds = True
            raise Exception("Faulty episode")
        
        # Calculate the reward in the timestep
        reward_function = self.env_config['reward_function']
        reward = reward_function(self.env_config, obs, infos)
        
        reward_dict = {}
        for agent in self.env_config['agent_ids']:
            reward_dict[agent] =  reward
        
        terminated["__all__"] = terminateds
        truncated["__all__"] = truncateds
        
        return obs, reward_dict, terminated, truncated, infos

    def close(self):
        if self.energyplus_runner is not None:
            self.energyplus_runner.stop()
    
    def render(self, mode="human"):
        pass