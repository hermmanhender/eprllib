"""# ENERGYPLUS RLLIB ENVIRONMENT

This script define the environment of EnergyPlus implemented in RLlib. To works need to define the
EnergyPlus Runner.
"""
import gymnasium as gym
# Used to define the environment base and the size of action and observation spaces.
from queue import Empty, Full, Queue
# Used to separate the execution in two threads and comunicate EnergyPlus with this environment.
from typing import Any, Dict, Optional
# To specify the types of variables espected.
from env.VENT_ep_runner import EnergyPlusRunner
# The EnergyPlus Runner.

class EnergyPlusEnv_v0(gym.Env):
    def __init__(
        self,
        env_config: Dict[str, Any]
        ):
        """Environment of a building that run with EnergyPlus Runner.

        Args:
            env_config (Dict[str, Any]): _description_
                'action_space'
                'observation_space'
        """
        super().__init__()
        # super init of the base class gym.Env.
        self.env_config = env_config
        # asigning the configuration of the environment.
        self.episode = -1
        # variable for the registry of the episode number.
        self.action_space = self.env_config['action_space']
        # asignation of the action space.
        self.observation_space = self.env_config['observation_space']
        # asignation of the observation space.
        self.last_obs = {}
        # dict to save the last observation in the environment.
        self.energyplus_runner: Optional[EnergyPlusRunner] = None
        # variable where the EnergyPlus Runner object will be save.
        self.obs_queue: Optional[Queue] = None
        # queue for observation communication between threads.
        self.act_queue: Optional[Queue] = None
        # queue for actions communication between threads.
        self.cooling_queue: Optional[Queue] = None
        # queue for cooling metric communication between threads.
        self.heating_queue: Optional[Queue] = None
        # queue for heating metric communication between threads.
        
        self.truncate_flag = False

    def reset(
        self, *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ):
        self.episode += 1
        # Increment the counting of episodes in 1.
        self.timestep = 0
        
        if not self.truncate_flag:
            # This flag is implemented in cases when the episode would be truncated.
            if self.energyplus_runner is not None and self.energyplus_runner.simulation_complete:
                # Condition implemented to restart a new epsiode when simulation is completed and EnergyPlus Runner is already inicialized.
                self.energyplus_runner.stop()
            
            # If the EnergyPlus Runner is not inicialized is a new simulation run.
            self.obs_queue = Queue(maxsize=1)
            self.act_queue = Queue(maxsize=1)
            self.cooling_queue = Queue(maxsize=1)
            self.heating_queue = Queue(maxsize=1)
            # Define the queues for flow control between threads in a max size of 1 because EnergyPlus 
            # time step will be processed at a time.

            
            self.energyplus_runner = EnergyPlusRunner(
                # Start EnergyPlusRunner whith the following configuration.
                episode=self.episode,
                env_config=self.env_config,
                obs_queue=self.obs_queue,
                act_queue=self.act_queue,
                cooling_queue=self.cooling_queue,
                heating_queue=self.heating_queue
            )
            
            self.energyplus_runner.start()
            # Divide the thread in two in this point.
            self.energyplus_runner.obs_event.wait()
            # Wait untill an observation is made.
            obs = self.obs_queue.get()
            # Get the observation.
            self.energyplus_runner.cooling_event.wait()
            # Wait untill an cooling metric read is made.
            ec = self.cooling_queue.get()
            # Get the cooling metric read. It is not used here, but yes in the step method and it is necesary
            # to liberate the space in teh queue.
            self.energyplus_runner.heating_event.wait()
            # Wait untill an heating metric read is made.
            eh = self.heating_queue.get()
            # Get the heating metric read. It is not used here, but yes in the step method and it is necesary
            # to liberate the space in teh queue.
            
            self.last_obs = obs
            # Save the observation as a last observation.
        
        else:
            obs = self.last_obs
            self.truncate_flag = False
            
        return obs, {}

    def step(self, action):
        self.timestep += 1
        terminated = False
        truncated = False
        # terminated variable is used to determine the end of a episode. Is stablished as False until the
        # environment present a terminal state.
        timeout = 40
        # timeout is set to 4s to handle end of simulation cases, which happens async
        # and materializes by worker thread waiting on this queue (EnergyPlus callback
        # not consuming yet/anymore).timeout value can be increased if E+ timestep takes longer.
        if self.energyplus_runner.simulation_complete:
            # simulation_complete is likely to happen after last env step()
            # is called, hence leading to waiting on queue for a timeout.
            if self.energyplus_runner.failed():
                # check for simulation errors.
                raise Exception("Faulty episode")
            terminated = True
            # if the simulation is complete, the episode is ended.
            obs = self.last_obs
            # we use the last observation as a observation for the timestep.

        else:
            # if the simulation is not complete, enqueue action (received by EnergyPlus through 
            # dedicated callback) and then wait to get next observation.
            try:
                self.act_queue.put(action,timeout=timeout)
                self.energyplus_runner.act_event.set()
                # Send the action to the EnergyPlus Runner flow.
                self.energyplus_runner.obs_event.wait(timeout=timeout)
                obs = self.obs_queue.get(timeout=timeout)
                # Get the return observation after the action is applied.
                self.last_obs = obs
                # Upgrade last observation.

            except (Full, Empty):
                terminated = True
                # Set the terminated variable into True to finish the episode.
                obs = self.last_obs
                # We use the last observation as a observation for the timestep.
        
        if self.energyplus_runner.failed():
            # Raise an exception if the episode is faulty.
            truncated = True
            raise Exception("Faulty episode")
        
        self.energyplus_runner.cooling_event.wait(10)
        if self.energyplus_runner.failed():
            raise Exception("Faulty episode")
        ec = self.cooling_queue.get()
        # Wait for the cooling energy consume in the timestep
        
        self.energyplus_runner.heating_event.wait(10)
        if self.energyplus_runner.failed():
            raise Exception("Faulty episode")
        eh = self.heating_queue.get()
        # Wait for the heating energy consume in the timestep
        
        e_C = (abs(ec))/(3600000)
        e_H = (abs(eh))/(3600000)
        energy = e_C + e_H
        # Transform energy variables in Joule to kWh
        
        reward = -energy
        # Calculate the reward in the timestep
        
        infos = {
            'energy': energy,
        }
        # Save energy, comfort (pmv) and ppd in the info dictionary, used after for analisys
        
        #truncated = self.timestep_cut(6*24*5, terminated)
        
        return obs, reward, terminated, truncated, infos

    def close(self):
        if self.energyplus_runner is not None:
            self.energyplus_runner.stop()
    
    def render(self, mode="human"):
        pass

    def timestep_cut(self, num_timestep: int, terminated: bool):
        if self.timestep >= num_timestep and not terminated:
            self.truncate_flag = True
            return True
        else:
            self.truncate_flag = False
            return False