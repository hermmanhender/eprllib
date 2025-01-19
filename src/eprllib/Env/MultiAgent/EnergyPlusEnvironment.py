"""
Multi-Agent Environment for EnergyPlus in RLlib
================================================

This script define the environment of EnergyPlus implemented in RLlib. To works 
need to define the EnergyPlus Runner.
"""
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from queue import Empty, Full, Queue
from typing import Any, Dict, Optional
from eprllib.Env.MultiAgent.EnergyPlusRunner import EnergyPlusRunner
from eprllib.RewardFunctions.RewardFunctions import RewardFunction
from eprllib.EpisodeFunctions.EpisodeFunctions import EpisodeFunction
from eprllib.ObservationFunctions.ObservationFunctions import ObservationFunction
from eprllib.ActionFunctions.ActionFunctions import ActionFunction

class EnergyPlusEnv_v0(MultiAgentEnv):
    """The EnergyPlusEnv_v0 class represents a multi-agent environment for 
    reinforcement learning tasks related to building energy simulation using 
    EnergyPlus software. It inherits from the MultiAgentEnv class, which 
    suggests that it supports multiple agents interacting with the environment.

    The class initializes with an env_config dictionary that contains various 
    configuration settings for the environment, such as the list of agent IDs, 
    action spaces, observable variables, actuators, meters, and other 
    EnergyPlus-related settings.
    
    The reset method is responsible for setting up a new episode of the environment. 
    It increments the episode counter, initializes queues for communication between 
    the environment and EnergyPlus, and starts an instance of the EnergyPlusRunner
    class, which likely handles the EnergyPlus simulation.
    
    The step method is the core of the environment, where agents take actions, and 
    the environment updates its state accordingly. It processes the provided actions, 
    communicates with the EnergyPlus simulation through queues, retrieves 
    observations and information from the simulation, calculates rewards based on a 
    specified reward function, and determines if the episode should terminate or truncate.
    
    The close method is used to stop the EnergyPlus simulation when the environment is 
    no longer needed.
    
    The render method is currently a placeholder and does not perform any rendering 
    functionality.
    
    Overall, this class encapsulates the logic for running EnergyPlus simulations as 
    part of a multi-agent reinforcement learning environment, allowing agents to 
    interact with the building energy simulation and receive observations, rewards, 
    and termination signals based on their actions.
    """
    def __init__(
        self,
        env_config: Dict[str, Any]
        ):
        """The __init__ method in the EnergyPlusEnv_v0 class is responsible for 
        initializing the multi-agent environment for the EnergyPlus reinforcement 
        learning task. Here's a summary of what it does:
            * 1. It assigns the env_config dictionary, which contains various 
            configuration settings for the environment, such as agent IDs, action 
            spaces, observable variables, actuators, meters, and other EnergyPlus-related 
            settings.
            * 2. It sets the agents attribute as a set of agent IDs from the env_config.
            * 3. It assigns the action_space attribute from the env_config.
            * 4. It calculates the length of the observation space based on the number of 
            observable variables, meters, actuators, time variables, weather variables, 
            and other relevant information specified in the env_config. It then creates a 
            Box space for the observation_space attribute.
            * 5. It initializes the energyplus_runner, obs_queue, act_queue, and infos_queue
            attributes to None. These will be used later for communication between the 
            environment and the EnergyPlus simulation.
            * 6. It sets up variables for tracking the episode number (episode), timestep 
            (timestep), termination status (terminateds), and truncation status (truncateds).
            * 7. It creates a dictionary last_obs and last_infos to store the last observation 
            and information for each agent.
        
        Overall, the __init__ method sets up the necessary data structures and configurations 
        for the EnergyPlus multi-agent environment, preparing it for running simulations 
        and interacting with agents.
        """
        self.env_config = env_config
        
        # Define all agent IDs that might even show up in your episodes.
        self.possible_agents = [key for key in self.env_config["agents_config"].keys()]
        # If your agents never change throughout the episode, set
        # `self.agents` to the same list as `self.possible_agents`.
        self.agents = self.possible_agents
        # Otherwise, you will have to adjust `self.agents` in `reset()` and `step()` to whatever the
        # currently "alive" agents are.
        
        # asigning the configuration of the environment.
        
        self.action_fn: Dict[str, ActionFunction] = {agent: None for agent in self.agents}
        self.reward_fn: Dict[str, RewardFunction] = {agent: None for agent in self.agents}
        for agent in self.agents:
            self.action_fn.update({agent: self.env_config["agents_config"][agent]["action"]['action_fn'](self.env_config["agents_config"][agent]["action"]["action_fn_config"])})
            self.reward_fn.update({agent: self.env_config["agents_config"][agent]["reward"]['reward_fn'](self.env_config["agents_config"][agent]["reward"]["reward_fn_config"])})
        
        self.observation_fn: ObservationFunction = self.env_config['observation_fn'](self.env_config["observation_fn_config"])
        self.episode_fn: EpisodeFunction = self.env_config['episode_fn'](self.env_config["episode_fn_config"])
        
        # asignation of environment action space.
        # self.action_space = {agent: None for agent in self.agents}
        # for agent in self.agents:
        #     self.action_space[agent] = self.action_fn[agent].get_action_space_dim()
        
        self.action_space = self.action_fn[self.agents[0]].get_action_space_dim()
        
        # asignation of the environment observation space.
        self.observation_space = self.observation_fn.get_agent_obs_dim(self.env_config)
        
        # super init of the base class (after the previos definition to avoid errors with agents argument).
        super().__init__()
        
        # EnergyPlusRunner class and queues for communication between MDP and EnergyPlus.
        self.energyplus_runner: Optional[EnergyPlusRunner] = None
        self.obs_queue: Optional[Queue] = None
        self.act_queue: Optional[Queue] = None
        self.infos_queue: Optional[Queue] = None
        
        # ===CONTROLS=== #
        # variable for the registry of the episode number.
        self.episode = -1
        self.timestep = 0
        self.terminateds = False
        self.truncateds = False
        self.env_config['num_time_steps_in_hour'] = 0
        # dict to save the last observation and infos in the environment.
        self.last_obs = {agent: [] for agent in self.agents}
        self.last_infos = {agent: {} for agent in self.agents}

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
            
            # TODO: Add here the availability to run EnergyPlus in Parallel.
            # Run EnergyPlus in Parallel
            # EnergyPlus is a multi-thread application but is not optimized for parallel runs on multi-core 
            # machines. However, multiple parametric runs can be launched in parallel if these runs are 
            # independent. One way to do it is to create multiple folders and copy files EnergyPlus.exe, 
            # Energy+.idd, DElight2.dll, libexpat.dll, bcvtb.dll, EPMacro.exe (if macros are used), 
            # ExpandObjects.exe (if HVACTemplate or Compact HVAC objects are used) from the original EnergyPlus 
            # installed folder. Inside each folder, copy IDF file as in.idf and EPW file as in.epw, then run 
            # EnergyPlus.exe from each folder. This is better handled with a batch file. If the Energy+.ini file 
            # exists in one of the created folders, make sure it is empty or points to the current EnergyPlus folder.

            # EP-Launch now does this automatically for Group or Parametric-Preprocessor runs.

            # The benefit of run time savings depends on computer configurations including number of CPUs, 
            # CPU clock speed, amount of RAM and cache, and hard drive speed. To be time efficient, the number 
            # of parallel EnergyPlus runs should not be more than the number of CPUs on a computer. The EnergyPlus 
            # utility program EP-Launch is being modified to add the parallel capability for group simulations. The 
            # long term goal is to run EnergyPlus in parallel on multi-core computers even for a single EnergyPlus 
            # run without degradation to accuracy.
            
            
            
            # Start EnergyPlusRunner whith the following configuration.
            self.energyplus_runner = EnergyPlusRunner(
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

    def step(self, action):
        # increment the timestep in 1.
        self.timestep += 1
        # ===CONTROLS=== #
        # terminated variable is used to determine the end of a episode. Is stablished as False until the
        # environment present a terminal state.
        terminated = {}
        truncated = {}
        # Truncate the simulation RunPeriod into shorter episodes defined in days. Default: 0
        cut_episode_len: int = self.env_config['cut_episode_len']
        if self.env_config["evaluation"]:
            self.truncateds = False
        elif cut_episode_len == 0:
            self.truncateds = False
        else:
            cut_episode_len_timesteps = cut_episode_len * 24 * self.env_config['num_time_steps_in_hour']
            if self.timestep % cut_episode_len_timesteps == 0:
                self.truncateds = True
        # timeout is set to 10s to handle the time of calculation of EnergyPlus simulation.
        # timeout value can be increased if EnergyPlus timestep takes longer.
        timeout = self.env_config["timeout"]
        
        # check for simulation errors.
        if self.energyplus_runner.failed():
            self.terminateds = True
            raise Exception('Simulation in EnergyPlus fallied.')
        # simulation_complete is likely to happen after last env step()
        # is called, hence leading to waiting on queue for a timeout.
        if self.energyplus_runner.simulation_complete:
            # if the simulation is complete, the episode is ended.
            self.terminateds = True
            # we use the last observation as a observation for the timestep.
            obs = self.last_obs
            infos = self.last_infos

        # if the simulation is not complete, enqueue action (received by EnergyPlus through 
        # dedicated callback) and then wait to get next observation.
        else:
            try:
                # Send the action to the EnergyPlus Runner flow.
                self.act_queue.put(action)
                self.energyplus_runner.act_event.set()
                # Modify the quantity of agents in the environment for this timestep, if apply.
                self.energyplus_runner.agents = self.episode_fn.get_timestep_agents(self.env_config, self.possible_agents)
                
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
                self.terminateds = True
                # We use the last observation as a observation for the timestep.
                obs = self.last_obs
                infos = self.last_infos
        
        # Calculate the reward in the timestep
        reward_dict = {}
        for agent in self.agents:
            reward_dict.update({agent: self.reward_fn[agent].get_reward(infos[agent], self.terminateds, self.truncateds)})
        
        terminated["__all__"] = self.terminateds
        truncated["__all__"] = self.truncateds
        # if self.timestep % 100 == 0:
        #     print(f"Action: {action}\nReward: {reward_dict}")
        return obs, reward_dict, terminated, truncated, infos

    def close(self):
        if self.energyplus_runner is not None:
            self.energyplus_runner.stop()
    
    def render(self, mode="human"):
        pass
