"""
Base Environment
=================

This script define the environment of EnergyPlus implemented in RLlib. To works 
need to define the EnergyPlus Runner.
"""
import tempfile
# import collections
import numpy as np
from gymnasium import spaces
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from queue import Empty, Full, Queue
from typing import Any, Dict, Optional, Tuple
from eprllib.Environment.EnvironmentConfig import EnvironmentConfig
from eprllib.Environment.EnvironmentRunner import EnvironmentRunner
from eprllib.Agents.Rewards.BaseReward import BaseReward
from eprllib.Agents.Filters.BaseFilter import BaseFilter
from eprllib.Agents.Triggers.BaseTrigger import BaseTrigger
from eprllib.AgentsConnectors.BaseConnector import BaseConnector
from eprllib.Episodes.BaseEpisode import BaseEpisode
from eprllib.Utils.annotations import override
from eprllib import logger

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

class Environment(MultiAgentEnv):
    """
    The BaseEnvironment class represents a multi-agent environment for 
    reinforcement learning tasks related to building energy simulation using 
    EnergyPlus software. It inherits from the MultiAgentEnv class, which 
    suggests that it supports multiple agents interacting with the environment.

    Attributes:
        env_config (Dict[str, Any]): Configuration settings for the environment, 
            such as the list of agent IDs, action spaces, observable variables, 
            actuators, meters, and other EnergyPlus-related settings.
        episode_counter (int): Counter for the number of episodes.
        queues (Dict[str, Queue]): Queues for communication between the environment 
            and EnergyPlus.
        runner (BaseRunner): Instance of the EnergyPlusRunner class that handles 
            the EnergyPlus simulation.

    Methods:
        reset() -> Dict[str, Any]:
            Sets up a new episode of the environment. Increments the episode counter, 
            initializes queues for communication, and starts an instance of the 
            EnergyPlusRunner class.
        
        step(actions: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, float], Dict[str, bool], Dict[str, bool], Dict[str, Any]]:
            The core method where agents take actions, and the environment updates 
            its state accordingly. Processes the provided actions, communicates with 
            the EnergyPlus simulation, retrieves observations and information, 
            calculates rewards, and determines if the episode should terminate or truncate.
        
        close() -> None:
            Stops the EnergyPlus simulation when the environment is no longer needed.
        
        render() -> None:
            Placeholder method for rendering functionality.
    """
    def __init__(
        self,
        env_config: Dict[str, Any]
        ) -> None:
        """
        Initializes the BaseEnvironment class.

        Args:
            env_config (Dict[str, Any]): Configuration settings for the environment.
        """
        self.env_config = env_config
        
        # Episode and multiagent functions
        self.episode_fn: BaseEpisode = self.env_config['episode_fn'](self.env_config["episode_fn_config"])
        logger.debug(f"Environment: Episode configuration: {self.episode_fn.get_episode_config(self.env_config)}")
        self.connector_fn: BaseConnector = self.env_config['connector_fn'](self.env_config["connector_fn_config"])
        logger.debug(f"Environment: Connector configuration: {self.connector_fn.connector_fn_config}")
        
        # === AGENTS === #
        # Define all agent IDs that might even show up in your episodes.
        self.possible_agents = [key for key in self.env_config["agents_config"].keys()]
        logger.info(f"Environment: Possible agents: {self.possible_agents}")
        # If your agents never change throughout the episode, set
        # `self.agents` to the same list as `self.possible_agents`.
        self.agents = self.possible_agents
        # Otherwise, you will have to adjust `self.agents` in `reset()` and `step()` to whatever the
        # currently "alive" agents are.
        
        # asigning the configuration of the environment.
        self.reward_fn: Dict[str, Optional[BaseReward]] = {agent: None for agent in self.agents}
        
        self.trigger_fn: Dict[str, BaseTrigger] = {
            agent: self.env_config["agents_config"][agent]["trigger"]['trigger_fn'](
                self.env_config["agents_config"][agent]["trigger"]["trigger_fn_config"]
            )
            for agent in self.agents
        }
        self.filter_fn: Dict[str, BaseFilter] = {
            agent: self.env_config["agents_config"][agent]["filter"]['filter_fn'](
                self.env_config["agents_config"][agent]["filter"]["filter_fn_config"]
            )
            for agent in self.agents
        }
        
        # asignation of environment action space.
        action_space: Dict[str, Any] = {agent: None for agent in self.agents}
        for agent in self.agents:
            
            assert self.trigger_fn[agent] is not None, f"Trigger function for agent {agent} is not defined."
            
            action_space[agent] = self.trigger_fn[agent].get_action_space_dim()
            
        self.action_space = spaces.Dict(action_space)
        logger.debug(f"Environment: Action space: {self.action_space}")
        
        # ==================================================================
        # TODO: Remove this commented section.
        # # Buffer to save the last "history_len" timesteps of observations
        # self.history_len: Dict[str, int] = {}
        # for agent in self.agents:
        #     self.history_len[agent] = self.env_config["agents_config"][agent]["observation"].get("history_len", 1) # Default history length is 1
        
        #     assert self.history_len[agent] > 0, f"History length must be greater than 0. Agent: {agent}, History length: {self.history_len[agent]}"
        #     assert isinstance(self.history_len[agent], int), f"History length must be an integer. Agent: {agent}, History length: {self.history_len[agent]}"
        # # Create a deque for each agent to store the last observations.
        # # The deque will have a maximum length of `history_len`.
        # self.observation_buffers: Dict[str, collections.deque[Any]] = {
        #     agent:collections.deque(maxlen=self.history_len[agent] ) for agent in self.agents
        # }
        # ==================================================================
        
        # asignation of environment observation space.
        self.observation_space = self.connector_fn.get_all_agents_obs_spaces_dict(self.env_config)
        
        # ==================================================================
        # TODO: Remove this commented section.
        # observation_space: Dict[str, Any]|spaces.Dict = {}
        # original_obs_space: spaces.Dict = self.connector_fn.get_all_agents_obs_spaces_dict(self.env_config)
        # logger.info(f"Originar observation space: {original_obs_space}")
        
        # for agent_id, obs_space_per_agent in original_obs_space.items():
        #     logger.info(f"Observation space for agent {agent_id}: {obs_space_per_agent}")
            
            
            # logger.info(f"History length for agent {agent_id}: {self.history_len[agent_id]}")
            # if self.history_len[agent_id] > 1:
            #     # Asumiendo que obs_space_per_agent es un Box(shape=(feature_dim,))
            #     # Si es un Box(shape=(X, Y)), necesitarás aplanar o ajustar.
            #     assert isinstance(obs_space_per_agent, spaces.Box), f"Observation space for agent {agent_id} must be a Box space."
            #     assert obs_space_per_agent.shape is not None, f"Observation space for agent {agent_id} must have a defined shape."
                
            #     feature_dim: int = obs_space_per_agent.shape[0] # La dimensión de una única observación
            #     observation_space[agent_id] = spaces.Box(
            #         low=obs_space_per_agent.low.min(), # Ajusta si tus rangos son por característica
            #         high=obs_space_per_agent.high.max(), # Ajusta si tus rangos son por característica
            #         shape=(self.history_len[agent_id], feature_dim), # Nueva forma: (N, feature_dim)
            #         dtype=obs_space_per_agent.dtype
            #     )
            #     logger.info(f"Observation space for agente {agent_id} with history: {observation_space[agent_id]}")
            # else:
            #     observation_space[agent_id] = obs_space_per_agent
            #     logger.info(f"Observation space for agent {agent_id} without history: {observation_space[agent_id]}")
            
        
        # observation_space[agent_id] = obs_space_per_agent
        # logger.info(f"Observation space for agent {agent_id} without history: {observation_space[agent_id]}")
        
        # self.observation_space = spaces.Dict(observation_space)
        # logger.info(f"Observation space: {self.observation_space}")
        # ==================================================================
        
        # super init of the base class (after the previos definition to avoid errors with agents argument).
        super().__init__()
        
        # EnergyPlusRunner class and queues for communication between MDP and EnergyPlus.
        self.runner: Optional[EnvironmentRunner] = None
        self.obs_queue: Optional[Queue[Dict[str, Any]]] = None
        self.act_queue: Optional[Queue[Any]] = None
        self.infos_queue: Optional[Queue[Dict[str, Any]]] = None
        
        # === CONTROLS === #
        # variable for the registry of the episode number.
        self.episode = -1
        self.timestep = 0
        self.terminateds = False
        self.truncateds = False
        self.env_config['num_time_steps_in_hour'] = 0
        # dict to save the last observation and infos in the environment.
        self.last_obs: Dict[str, Any] = {agent: None for agent in self.agents}
        self.last_infos: Dict[str, Any] = {agent: {} for agent in self.agents}
        
        # === DATA MANAGEMENT === #
        # output_path: Path to save the EnergyPlus simulation results.
        self.output_path = self.env_config["output_path"]
        if self.output_path is None:
            self.output_path = tempfile.gettempdir()
        logger.info(f"Environment: Output path for EnergyPlus simulation results: {self.output_path}")
        
        # If epjson_path is a IDF file, transform it into a epJSON file.
        if env_config['epjson_path'].endswith(".idf"):
            print("WARNING: Consider using epJSON files for full compatibility with the Episodes API.")

        logger.info("Environment: Environment initialization complete")
        
    @override(MultiAgentEnv)
    def reset(
        self, *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
        ):
        """
        Sets up a new episode of the environment.

        Returns:
            Dict[str, Any]: Initial observations for each agent.
        """
        # Call super's `reset()` method to (maybe) set the given `seed`.
        super().reset(seed=seed, options=options)
        # Set a list for the agents that are currently in the environment.
        # This list will be used to initialize the reward parameters.
        self.agents_to_inizialize_reward_parameters = self.possible_agents.copy()
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
            if self.runner is not None:
                self.close()
            # Define the queues for flow control between MDP and EnergyPlus threads in a max size 
            # of 1 because EnergyPlus timestep will be processed at a time.
            self.obs_queue = Queue(maxsize=1)
            self.act_queue = Queue(maxsize=1)
            self.infos_queue = Queue(maxsize=1)
            
            # episode_config_fn: Function that take the env_config as argument and upgrade the value
            # of env_config['epjson'] (str). Buid-in function allocated in tools.ep_episode_config
            try:
                self.env_config = self.episode_fn.get_episode_config(self.env_config)
                self.agents = self.episode_fn.get_episode_agents(self.env_config, self.possible_agents)
                logger.debug(f"Environment: Episode configuration: {self.env_config}")
            except (ValueError, FileNotFoundError):
                raise ValueError("The episode configuration is not valid. Please check the episode configuration.")
            
            # Update the rewards functions. The parameters of the reward function could be modify by the episode function, due that it's update here.
            for agent in self.agents:
                self.reward_fn[agent] = self.env_config["agents_config"][agent]["reward"]['reward_fn'](self.env_config["agents_config"][agent]["reward"]["reward_fn_config"])
            
            # Start EnergyPlusRunner whith the following configuration.
            self.runner = EnvironmentRunner(
                episode = self.episode,
                env_config = self.env_config,
                obs_queue = self.obs_queue,
                act_queue = self.act_queue,
                infos_queue = self.infos_queue,
                agents = self.agents,
                filter_fn = self.filter_fn,
                trigger_fn = self.trigger_fn,
                connector_fn = self.connector_fn
            )
            # Divide the thread in two in this point.
            self.runner.start()
            # Wait untill an observation and an infos are made, and get the values.
            self.runner.obs_event.wait()
            self.last_obs = self.obs_queue.get()
            self.runner.infos_event.wait()
            self.last_infos = self.infos_queue.get()
            # END OF THE LOOP
            
            # ==================================================================
            # TODO: Remove this commented section.
            # If the history length is greater than 1, we need to create a buffer for each agent
            # to store the last observations.
            # for agent in self.agents:
                # if self.history_len[agent] > 1:
                #     self.observation_buffers[agent].clear()
                #     # If the last observation is not empty, we append it to the buffer.
                #     # This is done to ensure that the buffer has the last `history_len` observations.
                #     single_obs_array = np.array(self.last_obs[agent])
                #     for _ in range(self.history_len[agent]):
                #         self.observation_buffers[agent].append(single_obs_array)
                #     self.last_obs[agent] = np.array(self.observation_buffers[agent])
            # If the history length is greater than 1, we need to return the last `history_len` observations
            # for each agent.
            # ==================================================================
        
        obs = self.last_obs
        infos = self.last_infos
        
        # set initial parameters for the reward function.
        for agent in obs.keys():
            reward_fn_instance = self.reward_fn.get(agent)
            if reward_fn_instance is not None:
                # set the initial parameters for the reward function.
                assert isinstance(reward_fn_instance, BaseReward), f"Reward function for agent {agent} must be a BaseReward instance."
                
                reward_fn_instance.set_initial_parameters(
                    agent,
                    self.connector_fn.obs_indexed[agent]
                    )
                logger.debug(f"Environment: Reward function for agent {agent} initialized with infos: {infos[agent]}")
                self.agents_to_inizialize_reward_parameters.remove(agent)
                logger.debug(f"Environment: Reward function initial parameters for agent {agent} initialized.")
                logger.debug(f"Environment: The agents to inizialize reward parameters: {self.agents_to_inizialize_reward_parameters}")
            else:
                logger.warning(f"Environment: No reward function defined for agent {agent}.")
                pass
        
        self.terminateds = False
        self.truncateds = False
            
        return obs, infos

    @override(MultiAgentEnv)
    def step(
        self, 
        actions: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, float], Dict[str, bool], Dict[str, bool], Dict[str, Any]]:
        """
        The core method where agents take actions, and the environment updates its state accordingly.

        Args:
            actions (Dict[str, Any]): Actions taken by each agent.

        Returns:
            Tuple[Dict[str, Any], Dict[str, float], Dict[str, bool], Dict[str, bool], Dict[str, Any]]: 
                Observations, rewards, done flags, truncated flags, and additional info for each agent.
        """
        # ===CONTROLS=== #
        # terminated variable is used to determine the end of a episode. Is stablished as False until the
        # environment present a terminal state.
        terminated: Dict[str, bool] = {}
        truncated: Dict[str, bool] = {}
        # Truncate the simulation RunPeriod into shorter episodes defined in days. Default: 0
        cut_episode_len: int = self.env_config['cut_episode_len']
        if self.env_config["evaluation"]:
            self.truncateds = False
        elif cut_episode_len == 0:
            self.truncateds = False
        else:
            # cut_episode_len_timesteps = cut_episode_len * 24 * self.env_config['num_time_steps_in_hour']
            if self.timestep % cut_episode_len == 0:
                self.truncateds = True
                logger.debug(f"Environment: Episode truncated after {cut_episode_len} timesteps.")
        # timeout is set to 10s to handle the time of calculation of EnergyPlus simulation.
        # timeout value can be increased if EnergyPlus timestep takes longer.
        timeout = self.env_config["timeout"]
        
        # check for simulation errors.
        assert self.runner is not None, "EnergyPlus Runner is not initialized. Please call reset() first."
        if self.runner.failed():
            self.terminateds = True
            logger.error("Environment: EnergyPlus failed with an error!")
        # simulation_complete is likely to happen after last env step()
        # is called, hence leading to waiting on queue for a timeout.
        if self.runner.simulation_complete:
            # if the simulation is complete, the episode is ended.
            self.terminateds = True
            # we use the last observation as a observation for the timestep.
            obs = self.last_obs
            infos = self.last_infos
            logger.info(f"Environment: Simulation completed after {self.timestep} timesteps.")

        # if the simulation is not complete, enqueue action (received by EnergyPlus through 
        # dedicated callback) and then wait to get next observation.
        else:
            try:
                assert isinstance(self.act_queue, Queue), "Action queue is not initialized. Please call reset() first."
                assert isinstance(self.obs_queue, Queue), "Observation queue is not initialized. Please call reset() first."
                assert isinstance(self.infos_queue, Queue), "Infos queue is not initialized. Please call reset() first."
                # Send the action to the EnergyPlus Runner flow.
                self.act_queue.put(actions)
                self.runner.act_event.set()
                # Modify the quantity of agents in the environment for this timestep, if apply.
                self.runner.agents = self.episode_fn.get_timestep_agents(self.env_config, self.possible_agents)
                
                # Get the return observation and infos after the action is applied.
                self.runner.obs_event.wait(timeout=timeout)
                current_obs = self.obs_queue.get(timeout=timeout)
                self.runner.infos_event.wait(timeout=timeout)
                infos = self.infos_queue.get(timeout=timeout)
                
                # If the history length is greater than 1, we need to create a buffer for each agent
                # to store the last observations.
                obs: Dict[str, Any] = {}
                for agent in self.agents:
                    # ============================================================================
                    # TODO: Remove this commented section.
                    # if self.history_len[agent] > 1:
                    #     # Append the observation to the buffer.
                    #     self.observation_buffers[agent].append(np.array(current_obs[agent]))
                    #     # Create the observation dict with the last `history_len` observations.
                    #     obs[agent] = np.array(self.observation_buffers[agent])
                    # else:
                    #     obs[agent] = current_obs[agent]
                    # ============================================================================
                        
                    obs[agent] = current_obs[agent]
                # logger.info(f"Observation for normal timestep {self.timestep}")
                # logger.info(f"Infos for timestep {self.timestep}: {infos}")
                # Check if the agents to initialize reward parameters is not empty.
                # If it is not empty, we initialize the reward parameters for each agent.
                if self.agents_to_inizialize_reward_parameters is not Empty:
                    for agent in obs.keys():
                        if agent in self.agents_to_inizialize_reward_parameters:
                            reward_fn_instance = self.reward_fn.get(agent)
                            if reward_fn_instance is not None:
                                reward_fn_instance.set_initial_parameters(
                                    agent,
                                    self.connector_fn.obs_indexed[agent]
                                    )
                                self.agents_to_inizialize_reward_parameters.remove(agent)
                
            except (Full, Empty):
                # logger.info("Queue timeout, ending episode early.")
                # Set the terminated variable into True to finish the episode.
                self.terminateds = True
                # We use the last observation as a observation for the timestep.
                obs = self.last_obs
                infos = self.last_infos
                # logger.info(f"Observation for terminal timestep {self.timestep}")
        
        # Calculate the reward in the timestep
        reward_dict: Dict[str, float] = {}
        for agent in obs.keys():
            # The reward function instance can be None if not defined for an agent.
            # We use .get() for safer access and check for None.
            
            assert type(agent) == str, f"Agent {agent} is not a string."
            
            reward_fn_instance = self.reward_fn.get(agent)
            if reward_fn_instance:
                # ============================================================================
                # TODO: Remove this commented section.
                # if self.history_len[agent] > 1:
                #     # If the history length is greater than 1, we use only the last observation in the buffer.
                #     obs_to_reward = np.array(self.last_obs[agent][-1])
                #     next_obs_to_reward = np.array(self.observation_buffers[agent][-1])
                # else:
                #     obs_to_reward = np.array(self.last_obs[agent])
                #     next_obs_to_reward = np.array(obs[agent])
                # ============================================================================
                
                obs_to_reward = self.last_obs[agent]
                next_obs_to_reward = obs[agent]
                    
                reward_dict[agent] = reward_fn_instance.get_reward(
                    obs_to_reward,
                    actions.get(agent),
                    next_obs_to_reward,
                    self.terminateds,
                    self.truncateds
                )
            else:
                # Assign a default reward of 0.0 if no reward function is found.
                reward_dict[agent] = 0.0
        
        terminated["__all__"] = self.terminateds
        truncated["__all__"] = self.truncateds
        
        # Upgrade last observation and infos dicts.
        self.last_obs = obs
        self.last_infos = infos
        
        # increment the timestep in 1.
        self.timestep += 1
        
        return obs, reward_dict, terminated, truncated, infos

    @override(MultiAgentEnv)
    def close(self):
        """
        Stops the EnergyPlus simulation when the environment is no longer needed.
        """
        if self.runner is not None:
            self.runner.stop()
    
    @override(MultiAgentEnv)
    def render(self, mode:str="human"):
        """
        Placeholder method for rendering functionality.
        """
        pass
    
    @staticmethod
    def from_checkpoint(
        cls,
        path: str
    ) -> "Environment":
        msg = "Environment: This method is not implemented yet. Please implement it in the child class."
        logger.error(msg)
        raise NotImplementedError(msg)
    
    def get_default_config(cls) -> EnvironmentConfig:
        return EnvironmentConfig()