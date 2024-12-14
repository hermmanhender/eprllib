"""

"""
from typing import Any, Dict, Tuple, Set
from eprllib.ObservationFunctions.ObservationFunctions import ObservationFunction

class IndependentLearning(ObservationFunction):
    def __init__(
        self,
        obs_fn_config: Dict[str,Any]
        ):
        super().__init__(obs_fn_config)
        
    def get_agent_obs_dim(
        self,
        env_config: Dict[str,Any],
        _agent_ids: Set,
        _thermal_zone_ids: Set,
        ) -> int:
        return NotImplementedError("You must implement this method.")
        
    def set_agent_obs_and_infos(
        self,
        env_config: Dict[str,Any],
        _agent_ids: Set,
        _thermal_zone_ids: Set,
        actuator_states: Dict[str,Any] = NotImplemented,
        actuator_infos: Dict[str,Any] = NotImplemented,
        site_state: Dict[str,Any] = NotImplemented,
        site_infos: Dict[str,Any] = NotImplemented,
        thermal_zone_states: Dict[str, Dict[str,Any]] = NotImplemented,
        thermal_zone_infos: Dict[str, Dict[str,Any]] = NotImplemented,
        agent_states: Dict[str, Dict[str,Any]] = NotImplemented,
        agent_infos: Dict[str, Dict[str,Any]] = NotImplemented,
        ) -> Tuple[Dict[str,Any],Dict[Dict[str,Any]]]:
        
        return NotImplementedError("You must implement this method.")
    
        
    def get_independent_agents_obs(
        self,
        obs_tz,
        infos_tz,
        agent_actions: Dict = NotImplemented):
        # Se asignan observaciones y infos a cada agente.
        agents_obs = {agent: [] for agent in self._agent_ids}
        agents_infos = {agent: {} for agent in self._agent_ids}
        
        for agent in self._agent_ids:
            # Agent properties
            agent_thermal_zone = self.env_config['agents_config'][agent]['thermal_zone']
            
            # Transform the observation in a numpy array to meet the condition expected in a RLlib Environment
            agents_obs[agent] = np.array(list(obs_tz[agent_thermal_zone].values()), dtype='float32')
            # if apply, add the actuator state.
            if self.env_config['use_actuator_state']:
                if agent_actions == NotImplemented:
                    NotImplementedError('The agent_actions argument is not implemented. Please provide a dictionary with the agent actions.')
                    return
                agents_obs[agent] = np.concatenate(
                    (
                        agents_obs[agent],
                        [agent_actions[agent]],
                    ),
                    dtype='float32'
                )
            # if apply, add the agent indicator.
            if self.env_config['use_agent_indicator']:
                agent_indicator = self.env_config['agents_config'][agent]['agent_indicator']
                agents_obs[agent] = np.concatenate(
                    (
                        agents_obs[agent],
                        [agent_indicator],
                    ),
                    dtype='float32'
                )
            # if apply, add the thermal zone indicator
            if self.env_config['use_thermal_zone_indicator']:
                thermal_zone_indicator = self.env_config['agents_config'][agent]['thermal_zone_indicator']
                agents_obs[agent] = np.concatenate(
                    (
                        agents_obs[agent],
                        [thermal_zone_indicator],
                    ),
                    dtype='float32'
                )
            # if apply, add the agent type.
            if self.env_config['use_agent_type']:
                agent_type = self.env_config['agents_config'][agent]['actuator_type']
                agents_obs[agent] = np.concatenate(
                    (
                        agents_obs[agent],
                        [agent_type],
                    ),
                    dtype='float32'
                )
            
            # Print the agents_obs array if one of the values are NaN or Inf
            if np.isnan(agents_obs[agent]).any() or np.isinf(agents_obs[agent]).any():
                print(f"NaN or Inf value found in agents_obs[{agent}]:\n{agents_obs[agent]}")
                    
            # Agent infos asignation
            agents_infos[agent] = infos_tz[agent_thermal_zone]
            
        return agents_obs, agents_infos
