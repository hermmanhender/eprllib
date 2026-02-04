"""
Policy Mapping Function
========================


"""
from typing import Dict, Any
from ray.rllib.utils.typing import AgentID, PolicyID
from ray.rllib.evaluation.rollout_worker import RolloutWorker
from ray.rllib.evaluation.episode_v2 import EpisodeV2

OCCUPANCY_INDEX = 2

def policy_mapping_fn(
    agent_id: AgentID,
    episode: EpisodeV2,
    worker: RolloutWorker,
    **kwargs: Dict[str, Any]
    ) -> PolicyID:
    """
    Selects a policy ID based on the current occupancy state.

    Args:
        agent_id (ray.rllib.utils.typing.AgentID): The ID of the agent (e.g., 'thermostat').
        episode (ray.rllib.evaluation.episode_v2.EpisodeV2): Current episode.
        worker (ray.rllib.evaluation.rollout_worker.RolloutWorker): The worker object.
        **kwargs: Additional keyword arguments.

    Returns:
        str: The ID of the policy to use ('energy_saving' or 'comfort_improving').
    """
    # Get the latest observation for the agent. Here is where we do not know how to read the latest obs.
    agents = episode.get_agents()
    
    if agent_id not in agents:
        print(f"Agent {agent_id} not found in episode.")
        return "comfort_improving"
    
    
    
    if latest_obs is None:
        print(f"latest_obs should not be None during policy mapping.")
        return "comfort_improving"
    
    else:
        print(f"latest_obs: {latest_obs}")
        occupancy_state = latest_obs[OCCUPANCY_INDEX]
        
        assert type(occupancy_state) in [int, float], "Occupancy state should be a numeric value."
        
        if occupancy_state > 0:
            # People are present (occupancy > 0) -> improve comfort
            return "comfort_improving"
        else:
            # No people present (occupancy == 0) -> save energy
            return "energy_saving"

