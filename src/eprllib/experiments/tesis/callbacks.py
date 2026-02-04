from typing import List
from ray.rllib.callbacks.callbacks import RLlibCallback
from ray.rllib.utils.annotations import override
import numpy as np



class ActionDistributionCallback(RLlibCallback):
    def __init__(self):
        super().__init__()
        
        self.action_taken: List[float] = []
    @override
    def on_episode_end(self, *, episode, prev_episode_chunks, env_runner, metrics_logger, env, env_index, rl_module, **kwargs):
        self.action_taken.append(episode.get_actions())
        
    @override    
    def on_train_result(self, *, algorithm, metrics_logger, result, **kwargs):
        # Compute the action distribution of self.action_taken
        action_dist = {}
        for actions in self.action_taken:
            for agent_id, agent_actions in actions.items():
                if agent_id not in action_dist:
                    action_dist[agent_id] = {}
                for action_value in agent_actions:
                    # Assuming action_value is a list/array, take the first element
                    # Adjust if your action space is different
                    key = round(action_value[0], 2) if isinstance(action_value, (list, tuple, np.ndarray)) else round(action_value, 2)
                    action_dist[agent_id][key] = action_dist[agent_id].get(key, 0) + 1
        self.action_taken = [] # Clear for next iteration
        
        # Print or store the action distribution
        print("Action Distribution: ", action_dist)
