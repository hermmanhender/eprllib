"""
"""
import os
os.environ['RAY_DEDUP_LOGS'] = '0'

import ray
from ray.tune import register_env
from ray.rllib.algorithms.ppo.ppo import PPOConfig
from ray.rllib.policy.policy import PolicySpec
from eprllib.Env.MultiAgent.EnergyPlusEnvironment import EnergyPlusEnv_v0
from datetime import datetime
from tempfile import TemporaryDirectory
from eprllib.RewardFunctions.RewardFunctions import RewardFunction
from eprllib.ActionFunctions.ActionFunctions import ActionFunction
from eprllib.Env.EnvConfig import EnvConfig, env_config_to_dict
from typing import Dict, Any

name = 'NaturalVentilationExample'

class reward_function(RewardFunction):
    def __inti__(self, reward_fn_config):
        super().__init__(reward_fn_config)
        self.max_energy = 0
        self.max_temperature = 0
    
    def calculate_reward(
        self,
        infos: Dict[str,Dict[str,Any]]
        ) -> Dict[str,float]:
        """
        This method must be implemented in the subclass.

        Args:
            infos (Dict[str,Dict[str,Any]]): The infos dictionary containing the necessary information for calculating the reward.

        Returns:
            Dict[str,float]: The calculated reward as a dictionary with the keys 'agent'.
        """
        agents = {agent for agent in self.reward_fn_config.keys()}
        _agent_rewards = {}
        
        for agent in agents:
            if (infos[agent]["Heating:DistrictHeatingWater"] + infos[agent]["Cooling:DistrictCooling"]) < self.max_energy:
                self.max_energy = (infos[agent]["Heating:DistrictHeatingWater"] + infos[agent]["Cooling:DistrictCooling"])
            if (infos[agent]["Zone Mean Air Temperature"]-22)**2 < self.max_temperature:
                self.max_temperature = (infos[agent]["Zone Mean Air Temperature"]-22)**2
        
        for agent in agents:    
            _agent_rewards[agent] = -0.5*(infos[agent]["Heating:DistrictHeatingWater"] + infos[agent]["Cooling:DistrictCooling"])/self.max_energy - 0.5*(infos[agent]["Zone People Occupant Count"])*((infos[agent]["Zone Mean Air Temperature"]-22)**2)

        return _agent_rewards

class window_opening_action_transformer(ActionFunction):
    def __init__(self, action_fn_config:Dict[str,Any]):
        super().__init__(action_fn_config)
    
    def transform_action(self, action:Dict[str,int]) -> Dict[str, float|int]:
        # create a dict to save the action per agent.
        action_transformed = {agent: 0. for agent in action.keys()}
        for agent in action_transformed.keys():
                action_transformed[agent] = action[agent]/10
        
        return action_transformed

GeneralModel = EnvConfig()
GeneralModel.generals(
    epjson_path = 'C:/Users/grhen/Documents/GitHub/natural_ventilation_EP_RLlib/src/eprllib/Examples/NaturalVentilation/natural_ventilation.idf',
    epw_path = "C:/Users/grhen/Documents/GitHub/natural_ventilation_EP_RLlib/src/eprllib/Examples/NaturalVentilation/GEF_Lujan_de_cuyo-hour-H4.epw",
    output_path = TemporaryDirectory("output","",'C:/Users/grhen/Documents/Resultados_RLforEP').name,
    ep_terminal_output = True,
    timeout = 10,
)
GeneralModel.agents(
    agents_config = {
        "window_north": {
            'ep_actuator_config': ("AirFlow Network Window/Door Opening", "Venting Opening Factor", "window_north"),
            'thermal_zone': 'Thermal Zone: Space 1',
            'agent_indicator': 1,
        },
        "window_south": {
            'ep_actuator_config': ("AirFlow Network Window/Door Opening", "Venting Opening Factor", "window_south"),
            'thermal_zone': 'Thermal Zone: Space 1',
            'agent_indicator': 2,
        },
    }
)
GeneralModel.observations(
    use_agent_indicator=True,
    ep_environment_variables = [
        "Site Outdoor Air Drybulb Temperature",
        "Site Wind Speed",
        "Site Outdoor Air Relative Humidity",
    ], 
    ep_thermal_zones_variables = [
        "Zone Mean Air Temperature",
        "Zone Air Relative Humidity",
        "Zone People Occupant Count",
    ],
    ep_meters = [
        "Electricity:Building",
        "Heating:DistrictHeatingWater",
        "Cooling:DistrictCooling",
    ],
    time_variables = [
        'hour',
        'day_of_year',
        'day_of_week',
        ],
    weather_variables = [
        "today_weather_horizontal_ir_at_time",
        ],
    infos_variables = {
        'Thermal Zone: Space 1':[
            'Heating:DistrictHeatingWater',
            'Cooling:DistrictCooling',
            'Zone People Occupant Count',
            "Zone Mean Air Temperature",
        ],
    },
)
GeneralModel.rewards(
    reward_fn = reward_function({}),
)
GeneralModel.actions(
    window_opening_action_transformer({})
)

ray.init(_temp_dir='C:/Users/grhen/ray_results/tmp')
register_env(name="EPEnv", env_creator=lambda args: EnergyPlusEnv_v0(args))

algo = PPOConfig()
algo.training(
    # General Algo Configs
    gamma = 0.8,
    lr = 0.0001,
    train_batch_size = 1000,
    model = {
        "fcnet_hiddens": [64,64],
        "fcnet_activation": "relu",
        },
    # PPO Configs
    lr_schedule = [[0, 0.3], [52561*8*1*2, 0.001], [52561*8*2*2, 0.0001]],
    use_critic = True,
    use_gae = True,
    lambda_ = 0.2,
    use_kl_loss = True,
    kl_coeff = 9,
    kl_target = 0.05,
    sgd_minibatch_size = 100,
    num_sgd_iter = 20,
    shuffle_sequences = True,
    vf_loss_coeff = 0.4,
    entropy_coeff = 10,
    entropy_coeff_schedule = [[0, 15], [52561*8*1*2, 5], [52561*8*2*2, 1]],
    clip_param = 0.1,
    vf_clip_param = 10,
)
algo.environment(
    env = "EPEnv",
    env_config = env_config_to_dict(GeneralModel),
)
algo.framework(
    framework = 'torch',
)
algo.fault_tolerance(
    recreate_failed_env_runners = True,
)
algo.env_runners(
    num_env_runners = 7,
    create_env_on_local_worker = True,
    rollout_fragment_length = 'auto',
    enable_connectors = True,
    num_envs_per_env_runner = 1,
    explore = True,
    exploration_config = {
        "type": "EpsilonGreedy",
        "initial_epsilon": 1.,
        "final_epsilon": 0.,
        "epsilon_timesteps": 52561*8*1,
    },
)
algo.evaluation(
    evaluation_interval = 100,
    evaluation_duration = 1,
    evaluation_duration_unit = 'episodes',
    evaluation_config = {
        "explore": False,
    },
    evaluation_num_env_runners = 0,
)
algo.multi_agent(
    policies = {
        'shared_policy': PolicySpec(),
    },
    policy_mapping_fn = lambda agent_id, episode, worker, **kwargs: "shared_policy",
)
algo.reporting(
    min_sample_timesteps_per_iteration = 100,
)
algo.checkpointing(
    export_native_model_files = True,
)
algo.debugging(
    log_level = "ERROR",
    # seed=0,
)
algo.resources(
    num_gpus = 0,
)

my_new_ppo = algo.build()
print(f"The algorithm was build with the following config:\n{my_new_ppo}")

now = datetime.now()
folder_name = now.strftime("%Y-%m-%d_%H-%M-%S")
new_dir_path = f"C:/Users/grhen/ray_results/{folder_name}_{name}"
try:
    os.makedirs(new_dir_path)
except OSError:
    print(f"Creation of directory {new_dir_path} failed")
else:
    print(f"Successfully created directory at {new_dir_path}")

for iteration in range(1000):
    print(f"\n\n********** Iteration: {iteration} **********")
    results = my_new_ppo.train()
    print(f"The results for the iteration {iteration} are:\nInfo:\n{results['info']}\nEnv Runners:\n{results['env_runners']}")
    if iteration % 10 == 0:
        save_result = my_new_ppo.save(f"{new_dir_path}/checkpoint-{iteration}")
        path_to_checkpoint = save_result.checkpoint.path
        print(
            "An Algorithm checkpoint has been created inside directory: "
            f"'{path_to_checkpoint}'."
        )

save_result = my_new_ppo.save(f"{new_dir_path}/checkpoint-final")
path_to_checkpoint = save_result.checkpoint.path
print("An Algorithm checkpoint has been created inside directory: "
    f"'{path_to_checkpoint}'.")
my_new_ppo.stop()
ray.shutdown()
print("Ray has been shut down.")
print("The training is finished.")
