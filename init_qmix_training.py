"""# DEEP REINFORCEMENT LEARNING FOR ENERGYPLUS MODELS

This script is used to execute an experiment of DRL using RLlib library and EnergyPlus Python API
to research in the field of building automation and the optimal use of bioclimatic strategies in
dwellings.

## How to use

This file create the algorithm for DRL in RLlib and the execute the experiment with Tune. Also, here is 
called the environment. The custom environment is builded in two differents scripts. One has the
RLlib requieriment to standarize an environment, based on the Farama Foundation Gymnasium format
and called EnergyPlus Gym Environment. The second script, called EnergyPlus Runner, is the engine 
of EnergyPlus Python API programmed to run as a Markov Dessicion Proccess.
"""
"""## DEFINE ENVIRONMENT VARIABLES

This is not always required, but here is an example of how to define 
a environmet variable: (NOTE: This must be implemented before import ray.)

>>> import os
>>> os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
"""
import os
os.environ['RAY_DEDUP_LOGS'] = '0'

"""## IMPORT THE NECESARY LIBRARIES
"""
from tempfile import TemporaryDirectory
# This library generate a tmp folder to save the the EnergyPlus output
import gymnasium as gym
# Used to configurate the action and observation spaces
import ray
# To init ray
from ray import air, tune
# To configurate the execution of the experiment
from ray.tune import register_env
# To register the custom environment. RLlib is not compatible with conventional Gym register of 
# custom environments.
from qmix.qmix import QMixConfig
# To config the QMix algorithm.
from ray.tune.schedulers import ASHAScheduler
# Early stop to tune the hyperparameters
from ray.tune.search.bayesopt import BayesOptSearch
# Search algorithm to tune the hyperparameters
from ray.tune.search import Repeater
# Tool to evaluate multiples seeds in a configuration of hyperparameters
from env.qmix_ep_gym_env import EnergyPlusEnv_v0
# The EnergyPlus Environment configuration. There is defined the reward function 
# and also is define the flux of execution of the MDP.
# TODO: Make a singular configuration for all the cases that I would to analise.

"""## DEFINE THE EXPERIMENT CONTROLS
"""
tune_runner  = False
# Define if the experiment tuning the variables or execute a unique configuration.
restore = False
# To define if is necesary to restore or not a previous experiment. Is necesary to stablish a 'restore_path'.
restore_path = ''
# Path to the folder where the experiment is located.
env_config={ 
    'weather_folder': 'C:/Users/grhen/Documents/GitHub/natural_ventilation_EP_RLlib/epw/GEF',
    'output': TemporaryDirectory("output","DQN_",'C:/Users/grhen/Documents/Resultados_RLforEP').name,
    'epjson_folderpath': 'C:/Users/grhen/Documents/GitHub/natural_ventilation_EP_RLlib/epjson',
    'epjson_output_folder': 'C:/Users/grhen/Documents/models',
    # Configure the directories for the experiment.
    'ep_terminal_output': False,
    # For dubugging is better to print in the terminal the outputs of the EnergyPlus simulation process.
    'is_test': False,
    # For evaluation process 'is_test=True' and for trainig False.
    'test_init_day': 1,
    'action_space': gym.spaces.Discrete(4),
    # action space for simple agent case
    'observation_space': gym.spaces.Box(float("-inf"), float("inf"), (303,)),
    # observation space for simple agent case
    
    # BUILDING CONFIGURATION
    'building_name': 'prot_1(natural)',
}

"""## INIT RAY AND REGISTER THE ENVIRONMENT
"""
ray.init()
# Inicialiced Ray Server
register_env(name="EPEnv", env_creator=lambda args: EnergyPlusEnv_v0(args))
# Register the environment.

"""## CONFIGURATION OF THE SELECTED ALGORITHM

Different algorithms are configurated in the following lines. It is possible to add other
algorithm configuration here or modify the presents.
"""

algo = QMixConfig().training(
    # General Algo Configs
    gamma = 0.99 if not tune_runner else tune.choice([0.7, 0.9, 0.99]),
    lr = 0.01 if not tune_runner else tune.choice([0.001, 0.01, 0.1]),
    grad_clip = 40,
    grad_clip_by = 'global_norm',
    train_batch_size = 96 if not tune_runner else tune.choice([8, 64, 128, 256]),
    model = {
        "fcnet_hiddens": [512,512,512],
        "fcnet_activation": "relu", #if not tune_runner else tune.choice(['tanh', 'relu', 'swish', 'linear']),
        },
    optimizer = {},
    
    #QMix Configs
    mixer = 'qmix',
    mixing_embed_dim = 512,
    double_q = True,
    target_network_update_freq = 4800,
    replay_buffer_config = {
        '_enable_replay_buffer_api': True,
        'type': 'MultiAgentPrioritizedReplayBuffer',
        'capacity': 500000,
        'prioritized_replay_alpha': 0.7,
        'prioritized_replay_beta': 0.6,
        'prioritized_replay_eps': 1e-6,
        'replay_sequence_length': 1,
    },
    optim_alpha = 0.99,
    optim_eps = 0.00001,
    grad_clip = 10,
).environment(
    env="EPEnv",
    env_config=env_config,
).framework(
    framework = 'torch',
).fault_tolerance(
    recreate_failed_workers = True,
    restart_failed_sub_environments=False,
).rollouts(
    num_rollout_workers = 7,
    create_env_on_local_worker=True,
    rollout_fragment_length = 'auto',
    enable_connectors = True,
    num_envs_per_worker=1,
).experimental(
    _enable_new_api_stack = False,
).reporting( # multi_agent config va aquí
    min_sample_timesteps_per_iteration = 1000,
).checkpointing(
    export_native_model_files = True,
).debugging(
    log_level = "ERROR",
    #seed=7,
).resources(
    num_gpus = 0,
)
algo.exploration(
    exploration_config={
        "type": "EpsilonGreedy",
        "initial_epsilon": 1.,
        "final_epsilon": 0.,
        "epsilon_timesteps": 24*365*200,
    }
)

"""## START EXPERIMENT
"""
def trial_str_creator(trial):
    """This method create a description for the folder where the outputs and checkpoints 
    will be save.

    Args:
        trial: A trial type of RLlib.

    Returns:
        str: Return a unique string for the folder of the trial.
    """
    return "3x512_dueT1x512_douT_{}_{}".format(trial.trainable_name, trial.trial_id)

if not restore:
    tune.Tuner(
        algorithm,
        tune_config=tune.TuneConfig(
            mode="max",
            metric="episode_reward_mean",
            #num_samples=1000,
            # This is necesary to iterative execute the search_alg to improve the hyperparameters
            reuse_actors=False,
            trial_name_creator=trial_str_creator,
            trial_dirname_creator=trial_str_creator,
            
            #search_alg = Repeater(BayesOptSearch(),repeat=10),
            #search_alg = BayesOptSearch(),
            # Search algorithm
            
            #scheduler = ASHAScheduler(time_attr = 'timesteps_total', max_t=6*24*365*30, grace_period=6*24*365*10),
            # Scheduler algorithm
            
        ),
        run_config=air.RunConfig(
            name='20240306_VN_prot_1_natural_'+str(algorithm),
            stop={"episodes_total": 1000},
            log_to_file=True,
            
            checkpoint_config=air.CheckpointConfig(
                checkpoint_at_end = True,
                checkpoint_frequency = 40,
                #num_to_keep = 20
            ),
            failure_config=air.FailureConfig(
                max_failures=100
                # Tries to recover a run up to this many times.
            ),
        ),
        param_space=algo.to_dict(),
    ).fit()

else:
    tune.Tuner.restore(
        path=restore_path,
        trainable = algorithm,
        resume_errored=True
    )

"""## END EXPERIMENT AND SHUTDOWN RAY SERVE
"""
ray.shutdown()