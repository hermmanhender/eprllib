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
import time
import sys
sys.path.append('C:/Users/grhen/Documents/GitHub/natural_ventilation_EP_RLlib/src')
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
from ray.rllib.algorithms.ppo.ppo import PPOConfig
# To config the PPO algorithm.
from ray.rllib.algorithms.dqn.dqn import DQNConfig
# To config the DQN algorithm.
from ray.rllib.algorithms.sac.sac import SACConfig
# To config the SAC algorithm.
from ray.tune.schedulers import ASHAScheduler
# Early stop to tune the hyperparameters
from ray.tune.search.bayesopt import BayesOptSearch
# Search algorithm to tune the hyperparameters
from ray.tune.search import Repeater
# Tool to evaluate multiples seeds in a configuration of hyperparameters
from ray.rllib.policy.policy import PolicySpec
from eprllib.env.multiagent.marl_ep_gym_env import EnergyPlusEnv_v0
from eprllib.tools import rewards
# The EnergyPlus Environment configuration. There is defined the reward function 
# and also is define the flux of execution of the MDP.
# TODO: Make a singular configuration for all the cases that I would to analise.

"""## DEFINE THE EXPERIMENT CONTROLS
"""
algorithm = 'DQN'
# Define the algorithm to use to train the policy. Options are: PPO, SAC, DQN.
tune_runner  = False
# Define if the experiment tuning the variables or execute a unique configuration.
restore = False
# To define if is necesary to restore or not a previous experiment. Is necesary to stablish a 'restore_path'.
restore_path = ''
# Path to the folder where the experiment is located.
env_config={
    # === ENERGYPLUS OPTIONS === #
    'epjson': "path_to_epjson_file",
    "epw_training": "path_to_epw_training_file",
    "epw": "path_to_epw_evaluation_file",
    # Configure the output directory for the EnergyPlus simulation.
    'output': TemporaryDirectory("output","DQN_",'C:/Users/grhen/Documents/Resultados_RLforEP').name,
    # For dubugging is better to print in the terminal the outputs of the EnergyPlus simulation process.
    'ep_terminal_output': False,
    
    # === EXPERIMENT OPTIONS === #
    # For evaluation process 'is_test=True' and for trainig False.
    'is_test': False,
    
    # === ENVIRONMENT OPTIONS === #
    # action space for simple agent case
    'action_space': gym.spaces.Discrete(2),
    # observation space for simple agent case
    # This is equal to the the sume of:
    #   + ep_variables
    #   + ep_meters
    #   + ep_actuators
    #   + weather_prob_days*144
    #   - no_observable_variables
    #   + 1 (agent_indicator)
    #   + 6 ('day_of_the_week','is_raining','sun_is_up','hora','simulation_day','rad')
    'observation_space': gym.spaces.Box(float("-inf"), float("inf"), (307,)),
    'reward_function': rewards.reward_function_T3,
    "ep_variables":{
        "To": ("Site Outdoor Air Drybulb Temperature", "Environment"),
        "Ti": ("Zone Mean Air Temperature", "Thermal Zone: Living"),
        "v": ("Site Wind Speed", "Environment"),
        "d": ("Site Wind Direction", "Environment"),
        "RHo": ("Site Outdoor Air Relative Humidity", "Environment"),
        "RHi": ("Zone Air Relative Humidity", "Thermal Zone: Living"),
        "pres": ("Site Outdoor Air Barometric Pressure", "Environment"),
        "occupancy": ("Zone People Occupant Count", "Thermal Zone: Living"),
        "ppd": ("Zone Thermal Comfort Fanger Model PPD", "Living Occupancy")
    },
    "ep_meters": {
        "electricity": "Electricity:Zone:THERMAL ZONE: LIVING",
        "gas": "NaturalGas:Zone:THERMAL ZONE: LIVING",
    },
    "ep_actuators": {
        "opening_window_1": ("AirFlow Network Window/Door Opening", "Venting Opening Factor", "living_NW_window"),
        "opening_window_2": ("AirFlow Network Window/Door Opening", "Venting Opening Factor", "living_E_window"),
    },
    "infos_variables": ["ppd", "occupancy", "Ti"],
    "no_observable_variables": ["ppd"],
    
    # === OPTIONAL === #
    "timeout": 5,
    "T_confort": 23.5,
    "weather_prob_days": 2
}
env_config["agent_ids"] = [key for key in env_config["ep_actuators"].keys()]

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

def policy_mapping_fn(agent_id, episode, worker, **kwargs):
    return "shared_policy"


if algorithm == 'PPO': # PPO Configuration
    algo = PPOConfig().training(
            # General Algo Configs
            gamma=0.72 if not tune_runner else tune.choice(0.7, 0.9, 0.99),
            # Float specifying the discount factor of the Markov Decision process.
            lr=0.04 if not tune_runner else tune.choice(0.001, 0.01, 0.1),
            # The learning rate (float) or learning rate schedule
            #model=,
            # Arguments passed into the policy model. See models/catalog.py for a full list of the 
            # available model options.
            train_batch_size=128 if not tune_runner else tune.choice([8, 64, 128, 256]),
            # PPO Configs
            lr_schedule=None, # List[List[int | float]] | None = NotProvided,
            # Learning rate schedule. In the format of [[timestep, lr-value], [timestep, lr-value], …] 
            # Intermediary timesteps will be assigned to interpolated learning rate values. A schedule 
            # should normally start from timestep 0.
            use_critic=True, # bool | None = NotProvided,
            # Should use a critic as a baseline (otherwise don’t use value baseline; required for using GAE).
            use_gae=True, # bool | None = NotProvided,
            # If true, use the Generalized Advantage Estimator (GAE) with a value function, 
            # see https://arxiv.org/pdf/1506.02438.pdf.
            lambda_=0.20216 if not tune_runner else tune.choice(0, 0.2, 0.5, 0.7, 1.0), # float | None = NotProvided,
            # The GAE (lambda) parameter.  The generalized advantage estimator for 0 < λ < 1 makes a 
            # compromise between bias and variance, controlled by parameter λ.
            use_kl_loss=True, # bool | None = NotProvided,
            # Whether to use the KL-term in the loss function.
            kl_coeff=9.9712 if not tune_runner else tune.uniform(0.3, 10.0), # float | None = NotProvided,
            # Initial coefficient for KL divergence.
            kl_target=0.054921 if not tune_runner else tune.uniform(0.001, 0.1), # float | None = NotProvided,
            # Target value for KL divergence.
            sgd_minibatch_size=48,# if not tune_runner else tune.choice([48, 128]), # int | None = NotProvided,
            # Total SGD batch size across all devices for SGD. This defines the minibatch size 
            # within each epoch.
            num_sgd_iter=6,# if not tune_runner else tune.randint(30, 60), # int | None = NotProvided,
            # Number of SGD iterations in each outer loop (i.e., number of epochs to execute per train batch).
            shuffle_sequences=True, # bool | None = NotProvided,
            # Whether to shuffle sequences in the batch when training (recommended).
            vf_loss_coeff=0.38584 if not tune_runner else tune.uniform(0.1, 1.0), # Tune this! float | None = NotProvided,
            # Coefficient of the value function loss. IMPORTANT: you must tune this if you set 
            # vf_share_layers=True inside your model’s config.
            entropy_coeff=10.319 if not tune_runner else tune.uniform(0.95, 15.0), # float | None = NotProvided,
            # Coefficient of the entropy regularizer.
            entropy_coeff_schedule=None, # List[List[int | float]] | None = NotProvided,
            # Decay schedule for the entropy regularizer.
            clip_param=0.22107 if not tune_runner else tune.uniform(0.1, 0.4), # float | None = NotProvided,
            # The PPO clip parameter.
            vf_clip_param=39.327 if not tune_runner else tune.uniform(0, 50), # float | None = NotProvided,
            # Clip param for the value function. Note that this is sensitive to the scale of the 
            # rewards. If your expected V is large, increase this.
            grad_clip=None, # float | None = NotProvided,
            # If specified, clip the global norm of gradients by this amount.
        ).environment(
            env="EPEnv",
            observation_space=gym.spaces.Box(float("-inf"), float("inf"), (49,)),
            action_space=gym.spaces.Discrete(4),
            env_config=env_config,
        ).framework(
            framework = 'torch',
        ).fault_tolerance(
            recreate_failed_workers = True,
            restart_failed_sub_environments=False,
        ).rollouts(
            num_rollout_workers = 1,# if not tune_runner else tune.grid_search([0, 1, 3]),
            create_env_on_local_worker=True,
            rollout_fragment_length = 'auto',
            enable_connectors = True,
            #batch_mode="truncate_episodes",
            num_envs_per_worker=1,
        ).experimental(
            _enable_new_api_stack = True,
        ).reporting( # multi_agent config va aquí
            min_sample_timesteps_per_iteration = 2000,
        ).checkpointing(
            export_native_model_files = True,
        ).debugging(
            log_level = "ERROR",
            #seed=7,# if not tune_runner else tune.grid_search([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
        ).resources(
            num_gpus = 0,
        )

elif algorithm == 'DQN': # DQN Configuration
    algo = DQNConfig().training(
        # General Algo Configs
        gamma = 0.99 if not tune_runner else tune.choice([0.7, 0.9, 0.99]),
        lr = 0.01 if not tune_runner else tune.choice([0.001, 0.01, 0.1]),
        grad_clip = 40,
        grad_clip_by = 'global_norm',
        train_batch_size = 256 if not tune_runner else tune.choice([8, 64, 128, 256]),
        model = {
            "fcnet_hiddens": [256,256,256],
            "fcnet_activation": "relu", #if not tune_runner else tune.choice(['tanh', 'relu', 'swish', 'linear']),
            
            # == LSTM ==
            # Whether to wrap the model with an LSTM.
            # Este parámetro booleano determina si se utilizará una LSTM en el modelo. Si se 
            # establece en True, la red neuronal del modelo se envolverá con una capa LSTM. Si 
            # se establece en False, se utilizará una arquitectura de red neuronal estándar sin LSTM.
            "use_lstm": False,
            # Max seq len for training the LSTM, defaults to 20.
            # Este parámetro define la longitud máxima de la secuencia de entrada para entrenar 
            # la LSTM. En otras palabras, especifica el número máximo de pasos de tiempo pasados 
            # que la LSTM recordará durante el entrenamiento. Si el entorno genera secuencias 
            # más largas que este valor, se truncarán.
            "max_seq_len": 20,
            # Size of the LSTM cell.
            # Este parámetro especifica el tamaño de la celda LSTM, es decir, el número de 
            # unidades o neuronas en la capa LSTM. Un tamaño de celda más grande puede permitir 
            # que la LSTM capture patrones más complejos en los d5atos, pero también aumentará 
            # la complejidad computacional del modelo.
            "lstm_cell_size": 256,
            # Whether to feed a_{t-1} to LSTM (one-hot encoded if discrete).
            # Este parámetro determina si la acción anterior, a_{t-1}, se debe alimentar a la 
            # LSTM como una entrada adicional. Esto puede ser útil en entornos donde la acción 
            # anterior es relevante para la toma de decisiones actual.
            "lstm_use_prev_action": False,
            # Whether to feed r_{t-1} to LSTM.
            # Este parámetro especifica si la recompensa anterior, r_{t-1}, se debe alimentar a 
            # la LSTM como una entrada adicional. Al igual que con la acción anterior, esto puede 
            # ser útil en entornos donde la recompensa anterior afecta la toma de decisiones actual.
            "lstm_use_prev_reward": False,
            # Whether the LSTM is time-major (TxBx..) or batch-major (BxTx..).
            # Este parámetro determina si la estructura de datos de entrada para la 
            # LSTM es de tipo "time-major" (tiempo x lote x ...) o "batch-major" 
            # (lote x tiempo x ...). Por lo general, no es necesario cambiar este parámetro, 
            # a menos que se tenga experiencia específica con la manipulación de datos LSTM.
            "_time_major": False,
            
            },
        optimizer = {},
        # DQN Configs
        num_atoms = 100,
        v_min = -343,
        v_max = 0,
        noisy = True,
        sigma0 = 0.7 if not tune_runner else tune.choice([0., 0.2, 0.5, 0.7, 1.]),
        dueling = True,
        hiddens = [256],
        double_q = True,
        n_step = 12,
        replay_buffer_config = {
            '_enable_replay_buffer_api': True,
            'type': 'MultiAgentPrioritizedReplayBuffer',
            'capacity': 5000000,
            'prioritized_replay_alpha': 0.7,
            'prioritized_replay_beta': 0.6,
            'prioritized_replay_eps': 1e-6,
            'replay_sequence_length': 1,
            },
        categorical_distribution_temperature = 0.5 if not tune_runner else tune.choice([0.2, 0.5, 0.7, 1.]),
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
    ).multi_agent(
        policies = {
            'shared_policy': PolicySpec(),
        },
        policy_mapping_fn = policy_mapping_fn,
    ).reporting(
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
            "epsilon_timesteps": 6*24*365*100,
        }
    )

elif algorithm == 'SAC': # SAC Configuration
    algo = SACConfig().training(
            # General Algo Configs
            gamma = 0.99 if not tune_runner else tune.uniform(0.7, 0.99),
            # Float specifying the discount factor of the Markov Decision process.
            lr = 0.1 if not tune_runner else tune.uniform(0.001, 0.1),
            # The learning rate (float) or learning rate schedule
            #grad_clip = None, #float
            # If None, no gradient clipping will be applied. Otherwise, depending on the setting of grad_clip_by, the (float) 
            # value of grad_clip will have the following effect: If grad_clip_by=value: Will clip all computed gradients 
            # individually inside the interval [-grad_clip, +`grad_clip`]. If grad_clip_by=norm, will compute the L2-norm of 
            # each weight/bias gradient tensor individually and then clip all gradients such that these L2-norms do not exceed 
            # grad_clip. The L2-norm of a tensor is computed via: sqrt(SUM(w0^2, w1^2, ..., wn^2)) where w[i] are the elements 
            # of the tensor (no matter what the shape of this tensor is). If grad_clip_by=global_norm, will compute the square 
            # of the L2-norm of each weight/bias gradient tensor individually, sum up all these squared L2-norms across all 
            # given gradient tensors (e.g. the entire module to be updated), square root that overall sum, and then clip all 
            # gradients such that this global L2-norm does not exceed the given value. The global L2-norm over a list of tensors 
            # (e.g. W and V) is computed via: sqrt[SUM(w0^2, w1^2, ..., wn^2) + SUM(v0^2, v1^2, ..., vm^2)], where w[i] and v[j] 
            # are the elements of the tensors W and V (no matter what the shapes of these tensors are).
            #grad_clip_by = 'global_norm', #str
            # See grad_clip for the effect of this setting on gradient clipping. Allowed values are value, norm, and global_norm.
            #train_batch_size = 128, # if not tune_runner else tune.randint(128, 257),
            #  Training batch size, if applicable.
            model = {
                "fcnet_hiddens": [256],
                "fcnet_activation": "relu",
                },
            # Arguments passed into the policy model. See models/catalog.py for a full list of the 
            # available model options. TODO: Provide ModelConfig objects instead of dicts
            #optimizer = None, #dict
            # Arguments to pass to the policy optimizer. This setting is not used when _enable_new_api_stack=True.
            #max_requests_in_flight_per_sampler_worker = None, #int
            # Max number of inflight requests to each sampling worker. See the FaultTolerantActorManager class for more details. 
            # Tuning these values is important when running experimens with large sample batches, where there is the risk that 
            # the object store may fill up, causing spilling of objects to disk. This can cause any asynchronous requests to 
            # become very slow, making your experiment run slow as well. You can inspect the object store during your experiment 
            # via a call to ray memory on your headnode, and by using the ray dashboard. If you’re seeing that the object store 
            # is filling up, turn down the number of remote requests in flight, or enable compression in your experiment of 
            # timesteps.
            #learner_class = None,
            # The Learner class to use for (distributed) updating of the RLModule. Only used when _enable_new_api_stack=True.
            
            # SAC Configs
            twin_q = True, #bool
            # Use two Q-networks (instead of one) for action-value estimation. Note: Each Q-network will have its own target network.
            #q_model_config = #~typing.Dict[str, ~typing.Any]
            # Model configs for the Q network(s). These will override MODEL_DEFAULTS. This is treated just as the top-level model 
            # dict in setting up the Q-network(s) (2 if twin_q=True). That means, you can do for different observation spaces: 
            # obs=Box(1D) -> Tuple(Box(1D) + Action) -> concat -> post_fcnet obs=Box(3D) -> Tuple(Box(3D) + Action) -> 
            # vision-net -> concat w/ action -> post_fcnet obs=Tuple(Box(1D), Box(3D)) -> Tuple(Box(1D), Box(3D), Action) -> 
            # vision-net -> concat w/ Box(1D) and action -> post_fcnet You can also have SAC use your custom_model as Q-model(s), 
            # by simply specifying the custom_model sub-key in below dict (just like you would do in the top-level model dict.
            #policy_model_config = #~typing.Dict[str, ~typing.Any]
            # Model options for the policy function (see q_model_config above for details). The difference to q_model_config above 
            # is that no action concat’ing is performed before the post_fcnet stack.
            tau = 1.0, #float
            # Update the target by au * policy + (1- au) * target_policy.
            initial_alpha = 0.5, #float
            # Initial value to use for the entropy weight alpha.
            target_entropy = 'auto', #str | float
            # Target entropy lower bound. If “auto”, will be set to -|A| (e.g. -2.0 for Discrete(2), -3.0 for Box(shape=(3,))). This 
            # is the inverse of reward scale, and will be optimized automatically.
            n_step = 10, # if not tune_runner else tune.randint(1, 11), #int
            # N-step target updates. If >1, sars’ tuples in trajectories will be postprocessed to become
            # sa[discounted sum of R][s t+n] tuples.
            store_buffer_in_checkpoints = True, #bool
            # Set this to True, if you want the contents of your buffer(s) to be stored in any saved checkpoints as well. Warnings 
            # will be created if: - This is True AND restoring from a checkpoint that contains no buffer data. - This is 
            # False AND restoring from a checkpoint that does contain buffer data.
            replay_buffer_config = {
                '_enable_replay_buffer_api': True,
                'type': 'MultiAgentPrioritizedReplayBuffer',
                'capacity': 50000,
                'prioritized_replay_alpha': 0.6,
                'prioritized_replay_beta': 0.4,
                'prioritized_replay_eps': 1e-6,
                'replay_sequence_length': 1,
                },
            # Replay buffer config. Examples: { “_enable_replay_buffer_api”: True, “type”: “MultiAgentReplayBuffer”, 
            # “capacity”: 50000, “replay_batch_size”: 32, “replay_sequence_length”: 1, } - OR - { “_enable_replay_buffer_api”: True, 
            # “type”: “MultiAgentPrioritizedReplayBuffer”, “capacity”: 50000, “prioritized_replay_alpha”: 0.6, 
            # “prioritized_replay_beta”: 0.4, “prioritized_replay_eps”: 1e-6, “replay_sequence_length”: 1, } - Where - 
            # prioritized_replay_alpha: Alpha parameter controls the degree of prioritization in the buffer. In other words, when 
            # a buffer sample has a higher temporal-difference error, with how much more probability should it drawn to use 
            # to update the parametrized Q-network. 0.0 corresponds to uniform probability. Setting much above 1.0 may quickly 
            # result as the sampling distribution could become heavily “pointy” with low entropy. prioritized_replay_beta: Beta 
            # parameter controls the degree of importance sampling which suppresses the influence of gradient updates from 
            # samples that have higher probability of being sampled via alpha parameter and the temporal-difference error. 
            # prioritized_replay_eps: Epsilon parameter sets the baseline probability for sampling so that when the 
            # temporal-difference error of a sample is zero, there is still a chance of drawing the sample.
            #training_intensity = #float
            # The intensity with which to update the model (vs collecting samples from the env). If None, uses “natural” values 
            # of: train_batch_size / (rollout_fragment_length x num_workers x num_envs_per_worker). If not None, will make sure 
            # that the ratio between timesteps inserted into and sampled from th buffer matches the given values. Example: 
            # training_intensity=1000.0 train_batch_size=250 rollout_fragment_length=1 num_workers=1 (or 0) num_envs_per_worker=1 -> 
            # natural value = 250 / 1 = 250.0 -> will make sure that replay+train op will be executed 4x asoften as rollout+insert 
            # op (4 * 250 = 1000). See: rllib/algorithms/dqn/dqn.py::calculate_rr_weights for further details.
            clip_actions = True, #bool
            # Whether to clip actions. If actions are already normalized, this should be set to False.
            #grad_clip = #float
            # If not None, clip gradients during optimization at this value.
            optimization_config = { #~typing.Dict[str, ~typing.Any]
                'actor_learning_rate': 0.005,
                'critic_learning_rate': 0.005,
                'entropy_learning_rate': 0.0001,
            },
            # Config dict for optimization. Set the supported keys actor_learning_rate, critic_learning_rate, and 
            # entropy_learning_rate in here.
            target_network_update_freq = 144, #int
            # Update the target network every target_network_update_freq steps.
            #_deterministic_loss = #bool
            # Whether the loss should be calculated deterministically (w/o the stochastic action sampling step). True only useful 
            # for continuous actions and for debugging.
            #_use_beta_distribution = #bool
            # Use a Beta-distribution instead of a SquashedGaussian for bounded, continuous action spaces (not recommended; for 
            # debugging only).
        ).environment(
            env="EPEnv",
            observation_space=gym.spaces.Box(float("-inf"), float("inf"), (49,)),
            action_space=gym.spaces.Discrete(4),
            env_config=env_config,
        ).framework(
            framework = 'torch',
        ).fault_tolerance(
            recreate_failed_workers = True,
            restart_failed_sub_environments=False,
        ).rollouts(
            num_rollout_workers = 1,# if not tune_runner else tune.grid_search([0, 1, 3]),
            create_env_on_local_worker=True,
            rollout_fragment_length = 'auto',
            enable_connectors = True,
            #batch_mode="truncate_episodes",
            num_envs_per_worker=1,
        ).experimental(
            _enable_new_api_stack = True,
        ).reporting( # multi_agent config va aquí
            min_sample_timesteps_per_iteration = 2000,
        ).checkpointing(
            export_native_model_files = True,
        ).debugging(
            log_level = "ERROR",
        ).resources(
            num_gpus = 0,
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
    return "3x256_dueT1x256_douT_{}_{}".format(trial.trainable_name, trial.trial_id)

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
            name="{date}_VN_marl_{algorithm}".format(
                date=str(time.time()),
                algorithm=str(algorithm)
            ),
            stop={"episodes_total": 200},
            log_to_file=True,
            
            checkpoint_config=air.CheckpointConfig(
                checkpoint_at_end = True,
                checkpoint_frequency = 50,
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