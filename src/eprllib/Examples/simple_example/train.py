
# 1. Import eprllib
from eprllib.Environment.EnvironmentConfig import EnvironmentConfig
from eprllib.Agents.AgentSpec import AgentSpec
from eprllib.Agents.ObservationSpec import ObservationSpec
from eprllib.Agents.ActionSpec import ActionSpec
from eprllib.Agents.Filters.FilterSpec import FilterSpec
from eprllib.Agents.ActionMappers.ActionMapperSpec import ActionMapperSpec
from eprllib.Agents.Rewards.RewardSpec import RewardSpec

# 2. Configure the Environment

environment = EnvironmentConfig()

# As in all EnergyPlus simulations, you need to provide the path to the EnergyPlus input files (IDF and EPW) and the output directory.
environment.generals(
    epjson_path = "path/to/your/model.epjson", 
    epw_path = "path/to/your/weather.epw", 
    output_path = "path/to/output/directory",
)

environment.connector(
    connector_fn: type[BaseConnector] = BaseConnector,
    connector_fn_config: Dict[str, Any] = {}
)

environment.agents(
    agents_config = {
        # This facilitate the configuration of the agent and provides a safety way of specifying the 
        # parameters of the agent. It is recommended to provides a intuitive name for the agent.
        "agent_1": AgentSpec(
            
            # The variables specified here will be the ones accesible from filter, action_mapper, and 
            # reward modules.
            observation = ObservationSpec(
                variables = [
                    ("variable_name_1", "variable_type_1"),
                    ("variable_name_2", "variable_type_2")
                ],
                internal_variables = [
                    ("variable_name_1", "variable_type_1"),
                    ("variable_name_2", "variable_type_2")
                ],
                meters = [
                    "meter_name_1",
                    "meter_name_2"
                ]
            ),
            
            # The action parameter is used to define the actuators that the agent can control in the 
            # EnergyPlus environment.
            action = ActionSpec(
                actuators = {
                    "actuator_name_1": ("actuator_key_1", "object_name_1", "field_name_1"),
                }
            ),
            
            # This object allows to transform the raw observations from EnergyPlus into a format that 
            # is more suitable for the RL algorithm. You can specify the type of filter to apply (e.g., 
            # normalization, scaling, etc.) and any relevant parameters for the chosen filter type.
            filter = FilterSpec(
                filter_fn = ,
                filter_fn_config = {}
            ),
            
            # The action mapper is responsible for mapping the actions chosen by the RL algorithm (the policy action) 
            # to the corresponding actuators in the EnergyPlus environment (actuator action). This module ensures that 
            # the actions taken by the agent are correctly translated into changes in the EnergyPlus simulation.
            action_mapper = ActionMapperSpec(
                
            ),
            
            # The reward module defines the reward function that the RL algorithm will use to evaluate the actions taken by the agent.
            reward = RewardSpec(
                
            )
        )
    }
)

# 3.  **Configure the RL Algorithm:** Set up an `RLlib algorithm <https://docs.ray.io/en/latest/rllib/rllib-algorithms.html>`_
#     to train the RL policy.
# 4.  **Execute Training:** Run the training process using `RLlib <https://docs.ray.io/en/latest/rllib/index.html>`_
#     or `Tune <https://docs.ray.io/en/latest/tune/index.html>`_.