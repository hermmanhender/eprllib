from ctypes import c_void_p
from eprllib.ActionFunctions.ActionFunctions import ActionFunction
from eprllib.Env.MultiAgent.EnergyPlusRunner import EnergyPlusRunner, api
from eprllib.ObservationFunctions.ObservationFunctions import ObservationFunction
from queue import Queue
from typing import Dict, Any, Set
from unittest.mock import MagicMock, patch
from unittest.mock import Mock
import numpy as np
import pytest
import time
import threading

class TestEnergyplusrunner:

    @pytest.fixture
    def energy_plus_runner(self):
        # Mock the necessary attributes for EnergyPlusRunner
        env_config = {
            'infos_variables': {
                'variables_env': ['var1', 'var2'],
                'simulation_parameters': ['param1', 'param2'],
                'variables_obj': {'agent1': ['obj_var1', 'obj_var2']},
                'meters': {'agent1': ['meter1', 'meter2']},
                'static_variables': {'zone1': ['static1', 'static2']},
                'variables_thz': {'zone1': ['thz_var1', 'thz_var2']},
                'zone_simulation_parameters': {'zone1': ['zone_param1', 'zone_param2']}
            }
        }
        return EnergyPlusRunner(env_config, 1, None, None, None, set(), set(), None, None)

    @pytest.fixture
    def energyplus_runner(self):
        env_config = {
            'zone_simulation_parameters': {
                'system_time_step': False,
                'zone_time_step': False,
                'zone_time_step_number': False
            },
            'infos_variables': {
                'zone_simulation_parameters': {}
            },
            'no_observable_variables': {
                'zone_simulation_parameters': {}
            }
        }
        runner = EnergyPlusRunner(
            env_config=env_config,
            episode=1,
            obs_queue=MagicMock(),
            act_queue=MagicMock(),
            infos_queue=MagicMock(),
            _agent_ids=set(),
            _thermal_zone_ids=set(),
            observation_fn=MagicMock(),
            action_fn=MagicMock()
        )
        return runner

    @pytest.fixture
    def mock_api(self):
        with patch('eprllib.Env.MultiAgent.EnergyPlusRunner.api') as mock_api:
            yield mock_api

    @pytest.fixture
    def mock_energy_plus_runner(self):
        env_config = {
            'agents_config': {
                'agent1': {'ep_actuator_config': ('actuator1', 'type1', 'key1')},
                'agent2': {'ep_actuator_config': ('actuator2', 'type2', 'key2')}
            }
        }
        return EnergyPlusRunner(env_config=env_config, episode=1, obs_queue=None, act_queue=None, 
                                infos_queue=None, _agent_ids={'agent1', 'agent2'}, _thermal_zone_ids=set(), 
                                observation_fn=None, action_fn=None)

    @pytest.fixture
    def mock_energyplus_api(self):
        with patch('eprllib.Env.MultiAgent.EnergyPlusRunner.api') as mock_api:
            yield mock_api

    @pytest.fixture
    def mock_energyplus_runner(self):
        # Create a mock EnergyPlusRunner instance
        runner = Mock(spec=EnergyPlusRunner)
        runner.first_observation = True
        runner._collect_obs = Mock()
        return runner

    @pytest.fixture
    def mock_energyplus_runner_2(self):
        # Create a mock EnergyPlusRunner instance
        runner = Mock(spec=EnergyPlusRunner)
        runner.handle_site_variables = {
            'OutdoorAirTemperature': 123,
            'OutdoorRelativeHumidity': 456
        }
        runner.env_config = {
            'variables_env': ['OutdoorAirTemperature', 'OutdoorRelativeHumidity'],
            'infos_variables': {'variables_env': ['OutdoorAirTemperature']},
            'no_observable_variables': {'variables_env': ['OutdoorRelativeHumidity']}
        }
        return runner

    @pytest.fixture
    def mock_energyplus_runner_3(self):
        env_config = {
            'variables_thz': ['Zone Mean Air Temperature', 'Zone Air Relative Humidity'],
            'infos_variables': {'variables_thz': {'ThermalZone1': ['Zone Mean Air Temperature']}},
            'no_observable_variables': {'variables_thz': {'ThermalZone1': ['Zone Air Relative Humidity']}}
        }
        runner = EnergyPlusRunner(env_config, 1, Mock(), Mock(), Mock(), set(), set(), Mock(), Mock())
        runner.handle_thermalzone_variables = {'ThermalZone1': {'Zone Mean Air Temperature': 1, 'Zone Air Relative Humidity': 2}}
        return runner

    @pytest.fixture
    def mock_energyplus_runner_4(self):
        # Create a mock EnergyPlusRunner instance
        runner = Mock(spec=EnergyPlusRunner)
        runner.env_config = {
            'variables_obj': {'agent1': {'var1': 'value1', 'var2': 'value2'}},
            'infos_variables': {'variables_obj': {'agent1': ['var1']}},
            'no_observable_variables': {'variables_obj': {'agent1': ['var2']}}
        }
        runner.handle_object_variables = {'agent1': {'var1': 1, 'var2': 2}}
        return runner

    @pytest.fixture
    def mock_energyplus_runner_5(self):
        env_config = {
            'simulation_parameters': {
                'actual_date_time': True,
                'current_time': True,
            },
            'no_observable_variables': {
                'simulation_parameters': []
            }
        }
        runner = EnergyPlusRunner(
            env_config=env_config,
            episode=1,
            obs_queue=MagicMock(),
            act_queue=MagicMock(),
            infos_queue=MagicMock(),
            _agent_ids=set(),
            _thermal_zone_ids=set(),
            observation_fn=MagicMock(),
            action_fn=MagicMock()
        )
        return runner

    @pytest.fixture
    def mock_energyplus_runner_6(self):
        # Create a mock EnergyPlusRunner instance with minimal required attributes
        runner = EnergyPlusRunner(
            env_config={
                'no_observable_variables': {
                    'variables_env': ['var1', 'var2'],
                    'simulation_parameters': ['param1', 'param2'],
                    'variables_obj': {'agent1': ['obj1', 'obj2']},
                    'meters': {'agent1': ['meter1', 'meter2']},
                    'static_variables': {'zone1': ['static1', 'static2']},
                    'variables_thz': {'zone1': ['thz1', 'thz2']},
                    'zone_simulation_parameters': {'zone1': ['zsp1', 'zsp2']}
                }
            },
            episode=1,
            obs_queue=None,
            act_queue=None,
            infos_queue=None,
            _agent_ids=set(),
            _thermal_zone_ids=set(),
            observation_fn=None,
            action_fn=None
        )
        return runner

    @pytest.fixture
    def mock_energyplus_runner_7(self):
        # Create a mock EnergyPlusRunner instance
        runner = Mock(spec=EnergyPlusRunner)
        runner._init_handles = Mock(return_value=True)
        runner.initialized = False
        runner.init_handles = False
        return runner

    @pytest.fixture
    def mock_energyplus_runner_8(self):
        env_config = {
            'epw_path': 'path/to/epw',
            'epjson_path': 'path/to/epjson',
            'output_path': 'path/to/output',
            'ep_terminal_output': False,
            'evaluation': False
        }
        return EnergyPlusRunner(env_config, 1, Mock(), Mock(), Mock(), set(), set(), Mock(), Mock())

    @pytest.fixture
    def mock_energyplus_runner_9(self):
        env_config = {
            'ep_terminal_output': False,
            'output_path': '/tmp',
            'epw_path': '/path/to/weather.epw',
            'epjson_path': '/path/to/model.epjson'
        }
        runner = EnergyPlusRunner(env_config, 1, MagicMock(), MagicMock(), MagicMock(), set(), set(), MagicMock(), MagicMock())
        runner.energyplus_state = MagicMock()
        return runner

    @pytest.fixture
    def mock_energyplusrunner(self):
        # Create a mock EnergyPlusRunner instance
        runner = Mock(spec=EnergyPlusRunner)
        runner.handle_site_variables = {'var1': 1, 'var2': 2}
        runner.env_config = {
            'variables_env': ['var1', 'var2'],
            'infos_variables': {'variables_env': ['var1']},
            'no_observable_variables': {'variables_env': ['var2']}
        }
        return runner

    @pytest.fixture
    def mock_runner(self):
        env_config = {
            'ep_terminal_output': False,
            'output_path': '/tmp',
            'epw_path': '/path/to/weather.epw',
            'epjson_path': '/path/to/model.epjson'
        }
        runner = EnergyPlusRunner(
            env_config=env_config,
            episode=1,
            obs_queue=Mock(),
            act_queue=Mock(),
            infos_queue=Mock(),
            _agent_ids=set(['agent1']),
            _thermal_zone_ids=set(['zone1']),
            observation_fn=Mock(),
            action_fn=Mock()
        )
        return runner

    @pytest.fixture
    def mock_runner_10(self):
        env_config = {
            'ep_terminal_output': False,
            'output_path': '/tmp',
            'epw_path': '/path/to/weather.epw',
            'epjson_path': '/path/to/model.epjson',
        }
        runner = EnergyPlusRunner(
            env_config=env_config,
            episode=1,
            obs_queue=MagicMock(),
            act_queue=MagicMock(),
            infos_queue=MagicMock(),
            _agent_ids=set(),
            _thermal_zone_ids=set(),
            observation_fn=MagicMock(),
            action_fn=MagicMock(),
        )
        return runner

    @pytest.fixture
    def mock_runner_11(self):
        env_config = {
            'ep_terminal_output': False,
            'output_path': '/tmp',
            'epw_path': '/path/to/weather.epw',
            'epjson_path': '/path/to/model.epjson',
            'evaluation': False
        }
        runner = EnergyPlusRunner(env_config, 1, Mock(), Mock(), Mock(), set(), set(), Mock(), Mock())
        runner.energyplus_state = Mock()
        runner.energyplus_exec_thread = Mock()
        return runner

    @pytest.fixture
    def mock_runner_2(self):
        env_config = {
            'variables_env': ['OutdoorDryBulb'],
            'variables_thz': ['ZoneAirTemperature'],
            'variables_obj': {'agent1': {'Temperature': 'Object1'}},
            'meters': {'agent1': ['Electricity:Facility']},
            'static_variables': {'Zone1': {'Zone Floor Area': 'Zone Floor Area'}},
            'simulation_parameters': {'hour': True},
            'zone_simulation_parameters': {'zone_time_step': True},
            'agents_config': {'agent1': {'thermal_zone': 'Zone1'}},
            'ep_terminal_output': False,
            'observation': {'use_actuator_state': True},
        }
        runner = EnergyPlusRunner(env_config, 1, Mock(), Mock(), Mock(), {'agent1'}, {'Zone1'}, Mock(), Mock())
        runner._init_callback = Mock(return_value=True)
        runner.simulation_complete = False
        return runner

    @pytest.fixture
    def mock_runner_3(self):
        env_config = {
            'ep_terminal_output': False,
            'agents_config': {'agent1': {'ep_actuator_config': ('actuator', 'type', 'name')}},
            'timeout': 1
        }
        runner = EnergyPlusRunner(
            env_config=env_config,
            episode=1,
            obs_queue=Queue(),
            act_queue=Queue(),
            infos_queue=Queue(),
            _agent_ids={'agent1'},
            _thermal_zone_ids={'zone1'},
            observation_fn=MagicMock(),
            action_fn=MagicMock()
        )
        runner.actuator_handles = {'agent1': 1}
        runner._init_callback = MagicMock(return_value=True)
        runner.simulation_complete = False
        runner.first_observation = False
        runner.act_event = MagicMock()
        runner.act_queue = MagicMock()
        runner.action_fn = MagicMock()
        return runner

    @pytest.fixture
    def mock_runner_4(self):
        env_config = {
            'variables_thz': ['Zone Mean Air Temperature', 'Zone Air Relative Humidity'],
            'infos_variables': {'variables_thz': {'LIVING ZONE': ['Zone Mean Air Temperature']}},
            'no_observable_variables': {'variables_thz': {'LIVING ZONE': ['Zone Air Relative Humidity']}}
        }
        runner = EnergyPlusRunner(env_config, 1, None, None, None, set(), set(['LIVING ZONE']), None, None)
        runner.handle_thermalzone_variables = {'LIVING ZONE': {'Zone Mean Air Temperature': 1, 'Zone Air Relative Humidity': 2}}
        return runner

    @pytest.fixture
    def mock_runner_5(self):
        # Create a mock EnergyPlusRunner object
        env_config = {
            'variables_obj': {'agent1': {'var1': 'value1'}},
            'infos_variables': {'variables_obj': {'agent1': []}},
            'no_observable_variables': {'variables_obj': {'agent1': []}}
        }
        runner = EnergyPlusRunner(env_config, 1, None, None, None, {'agent1'}, {'zone1'}, None, None)
        runner.handle_object_variables = {'agent1': {'var1': 1}}
        return runner

    @pytest.fixture
    def mock_runner_6(self):
        env_config = {
            'meters': ['Electricity:Facility', 'Gas:Facility'],
            'no_observable_variables': {'meters': {'agent1': []}},
            'infos_variables': {'meters': {'agent1': []}}
        }
        runner = EnergyPlusRunner(env_config, 1, Mock(), Mock(), Mock(), {'agent1'}, {'zone1'}, Mock(), Mock())
        runner.meter_handles = {'agent1': {'Electricity:Facility': 1, 'Gas:Facility': 2}}
        return runner

    @pytest.fixture
    def mock_runner_7(self):
        # Create a mock EnergyPlusRunner instance
        env_config = {
            'agents_config': {
                'agent1': {'ep_actuator_config': ('Component', 'Control Type', 'Key')},
                'agent2': {'ep_actuator_config': ('Component', 'Control Type', 'Key')}
            }
        }
        runner = EnergyPlusRunner(env_config, 1, None, None, None, {'agent1', 'agent2'}, {'zone1', 'zone2'}, None, None)
        runner.actuator_handles = {'agent1': 1, 'agent2': 2}
        return runner

    @pytest.fixture
    def mock_runner_8(self):
        env_config = {
            'zone_simulation_parameters': {
                'system_time_step': True,
                'zone_time_step': True,
                'zone_time_step_number': True
            },
            'no_observable_variables': {
                'zone_simulation_parameters': {}
            }
        }
        runner = EnergyPlusRunner(env_config, 1, None, None, None, set(), set(), None, None)
        return runner

    @pytest.fixture
    def mock_runner_9(self):
        env_config = {
            'use_one_day_weather_prediction': True,
            'prediction_variables': {
                'outdoor_dry_bulb': True,
                'outdoor_relative_humidity': True,
                'wind_speed': True
            },
            'prediction_hours': 24
        }
        runner = EnergyPlusRunner(env_config, 1, Mock(), Mock(), Mock(), set(), set(), Mock(), Mock())
        return runner

    @pytest.fixture
    def runner(self):
        env_config = {
            'variables_obj': {'agent1': {'var1': 'value1', 'var2': 'value2'}},
            'infos_variables': {'variables_obj': {'agent1': ['var1']}},
            'no_observable_variables': {'variables_obj': {'agent1': ['var2']}}
        }
        return EnergyPlusRunner(env_config, 1, Mock(), Mock(), Mock(), {'agent1'}, {'zone1'}, Mock(), Mock())

    @pytest.fixture
    def runner_2(self):
        # Mock EnergyPlusRunner instance
        env_config = {
            'static_variables': ['Variable1', 'Variable2'],
            'no_observable_variables': {'static_variables': {'Zone1': ['Variable2']}},
            'infos_variables': {'static_variables': {'Zone1': ['Variable1']}}
        }
        return EnergyPlusRunner(env_config, 1, None, None, None, set(), {'Zone1'}, None, None)

    def test___init___empty_env_config(self):
        """
        Test initialization with an empty environment configuration.
        """
        with pytest.raises(KeyError):
            EnergyPlusRunner({}, 1, Queue(), Queue(), Queue(), set(), set(), ObservationFunction(), ActionFunction())

    def test___init___initialization(self):
        """
        Test that EnergyPlusRunner initializes correctly with all required parameters.
        """
        # Arrange
        env_config: Dict[str, Any] = {
            "ep_terminal_output": False,
            "output_path": "./output",
            "epw_path": "./weather.epw",
            "epjson_path": "./model.epjson",
            "variables_env": [],
            "variables_thz": [],
            "variables_obj": {},
            "static_variables": {},
            "meters": [],
            "agents_config": {},
            "simulation_parameters": {},
            "zone_simulation_parameters": {},
            "prediction_variables": {},
            "infos_variables": {
                "variables_env": [],
                "simulation_parameters": [],
                "variables_obj": {},
                "meters": {},
                "static_variables": {},
                "variables_thz": {},
                "zone_simulation_parameters": {}
            },
            "no_observable_variables": {
                "variables_env": [],
                "simulation_parameters": [],
                "variables_obj": {},
                "meters": {},
                "static_variables": {},
                "variables_thz": {},
                "zone_simulation_parameters": {}
            },
            "use_one_day_weather_prediction": False,
            "use_building_properties": False,
            "evaluation": False
        }
        episode = 1
        obs_queue = Queue()
        act_queue = Queue()
        infos_queue = Queue()
        _agent_ids: Set = {"agent1", "agent2"}
        _thermal_zone_ids: Set = {"zone1", "zone2"}
        observation_fn = ObservationFunction()
        action_fn = ActionFunction()

        # Act
        runner = EnergyPlusRunner(
            env_config,
            episode,
            obs_queue,
            act_queue,
            infos_queue,
            _agent_ids,
            _thermal_zone_ids,
            observation_fn,
            action_fn
        )

        # Assert
        assert runner.env_config == env_config
        assert runner.episode == episode
        assert runner.obs_queue == obs_queue
        assert runner.act_queue == act_queue
        assert runner.infos_queue == infos_queue
        assert runner._agent_ids == _agent_ids
        assert runner._thermal_zone_ids == _thermal_zone_ids
        assert runner.observation_fn == observation_fn
        assert runner.action_fn == action_fn
        assert runner.initialized is False
        assert runner.init_handles is False
        assert runner.simulation_complete is False
        assert runner.first_observation is True
        assert isinstance(runner.obs_event, threading.Event)
        assert isinstance(runner.act_event, threading.Event)
        assert isinstance(runner.infos_event, threading.Event)

    def test___init___invalid_action_function(self):
        """
        Test initialization with an invalid action function.
        """
        env_config = {"ep_terminal_output": True, "epw_path": "path/to/weather.epw", "output_path": "path/to/output", "epjson_path": "path/to/model.epjson"}
        with pytest.raises(TypeError):
            EnergyPlusRunner(env_config, 1, Queue(), Queue(), Queue(), set(), set(), ObservationFunction(), "invalid")

    def test___init___invalid_agent_ids(self):
        """
        Test initialization with invalid agent IDs.
        """
        env_config = {"ep_terminal_output": True, "epw_path": "path/to/weather.epw", "output_path": "path/to/output", "epjson_path": "path/to/model.epjson"}
        with pytest.raises(TypeError):
            EnergyPlusRunner(env_config, 1, Queue(), Queue(), Queue(), "invalid", set(), ObservationFunction(), ActionFunction())

    def test___init___invalid_episode(self):
        """
        Test initialization with an invalid episode number.
        """
        env_config = {"ep_terminal_output": True, "epw_path": "path/to/weather.epw", "output_path": "path/to/output", "epjson_path": "path/to/model.epjson"}
        with pytest.raises(TypeError):
            EnergyPlusRunner(env_config, "invalid", Queue(), Queue(), Queue(), set(), set(), ObservationFunction(), ActionFunction())

    def test___init___invalid_observation_function(self):
        """
        Test initialization with an invalid observation function.
        """
        env_config = {"ep_terminal_output": True, "epw_path": "path/to/weather.epw", "output_path": "path/to/output", "epjson_path": "path/to/model.epjson"}
        with pytest.raises(TypeError):
            EnergyPlusRunner(env_config, 1, Queue(), Queue(), Queue(), set(), set(), "invalid", ActionFunction())

    def test___init___invalid_path_types(self):
        """
        Test initialization with invalid path types in the configuration.
        """
        env_config = {"ep_terminal_output": True, "epw_path": 123, "output_path": 456, "epjson_path": 789}
        with pytest.raises(TypeError):
            EnergyPlusRunner(env_config, 1, Queue(), Queue(), Queue(), set(), set(), ObservationFunction(), ActionFunction())

    def test___init___invalid_queue_types(self):
        """
        Test initialization with invalid queue types.
        """
        env_config = {"ep_terminal_output": True, "epw_path": "path/to/weather.epw", "output_path": "path/to/output", "epjson_path": "path/to/model.epjson"}
        with pytest.raises(AttributeError):
            EnergyPlusRunner(env_config, 1, [], [], [], set(), set(), ObservationFunction(), ActionFunction())

    def test___init___invalid_thermal_zone_ids(self):
        """
        Test initialization with invalid thermal zone IDs.
        """
        env_config = {"ep_terminal_output": True, "epw_path": "path/to/weather.epw", "output_path": "path/to/output", "epjson_path": "path/to/model.epjson"}
        with pytest.raises(TypeError):
            EnergyPlusRunner(env_config, 1, Queue(), Queue(), Queue(), set(), "invalid", ObservationFunction(), ActionFunction())

    def test___init___missing_required_config(self):
        """
        Test initialization with missing required configuration.
        """
        env_config = {"ep_terminal_output": True}  # Missing required paths
        with pytest.raises(KeyError):
            EnergyPlusRunner(env_config, 1, Queue(), Queue(), Queue(), set(), set(), ObservationFunction(), ActionFunction())

    def test___init___negative_episode_number(self):
        """
        Test initialization with a negative episode number.
        """
        env_config = {"ep_terminal_output": True, "epw_path": "path/to/weather.epw", "output_path": "path/to/output", "epjson_path": "path/to/model.epjson"}
        with pytest.raises(ValueError):
            EnergyPlusRunner(env_config, -1, Queue(), Queue(), Queue(), set(), set(), ObservationFunction(), ActionFunction())

    def test__collect_first_obs_1(self):
        """
        Test that _collect_first_obs does nothing when self.first_observation is False
        """
        # Arrange
        runner = EnergyPlusRunner(
            env_config={},
            episode=1,
            obs_queue=MagicMock(),
            act_queue=MagicMock(),
            infos_queue=MagicMock(),
            _agent_ids=set(),
            _thermal_zone_ids=set(),
            observation_fn=MagicMock(),
            action_fn=MagicMock()
        )
        runner.first_observation = False
        runner._collect_obs = MagicMock()
        state_argument = c_void_p()

        # Act
        runner._collect_first_obs(state_argument)

        # Assert
        runner._collect_obs.assert_not_called()
        assert runner.first_observation is False

    def test__collect_first_obs_2(self):
        """
        Test that _collect_first_obs calls _collect_obs and sets first_observation to False
        when self.first_observation is True
        """
        # Arrange
        runner = EnergyPlusRunner(
            env_config={},
            episode=1,
            obs_queue=MagicMock(),
            act_queue=MagicMock(),
            infos_queue=MagicMock(),
            _agent_ids=set(),
            _thermal_zone_ids=set(),
            observation_fn=MagicMock(),
            action_fn=MagicMock()
        )
        runner.first_observation = True
        runner._collect_obs = MagicMock()
        state_argument = c_void_p()

        # Act
        runner._collect_first_obs(state_argument)

        # Assert
        runner._collect_obs.assert_called_once_with(state_argument)
        assert runner.first_observation is False

    def test__collect_first_obs_when_not_first_observation(self):
        """
        Test _collect_first_obs when it's not the first observation.
        """
        # Arrange
        env_config = {
            'ep_terminal_output': False,
            'output_path': '/tmp',
            'epjson_path': '/path/to/epjson',
            'epw_path': '/path/to/epw',
            'csv': False,
            'evaluation': False
        }
        runner = EnergyPlusRunner(
            env_config=env_config,
            episode=1,
            obs_queue=Mock(),
            act_queue=Mock(),
            infos_queue=Mock(),
            _agent_ids=set(),
            _thermal_zone_ids=set(),
            observation_fn=Mock(),
            action_fn=Mock()
        )
        runner.first_observation = False
        state_argument = Mock()

        # Act
        runner._collect_first_obs(state_argument)

        # Assert
        # Verify that _collect_obs is not called when it's not the first observation
        assert not hasattr(runner, '_collect_obs'), "_collect_obs should not be called"
        assert runner.first_observation is False, "first_observation should remain False"

    def test__collect_obs_2(self):
        """
        Test _collect_obs when simulation is ready and NaN or Inf values are present in observations
        """
        # Mock the necessary objects and methods
        mock_env_config = {
            'ep_terminal_output': False,
            'variables_env': ['OutdoorAirTemperature'],
            'variables_thz': ['ZoneAirTemperature'],
            'variables_obj': {'agent1': {'ObjectVariable': 'Value'}},
            'meters': {'agent1': ['Electricity:Facility']},
            'static_variables': {'Zone1': {'ZoneVolume': 'Value'}},
            'simulation_parameters': {'day_of_year': True},
            'zone_simulation_parameters': {'zone_time_step': True},
            'use_one_day_weather_prediction': False,
            'use_building_properties': False,
            'infos_variables': {
                'variables_env': [],
                'simulation_parameters': [],
                'variables_obj': {'agent1': []},
                'meters': {'agent1': []},
                'static_variables': {'Zone1': []},
                'variables_thz': {'Zone1': []},
                'zone_simulation_parameters': {'Zone1': []}
            },
            'no_observable_variables': {
                'variables_env': [],
                'simulation_parameters': [],
                'variables_obj': {'agent1': []},
                'meters': {'agent1': []},
                'static_variables': {'Zone1': []},
                'variables_thz': {'Zone1': []},
                'zone_simulation_parameters': {'Zone1': []}
            }
        }

        runner = EnergyPlusRunner(
            env_config=mock_env_config,
            episode=1,
            obs_queue=MagicMock(),
            act_queue=MagicMock(),
            infos_queue=MagicMock(),
            _agent_ids={'agent1'},
            _thermal_zone_ids={'Zone1'},
            observation_fn=MagicMock(),
            action_fn=MagicMock()
        )

        # Mock the necessary methods
        runner._init_callback = MagicMock(return_value=True)
        runner.simulation_complete = False
        runner.get_actuators_state = MagicMock(return_value=({}, {}))
        runner.get_site_variables_state = MagicMock(return_value=({'OutdoorAirTemperature': 25.0}, {}))
        runner.get_simulation_parameters_values = MagicMock(return_value=({'day_of_year': 1}, {}))
        runner.get_weather_prediction = MagicMock(return_value=({}, {}))
        runner.get_thermalzone_variables_state = MagicMock(return_value=({'ZoneAirTemperature': np.nan}, {}))
        runner.get_static_variables_state = MagicMock(return_value=({'ZoneVolume': np.inf}, {}))
        runner.get_zone_simulation_parameters_values = MagicMock(return_value=({'zone_time_step': 1}, {}))
        runner.get_buiding_properties = MagicMock(return_value=({}, {}))
        runner.get_object_variables_state = MagicMock(return_value=({'ObjectVariable': 1.0}, {}))
        runner.get_meters_state = MagicMock(return_value=({'Electricity:Facility': 100.0}, {}))

        runner.multiagent_method = MagicMock()
        runner.multiagent_method.set_agent_obs.return_value = (
            {'agent1': np.array([np.nan, np.inf, 1.0])},
            {'agent1': {}}
        )

        # Call the method under test
        with patch('builtins.print') as mock_print:
            runner._collect_obs(c_void_p())

        # Assertions
        assert runner.obs_queue.put.called
        assert runner.obs_event.set.called
        assert runner.infos_queue.put.called
        assert runner.infos_event.set.called
        
        # Check if print was called for NaN and Inf values
        mock_print.assert_any_call("NaN or Inf value found in agent1: [nan inf  1.]")

    def test__collect_obs_3(self):
        """
        Test _collect_obs method when simulation is ready and NaN/Inf values are present in observations
        """
        # Create a mock EnergyPlusRunner instance
        runner = EnergyPlusRunner(
            env_config={},
            episode=1,
            obs_queue=MagicMock(),
            act_queue=MagicMock(),
            infos_queue=MagicMock(),
            _agent_ids=set(['agent1']),
            _thermal_zone_ids=set(['zone1']),
            observation_fn=MagicMock(),
            action_fn=MagicMock()
        )

        # Mock necessary methods and attributes
        runner._init_callback = MagicMock(return_value=True)
        runner.simulation_complete = False
        runner.get_actuators_state = MagicMock(return_value=({}, {}))
        runner.get_site_variables_state = MagicMock(return_value=({}, {}))
        runner.get_simulation_parameters_values = MagicMock(return_value=({}, {}))
        runner.get_weather_prediction = MagicMock(return_value=({}, {}))
        runner.get_thermalzone_variables_state = MagicMock(return_value=({}, {}))
        runner.get_static_variables_state = MagicMock(return_value=({}, {}))
        runner.get_zone_simulation_parameters_values = MagicMock(return_value=({}, {}))
        runner.get_buiding_properties = MagicMock(return_value=({}, {}))
        runner.get_object_variables_state = MagicMock(return_value=({}, {}))
        runner.get_meters_state = MagicMock(return_value=({}, {}))

        # Mock multiagent_method to return observations with NaN and Inf values
        runner.multiagent_method = MagicMock()
        runner.multiagent_method.set_agent_obs.return_value = (
            {'agent1': np.array([1.0, np.nan, 3.0, np.inf])},
            {'agent1': {'info1': 1, 'info2': np.inf}}
        )

        # Call the method under test
        with patch('builtins.print') as mock_print:
            runner._collect_obs(c_void_p())

        # Assert that the method detected and printed NaN/Inf values
        mock_print.assert_any_call("NaN or Inf value found in agent1: [1.         nan         3.                inf]")
        mock_print.assert_any_call("NaN or Inf value found in agent1: {'info1': 1, 'info2': inf}")

        # Assert that observations and infos were put in the queues
        runner.obs_queue.put.assert_called_once()
        runner.obs_event.set.assert_called_once()
        runner.infos_queue.put.assert_called_once()
        runner.infos_event.set.assert_called_once()

    def test__collect_obs_4(self):
        """
        Test _collect_obs method when simulation is active and NaN/Inf values are present in observations
        """
        # Create a mock EnergyPlusRunner instance
        runner = EnergyPlusRunner(
            env_config={},
            episode=1,
            obs_queue=MagicMock(),
            act_queue=MagicMock(),
            infos_queue=MagicMock(),
            _agent_ids={"agent1"},
            _thermal_zone_ids={"zone1"},
            observation_fn=MagicMock(),
            action_fn=MagicMock()
        )

        # Mock dependencies
        runner._init_callback = MagicMock(return_value=True)
        runner.simulation_complete = False
        runner.get_actuators_state = MagicMock(return_value=({}, {}))
        runner.get_site_variables_state = MagicMock(return_value=({}, {}))
        runner.get_simulation_parameters_values = MagicMock(return_value=({}, {}))
        runner.get_weather_prediction = MagicMock(return_value=({}, {}))
        runner.get_thermalzone_variables_state = MagicMock(return_value=({}, {}))
        runner.get_static_variables_state = MagicMock(return_value=({}, {}))
        runner.get_zone_simulation_parameters_values = MagicMock(return_value=({}, {}))
        runner.get_buiding_properties = MagicMock(return_value=({}, {}))
        runner.get_object_variables_state = MagicMock(return_value=({}, {}))
        runner.get_meters_state = MagicMock(return_value=({}, {}))

        # Mock multiagent_method to return NaN and Inf values
        runner.multiagent_method = MagicMock()
        runner.multiagent_method.set_agent_obs.return_value = (
            {"agent1": np.array([np.nan, np.inf, 1.0])},
            {"agent1": {"info": np.array([np.nan, np.inf, 2.0])}}
        )

        # Call the method
        with patch('builtins.print') as mock_print:
            runner._collect_obs(c_void_p())

        # Assertions
        assert mock_print.call_count == 2
        mock_print.assert_any_call("NaN or Inf value found in agent1: [nan inf  1.]")
        mock_print.assert_any_call("NaN or Inf value found in agent1: [nan inf  2.]")

        runner.obs_queue.put.assert_called_once()
        runner.obs_event.set.assert_called_once()
        runner.infos_queue.put.assert_called_once()
        runner.infos_event.set.assert_called_once()

    def test__collect_obs_when_not_initialized_or_simulation_complete(self):
        """
        Test that _collect_obs returns early when not initialized or simulation is complete.
        """
        # Setup
        mock_env_config = {
            'ep_terminal_output': False,
            'output_path': '/tmp',
            'epw_path': '/path/to/weather.epw',
            'epjson_path': '/path/to/model.epjson',
            'evaluation': False,
        }
        mock_episode = 1
        mock_obs_queue = Mock()
        mock_act_queue = Mock()
        mock_infos_queue = Mock()
        mock_agent_ids = set(['agent1'])
        mock_thermal_zone_ids = set(['zone1'])
        mock_observation_fn = Mock()
        mock_action_fn = Mock()

        runner = EnergyPlusRunner(
            mock_env_config, mock_episode, mock_obs_queue, mock_act_queue, mock_infos_queue,
            mock_agent_ids, mock_thermal_zone_ids, mock_observation_fn, mock_action_fn
        )

        # Mock the _init_callback method to return False
        runner._init_callback = Mock(return_value=False)
        
        # Test when not initialized
        mock_state_argument = c_void_p()
        runner._collect_obs(mock_state_argument)
        
        # Assert that the obs_queue and infos_queue were not called
        mock_obs_queue.put.assert_not_called()
        mock_infos_queue.put.assert_not_called()

        # Test when simulation is complete
        runner.simulation_complete = True
        runner._collect_obs(mock_state_argument)
        
        # Assert that the obs_queue and infos_queue were not called
        mock_obs_queue.put.assert_not_called()
        mock_infos_queue.put.assert_not_called()

    def test__flush_queues_1(self):
        """
        Test that _flush_queues method empties all queues
        """
        # Create a mock EnergyPlusRunner instance
        runner = EnergyPlusRunner(
            env_config={},
            episode=1,
            obs_queue=Queue(),
            act_queue=Queue(),
            infos_queue=Queue(),
            _agent_ids=set(),
            _thermal_zone_ids=set(),
            observation_fn=MagicMock(),
            action_fn=MagicMock()
        )

        # Add some items to the queues
        runner.obs_queue.put("obs1")
        runner.obs_queue.put("obs2")
        runner.act_queue.put("act1")
        runner.infos_queue.put("info1")
        runner.infos_queue.put("info2")
        runner.infos_queue.put("info3")

        # Call the _flush_queues method
        runner._flush_queues()

        # Assert that all queues are empty
        assert runner.obs_queue.empty()
        assert runner.act_queue.empty()
        assert runner.infos_queue.empty()

    def test__flush_queues_2(self):
        """
        Test that _flush_queues empties non-empty queues
        """
        # Create a mock EnergyPlusRunner instance
        runner = MagicMock(spec=EnergyPlusRunner)
        
        # Create non-empty queues
        runner.obs_queue = Queue()
        runner.act_queue = Queue()
        runner.infos_queue = Queue()
        
        # Add items to the queues
        runner.obs_queue.put("obs1")
        runner.act_queue.put("act1")
        runner.infos_queue.put("info1")
        
        # Call the _flush_queues method
        EnergyPlusRunner._flush_queues(runner)
        
        # Assert that all queues are empty after flushing
        assert runner.obs_queue.empty()
        assert runner.act_queue.empty()
        assert runner.infos_queue.empty()

    def test__flush_queues_3(self):
        """
        Test that _flush_queues handles empty queues correctly.
        """
        # Create a mock EnergyPlusRunner instance
        runner = EnergyPlusRunner(
            env_config={},
            episode=1,
            obs_queue=Queue(),
            act_queue=Queue(),
            infos_queue=Queue(),
            _agent_ids=set(),
            _thermal_zone_ids=set(),
            observation_fn=None,
            action_fn=None
        )

        # Ensure all queues are empty
        assert runner.obs_queue.empty()
        assert runner.act_queue.empty()
        assert runner.infos_queue.empty()

        # Call _flush_queues
        runner._flush_queues()

        # Verify that the queues are still empty after flushing
        assert runner.obs_queue.empty()
        assert runner.act_queue.empty()
        assert runner.infos_queue.empty()

    def test__flush_queues_with_empty_queues(self):
        """
        Test _flush_queues with empty queues
        """
        runner = EnergyPlusRunner(
            env_config={},
            episode=1,
            obs_queue=Queue(),
            act_queue=Queue(),
            infos_queue=Queue(),
            _agent_ids=set(),
            _thermal_zone_ids=set(),
            observation_fn=Mock(),
            action_fn=Mock()
        )

        runner._flush_queues()
        
        assert runner.obs_queue.empty()
        assert runner.act_queue.empty()
        assert runner.infos_queue.empty()

    @patch('queue.Queue.get')
    def test__flush_queues_with_exception_during_get(self, mock_get):
        """
        Test _flush_queues when an exception occurs during queue.get()
        """
        runner = EnergyPlusRunner(
            env_config={},
            episode=1,
            obs_queue=Queue(),
            act_queue=Queue(),
            infos_queue=Queue(),
            _agent_ids=set(),
            _thermal_zone_ids=set(),
            observation_fn=Mock(),
            action_fn=Mock()
        )

        runner.obs_queue.put("obs")
        mock_get.side_effect = Exception("Test exception")

        with pytest.raises(Exception, match="Test exception"):
            runner._flush_queues()

    def test__flush_queues_with_invalid_queue_object(self):
        """
        Test _flush_queues with invalid queue object
        """
        runner = EnergyPlusRunner(
            env_config={},
            episode=1,
            obs_queue=Queue(),
            act_queue=Queue(),
            infos_queue=Queue(),
            _agent_ids=set(),
            _thermal_zone_ids=set(),
            observation_fn=Mock(),
            action_fn=Mock()
        )

        runner.obs_queue = "not a queue"

        with pytest.raises(AttributeError):
            runner._flush_queues()

    def test__flush_queues_with_large_queue(self):
        """
        Test _flush_queues with a large number of items in the queue
        """
        runner = EnergyPlusRunner(
            env_config={},
            episode=1,
            obs_queue=Queue(),
            act_queue=Queue(),
            infos_queue=Queue(),
            _agent_ids=set(),
            _thermal_zone_ids=set(),
            observation_fn=Mock(),
            action_fn=Mock()
        )

        for _ in range(10000):
            runner.obs_queue.put("obs")
            runner.act_queue.put("act")
            runner.infos_queue.put("info")

        runner._flush_queues()
        
        assert runner.obs_queue.empty()
        assert runner.act_queue.empty()
        assert runner.infos_queue.empty()

    def test__flush_queues_with_non_empty_queues(self):
        """
        Test _flush_queues with non-empty queues
        """
        runner = EnergyPlusRunner(
            env_config={},
            episode=1,
            obs_queue=Queue(),
            act_queue=Queue(),
            infos_queue=Queue(),
            _agent_ids=set(),
            _thermal_zone_ids=set(),
            observation_fn=Mock(),
            action_fn=Mock()
        )

        runner.obs_queue.put("obs")
        runner.act_queue.put("act")
        runner.infos_queue.put("info")

        runner._flush_queues()
        
        assert runner.obs_queue.empty()
        assert runner.act_queue.empty()
        assert runner.infos_queue.empty()

    @patch('eprllib.Env.MultiAgent.EnergyPlusRunner.api')
    def test__init_callback_during_warmup(self, mock_api, mock_energyplus_runner_7):
        """
        Test that _init_callback returns False during warmup period
        """
        # Arrange
        mock_state_argument = Mock()
        mock_api.exchange.warmup_flag.return_value = True
        mock_energyplus_runner_7._init_handles.return_value = True

        # Act
        result = EnergyPlusRunner._init_callback(mock_energyplus_runner_7, mock_state_argument)

        # Assert
        assert result is False
        assert mock_energyplus_runner_7.init_handles == True
        assert mock_energyplus_runner_7.initialized == False

    @patch('eprllib.Env.MultiAgent.EnergyPlusRunner.api')
    def test__init_callback_handles_not_initialized(self, mock_api, mock_energyplus_runner_7):
        """
        Test that _init_callback returns False when handles are not initialized
        """
        # Arrange
        mock_state_argument = Mock()
        mock_api.exchange.warmup_flag.return_value = False
        mock_energyplus_runner_7._init_handles.return_value = False

        # Act
        result = EnergyPlusRunner._init_callback(mock_energyplus_runner_7, mock_state_argument)

        # Assert
        assert result is False
        assert mock_energyplus_runner_7.init_handles == False
        assert mock_energyplus_runner_7.initialized == False

    @patch.object(EnergyPlusRunner, '_init_handles')
    @patch.object(api.exchange, 'warmup_flag')
    def test__init_callback_init_handles_false(self, mock_warmup_flag, mock_init_handles, mock_runner_10):
        """
        Test _init_callback when _init_handles returns False
        """
        mock_init_handles.return_value = False
        mock_warmup_flag.return_value = False
        state_argument = c_void_p()

        result = mock_runner_10._init_callback(state_argument)

        assert result is False
        assert mock_runner_10.initialized is False

    @patch('eprllib.Env.MultiAgent.EnergyPlusRunner.api')
    def test__init_callback_initializes_handles_and_sets_flags(self, mock_api, mock_energyplus_runner_7):
        """
        Test that _init_callback initializes handles and sets flags correctly
        """
        # Arrange
        mock_state_argument = Mock()
        mock_api.exchange.warmup_flag.return_value = False

        # Act
        result = EnergyPlusRunner._init_callback(mock_energyplus_runner_7, mock_state_argument)

        # Assert
        assert result is True
        assert mock_energyplus_runner_7.init_handles is True
        assert mock_energyplus_runner_7.initialized is True
        mock_energyplus_runner_7._init_handles.assert_called_once_with(mock_state_argument)
        mock_api.exchange.warmup_flag.assert_called_once_with(mock_state_argument)

    def test__init_callback_invalid_state_argument(self, mock_runner_10):
        """
        Test _init_callback with an invalid state_argument (None)
        """
        with pytest.raises(AttributeError):
            mock_runner_10._init_callback(None)

    def test__init_callback_invalid_state_argument_type(self, mock_runner_10):
        """
        Test _init_callback with an invalid state_argument type (int instead of c_void_p)
        """
        with pytest.raises(TypeError):
            mock_runner_10._init_callback(42)

    def test__init_callback_simulation_complete(self, mock_runner_10):
        """
        Test _init_callback when simulation is already complete
        """
        mock_runner_10.simulation_complete = True
        state_argument = c_void_p()

        result = mock_runner_10._init_callback(state_argument)

        assert result is False
        assert mock_runner_10.initialized is False

    @patch.object(EnergyPlusRunner, '_init_handles')
    @patch.object(api.exchange, 'warmup_flag')
    def test__init_callback_warmup_flag_true(self, mock_warmup_flag, mock_init_handles, mock_runner_10):
        """
        Test _init_callback when warmup_flag is True
        """
        mock_init_handles.return_value = True
        mock_warmup_flag.return_value = True
        state_argument = c_void_p()

        result = mock_runner_10._init_callback(state_argument)

        assert result is False
        assert mock_runner_10.initialized is False

    def test__init_handles_2(self):
        """
        Test _init_handles when api_data_fully_ready is True but actuator_handles contain -1 values.
        """
        # Mock the necessary objects and methods
        mock_state = Mock()
        mock_env_config = {
            'agents_config': {'agent1': {'ep_actuator_config': ('Type', 'Key', 'Value')}},
            'variables_env': [],
            'variables_obj': {},
            'meters': {},
            'static_variables': {},
            'variables_thz': {}
        }
        mock_obs_queue = Mock()
        mock_act_queue = Mock()
        mock_infos_queue = Mock()

        # Create an instance of EnergyPlusRunner
        runner = EnergyPlusRunner(
            env_config=mock_env_config,
            episode=1,
            obs_queue=mock_obs_queue,
            act_queue=mock_act_queue,
            infos_queue=mock_infos_queue,
            _agent_ids={'agent1'},
            _thermal_zone_ids=set(),
            observation_fn=Mock(),
            action_fn=Mock()
        )

        # Mock the necessary API calls
        with patch.object(api.exchange, 'api_data_fully_ready', return_value=True), \
             patch.object(api.exchange, 'get_actuator_handle', return_value=-1), \
             patch.object(api.exchange, 'list_available_api_data_csv', return_value=b''):

            # Call the method under test
            result = runner._init_handles(mock_state)

        # Assert the expected behavior
        assert result is False
        assert runner.actuator_handles == {'agent1': -1}

    def test__init_handles_3(self):
        """
        Test _init_handles when actuator handles are not initialized correctly.
        """
        # Mock the necessary objects and methods
        mock_state = MagicMock()
        mock_env_config = {
            'variables_env': [],
            'variables_thz': [],
            'variables_obj': {},
            'meters': [],
            'static_variables': {},
            'agents_config': {'agent1': {'ep_actuator_config': ('Component', 'Control Type', 'Key')}},
        }
        mock_obs_queue = MagicMock()
        mock_act_queue = MagicMock()
        mock_infos_queue = MagicMock()
        mock_agent_ids = {'agent1'}
        mock_thermal_zone_ids = {'zone1'}
        mock_observation_fn = MagicMock()
        mock_action_fn = MagicMock()

        # Create an instance of EnergyPlusRunner
        runner = EnergyPlusRunner(
            mock_env_config, 1, mock_obs_queue, mock_act_queue, mock_infos_queue,
            mock_agent_ids, mock_thermal_zone_ids, mock_observation_fn, mock_action_fn
        )

        # Set up the mocked behavior
        runner.init_handles = True
        runner.actuator_handles = {'agent1': -1}  # Simulate a failed handle initialization

        # Mock the api.exchange methods
        with patch.object(api.exchange, 'api_data_fully_ready', return_value=True), \
             patch.object(api.exchange, 'get_actuator_handle', return_value=-1), \
             patch.object(api.exchange, 'list_available_api_data_csv', return_value=b'mock_data'):

            # Call the method under test
            result = runner._init_handles(mock_state)

        # Assert the result
        assert result is False

        # Verify that the error message was created (we can't check if it was raised because it's not actually raised in the method)
        assert any([v == -1 for v in runner.actuator_handles.values()])

    @patch('eprllib.Env.MultiAgent.EnergyPlusRunner.api')
    def test__init_handles_4(self, mock_api):
        """
        Test _init_handles when handle_site_variables contains -1 values.
        """
        # Create a mock EnergyPlusRunner instance
        runner = MagicMock(spec=EnergyPlusRunner)
        runner.init_handles = False
        runner.actuator_handles = {'actuator1': 1, 'actuator2': 2}
        runner.handle_site_variables = {'var1': -1, 'var2': 2}
        runner.dict_site_variables = {'var1': ('Var1', 'Env'), 'var2': ('Var2', 'Env')}

        # Mock api.exchange methods
        mock_api.exchange.api_data_fully_ready.return_value = True
        mock_api.exchange.get_actuator_handle.return_value = 1
        mock_api.exchange.get_variable_handle.side_effect = [-1, 2]
        mock_api.exchange.list_available_api_data_csv.return_value = b"Available data"

        # Call the method
        result = EnergyPlusRunner._init_handles(runner, MagicMock())

        # Assertions
        assert result is False
        mock_api.exchange.api_data_fully_ready.assert_called_once()
        mock_api.exchange.get_variable_handle.assert_called()
        mock_api.exchange.list_available_api_data_csv.assert_called_once()

    def test__init_handles_5(self):
        """
        Test _init_handles when object variables or meter handles are invalid
        """
        # Mock EnergyPlusRunner instance
        runner = MagicMock(spec=EnergyPlusRunner)
        runner.init_handles = False
        runner._agent_ids = ['agent1']
        runner._thermal_zone_ids = ['zone1']
        runner.actuators = {'agent1': ('Component', 'Control Type', 'Key')}
        runner.dict_site_variables = {'var1': ('Variable', 'Key')}
        runner.dict_object_variables = {'var2': ('Variable', 'Key')}
        runner.meters = {'meter1': 'Meter'}
        runner.dict_thermalzone_variables = {'var3': ('Variable', 'Key')}
        runner.dict_static_variables = {'zone1': {'var4': ('Variable', 'Key')}}

        # Mock api calls
        api.exchange.api_data_fully_ready = MagicMock(return_value=True)
        api.exchange.get_actuator_handle = MagicMock(return_value=1)
        api.exchange.get_variable_handle = MagicMock(return_value=1)
        api.exchange.get_meter_handle = MagicMock(return_value=-1)  # Simulate invalid meter handle
        api.exchange.get_internal_variable_handle = MagicMock(return_value=1)
        api.exchange.list_available_api_data_csv = MagicMock(return_value=b'available_data')

        # Call the method
        result = EnergyPlusRunner._init_handles(runner, MagicMock())

        # Assertions
        assert result is False
        api.exchange.api_data_fully_ready.assert_called_once()
        api.exchange.get_actuator_handle.assert_called_once()
        api.exchange.get_variable_handle.assert_called()
        api.exchange.get_meter_handle.assert_called_once()
        api.exchange.list_available_api_data_csv.assert_called_once()

    def test__init_handles_6(self):
        """
        Test that _init_handles returns False when thermal zone variables or static variables have -1 handles.
        """
        # Mock the necessary objects and methods
        mock_state = MagicMock()
        mock_env_config = {
            'variables_env': [],
            'variables_thz': [],
            'variables_obj': {},
            'meters': [],
            'static_variables': {},
            'agents_config': {}
        }
        mock_obs_queue = MagicMock()
        mock_act_queue = MagicMock()
        mock_infos_queue = MagicMock()
        mock_agent_ids = set(['agent1'])
        mock_thermal_zone_ids = set(['zone1'])
        mock_observation_fn = MagicMock()
        mock_action_fn = MagicMock()

        # Create an instance of EnergyPlusRunner
        runner = EnergyPlusRunner(
            mock_env_config, 1, mock_obs_queue, mock_act_queue, mock_infos_queue,
            mock_agent_ids, mock_thermal_zone_ids, mock_observation_fn, mock_action_fn
        )

        # Mock the necessary attributes and methods
        runner.init_handles = False
        runner.actuator_handles = {'agent1': 1}
        runner.handle_site_variables = {'var1': 1}
        runner.handle_object_variables = {'agent1': {'var1': 1}}
        runner.meter_handles = {'agent1': {'meter1': 1}}
        runner.handle_thermalzone_variables = {'zone1': {'var1': -1}}  # Set to -1 to trigger the condition
        runner.handle_static_variables = {'zone1': {'var1': 1}}
        runner.dict_thermalzone_variables = {'var1': ('var1', 'zone1')}
        runner.dict_static_variables = {'zone1': {'var1': 'var1'}}

        # Mock the api.exchange methods
        api.exchange.api_data_fully_ready = MagicMock(return_value=True)
        api.exchange.get_variable_handle = MagicMock(return_value=1)
        api.exchange.get_internal_variable_handle = MagicMock(return_value=1)
        api.exchange.list_available_api_data_csv = MagicMock(return_value=b'available_data')

        # Call the method under test
        result = runner._init_handles(mock_state)

        # Assert the result
        assert result is False

        # Verify that the appropriate methods were called
        api.exchange.api_data_fully_ready.assert_called_once_with(mock_state)
        api.exchange.get_variable_handle.assert_called()
        api.exchange.get_internal_variable_handle.assert_called()
        api.exchange.list_available_api_data_csv.assert_called_once_with(mock_state)

    def test__init_handles_api_data_not_ready(self, mock_energyplus_runner_8):
        """
        Test when API data is not fully ready
        """
        with patch('eprllib.Env.MultiAgent.EnergyPlusRunner.api') as mock_api:
            mock_api.exchange.api_data_fully_ready.return_value = False
            state_argument = c_void_p()
            
            result = mock_energyplus_runner_8._init_handles(state_argument)
            
            assert result is False
            mock_api.exchange.api_data_fully_ready.assert_called_once_with(state_argument)

    def test__init_handles_data_not_ready(self):
        """
        Test _init_handles when api_data is not fully ready.
        """
        # Mock the necessary objects and methods
        mock_state = Mock()
        mock_env_config = {
            'variables_env': [],
            'variables_obj': {},
            'meters': {},
            'static_variables': {},
            'variables_thz': [],
            'agents_config': {}
        }
        
        # Create an instance of EnergyPlusRunner with mocked dependencies
        runner = EnergyPlusRunner(
            env_config=mock_env_config,
            episode=1,
            obs_queue=Mock(),
            act_queue=Mock(),
            infos_queue=Mock(),
            _agent_ids=set(),
            _thermal_zone_ids=set(),
            observation_fn=Mock(),
            action_fn=Mock()
        )
        
        # Mock the api.exchange.api_data_fully_ready method to return False
        with patch.object(api.exchange, 'api_data_fully_ready', return_value=False):
            result = runner._init_handles(mock_state)
        
        # Assert that the method returns False when api_data is not fully ready
        assert result is False

        # Verify that the init_handles attribute remains False
        assert runner.init_handles is False

    def test__init_handles_invalid_actuator_handle(self, mock_energyplus_runner_8):
        """
        Test when an invalid actuator handle is returned
        """
        with patch('eprllib.Env.MultiAgent.EnergyPlusRunner.api') as mock_api:
            mock_api.exchange.api_data_fully_ready.return_value = True
            mock_api.exchange.get_actuator_handle.return_value = -1
            mock_energyplus_runner_8.actuators = {'test_actuator': ('component', 'control_type', 'actuator_key')}
            state_argument = c_void_p()
            
            with pytest.raises(ValueError):
                mock_energyplus_runner_8._init_handles(state_argument)

    def test__init_handles_invalid_meter_handle(self, mock_energyplus_runner_8):
        """
        Test when an invalid meter handle is returned
        """
        with patch('eprllib.Env.MultiAgent.EnergyPlusRunner.api') as mock_api:
            mock_api.exchange.api_data_fully_ready.return_value = True
            mock_api.exchange.get_actuator_handle.return_value = 1
            mock_api.exchange.get_variable_handle.return_value = 1
            mock_api.exchange.get_meter_handle.return_value = -1
            mock_energyplus_runner_8.actuators = {'test_actuator': ('component', 'control_type', 'actuator_key')}
            mock_energyplus_runner_8.dict_site_variables = {'test_var': ('var_type', 'var_key')}
            mock_energyplus_runner_8.dict_object_variables = {'test_obj_var': ('var_type', 'var_key')}
            mock_energyplus_runner_8.meters = {'test_meter': 'meter_name'}
            mock_energyplus_runner_8._agent_ids = {'agent1'}
            state_argument = c_void_p()
            
            with pytest.raises(ValueError):
                mock_energyplus_runner_8._init_handles(state_argument)

    def test__init_handles_invalid_object_variable_handle(self, mock_energyplus_runner_8):
        """
        Test when an invalid object variable handle is returned
        """
        with patch('eprllib.Env.MultiAgent.EnergyPlusRunner.api') as mock_api:
            mock_api.exchange.api_data_fully_ready.return_value = True
            mock_api.exchange.get_actuator_handle.return_value = 1
            mock_api.exchange.get_variable_handle.side_effect = [1, -1]
            mock_energyplus_runner_8.actuators = {'test_actuator': ('component', 'control_type', 'actuator_key')}
            mock_energyplus_runner_8.dict_site_variables = {'test_var': ('var_type', 'var_key')}
            mock_energyplus_runner_8.dict_object_variables = {'test_obj_var': ('var_type', 'var_key')}
            mock_energyplus_runner_8._agent_ids = {'agent1'}
            state_argument = c_void_p()
            
            with pytest.raises(ValueError):
                mock_energyplus_runner_8._init_handles(state_argument)

    def test__init_handles_invalid_site_variable_handle(self, mock_energyplus_runner_8):
        """
        Test when an invalid site variable handle is returned
        """
        with patch('eprllib.Env.MultiAgent.EnergyPlusRunner.api') as mock_api:
            mock_api.exchange.api_data_fully_ready.return_value = True
            mock_api.exchange.get_actuator_handle.return_value = 1
            mock_api.exchange.get_variable_handle.return_value = -1
            mock_energyplus_runner_8.actuators = {'test_actuator': ('component', 'control_type', 'actuator_key')}
            mock_energyplus_runner_8.dict_site_variables = {'test_var': ('var_type', 'var_key')}
            state_argument = c_void_p()
            
            with pytest.raises(ValueError):
                mock_energyplus_runner_8._init_handles(state_argument)

    def test__init_handles_success(self, mock_energyplus_runner_8):
        """
        Test successful initialization of handles
        """
        with patch('eprllib.Env.MultiAgent.EnergyPlusRunner.api') as mock_api:
            mock_api.exchange.api_data_fully_ready.return_value = True
            mock_api.exchange.get_actuator_handle.return_value = 1
            mock_api.exchange.get_variable_handle.return_value = 1
            mock_api.exchange.get_meter_handle.return_value = 1
            mock_api.exchange.get_internal_variable_handle.return_value = 1
            mock_energyplus_runner_8.actuators = {'test_actuator': ('component', 'control_type', 'actuator_key')}
            mock_energyplus_runner_8.dict_site_variables = {'test_var': ('var_type', 'var_key')}
            mock_energyplus_runner_8.dict_object_variables = {'test_obj_var': ('var_type', 'var_key')}
            mock_energyplus_runner_8.meters = {'test_meter': 'meter_name'}
            mock_energyplus_runner_8._agent_ids = {'agent1'}
            mock_energyplus_runner_8._thermal_zone_ids = {'zone1'}
            mock_energyplus_runner_8.dict_thermalzone_variables = {'test_thz_var': ('var_type', 'var_key')}
            mock_energyplus_runner_8.dict_static_variables = {'zone1': {'test_static_var': ('var_type', 'var_key')}}
            state_argument = c_void_p()
            
            result = mock_energyplus_runner_8._init_handles(state_argument)
            
            assert result is True
            mock_api.exchange.api_data_fully_ready.assert_called_once_with(state_argument)
            mock_api.exchange.get_actuator_handle.assert_called()
            mock_api.exchange.get_variable_handle.assert_called()
            mock_api.exchange.get_meter_handle.assert_called()
            mock_api.exchange.get_internal_variable_handle.assert_called()

    def test__run_energyplus_exception(self, mock_energyplus_runner_9):
        """Test _run_energyplus when an exception is raised"""
        mock_energyplus_runner_9.make_eplus_args = MagicMock(return_value=["-w", "nonexistent.epw"])
        
        with patch.object(api.runtime, 'run_energyplus', side_effect=Exception("EnergyPlus error")):
            with pytest.raises(Exception, match="EnergyPlus error"):
                mock_energyplus_runner_9._run_energyplus()

    def test__run_energyplus_invalid_args(self, mock_energyplus_runner_9):
        """Test _run_energyplus with invalid command line arguments"""
        mock_energyplus_runner_9.make_eplus_args = MagicMock(return_value=[])
        
        with patch.object(api.runtime, 'run_energyplus', return_value=1) as mock_run:
            mock_energyplus_runner_9._run_energyplus()
            
        assert mock_energyplus_runner_9.sim_results == 1
        assert mock_energyplus_runner_9.simulation_complete is True

    def test__run_energyplus_invalid_epw_path(self, mock_energyplus_runner_9):
        """Test _run_energyplus with an invalid EPW file path"""
        mock_energyplus_runner_9.env_config['epw_path'] = '/nonexistent/path/weather.epw'
        
        with patch.object(api.runtime, 'run_energyplus', return_value=1) as mock_run:
            mock_energyplus_runner_9._run_energyplus()
            
        assert mock_energyplus_runner_9.sim_results == 1
        assert mock_energyplus_runner_9.simulation_complete is True

    def test__run_energyplus_invalid_state(self, mock_energyplus_runner_9):
        """Test _run_energyplus with an invalid EnergyPlus state"""
        mock_energyplus_runner_9.energyplus_state = None
        
        with pytest.raises(AttributeError):
            mock_energyplus_runner_9._run_energyplus()

    def test__run_energyplus_simulation_failure(self, mock_energyplus_runner_9):
        """Test _run_energyplus when the simulation fails"""
        mock_energyplus_runner_9.make_eplus_args = MagicMock(return_value=["-w", "weather.epw", "-d", "output", "model.epjson"])
        
        with patch.object(api.runtime, 'run_energyplus', return_value=1) as mock_run:
            mock_energyplus_runner_9._run_energyplus()
            
        assert mock_energyplus_runner_9.sim_results == 1
        assert mock_energyplus_runner_9.simulation_complete is True

    @patch('eprllib.Env.MultiAgent.EnergyPlusRunner.api')
    def test__run_energyplus_successful_execution(self, mock_api):
        """
        Test _run_energyplus for successful execution of EnergyPlus simulation
        """
        # Arrange
        mock_runner = Mock(spec=EnergyPlusRunner)
        mock_runner.make_eplus_args.return_value = ["-w", "weather.epw", "-d", "output_dir", "input.epjson"]
        mock_runner.energyplus_state = "mock_state"
        mock_api.runtime.run_energyplus.return_value = 0

        # Act
        EnergyPlusRunner._run_energyplus(mock_runner)

        # Assert
        mock_runner.make_eplus_args.assert_called_once()
        mock_api.runtime.run_energyplus.assert_called_once_with("mock_state", ["-w", "weather.epw", "-d", "output_dir", "input.epjson"])
        assert mock_runner.sim_results == 0
        assert mock_runner.simulation_complete is True

    def test__send_actions_first_observation(self, mock_runner_3):
        mock_runner_3.first_observation = True
        mock_runner_3._collect_first_obs = MagicMock()
        mock_runner_3._send_actions(MagicMock())
        mock_runner_3._collect_first_obs.assert_called_once()

    def test__send_actions_incorrect_type(self, mock_runner_3):
        mock_runner_3.act_event.wait.return_value = True
        mock_runner_3.act_queue.get.return_value = {'agent1': 'invalid'}
        mock_runner_3.action_fn.transform_action.side_effect = TypeError
        with pytest.raises(TypeError):
            mock_runner_3._send_actions(MagicMock())

    def test__send_actions_invalid_action(self, mock_runner_3):
        mock_runner_3.act_event.wait.return_value = True
        mock_runner_3.act_queue.get.return_value = {'invalid_agent': 0}
        mock_runner_3.action_fn.transform_action.return_value = {'invalid_agent': 0}
        with pytest.raises(KeyError):
            mock_runner_3._send_actions(MagicMock())

    @patch('eprllib.Env.MultiAgent.EnergyPlusRunner.api')
    def test__send_actions_out_of_bounds(self, mock_api, mock_runner_3):
        mock_runner_3.act_event.wait.return_value = True
        mock_runner_3.act_queue.get.return_value = {'agent1': 100}  # Assuming 100 is out of bounds
        mock_runner_3.action_fn.transform_action.return_value = {'agent1': 100}
        mock_runner_3._send_actions(MagicMock())
        mock_api.exchange.set_actuator_value.assert_called_once()
        # Note: The method doesn't check for out of bounds values, so it will still set the actuator value

    def test__send_actions_simulation_complete(self, mock_runner_3):
        mock_runner_3.simulation_complete = True
        mock_runner_3._send_actions(MagicMock())
        mock_runner_3.act_event.wait.assert_not_called()

    def test__send_actions_simulation_not_initialized(self, mock_runner_3):
        mock_runner_3._init_callback.return_value = False
        mock_runner_3._send_actions(MagicMock())
        mock_runner_3.act_event.wait.assert_not_called()

    @patch('eprllib.Env.MultiAgent.EnergyPlusRunner.api')
    def test__send_actions_successful(self, mock_api, mock_runner_3):
        mock_runner_3.act_event.wait.return_value = True
        mock_runner_3.act_queue.get.return_value = {'agent1': 0}
        mock_runner_3.action_fn.transform_action.return_value = {'agent1': 0}
        mock_runner_3._send_actions(MagicMock())
        mock_api.exchange.set_actuator_value.assert_called_once()

    def test__send_actions_timeout(self, mock_runner_3):
        mock_runner_3.act_event.wait.return_value = False
        with pytest.raises(ValueError):
            mock_runner_3._send_actions(MagicMock())

    def test__send_actions_when_not_initialized_or_simulation_complete(self):
        """
        Test _send_actions when not initialized or simulation is complete.
        Expects the method to return without performing any actions.
        """
        # Mock the necessary objects and configurations
        env_config = {
            'epw_path': 'path/to/weather.epw',
            'epjson_path': 'path/to/model.epjson',
            'output_path': 'path/to/output',
            'ep_terminal_output': False,
            'timeout': 10,
            'agents_config': {},
        }
        obs_queue = MagicMock()
        act_queue = MagicMock()
        infos_queue = MagicMock()
        agent_ids = set(['agent1', 'agent2'])
        thermal_zone_ids = set(['zone1', 'zone2'])
        observation_fn = MagicMock()
        action_fn = MagicMock()

        # Create an instance of EnergyPlusRunner
        runner = EnergyPlusRunner(
            env_config, 1, obs_queue, act_queue, infos_queue,
            agent_ids, thermal_zone_ids, observation_fn, action_fn
        )

        # Mock the _init_callback method to return False
        runner._init_callback = MagicMock(return_value=False)
        
        # Create a mock state_argument
        state_argument = c_void_p()

        # Call the _send_actions method
        runner._send_actions(state_argument)

        # Assert that _init_callback was called
        runner._init_callback.assert_called_once_with(state_argument)

        # Assert that act_event.wait() was not called
        runner.act_event.wait.assert_not_called()

        # Set simulation_complete to True and test again
        runner.simulation_complete = True
        runner._init_callback.reset_mock()

        runner._send_actions(state_argument)

        # Assert that _init_callback was called
        runner._init_callback.assert_called_once_with(state_argument)

        # Assert that act_event.wait() was not called
        runner.act_event.wait.assert_not_called()

    @patch('eprllib.Env.MultiAgent.EnergyPlusRunner.api')
    def test_collect_first_obs_api_error(self, mock_api, mock_energyplus_runner):
        """
        Test _collect_first_obs when the EnergyPlus API raises an error
        """
        mock_api.exchange.get_variable_value.side_effect = RuntimeError("API Error")
        state_argument = c_void_p(1234)  # Mock state argument
        
        with pytest.raises(RuntimeError, match="API Error"):
            mock_energyplus_runner._collect_first_obs(state_argument)

    def test_collect_first_obs_exception_in_collect_obs(self, mock_energyplus_runner):
        """
        Test _collect_first_obs when _collect_obs raises an exception
        """
        mock_energyplus_runner._collect_obs.side_effect = Exception("Mocked exception")
        state_argument = c_void_p(1234)  # Mock state argument
        
        with pytest.raises(Exception, match="Mocked exception"):
            mock_energyplus_runner._collect_first_obs(state_argument)

    def test_collect_first_obs_incorrect_type(self, mock_energyplus_runner):
        """
        Test _collect_first_obs with an incorrect type for state_argument
        """
        with pytest.raises(TypeError):
            mock_energyplus_runner._collect_first_obs("invalid_state")

    def test_collect_first_obs_invalid_state_argument(self, mock_energyplus_runner):
        """
        Test _collect_first_obs with an invalid state_argument (None)
        """
        with pytest.raises(TypeError):
            mock_energyplus_runner._collect_first_obs(None)

    def test_collect_first_obs_second_call(self, mock_energyplus_runner):
        """
        Test _collect_first_obs when called for the second time (first_observation is False)
        """
        mock_energyplus_runner.first_observation = False
        state_argument = c_void_p(1234)  # Mock state argument
        
        mock_energyplus_runner._collect_first_obs(state_argument)
        
        mock_energyplus_runner._collect_obs.assert_not_called()

    @patch('eprllib.Env.MultiAgent.EnergyPlusRunner.api')
    def test_collect_obs_api_error(self, mock_api, mock_runner_2):
        """Test _collect_obs when API throws an unexpected error"""
        mock_api.exchange.get_variable_value.side_effect = RuntimeError("API Error")
        
        with pytest.raises(RuntimeError):
            mock_runner_2._collect_obs(c_void_p())

    @patch('eprllib.Env.MultiAgent.EnergyPlusRunner.api')
    def test_collect_obs_inf_values(self, mock_api, mock_runner_2):
        """Test _collect_obs with Inf values in observation"""
        mock_api.exchange.get_variable_value.return_value = float('inf')
        mock_api.exchange.get_meter_value.return_value = float('inf')
        mock_api.exchange.get_actuator_value.return_value = float('inf')

        with patch('builtins.print') as mock_print:
            mock_runner_2._collect_obs(c_void_p())

        mock_print.assert_called()

    def test_collect_obs_invalid_state_argument(self, mock_runner_2):
        """Test _collect_obs with invalid state_argument"""
        with pytest.raises(AttributeError):
            mock_runner_2._collect_obs(None)

    @patch('eprllib.Env.MultiAgent.EnergyPlusRunner.api')
    def test_collect_obs_missing_required_data(self, mock_api, mock_runner_2):
        """Test _collect_obs with missing required data"""
        mock_api.exchange.get_variable_value.side_effect = [KeyError, 20.0]
        
        with pytest.raises(KeyError):
            mock_runner_2._collect_obs(c_void_p())

    @patch('eprllib.Env.MultiAgent.EnergyPlusRunner.api')
    def test_collect_obs_nan_values(self, mock_api, mock_runner_2):
        """Test _collect_obs with NaN values in observation"""
        mock_api.exchange.get_variable_value.return_value = float('nan')
        mock_api.exchange.get_meter_value.return_value = float('nan')
        mock_api.exchange.get_actuator_value.return_value = float('nan')

        with patch('builtins.print') as mock_print:
            mock_runner_2._collect_obs(c_void_p())

        mock_print.assert_called()

    def test_collect_obs_simulation_complete(self, mock_runner_2):
        """Test _collect_obs when simulation is complete"""
        mock_runner_2.simulation_complete = True
        mock_runner_2._collect_obs(c_void_p())
        mock_runner_2.obs_queue.put.assert_not_called()
        mock_runner_2.obs_event.set.assert_not_called()

    def test_collect_obs_uninitialized(self, mock_runner_2):
        """Test _collect_obs when not initialized"""
        mock_runner_2._init_callback.return_value = False
        mock_runner_2._collect_obs(c_void_p())
        mock_runner_2.obs_queue.put.assert_not_called()
        mock_runner_2.obs_event.set.assert_not_called()

    def test_delete_not_observable_variables_empty_config(self, mock_energyplus_runner_6):
        """Test delete_not_observable_variables with empty config for a category"""
        mock_energyplus_runner_6.env_config['no_observable_variables']['variables_env'] = []
        input_dict = {"var1": 1, "var2": 2}
        result = mock_energyplus_runner_6.delete_not_observable_variables(input_dict, belong_to="variables_env")
        assert result == input_dict, "Should not modify input when no variables are specified for deletion"

    def test_delete_not_observable_variables_empty_input(self, mock_energyplus_runner_6):
        """Test delete_not_observable_variables with empty input"""
        with pytest.raises(ValueError, match="The 'belong_to' argument must be specified."):
            mock_energyplus_runner_6.delete_not_observable_variables({})

    def test_delete_not_observable_variables_incorrect_type(self, mock_energyplus_runner_6):
        """Test delete_not_observable_variables with incorrect input type"""
        with pytest.raises(AttributeError):
            mock_energyplus_runner_6.delete_not_observable_variables("not a dict", belong_to="variables_env")

    def test_delete_not_observable_variables_invalid_belong_to(self, mock_energyplus_runner_6):
        """Test delete_not_observable_variables with invalid belong_to"""
        with pytest.raises(ValueError, match="The 'belong_to' argument must be one of the following:"):
            mock_energyplus_runner_6.delete_not_observable_variables({}, belong_to="invalid")

    def test_delete_not_observable_variables_missing_reference(self, mock_energyplus_runner_6):
        """Test delete_not_observable_variables with missing reference for variables_obj"""
        with pytest.raises(ValueError, match="The 'reference' argument must be specified."):
            mock_energyplus_runner_6.delete_not_observable_variables({}, belong_to="variables_obj")

    def test_delete_not_observable_variables_nonexistent_key(self, mock_energyplus_runner_6):
        """Test delete_not_observable_variables with nonexistent key"""
        input_dict = {"existing_var": 1, "non_existing_var": 2}
        result = mock_energyplus_runner_6.delete_not_observable_variables(input_dict, belong_to="variables_env")
        assert "non_existing_var" in result, "Should not delete keys that don't exist in no_observable_variables"

    def test_delete_not_observable_variables_raises_value_error_when_belong_to_is_none(self):
        """
        Test that delete_not_observable_variables raises a ValueError when belong_to is None.
        """
        # Arrange
        runner = EnergyPlusRunner(
            env_config={},
            episode=1,
            obs_queue=None,
            act_queue=None,
            infos_queue=None,
            _agent_ids=set(),
            _thermal_zone_ids=set(),
            observation_fn=None,
            action_fn=None
        )
        dict_parameters: Dict[str, Any] = {"test_key": "test_value"}

        # Act & Assert
        with pytest.raises(ValueError, match="The 'belong_to' argument must be specified."):
            runner.delete_not_observable_variables(dict_parameters, belong_to=None)

    def test_delete_not_observable_variables_reference_mismatch(self, mock_energyplus_runner_6):
        """Test delete_not_observable_variables with mismatched reference"""
        with pytest.raises(KeyError):
            mock_energyplus_runner_6.delete_not_observable_variables({"obj1": 1}, belong_to="variables_obj", reference="non_existing_agent")

    def test_failed_simulation(self):
        """
        Test that failed() returns True when sim_results is non-zero
        """
        # Create a mock EnergyPlusRunner instance
        runner = EnergyPlusRunner(
            env_config={},
            episode=1,
            obs_queue=None,
            act_queue=None,
            infos_queue=None,
            _agent_ids=set(),
            _thermal_zone_ids=set(),
            observation_fn=None,
            action_fn=None
        )
        
        # Set sim_results to a non-zero value
        runner.sim_results = 1
        
        # Check that failed() returns True
        assert runner.failed() is True

    def test_failed_simulation_results_bool(self):
        """
        Test that failed handles boolean values correctly.
        """
        runner = EnergyPlusRunner({}, 1, None, None, None, set(), set(), None, None)
        runner.sim_results = True
        assert runner.failed() is True

    def test_failed_simulation_results_float(self):
        """
        Test that failed handles float values correctly.
        """
        runner = EnergyPlusRunner({}, 1, None, None, None, set(), set(), None, None)
        runner.sim_results = 1.5
        assert runner.failed() is True

    def test_failed_simulation_results_large_integer(self):
        """
        Test that failed handles large integer values correctly.
        """
        runner = EnergyPlusRunner({}, 1, None, None, None, set(), set(), None, None)
        runner.sim_results = 1000000000
        assert runner.failed() is True

    def test_failed_simulation_results_negative(self):
        """
        Test that failed returns True when sim_results is negative.
        """
        runner = EnergyPlusRunner({}, 1, None, None, None, set(), set(), None, None)
        runner.sim_results = -1
        assert runner.failed() is True

    def test_failed_simulation_results_non_zero(self):
        """
        Test that failed returns True when sim_results is non-zero.
        """
        runner = EnergyPlusRunner({}, 1, None, None, None, set(), set(), None, None)
        runner.sim_results = 1
        assert runner.failed() is True

    def test_failed_simulation_results_none(self):
        """
        Test that failed handles None value correctly.
        """
        runner = EnergyPlusRunner({}, 1, None, None, None, set(), set(), None, None)
        runner.sim_results = None
        with pytest.raises(TypeError):
            runner.failed()

    def test_failed_simulation_results_string(self):
        """
        Test that failed handles string values correctly.
        """
        runner = EnergyPlusRunner({}, 1, None, None, None, set(), set(), None, None)
        runner.sim_results = "0"
        with pytest.raises(TypeError):
            runner.failed()

    def test_failed_simulation_results_zero(self):
        """
        Test that failed returns False when sim_results is zero.
        """
        runner = EnergyPlusRunner({}, 1, None, None, None, set(), set(), None, None)
        runner.sim_results = 0
        assert runner.failed() is False

    @patch('eprllib.Env.MultiAgent.EnergyPlusRunner.api')
    def test_get_actuators_state_1(self, mock_api):
        """
        Test that get_actuators_state returns the correct dictionary of actuator values
        """
        # Arrange
        runner = EnergyPlusRunner(
            env_config={},
            episode=1,
            obs_queue=Mock(),
            act_queue=Mock(),
            infos_queue=Mock(),
            _agent_ids=set(['agent1', 'agent2']),
            _thermal_zone_ids=set(['zone1', 'zone2']),
            observation_fn=Mock(),
            action_fn=Mock()
        )
        runner.actuator_handles = {
            'agent1': 1,
            'agent2': 2
        }
        mock_state_argument = c_void_p(1234)
        mock_api.exchange.get_actuator_value.side_effect = [10.0, 20.0]

        # Act
        result, _ = runner.get_actuators_state(mock_state_argument)

        # Assert
        assert result == {'agent1': 10.0, 'agent2': 20.0}
        mock_api.exchange.get_actuator_value.assert_any_call(mock_state_argument, 1)
        mock_api.exchange.get_actuator_value.assert_any_call(mock_state_argument, 2)
        assert mock_api.exchange.get_actuator_value.call_count == 2

    def test_get_actuators_state_empty_handles(self, mock_runner_7):
        """Test get_actuators_state with empty actuator handles"""
        mock_runner_7.actuator_handles = {}
        result, _ = mock_runner_7.get_actuators_state(c_void_p(1))
        assert result == {}, "Expected empty dictionary when actuator_handles is empty"

    def test_get_actuators_state_exception_handling(self, mock_runner_7, monkeypatch):
        """Test exception handling in get_actuators_state"""
        def mock_get_actuator_value(*args):
            raise Exception("Mocked exception")

        monkeypatch.setattr(api.exchange, 'get_actuator_value', mock_get_actuator_value)

        with pytest.raises(Exception):
            mock_runner_7.get_actuators_state(c_void_p(1))

    def test_get_actuators_state_incorrect_type(self, mock_runner_7):
        """Test get_actuators_state with incorrect input type"""
        with pytest.raises(TypeError):
            mock_runner_7.get_actuators_state("not_a_c_void_p")

    def test_get_actuators_state_invalid_input(self, mock_runner_7):
        """Test get_actuators_state with invalid input"""
        with pytest.raises(TypeError):
            mock_runner_7.get_actuators_state(None)

    def test_get_actuators_state_negative_handle_value(self, mock_runner_7, monkeypatch):
        """Test get_actuators_state with a negative handle value"""
        mock_runner_7.actuator_handles = {'agent1': -1}

        def mock_get_actuator_value(*args):
            return None

        monkeypatch.setattr(api.exchange, 'get_actuator_value', mock_get_actuator_value)

        result, _ = mock_runner_7.get_actuators_state(c_void_p(1))
        assert result == {'agent1': None}, "Expected None value for negative handle"

    def test_get_buiding_properties_2(self):
        """
        Test get_buiding_properties when use_building_properties is True
        """
        # Arrange
        env_config = {
            'use_building_properties': True,
            'building_properties': {
                'Zone1': {'area': 100, 'volume': 300},
                'Zone2': {'area': 150, 'volume': 450}
            }
        }
        runner = EnergyPlusRunner(env_config=env_config, episode=1, obs_queue=None, act_queue=None, 
                                  infos_queue=None, _agent_ids=set(), _thermal_zone_ids={'Zone1', 'Zone2'},
                                  observation_fn=None, action_fn=None)

        # Act
        result_zone1, info_zone1 = runner.get_buiding_properties(thermal_zone='Zone1')
        result_zone2, info_zone2 = runner.get_buiding_properties(thermal_zone='Zone2')

        # Assert
        assert result_zone1 == {'area': 100, 'volume': 300}
        assert info_zone1 == {}
        assert result_zone2 == {'area': 150, 'volume': 450}
        assert info_zone2 == {}

    def test_get_buiding_properties_raises_assertion_error(self):
        """
        Test get_buiding_properties raises AssertionError when thermal_zones_id don't match building_properties keys
        """
        # Arrange
        env_config = {
            'use_building_properties': True,
            'building_properties': {
                'Zone1': {'area': 100, 'volume': 300},
            }
        }
        runner = EnergyPlusRunner(env_config=env_config, episode=1, obs_queue=None, act_queue=None, 
                                  infos_queue=None, _agent_ids=set(), _thermal_zone_ids={'Zone1', 'Zone2'},
                                  observation_fn=None, action_fn=None)

        # Act & Assert
        with pytest.raises(AssertionError):
            runner.get_buiding_properties(thermal_zone='Zone1')

    def test_get_buiding_properties_when_not_using_building_properties(self):
        """
        Test that get_buiding_properties returns empty dictionaries when use_building_properties is False
        """
        # Arrange
        mock_env_config = {'use_building_properties': False}
        runner = EnergyPlusRunner(
            env_config=mock_env_config,
            episode=1,
            obs_queue=MagicMock(),
            act_queue=MagicMock(),
            infos_queue=MagicMock(),
            _agent_ids=set(),
            _thermal_zone_ids=set(),
            observation_fn=MagicMock(),
            action_fn=MagicMock()
        )

        # Act
        result = runner.get_buiding_properties()

        # Assert
        assert result == ({}, {}), "Expected empty dictionaries when use_building_properties is False"

    def test_get_building_properties_empty_input(self):
        """
        Test get_buiding_properties with empty input.
        """
        runner = EnergyPlusRunner({}, 1, None, None, None, set(), set(), None, None)
        runner.env_config = {'use_building_properties': True, 'building_properties': {}}
        runner._thermal_zone_ids = set(['Zone1'])
        
        with pytest.raises(AssertionError):
            runner.get_buiding_properties()

    def test_get_building_properties_incorrect_type(self):
        """
        Test get_buiding_properties with incorrect input type.
        """
        runner = EnergyPlusRunner({}, 1, None, None, None, set(), set(['Zone1']), None, None)
        runner.env_config = {'use_building_properties': True, 'building_properties': {'Zone1': {}}}
        
        with pytest.raises(TypeError):
            runner.get_buiding_properties(123)

    def test_get_building_properties_invalid_thermal_zone(self):
        """
        Test get_buiding_properties with invalid thermal zone.
        """
        runner = EnergyPlusRunner({}, 1, None, None, None, set(), set(['Zone1']), None, None)
        runner.env_config = {'use_building_properties': True, 'building_properties': {'Zone1': {}}}
        
        with pytest.raises(KeyError):
            runner.get_buiding_properties('InvalidZone')

    def test_get_building_properties_missing_thermal_zone(self):
        """
        Test get_buiding_properties when a thermal zone is missing from building_properties.
        """
        runner = EnergyPlusRunner({}, 1, None, None, None, set(), set(['Zone1', 'Zone2']), None, None)
        runner.env_config = {'use_building_properties': True, 'building_properties': {'Zone1': {}}}
        
        with pytest.raises(AssertionError):
            runner.get_buiding_properties('Zone1')

    def test_get_building_properties_use_building_properties_false(self):
        """
        Test get_buiding_properties when use_building_properties is False.
        """
        runner = EnergyPlusRunner({}, 1, None, None, None, set(), set(['Zone1']), None, None)
        runner.env_config = {'use_building_properties': False, 'building_properties': {'Zone1': {}}}
        
        result, _ = runner.get_buiding_properties('Zone1')
        assert result == {}, "Should return empty dict when use_building_properties is False"

    def test_get_meters_state_agent_is_none(self):
        """
        Test that get_meters_state raises a ValueError when agent is None.
        """
        # Create a mock EnergyPlusRunner instance
        runner = MagicMock(spec=EnergyPlusRunner)
        
        # Create a mock state_argument
        state_argument = MagicMock(spec=c_void_p)

        # Call the method with agent=None
        with pytest.raises(ValueError, match="The agent must be defined."):
            EnergyPlusRunner.get_meters_state(runner, state_argument, agent=None)

    def test_get_meters_state_agent_not_defined(self, mock_runner_6):
        """Test get_meters_state when agent is not defined"""
        with pytest.raises(ValueError, match="The agent must be defined."):
            mock_runner_6.get_meters_state(c_void_p(), None)

    def test_get_meters_state_all_meters_not_observable(self, mock_runner_6):
        """Test get_meters_state when all meters are not observable"""
        mock_runner_6.env_config['no_observable_variables']['meters']['agent1'] = ['Electricity:Facility', 'Gas:Facility']
        variables, infos = mock_runner_6.get_meters_state(c_void_p(), 'agent1')
        assert variables == {}
        assert infos == {}

    def test_get_meters_state_empty_meter_handles(self, mock_runner_6):
        """Test get_meters_state with empty meter handles"""
        mock_runner_6.meter_handles['agent1'] = {}
        variables, infos = mock_runner_6.get_meters_state(c_void_p(), 'agent1')
        assert variables == {}
        assert infos == {}

    def test_get_meters_state_invalid_meter_handle(self, mock_runner_6):
        """Test get_meters_state with invalid meter handle"""
        mock_runner_6.meter_handles['agent1']['Invalid:Meter'] = -1
        with patch('eprllib.Env.MultiAgent.EnergyPlusRunner.api.exchange.get_meter_value', side_effect=Exception("Invalid meter handle")):
            with pytest.raises(Exception, match="Invalid meter handle"):
                mock_runner_6.get_meters_state(c_void_p(), 'agent1')

    def test_get_meters_state_invalid_state_argument(self, mock_runner_6):
        """Test get_meters_state with invalid state_argument"""
        with patch('eprllib.Env.MultiAgent.EnergyPlusRunner.api.exchange.get_meter_value', side_effect=Exception("Invalid state")):
            with pytest.raises(Exception, match="Invalid state"):
                mock_runner_6.get_meters_state(c_void_p(), 'agent1')

    def test_get_meters_state_nonexistent_agent(self, mock_runner_6):
        """Test get_meters_state with a nonexistent agent"""
        with pytest.raises(KeyError):
            mock_runner_6.get_meters_state(c_void_p(), 'nonexistent_agent')

    @patch('eprllib.Env.MultiAgent.EnergyPlusRunner.api.exchange.get_meter_value')
    def test_get_meters_state_with_valid_agent(self, mock_get_meter_value):
        """
        Test get_meters_state with a valid agent.
        """
        # Create a mock EnergyPlusRunner instance
        runner = MagicMock(spec=EnergyPlusRunner)
        
        # Set up mock meter handles
        runner.meter_handles = {
            'agent1': {
                'meter1': 1,
                'meter2': 2
            }
        }

        # Set up mock return values for get_meter_value
        mock_get_meter_value.side_effect = [10.0, 20.0]

        # Create a mock state_argument
        state_argument = MagicMock(spec=c_void_p)

        # Mock the update_infos and delete_not_observable_variables methods
        runner.update_infos.return_value = {'info1': 'value1'}
        runner.delete_not_observable_variables.return_value = {'meter1': 10.0}

        # Call the method
        variables, infos = EnergyPlusRunner.get_meters_state(runner, state_argument, agent='agent1')

        # Assert the results
        assert variables == {'meter1': 10.0}
        assert infos == {'info1': 'value1'}

        # Verify method calls
        mock_get_meter_value.assert_any_call(state_argument, 1)
        mock_get_meter_value.assert_any_call(state_argument, 2)
        runner.update_infos.assert_called_once_with({'meter1': 10.0, 'meter2': 20.0}, 'meters', 'agent1')
        runner.delete_not_observable_variables.assert_called_once_with({'meter1': 10.0, 'meter2': 20.0}, 'meters', 'agent1')

    @patch('eprllib.Env.MultiAgent.EnergyPlusRunner.api')
    def test_get_object_variables_state_2(self, mock_api, mock_energyplus_runner_4):
        """
        Test get_object_variables_state when agent is not None
        """
        # Arrange
        state_argument = c_void_p(1234)
        agent = 'agent1'
        mock_api.exchange.get_variable_value.side_effect = [10.0, 20.0]

        # Act
        variables, infos = EnergyPlusRunner.get_object_variables_state(mock_energyplus_runner_4, state_argument, agent)

        # Assert
        assert isinstance(variables, dict)
        assert isinstance(infos, dict)
        assert variables == {'var1': 10.0}
        assert infos == {'var1': 10.0}
        mock_api.exchange.get_variable_value.assert_any_call(state_argument, 1)
        mock_api.exchange.get_variable_value.assert_any_call(state_argument, 2)
        assert mock_api.exchange.get_variable_value.call_count == 2

        # Check that var2 is not in variables due to no_observable_variables
        assert 'var2' not in variables

    def test_get_object_variables_state_agent_none(self, runner):
        """
        Test get_object_variables_state when agent is None
        """
        state_argument = c_void_p()

        with pytest.raises(ValueError, match="The agent must be defined."):
            runner.get_object_variables_state(state_argument)

    def test_get_object_variables_state_empty_handles(self, mock_runner_5):
        """Test get_object_variables_state with empty handles"""
        mock_runner_5.handle_object_variables['agent1'] = {}
        variables, infos = mock_runner_5.get_object_variables_state(c_void_p(), "agent1")
        assert variables == {}
        assert infos == {}

    def test_get_object_variables_state_exception_in_get_variable_value(self, mock_runner_5, monkeypatch):
        """Test get_object_variables_state when get_variable_value raises an exception"""
        def mock_get_variable_value(*args):
            raise Exception("Test exception")

        monkeypatch.setattr("pyenergyplus.api.exchange.get_variable_value", mock_get_variable_value)
        
        with pytest.raises(Exception, match="Test exception"):
            mock_runner_5.get_object_variables_state(c_void_p(), "agent1")

    def test_get_object_variables_state_incorrect_type(self, mock_runner_5):
        """Test get_object_variables_state with incorrect input types"""
        with pytest.raises(AttributeError):
            mock_runner_5.get_object_variables_state("not_a_c_void_p", "agent1")

    def test_get_object_variables_state_invalid_agent(self, mock_runner_5):
        """Test get_object_variables_state with an invalid agent"""
        with pytest.raises(KeyError):
            mock_runner_5.get_object_variables_state(c_void_p(), "invalid_agent")

    def test_get_object_variables_state_no_agent(self, mock_runner_5):
        """Test get_object_variables_state with no agent specified"""
        with pytest.raises(ValueError, match="The agent must be defined."):
            mock_runner_5.get_object_variables_state(c_void_p(), None)

    @patch('eprllib.Env.MultiAgent.EnergyPlusRunner.api.exchange.get_variable_value')
    def test_get_object_variables_state_no_variables(self, mock_get_variable_value, runner):
        """
        Test get_object_variables_state when there are no variables for the agent
        """
        state_argument = c_void_p()
        agent = 'agent2'

        runner.handle_object_variables = {'agent2': {}}

        variables, infos = runner.get_object_variables_state(state_argument, agent)

        assert variables == {}
        assert infos == {}
        assert mock_get_variable_value.call_count == 0

    @patch('eprllib.Env.MultiAgent.EnergyPlusRunner.api.exchange.get_variable_value')
    def test_get_object_variables_state_valid_agent(self, mock_get_variable_value, runner):
        """
        Test get_object_variables_state with a valid agent
        """
        state_argument = c_void_p()
        agent = 'agent1'
        mock_get_variable_value.side_effect = [10, 20]

        runner.handle_object_variables = {'agent1': {'var1': 1, 'var2': 2}}

        variables, infos = runner.get_object_variables_state(state_argument, agent)

        assert variables == {'var1': 10}
        assert infos == {'var1': 10}
        assert mock_get_variable_value.call_count == 2

    def test_get_simulation_parameters_values_1(self):
        """
        Test the get_simulation_parameters_values method when some simulation parameters are set to True.
        """
        # Mock the EnergyPlusRunner instance
        runner = MagicMock(spec=EnergyPlusRunner)
        
        # Set up the environment configuration
        runner.env_config = {
            'simulation_parameters': {
                'actual_date_time': True,
                'current_time': True,
                'day_of_year': False,
                'hour': True
            }
        }
        
        # Mock the api.exchange methods
        with patch('eprllib.Env.MultiAgent.EnergyPlusRunner.api.exchange') as mock_exchange:
            mock_exchange.actual_date_time.return_value = 20230601120000
            mock_exchange.current_time.return_value = 12.5
            mock_exchange.hour.return_value = 12
            
            # Call the method under test
            variables, infos = EnergyPlusRunner.get_simulation_parameters_values(runner, c_void_p())
        
        # Assert the expected results
        expected_variables = {
            'actual_date_time': 20230601120000,
            'current_time': 12.5,
            'hour': 12
        }
        assert variables == expected_variables
        assert infos == {}  # Assuming no infos are set in this case
        
        # Verify that the methods were called
        mock_exchange.actual_date_time.assert_called_once()
        mock_exchange.current_time.assert_called_once()
        mock_exchange.hour.assert_called_once()
        mock_exchange.day_of_year.assert_not_called()

    def test_get_simulation_parameters_values_2(self):
        """
        Test get_simulation_parameters_values when no parameters are selected.
        """
        # Mock the necessary objects and methods
        mock_state = MagicMock(spec=c_void_p)
        mock_api = MagicMock()
        
        # Create a mock EnergyPlusRunner instance
        runner = EnergyPlusRunner(
            env_config={
                'simulation_parameters': {
                    'hour': False,
                    'day_of_year': False,
                    'is_raining': False
                }
            },
            episode=1,
            obs_queue=MagicMock(),
            act_queue=MagicMock(),
            infos_queue=MagicMock(),
            _agent_ids=set(),
            _thermal_zone_ids=set(),
            observation_fn=MagicMock(),
            action_fn=MagicMock()
        )
        
        # Patch the api object in the EnergyPlusRunner
        with patch.object(runner, 'api', mock_api):
            # Call the method under test
            variables, infos = runner.get_simulation_parameters_values(mock_state)
        
        # Assert that the returned dictionaries are empty
        assert variables == {}
        assert infos == {}
        
        # Verify that no API calls were made
        mock_api.exchange.hour.assert_not_called()
        mock_api.exchange.day_of_year.assert_not_called()
        mock_api.exchange.is_raining.assert_not_called()

    @patch('eprllib.Env.MultiAgent.EnergyPlusRunner.api.exchange')
    def test_get_simulation_parameters_values_api_error(self, mock_api_exchange, mock_energyplus_runner_5):
        """
        Test get_simulation_parameters_values when API calls raise exceptions
        """
        mock_api_exchange.actual_date_time.side_effect = RuntimeError("API Error")
        
        with pytest.raises(RuntimeError):
            mock_energyplus_runner_5.get_simulation_parameters_values(c_void_p())

    def test_get_simulation_parameters_values_empty_config(self, mock_energyplus_runner_5):
        """
        Test get_simulation_parameters_values with an empty simulation_parameters config
        """
        mock_energyplus_runner_5.env_config['simulation_parameters'] = {}
        result, infos = mock_energyplus_runner_5.get_simulation_parameters_values(c_void_p())
        assert result == {}
        assert infos == {}

    @patch('eprllib.Env.MultiAgent.EnergyPlusRunner.api.exchange')
    def test_get_simulation_parameters_values_incorrect_return_type(self, mock_api_exchange, mock_energyplus_runner_5):
        """
        Test get_simulation_parameters_values when API returns incorrect type
        """
        mock_api_exchange.actual_date_time.return_value = "Not a number"
        mock_api_exchange.current_time.return_value = 10.5
        
        result, _ = mock_energyplus_runner_5.get_simulation_parameters_values(c_void_p())
        assert isinstance(result['actual_date_time'], str)
        assert isinstance(result['current_time'], float)

    @patch('eprllib.Env.MultiAgent.EnergyPlusRunner.api.exchange')
    def test_get_simulation_parameters_values_invalid_parameter(self, mock_api_exchange, mock_energyplus_runner_5):
        """
        Test get_simulation_parameters_values with an invalid parameter in the config
        """
        mock_energyplus_runner_5.env_config['simulation_parameters']['invalid_param'] = True
        
        with pytest.raises(KeyError):
            mock_energyplus_runner_5.get_simulation_parameters_values(c_void_p())

    def test_get_simulation_parameters_values_invalid_state_argument(self, mock_energyplus_runner_5):
        """
        Test get_simulation_parameters_values with an invalid state_argument
        """
        with pytest.raises(AttributeError):
            mock_energyplus_runner_5.get_simulation_parameters_values(None)

    @patch('eprllib.Env.MultiAgent.EnergyPlusRunner.api.exchange')
    def test_get_simulation_parameters_values_no_observable_variables(self, mock_api_exchange, mock_energyplus_runner_5):
        """
        Test get_simulation_parameters_values with no_observable_variables set
        """
        mock_api_exchange.actual_date_time.return_value = 20210101
        mock_api_exchange.current_time.return_value = 12.5
        
        mock_energyplus_runner_5.env_config['no_observable_variables']['simulation_parameters'] = ['current_time']
        
        result, infos = mock_energyplus_runner_5.get_simulation_parameters_values(c_void_p())
        assert 'actual_date_time' in result
        assert 'current_time' not in result
        assert 'current_time' in infos

    def test_get_site_variables_state_1(self, mock_energyplus_runner_2):
        """
        Test get_site_variables_state method returns correct variables and infos
        """
        # Arrange
        state_argument = c_void_p(1234)  # Mock state argument
        expected_variables = {'OutdoorAirTemperature': 25.0}
        expected_infos = {'OutdoorAirTemperature': 25.0}

        # Mock the api.exchange.get_variable_value method
        with patch('eprllib.Env.MultiAgent.EnergyPlusRunner.api.exchange.get_variable_value', return_value=25.0):
            # Act
            variables, infos = EnergyPlusRunner.get_site_variables_state(mock_energyplus_runner_2, state_argument)

        # Assert
        assert variables == expected_variables
        assert infos == expected_infos
        assert 'OutdoorRelativeHumidity' not in variables

    @patch('eprllib.Env.MultiAgent.EnergyPlusRunner.api.exchange.get_variable_value')
    def test_get_site_variables_state_api_exception(self, mock_get_variable_value, mock_energyplusrunner):
        """Test get_site_variables_state when API raises an exception"""
        mock_get_variable_value.side_effect = Exception("API Error")
        with pytest.raises(Exception, match="API Error"):
            mock_energyplusrunner.get_site_variables_state(c_void_p())

    def test_get_site_variables_state_empty_handles(self, mock_energyplusrunner):
        """Test get_site_variables_state with empty handle_site_variables"""
        mock_energyplusrunner.handle_site_variables = {}
        result, infos = mock_energyplusrunner.get_site_variables_state(c_void_p())
        assert result == {}
        assert infos == {}

    def test_get_site_variables_state_invalid_input(self, mock_energyplusrunner):
        """Test get_site_variables_state with invalid input"""
        with pytest.raises(TypeError):
            mock_energyplusrunner.get_site_variables_state(None)

    @patch('eprllib.Env.MultiAgent.EnergyPlusRunner.api.exchange.get_variable_value')
    def test_get_site_variables_state_invalid_variable_value(self, mock_get_variable_value, mock_energyplusrunner):
        """Test get_site_variables_state when API returns invalid variable value"""
        mock_get_variable_value.return_value = float('nan')
        result, infos = mock_energyplusrunner.get_site_variables_state(c_void_p())
        assert 'var1' not in result
        assert 'var2' not in result

    def test_get_site_variables_state_missing_config(self, mock_energyplusrunner):
        """Test get_site_variables_state with missing configuration"""
        del mock_energyplusrunner.env_config['variables_env']
        with pytest.raises(KeyError):
            mock_energyplusrunner.get_site_variables_state(c_void_p())

    def test_get_static_variables_state_all_variables_not_observable(self, runner_2):
        """Test get_static_variables_state when all variables are not observable"""
        runner_2.env_config['no_observable_variables']['static_variables']['Zone1'] = ['Variable1', 'Variable2']
        runner_2.handle_static_variables = {'Zone1': {'Variable1': 1, 'Variable2': 2}}
        variables, infos = runner_2.get_static_variables_state(c_void_p(), 'Zone1')
        assert variables == {}
        assert infos == {}

    def test_get_static_variables_state_empty_variables(self, runner_2):
        """Test get_static_variables_state with empty variables"""
        runner_2.handle_static_variables = {'Zone1': {}}
        variables, infos = runner_2.get_static_variables_state(c_void_p(), 'Zone1')
        assert variables == {}
        assert infos == {}

    def test_get_static_variables_state_invalid_thermal_zone(self, runner_2):
        """Test get_static_variables_state with invalid thermal_zone"""
        with pytest.raises(ValueError):
            runner_2.get_static_variables_state(c_void_p(), 'InvalidZone')

    def test_get_static_variables_state_no_variables(self):
        """
        Test get_static_variables_state when there are no static variables for the thermal zone.
        """
        # Mock the EnergyPlusRunner instance
        runner = MagicMock(spec=EnergyPlusRunner)
        runner.handle_static_variables = {'Zone1': {}}
        runner.env_config = {
            'infos_variables': {'static_variables': {'Zone1': []}},
            'no_observable_variables': {'static_variables': {'Zone1': []}}
        }

        # Mock the state_argument
        state_argument = c_void_p()

        # Call the method
        variables, infos = EnergyPlusRunner.get_static_variables_state(runner, state_argument, thermal_zone='Zone1')

        # Assert the results
        assert variables == {}
        assert infos == {}

    def test_get_static_variables_state_null_state_argument(self, runner_2):
        """Test get_static_variables_state with null state_argument"""
        with pytest.raises(ValueError):
            runner_2.get_static_variables_state(None, 'Zone1')

    def test_get_static_variables_state_null_thermal_zone(self, runner_2):
        """Test get_static_variables_state with null thermal_zone"""
        with pytest.raises(ValueError):
            runner_2.get_static_variables_state(c_void_p(), None)

    def test_get_static_variables_state_thermal_zone_none(self):
        """
        Test get_static_variables_state when thermal_zone is None.
        It should raise a ValueError.
        """
        # Mock the EnergyPlusRunner instance
        runner = MagicMock(spec=EnergyPlusRunner)
        
        # Mock the state_argument
        state_argument = c_void_p()

        # Call the method and expect a ValueError
        with pytest.raises(ValueError, match="The thermal zone must be defined."):
            EnergyPlusRunner.get_static_variables_state(runner, state_argument, thermal_zone=None)

    def test_get_static_variables_state_uninitialized_handles(self, runner_2):
        """Test get_static_variables_state with uninitialized handles"""
        runner_2.handle_static_variables = {}
        with pytest.raises(KeyError):
            runner_2.get_static_variables_state(c_void_p(), 'Zone1')

    @patch('eprllib.Env.MultiAgent.EnergyPlusRunner.api.exchange.get_internal_variable_value')
    def test_get_static_variables_state_with_thermal_zone(self, mock_get_internal_variable_value):
        """
        Test get_static_variables_state with a valid thermal_zone.
        """
        # Mock the EnergyPlusRunner instance
        runner = MagicMock(spec=EnergyPlusRunner)
        runner.handle_static_variables = {
            'Zone1': {'var1': 'handle1', 'var2': 'handle2'}
        }
        runner.env_config = {
            'infos_variables': {'static_variables': {'Zone1': ['var1']}},
            'no_observable_variables': {'static_variables': {'Zone1': ['var2']}}
        }

        # Mock the state_argument
        state_argument = c_void_p()

        # Mock the get_internal_variable_value function
        mock_get_internal_variable_value.side_effect = [10, 20]

        # Call the method
        variables, infos = EnergyPlusRunner.get_static_variables_state(runner, state_argument, thermal_zone='Zone1')

        # Assert the results
        assert variables == {'var1': 10}
        assert infos == {'var1': 10}

        # Verify that get_internal_variable_value was called correctly
        mock_get_internal_variable_value.assert_any_call(state_argument, 'handle1')
        mock_get_internal_variable_value.assert_any_call(state_argument, 'handle2')

    def test_get_thermalzone_variables_state_2(self):
        """
        Test get_thermalzone_variables_state when thermal_zone is provided.
        """
        # Mock EnergyPlusRunner instance
        runner = Mock(spec=EnergyPlusRunner)
        runner.env_config = {
            'variables_thz': ['Temperature', 'Humidity'],
            'infos_variables': {'variables_thz': {'Zone1': ['Temperature']}},
            'no_observable_variables': {'variables_thz': {'Zone1': ['Humidity']}}
        }
        runner.handle_thermalzone_variables = {
            'Zone1': {
                'Temperature': 1,
                'Humidity': 2
            }
        }

        # Mock api.exchange.get_variable_value
        with patch('eprllib.Env.MultiAgent.EnergyPlusRunner.api.exchange.get_variable_value') as mock_get_value:
            mock_get_value.side_effect = [20.0, 50.0]

            # Call the method
            state_argument = c_void_p()
            thermal_zone = 'Zone1'
            variables, infos = EnergyPlusRunner.get_thermalzone_variables_state(runner, state_argument, thermal_zone)

        # Assertions
        assert isinstance(variables, dict)
        assert isinstance(infos, dict)
        assert variables == {'Temperature': 20.0}
        assert infos == {'Temperature': 20.0}
        mock_get_value.assert_called_with(state_argument, 2)

    @patch('eprllib.Env.MultiAgent.EnergyPlusRunner.api.exchange.get_variable_value')
    def test_get_thermalzone_variables_state_empty_result(self, mock_get_variable_value, mock_runner_4):
        """Test get_thermalzone_variables_state when no variables are returned"""
        mock_get_variable_value.return_value = None
        variables, infos = mock_runner_4.get_thermalzone_variables_state(c_void_p(), "LIVING ZONE")
        assert variables == {}
        assert infos == {'Zone Mean Air Temperature': None}

    @patch('eprllib.Env.MultiAgent.EnergyPlusRunner.api.exchange.get_variable_value')
    def test_get_thermalzone_variables_state_filtered_variables(self, mock_get_variable_value, mock_runner_4):
        """Test get_thermalzone_variables_state with filtered variables"""
        mock_get_variable_value.side_effect = [25.0, 50.0]
        variables, infos = mock_runner_4.get_thermalzone_variables_state(c_void_p(), "LIVING ZONE")
        assert variables == {'Zone Mean Air Temperature': 25.0}
        assert infos == {'Zone Mean Air Temperature': 25.0}
        assert 'Zone Air Relative Humidity' not in variables

    @patch('eprllib.Env.MultiAgent.EnergyPlusRunner.api.exchange.get_variable_value')
    def test_get_thermalzone_variables_state_invalid_handle(self, mock_get_variable_value, mock_runner_4):
        """Test get_thermalzone_variables_state with an invalid handle"""
        mock_get_variable_value.side_effect = RuntimeError("Invalid handle")
        with pytest.raises(RuntimeError, match="Invalid handle"):
            mock_runner_4.get_thermalzone_variables_state(c_void_p(), "LIVING ZONE")

    def test_get_thermalzone_variables_state_invalid_input(self, mock_runner_4):
        """Test get_thermalzone_variables_state with invalid input"""
        with pytest.raises(ValueError, match="The thermal zone must be defined."):
            mock_runner_4.get_thermalzone_variables_state(c_void_p(), None)

    def test_get_thermalzone_variables_state_nonexistent_zone(self, mock_runner_4):
        """Test get_thermalzone_variables_state with a non-existent thermal zone"""
        with pytest.raises(KeyError):
            mock_runner_4.get_thermalzone_variables_state(c_void_p(), "NON_EXISTENT_ZONE")

    def test_get_thermalzone_variables_state_with_none_thermal_zone(self, mock_energyplus_runner_3):
        """
        Test get_thermalzone_variables_state method when thermal_zone is None.
        """
        state_argument = c_void_p()
        
        with pytest.raises(ValueError) as exc_info:
            mock_energyplus_runner_3.get_thermalzone_variables_state(state_argument, None)
        
        assert str(exc_info.value) == "The thermal zone must be defined."

    @patch('eprllib.Env.MultiAgent.EnergyPlusRunner.api.exchange.get_variable_value')
    def test_get_thermalzone_variables_state_with_valid_thermal_zone(self, mock_get_variable_value, mock_energyplus_runner_3):
        """
        Test get_thermalzone_variables_state method with a valid thermal_zone.
        """
        state_argument = c_void_p()
        thermal_zone = 'ThermalZone1'
        mock_get_variable_value.side_effect = [20.5, 60.0]  # Mocked values for temperature and humidity

        variables, infos = mock_energyplus_runner_3.get_thermalzone_variables_state(state_argument, thermal_zone)

        assert isinstance(variables, dict)
        assert isinstance(infos, dict)
        assert 'Zone Mean Air Temperature' in variables
        assert 'Zone Air Relative Humidity' not in variables  # Should be removed as not observable
        assert variables['Zone Mean Air Temperature'] == 20.5
        assert 'Zone Mean Air Temperature' in infos
        assert infos['Zone Mean Air Temperature'] == 20.5

        mock_get_variable_value.assert_called()

    def test_get_weather_prediction_3(self):
        """
        Test get_weather_prediction when use_one_day_weather_prediction is True,
        prediction_hour < 24, and prediction_variables[key] is True
        """
        # Mock the EnergyPlusRunner instance
        runner = Mock(spec=EnergyPlusRunner)
        runner.env_config = {
            'use_one_day_weather_prediction': True,
            'prediction_hours': 5,
            'prediction_variables': {
                'outdoor_dry_bulb': True,
                'outdoor_relative_humidity': False
            }
        }

        # Mock the api.exchange methods
        with patch('eprllib.Env.MultiAgent.EnergyPlusRunner.api.exchange') as mock_exchange:
            mock_exchange.hour.return_value = 10
            mock_exchange.zone_time_step_number.return_value = 1
            mock_exchange.today_weather_outdoor_dry_bulb_at_time.return_value = 25.5

            # Call the method
            state_argument = c_void_p()
            result, _ = EnergyPlusRunner.get_weather_prediction(runner, state_argument)

        # Assert the results
        assert isinstance(result, dict)
        assert len(result) == 5  # 5 prediction hours
        assert 'today_weather_outdoor_dry_bulb_at_time_11' in result
        assert 'today_weather_outdoor_dry_bulb_at_time_12' in result
        assert 'today_weather_outdoor_dry_bulb_at_time_13' in result
        assert 'today_weather_outdoor_dry_bulb_at_time_14' in result
        assert 'today_weather_outdoor_dry_bulb_at_time_15' in result
        assert all(value == 25.5 for value in result.values())
        assert 'today_weather_outdoor_relative_humidity_at_time_11' not in result

        # Verify that the mocked methods were called correctly
        mock_exchange.hour.assert_called_once_with(state_argument)
        mock_exchange.zone_time_step_number.assert_called_once_with(state_argument)
        assert mock_exchange.today_weather_outdoor_dry_bulb_at_time.call_count == 5

    def test_get_weather_prediction_4(self):
        """
        Test get_weather_prediction when use_one_day_weather_prediction is True,
        prediction_hour < 24, and prediction_variables[key] is False.
        """
        # Mock the EnergyPlusRunner instance
        runner = MagicMock(spec=EnergyPlusRunner)
        runner.env_config = {
            'use_one_day_weather_prediction': True,
            'prediction_hours': 5,
            'prediction_variables': {'outdoor_dry_bulb': False}
        }

        # Mock the api.exchange methods
        with patch('eprllib.Env.MultiAgent.EnergyPlusRunner.api.exchange') as mock_exchange:
            mock_exchange.hour.return_value = 10
            mock_exchange.zone_time_step_number.return_value = 1
            mock_exchange.today_weather_outdoor_dry_bulb_at_time.return_value = 25.0

            # Call the method
            result, _ = EnergyPlusRunner.get_weather_prediction(runner, c_void_p())

        # Assert the result
        assert result == {}, "Expected an empty dictionary when prediction_variables[key] is False"

        # Verify that the mocked methods were called
        mock_exchange.hour.assert_called_once()
        mock_exchange.zone_time_step_number.assert_called_once()
        mock_exchange.today_weather_outdoor_dry_bulb_at_time.assert_not_called()

    def test_get_weather_prediction_5(self, mock_api):
        """
        Test get_weather_prediction when use_one_day_weather_prediction is True,
        prediction_hour >= 24, and some prediction variables are True.
        """
        # Arrange
        env_config = {
            'use_one_day_weather_prediction': True,
            'prediction_hours': 25,
            'prediction_variables': {
                'outdoor_dry_bulb': True,
                'outdoor_relative_humidity': False,
                'wind_speed': True
            }
        }
        runner = EnergyPlusRunner(env_config, 1, Mock(), Mock(), Mock(), set(), set(), Mock(), Mock())
        
        state_argument = c_void_p()
        mock_api.exchange.hour.return_value = 23
        mock_api.exchange.zone_time_step_number.return_value = 1
        
        mock_api.exchange.tomorrow_weather_outdoor_dry_bulb_at_time.return_value = 25.0
        mock_api.exchange.tomorrow_weather_wind_speed_at_time.return_value = 5.0

        # Act
        result, _ = runner.get_weather_prediction(state_argument)

        # Assert
        assert 'tomorrow_weather_outdoor_dry_bulb_at_time_0' in result
        assert 'tomorrow_weather_wind_speed_at_time_0' in result
        assert 'tomorrow_weather_outdoor_relative_humidity_at_time_0' not in result
        assert result['tomorrow_weather_outdoor_dry_bulb_at_time_0'] == 25.0
        assert result['tomorrow_weather_wind_speed_at_time_0'] == 5.0
        
        mock_api.exchange.hour.assert_called_once_with(state_argument)
        mock_api.exchange.zone_time_step_number.assert_called_once_with(state_argument)
        mock_api.exchange.tomorrow_weather_outdoor_dry_bulb_at_time.assert_called_once_with(state_argument, 23, 1)
        mock_api.exchange.tomorrow_weather_wind_speed_at_time.assert_called_once_with(state_argument, 23, 1)

    @patch('eprllib.Env.MultiAgent.EnergyPlusRunner.api.exchange')
    def test_get_weather_prediction_api_exception(self, mock_exchange, mock_runner_9):
        """Test exception handling from API calls"""
        mock_exchange.hour.side_effect = Exception("API Error")
        state_argument = c_void_p()
        with pytest.raises(Exception, match="API Error"):
            mock_runner_9.get_weather_prediction(state_argument)

    def test_get_weather_prediction_disabled(self, mock_runner_9):
        """Test when weather prediction is disabled"""
        mock_runner_9.env_config['use_one_day_weather_prediction'] = False
        state_argument = c_void_p()
        result, _ = mock_runner_9.get_weather_prediction(state_argument)
        assert result == {}, "Weather prediction should return empty dict when disabled"

    @patch('eprllib.Env.MultiAgent.EnergyPlusRunner.api.exchange')
    def test_get_weather_prediction_empty_prediction_variables(self, mock_exchange, mock_runner_9):
        """Test with empty prediction variables"""
        mock_runner_9.env_config['prediction_variables'] = {}
        mock_exchange.hour.return_value = 12
        mock_exchange.zone_time_step_number.return_value = 1
        state_argument = c_void_p()
        result, _ = mock_runner_9.get_weather_prediction(state_argument)
        assert result == {}, "Weather prediction should return empty dict with no prediction variables"

    @patch('eprllib.Env.MultiAgent.EnergyPlusRunner.api.exchange')
    def test_get_weather_prediction_incorrect_type(self, mock_exchange, mock_runner_9):
        """Test with incorrect input type"""
        mock_exchange.hour.return_value = "12"  # String instead of int
        state_argument = c_void_p()
        with pytest.raises(TypeError):
            mock_runner_9.get_weather_prediction(state_argument)

    @patch('eprllib.Env.MultiAgent.EnergyPlusRunner.api.exchange')
    def test_get_weather_prediction_invalid_hour(self, mock_exchange, mock_runner_9):
        """Test with invalid hour input"""
        mock_exchange.hour.return_value = 25  # Invalid hour
        state_argument = c_void_p()
        with pytest.raises(KeyError):
            mock_runner_9.get_weather_prediction(state_argument)

    def test_get_weather_prediction_when_not_using_one_day_prediction(self):
        """
        Test get_weather_prediction when use_one_day_weather_prediction is False.
        It should return an empty dictionary.
        """
        # Arrange
        mock_env_config = {'use_one_day_weather_prediction': False}
        runner = EnergyPlusRunner(
            env_config=mock_env_config,
            episode=1,
            obs_queue=MagicMock(),
            act_queue=MagicMock(),
            infos_queue=MagicMock(),
            _agent_ids=set(),
            _thermal_zone_ids=set(),
            observation_fn=MagicMock(),
            action_fn=MagicMock()
        )
        mock_state_argument = c_void_p()

        # Act
        result = runner.get_weather_prediction(mock_state_argument)

        # Assert
        assert result == ({}, {}), "Expected an empty tuple of dictionaries when use_one_day_weather_prediction is False"

    def test_get_weather_prediction_when_not_using_one_day_prediction_2(self):
        """
        Test that get_weather_prediction returns an empty dictionary when use_one_day_weather_prediction is False.
        """
        # Arrange
        mock_env_config = {'use_one_day_weather_prediction': False}
        runner = EnergyPlusRunner(
            env_config=mock_env_config,
            episode=1,
            obs_queue=MagicMock(),
            act_queue=MagicMock(),
            infos_queue=MagicMock(),
            _agent_ids=set(),
            _thermal_zone_ids=set(),
            observation_fn=MagicMock(),
            action_fn=MagicMock()
        )
        mock_state_argument = MagicMock()

        # Act
        result = runner.get_weather_prediction(mock_state_argument)

        # Assert
        assert result == ({}, {}), "Expected an empty tuple of dictionaries when use_one_day_weather_prediction is False"

    @patch('eprllib.Env.MultiAgent.EnergyPlusRunner.api.exchange')
    def test_get_weather_prediction_zero_prediction_hours(self, mock_exchange, mock_runner_9):
        """Test with zero prediction hours"""
        mock_runner_9.env_config['prediction_hours'] = 0
        mock_exchange.hour.return_value = 12
        mock_exchange.zone_time_step_number.return_value = 1
        state_argument = c_void_p()
        result, _ = mock_runner_9.get_weather_prediction(state_argument)
        assert result == {}, "Weather prediction should return empty dict with zero prediction hours"

    def test_get_zone_simulation_parameters_values_1(self):
        """
        Test that get_zone_simulation_parameters_values returns correct variables and infos
        when zone_simulation_parameters are configured in env_config.
        """
        # Mock the EnergyPlusRunner instance
        runner = Mock(spec=EnergyPlusRunner)
        runner.env_config = {
            'zone_simulation_parameters': {
                'system_time_step': True,
                'zone_time_step': True,
                'zone_time_step_number': False
            },
            'infos_variables': {
                'zone_simulation_parameters': {}
            },
            'no_observable_variables': {
                'zone_simulation_parameters': {}
            }
        }

        # Mock the api.exchange methods
        api.exchange.system_time_step = Mock(return_value=60)
        api.exchange.zone_time_step = Mock(return_value=900)
        api.exchange.zone_time_step_number = Mock(return_value=4)

        # Call the method
        state_argument = c_void_p()
        variables, infos = EnergyPlusRunner.get_zone_simulation_parameters_values(runner, state_argument)

        # Assert the results
        expected_variables = {
            'system_time_step': 60,
            'zone_time_step': 900
        }
        expected_infos = {}

        assert variables == expected_variables
        assert infos == expected_infos

        # Verify that the mocked methods were called
        api.exchange.system_time_step.assert_called_once_with(state_argument)
        api.exchange.zone_time_step.assert_called_once_with(state_argument)
        api.exchange.zone_time_step_number.assert_not_called()

    def test_get_zone_simulation_parameters_values_2(self, energyplus_runner, mock_energyplus_api):
        """
        Test get_zone_simulation_parameters_values when all parameters are set to False.
        """
        # Arrange
        state_argument = c_void_p()

        # Act
        variables, infos = energyplus_runner.get_zone_simulation_parameters_values(state_argument)

        # Assert
        assert variables == {}, "Expected empty variables dictionary when all parameters are False"
        assert infos == {}, "Expected empty infos dictionary when all parameters are False"
        
        # Verify that no API calls were made
        mock_energyplus_api.exchange.system_time_step.assert_not_called()
        mock_energyplus_api.exchange.zone_time_step.assert_not_called()
        mock_energyplus_api.exchange.zone_time_step_number.assert_not_called()

    @patch('eprllib.Env.MultiAgent.EnergyPlusRunner.api')
    def test_get_zone_simulation_parameters_values_api_error(self, mock_api, mock_runner_8):
        """
        Test get_zone_simulation_parameters_values when API calls raise exceptions
        """
        mock_api.exchange.system_time_step.side_effect = RuntimeError("API Error")
        state = MagicMock(spec=c_void_p)
        
        with pytest.raises(RuntimeError):
            mock_runner_8.get_zone_simulation_parameters_values(state)

    def test_get_zone_simulation_parameters_values_empty_config(self, mock_runner_8):
        """
        Test get_zone_simulation_parameters_values with an empty zone_simulation_parameters config
        """
        mock_runner_8.env_config['zone_simulation_parameters'] = {}
        state = MagicMock(spec=c_void_p)
        variables, infos = mock_runner_8.get_zone_simulation_parameters_values(state)
        assert variables == {}
        assert infos == {}

    @patch('eprllib.Env.MultiAgent.EnergyPlusRunner.api')
    def test_get_zone_simulation_parameters_values_incorrect_return_type(self, mock_api, mock_runner_8):
        """
        Test get_zone_simulation_parameters_values when API calls return incorrect types
        """
        mock_api.exchange.system_time_step.return_value = "not a number"
        mock_api.exchange.zone_time_step.return_value = 1.0
        mock_api.exchange.zone_time_step_number.return_value = 1
        state = MagicMock(spec=c_void_p)
        
        variables, infos = mock_runner_8.get_zone_simulation_parameters_values(state)
        assert isinstance(variables['system_time_step'], str)
        assert isinstance(variables['zone_time_step'], float)
        assert isinstance(variables['zone_time_step_number'], int)

    def test_get_zone_simulation_parameters_values_invalid_state(self, mock_runner_8):
        """
        Test get_zone_simulation_parameters_values with an invalid state argument
        """
        with pytest.raises(AttributeError):
            mock_runner_8.get_zone_simulation_parameters_values(None)

    @patch('eprllib.Env.MultiAgent.EnergyPlusRunner.api')
    def test_get_zone_simulation_parameters_values_negative_values(self, mock_api, mock_runner_8):
        """
        Test get_zone_simulation_parameters_values with negative return values from API
        """
        mock_api.exchange.system_time_step.return_value = -1.0
        mock_api.exchange.zone_time_step.return_value = -2.0
        mock_api.exchange.zone_time_step_number.return_value = -3
        state = MagicMock(spec=c_void_p)
        
        variables, infos = mock_runner_8.get_zone_simulation_parameters_values(state)
        assert variables['system_time_step'] == -1.0
        assert variables['zone_time_step'] == -2.0
        assert variables['zone_time_step_number'] == -3

    def test_get_zone_simulation_parameters_values_no_observable_variables(self, mock_runner_8):
        """
        Test get_zone_simulation_parameters_values with no_observable_variables set
        """
        mock_runner_8.env_config['no_observable_variables']['zone_simulation_parameters'] = {'system_time_step'}
        state = MagicMock(spec=c_void_p)
        
        with patch('eprllib.Env.MultiAgent.EnergyPlusRunner.api') as mock_api:
            mock_api.exchange.system_time_step.return_value = 1.0
            mock_api.exchange.zone_time_step.return_value = 2.0
            mock_api.exchange.zone_time_step_number.return_value = 3
            
            variables, infos = mock_runner_8.get_zone_simulation_parameters_values(state)
            
            assert 'system_time_step' not in variables
            assert 'zone_time_step' in variables
            assert 'zone_time_step_number' in variables

    def test_make_eplus_args_csv_flag(self):
        """
        Test make_eplus_args with CSV flag set to True.
        """
        env_config = {
            "epw_path": "/path/to/weather.epw",
            "output_path": "/tmp",
            "epjson_path": "/path/to/model.epjson",
            "csv": True,
            "evaluation": False
        }
        runner = EnergyPlusRunner(env_config, 1, Queue(), Queue(), Queue(), set(), set(), None, None)
        args = runner.make_eplus_args()
        assert "-r" in args

    def test_make_eplus_args_evaluation_mode(self):
        """
        Test make_eplus_args in evaluation mode.
        """
        env_config = {
            "epw_path": "/path/to/weather.epw",
            "output_path": "/tmp",
            "epjson_path": "/path/to/model.epjson",
            "csv": False,
            "evaluation": True
        }
        runner = EnergyPlusRunner(env_config, 1, Queue(), Queue(), Queue(), set(), set(), None, None)
        args = runner.make_eplus_args()
        assert "evaluation-episode" in args[args.index("-d") + 1]

    def test_make_eplus_args_invalid_epjson_path(self):
        """
        Test make_eplus_args with an invalid EPJSON file path.
        """
        env_config = {
            "epw_path": "/path/to/weather.epw",
            "output_path": "/tmp",
            "epjson_path": "/invalid/path/to/model.epjson",
            "csv": False,
            "evaluation": False
        }
        runner = EnergyPlusRunner(env_config, 1, Queue(), Queue(), Queue(), set(), set(), None, None)
        args = runner.make_eplus_args()
        assert "/invalid/path/to/model.epjson" in args

    def test_make_eplus_args_invalid_epw_path(self):
        """
        Test make_eplus_args with an invalid EPW file path.
        """
        env_config = {
            "epw_path": "/invalid/path/to/weather.epw",
            "output_path": "/tmp",
            "epjson_path": "/path/to/model.epjson",
            "csv": False,
            "evaluation": False
        }
        runner = EnergyPlusRunner(env_config, 1, Queue(), Queue(), Queue(), set(), set(), None, None)
        args = runner.make_eplus_args()
        assert "-w" in args
        assert "/invalid/path/to/weather.epw" in args

    def test_make_eplus_args_invalid_output_path(self):
        """
        Test make_eplus_args with an invalid output path.
        """
        env_config = {
            "epw_path": "/path/to/weather.epw",
            "output_path": "/invalid/output/path",
            "epjson_path": "/path/to/model.epjson",
            "csv": False,
            "evaluation": False
        }
        runner = EnergyPlusRunner(env_config, 1, Queue(), Queue(), Queue(), set(), set(), None, None)
        args = runner.make_eplus_args()
        assert "-d" in args
        assert "/invalid/output/path" in args[args.index("-d") + 1]

    def test_make_eplus_args_missing_required_config(self):
        """
        Test make_eplus_args when required configuration is missing.
        """
        env_config = {}
        runner = EnergyPlusRunner(env_config, 1, Queue(), Queue(), Queue(), set(), set(), None, None)
        with pytest.raises(KeyError):
            runner.make_eplus_args()

    def test_make_eplus_args_with_csv_and_evaluation(self):
        """
        Test make_eplus_args method with csv=True and evaluation=True
        """
        # Mock the necessary attributes and methods
        runner = EnergyPlusRunner.__new__(EnergyPlusRunner)
        runner.env_config = {
            "csv": True,
            "epw_path": "/path/to/weather.epw",
            "output_path": "/path/to/output",
            "evaluation": True,
            "epjson_path": "/path/to/model.epjson"
        }
        runner.episode = 1
        runner.unique_id = 123456789

        # Mock os.getpid() to return a fixed value
        with patch('os.getpid', return_value=12345):
            result = runner.make_eplus_args()

        expected = [
            "-r",
            "-w",
            "/path/to/weather.epw",
            "-d",
            f"/path/to/output/evaluation-episode-00000001-12345-123456789",
            "/path/to/model.epjson"
        ]

        assert result == expected, f"Expected {expected}, but got {result}"

    def test_set_actuators_empty_agent_config(self, mock_energy_plus_runner):
        """Test set_actuators with empty agents_config"""
        mock_energy_plus_runner.env_config['agents_config'] = {}
        actuators, actuator_handles = mock_energy_plus_runner.set_actuators()
        assert actuators == {}
        assert actuator_handles == {}

    def test_set_actuators_extra_agent_in_config(self, mock_energy_plus_runner):
        """Test set_actuators with an extra agent in config not in _agent_ids"""
        mock_energy_plus_runner.env_config['agents_config']['extra_agent'] = {'ep_actuator_config': ('extra', 'type', 'key')}
        actuators, actuator_handles = mock_energy_plus_runner.set_actuators()
        assert 'extra_agent' not in actuators
        assert len(actuators) == 2

    def test_set_actuators_incorrect_type(self, mock_energy_plus_runner):
        """Test set_actuators with incorrect type for ep_actuator_config"""
        mock_energy_plus_runner.env_config['agents_config']['agent1']['ep_actuator_config'] = 'incorrect_type'
        with pytest.raises(TypeError):
            mock_energy_plus_runner.set_actuators()

    def test_set_actuators_invalid_agent_config(self, mock_energy_plus_runner):
        """Test set_actuators with invalid agent config"""
        mock_energy_plus_runner.env_config['agents_config']['invalid_agent'] = {'invalid_key': 'invalid_value'}
        with pytest.raises(KeyError):
            mock_energy_plus_runner.set_actuators()

    def test_set_actuators_missing_agent(self, mock_energy_plus_runner):
        """Test set_actuators with a missing agent in _agent_ids"""
        mock_energy_plus_runner._agent_ids.add('missing_agent')
        with pytest.raises(KeyError):
            mock_energy_plus_runner.set_actuators()

    def test_set_actuators_returns_correct_types(self):
        """
        Test that set_actuators returns the correct types for actuators and actuator_handles.
        """
        # Mock the necessary attributes and configuration
        runner = EnergyPlusRunner.__new__(EnergyPlusRunner)
        runner.env_config = {
            'agents_config': {
                'agent1': {'ep_actuator_config': ('component_type', 'control_type', 'actuator_key')},
                'agent2': {'ep_actuator_config': ('component_type', 'control_type', 'actuator_key')}
            }
        }
        runner._agent_ids = {'agent1', 'agent2'}

        # Call the method
        actuators, actuator_handles = runner.set_actuators()

        # Assert the correct types are returned
        assert isinstance(actuators, dict)
        assert isinstance(actuator_handles, dict)
        assert all(isinstance(v, tuple) for v in actuators.values())
        assert all(isinstance(v, tuple) and len(v) == 3 for v in actuators.values())
        assert len(actuator_handles) == 0  # Should be empty as per the method implementation

        # Assert the correct structure of the actuators dictionary
        assert set(actuators.keys()) == runner._agent_ids
        for agent_id in runner._agent_ids:
            assert actuators[agent_id] == runner.env_config['agents_config'][agent_id]['ep_actuator_config']

    def test_set_meters_1(self):
        """
        Test set_meters method when env_config['meters'] is not empty.
        """
        # Mock environment configuration
        env_config: Dict[str, Any] = {
            'meters': ['Electricity:Facility', 'Gas:Facility'],
            'output_path': '/tmp',
            'epw_path': '/path/to/weather.epw',
            'epjson_path': '/path/to/model.epjson',
            'evaluation': False
        }
        
        # Mock other required attributes
        thermal_zone_ids: Set[str] = {'Zone1', 'Zone2'}
        
        # Create EnergyPlusRunner instance with mocked attributes
        runner = EnergyPlusRunner(
            env_config=env_config,
            episode=1,
            obs_queue=None,
            act_queue=None,
            infos_queue=None,
            _agent_ids=set(),
            _thermal_zone_ids=thermal_zone_ids,
            observation_fn=None,
            action_fn=None
        )
        
        # Call the method under test
        meters, meter_handles = runner.set_meters()
        
        # Assert the results
        expected_meters = {
            'Zone1': {'Electricity:Facility': 'Electricity:Facility', 'Gas:Facility': 'Gas:Facility'},
            'Zone2': {'Electricity:Facility': 'Electricity:Facility', 'Gas:Facility': 'Gas:Facility'}
        }
        expected_meter_handles = {'Zone1': {}, 'Zone2': {}}
        
        assert meters == expected_meters
        assert meter_handles == expected_meter_handles

    def test_set_meters_2(self):
        """
        Test set_meters method when env_config['meters'] is not empty.
        """
        # Mock the EnergyPlusRunner instance
        runner = MagicMock(spec=EnergyPlusRunner)
        
        # Set up the test scenario
        runner.env_config = {'meters': ['Electricity:Facility', 'Gas:Facility']}
        runner._thermal_zone_ids = ['Zone1', 'Zone2']
        
        # Call the method under test
        meters, meter_handles = EnergyPlusRunner.set_meters(runner)
        
        # Assert the results
        expected_meters = {
            'Zone1': {'Electricity:Facility': 'Electricity:Facility', 'Gas:Facility': 'Gas:Facility'},
            'Zone2': {'Electricity:Facility': 'Electricity:Facility', 'Gas:Facility': 'Gas:Facility'}
        }
        expected_meter_handles = {'Zone1': {}, 'Zone2': {}}
        
        assert meters == expected_meters
        assert meter_handles == expected_meter_handles

    def test_set_meters_empty_input(self):
        """
        Test set_meters with empty input for env_config['meters']
        """
        runner = EnergyPlusRunner({
            'meters': [],
            'output_path': '/tmp',
            'epw_path': '/path/to/weather.epw',
            'epjson_path': '/path/to/model.epjson',
            'evaluation': False
        }, 1, None, None, None, {'agent1'}, {'zone1'}, None, None)
        meters, meter_handles = runner.set_meters()
        assert meters == {'zone1': {}}, "Meters should be empty when no meters are specified"
        assert meter_handles == {'zone1': {}}, "Meter handles should be empty when no meters are specified"

    def test_set_meters_incorrect_type(self):
        """
        Test set_meters with incorrect type for env_config['meters']
        """
        with pytest.raises(TypeError):
            runner = EnergyPlusRunner({
                'meters': {'invalid': 'type'},
                'output_path': '/tmp',
                'epw_path': '/path/to/weather.epw',
                'epjson_path': '/path/to/model.epjson',
                'evaluation': False
            }, 1, None, None, None, {'agent1'}, {'zone1'}, None, None)
            runner.set_meters()

    def test_set_meters_invalid_input(self):
        """
        Test set_meters with invalid input for env_config['meters']
        """
        with pytest.raises(TypeError):
            runner = EnergyPlusRunner({
                'meters': 'invalid',
                'output_path': '/tmp',
                'epw_path': '/path/to/weather.epw',
                'epjson_path': '/path/to/model.epjson',
                'evaluation': False
            }, 1, None, None, None, {'agent1'}, {'zone1'}, None, None)
            runner.set_meters()

    def test_set_meters_mismatched_thermal_zones(self):
        """
        Test set_meters with mismatched thermal zones
        """
        runner = EnergyPlusRunner({
            'meters': ['meter1', 'meter2'],
            'output_path': '/tmp',
            'epw_path': '/path/to/weather.epw',
            'epjson_path': '/path/to/model.epjson',
            'evaluation': False
        }, 1, None, None, None, {'agent1'}, {'zone1', 'zone2'}, None, None)
        meters, meter_handles = runner.set_meters()
        assert set(meters.keys()) == {'zone1', 'zone2'}, "Meters should be created for all thermal zones"
        assert set(meter_handles.keys()) == {'zone1', 'zone2'}, "Meter handles should be created for all thermal zones"

    def test_set_meters_missing_config(self):
        """
        Test set_meters with missing 'meters' in env_config
        """
        runner = EnergyPlusRunner({
            'output_path': '/tmp',
            'epw_path': '/path/to/weather.epw',
            'epjson_path': '/path/to/model.epjson',
            'evaluation': False
        }, 1, None, None, None, {'agent1'}, {'zone1'}, None, None)
        with pytest.raises(KeyError):
            runner.set_meters()

    def test_set_object_variables_1(self):
        """
        Test set_object_variables method when variables_obj is not empty.
        """
        # Mock env_config and other required attributes
        env_config = {
            'variables_obj': {
                'agent1': {'var1': 'key1', 'var2': 'key2'},
                'agent2': {'var3': 'key3'}
            }
        }
        thermal_zone_ids = {'zone1', 'zone2'}
        agent_ids = {'agent1', 'agent2'}

        # Create EnergyPlusRunner instance with mocked attributes
        runner = EnergyPlusRunner.__new__(EnergyPlusRunner)
        runner.env_config = env_config
        runner._thermal_zone_ids = thermal_zone_ids
        runner._agent_ids = agent_ids

        # Call the method
        variables, var_handles = runner.set_object_variables()

        # Assert the results
        expected_variables = {
            'agent1': {'var1': ('var1', 'key1'), 'var2': ('var2', 'key2')},
            'agent2': {'var3': ('var3', 'key3')}
        }
        expected_var_handles = {zone: {} for zone in thermal_zone_ids}

        assert variables == expected_variables
        assert var_handles == expected_var_handles
        assert set(variables.keys()) == agent_ids

    def test_set_object_variables_1_2(self):
        """
        Testcase 1 for def set_object_variables(self) -> Tuple[Dict[str, Tuple [str, str]], Dict[str, int]]:
        Path constraints: not len(self.env_config['variables_obj']) == 0
        """
        # Mock EnergyPlusRunner instance
        runner = EnergyPlusRunner.__new__(EnergyPlusRunner)
        runner.env_config = {
            'variables_obj': {
                'Agent1': {'Var1': 'Key1', 'Var2': 'Key2'},
                'Agent2': {'Var3': 'Key3'}
            }
        }
        runner._thermal_zone_ids = {'Zone1', 'Zone2'}
        runner._agent_ids = {'Agent1', 'Agent2'}

        # Call the method
        variables, var_handles = runner.set_object_variables()

        # Assert that variables contain the correct data
        expected_variables = {
            'Agent1': {'Var1': ('Var1', 'Key1'), 'Var2': ('Var2', 'Key2')},
            'Agent2': {'Var3': ('Var3', 'Key3')}
        }
        assert variables == expected_variables

        # Assert that var_handles is an empty dictionary for each thermal zone
        assert var_handles == {thermal_zone: {} for thermal_zone in runner._thermal_zone_ids}

    def test_set_object_variables_2(self):
        """
        Test set_object_variables when env_config['variables_obj'] is empty.
        """
        # Mock EnergyPlusRunner instance
        runner = EnergyPlusRunner.__new__(EnergyPlusRunner)
        runner.env_config = {'variables_obj': {}}
        runner._thermal_zone_ids = {'Zone1', 'Zone2'}
        runner._agent_ids = {'Agent1', 'Agent2'}

        # Call the method
        variables, var_handles = runner.set_object_variables()

        # Assert that variables and var_handles are empty dictionaries for each thermal zone
        assert variables == {thermal_zone: {} for thermal_zone in runner._thermal_zone_ids}
        assert var_handles == {thermal_zone: {} for thermal_zone in runner._thermal_zone_ids}

    def test_set_object_variables_empty_input(self):
        """Test set_object_variables with empty input"""
        runner = EnergyPlusRunner({}, 1, None, None, None, set(), set(), None, None)
        runner.env_config = {'variables_obj': {}}
        runner._agent_ids = set()
        
        variables, var_handles = runner.set_object_variables()
        
        assert variables == {}
        assert var_handles == {}

    def test_set_object_variables_extra_agent(self):
        """Test set_object_variables with extra agent in variables_obj"""
        runner = EnergyPlusRunner({}, 1, None, None, None, {'agent1'}, set(), None, None)
        runner.env_config = {'variables_obj': {'agent1': {}, 'agent2': {}}}
        
        with pytest.raises(AssertionError, match="The variables_obj must include all agent_ids:"):
            runner.set_object_variables()

    def test_set_object_variables_incorrect_type(self):
        """Test set_object_variables with incorrect input type"""
        runner = EnergyPlusRunner({}, 1, None, None, None, set(), set(), None, None)
        runner.env_config = {'variables_obj': 'not a dict'}
        runner._agent_ids = set()
        
        with pytest.raises(AttributeError):
            runner.set_object_variables()

    def test_set_object_variables_invalid_agents(self):
        """
        Test set_object_variables when env_config['variables_obj'] contains invalid agents.
        """
        # Mock EnergyPlusRunner instance
        runner = EnergyPlusRunner.__new__(EnergyPlusRunner)
        runner.env_config = {
            'variables_obj': {
                'Agent1': {'Var1': 'Key1'},
                'InvalidAgent': {'Var2': 'Key2'}
            }
        }
        runner._thermal_zone_ids = {'Zone1'}
        runner._agent_ids = {'Agent1', 'Agent2'}

        # Call the method and expect an AssertionError
        with pytest.raises(AssertionError):
            runner.set_object_variables()

    def test_set_object_variables_invalid_format(self):
        """Test set_object_variables with invalid input format"""
        runner = EnergyPlusRunner({}, 1, None, None, None, {'agent1'}, set(), None, None)
        runner.env_config = {'variables_obj': {'agent1': 'not a dict'}}
        
        with pytest.raises(AttributeError):
            runner.set_object_variables()

    def test_set_object_variables_missing_agent(self):
        """Test set_object_variables with missing agent in variables_obj"""
        runner = EnergyPlusRunner({}, 1, None, None, None, {'agent1'}, set(), None, None)
        runner.env_config = {'variables_obj': {}}
        
        with pytest.raises(AssertionError, match="The variables_obj must include all agent_ids:"):
            runner.set_object_variables()

    def test_set_site_variables_1(self):
        """
        Test set_site_variables method when env_config['variables_env'] is not empty.
        """
        # Create a mock EnergyPlusRunner instance
        runner = EnergyPlusRunner.__new__(EnergyPlusRunner)
        
        # Set up mock env_config and thermal_zone_ids
        runner.env_config = {
            'variables_env': ['Site Outdoor Air Drybulb Temperature', 'Site Wind Speed']
        }
        runner._thermal_zone_ids = ['Zone1', 'Zone2']

        # Call the method
        variables, var_handles = runner.set_site_variables()

        # Assert the expected output
        expected_variables = {
            'Site Outdoor Air Drybulb Temperature': ('Site Outdoor Air Drybulb Temperature', 'Environment'),
            'Site Wind Speed': ('Site Wind Speed', 'Environment')
        }
        expected_var_handles = {'Zone1': {}, 'Zone2': {}}

        assert variables == expected_variables
        assert var_handles == expected_var_handles

    def test_set_site_variables_2(self):
        """
        Test set_site_variables when env_config['variables_env'] is not empty.
        """
        # Create a mock EnergyPlusRunner instance
        runner = EnergyPlusRunner(
            env_config={
                'variables_env': ['OutdoorAirTemperature', 'OutdoorRelativeHumidity'],
            },
            episode=1,
            obs_queue=None,
            act_queue=None,
            infos_queue=None,
            _agent_ids=set(),
            _thermal_zone_ids=set(['Zone1', 'Zone2']),
            observation_fn=None,
            action_fn=None
        )

        # Call the method
        variables, var_handles = runner.set_site_variables()

        # Assert the returned values
        expected_variables = {
            'OutdoorAirTemperature': ('OutdoorAirTemperature', 'Environment'),
            'OutdoorRelativeHumidity': ('OutdoorRelativeHumidity', 'Environment')
        }
        expected_var_handles = {'Zone1': {}, 'Zone2': {}}

        assert variables == expected_variables
        assert var_handles == expected_var_handles

    def test_set_site_variables_with_empty_env_config(self):
        """
        Test set_site_variables with an empty env_config.
        """
        runner = EnergyPlusRunner(env_config={}, episode=1, obs_queue=None, act_queue=None, 
                                  infos_queue=None, _agent_ids=set(), _thermal_zone_ids=set(), 
                                  observation_fn=None, action_fn=None)
        variables, var_handles = runner.set_site_variables()
        assert variables == {}
        assert var_handles == {}

    def test_set_site_variables_with_empty_thermal_zone_ids(self):
        """
        Test set_site_variables with empty thermal_zone_ids.
        """
        runner = EnergyPlusRunner(env_config={'variables_env': ['var1']}, episode=1, 
                                  obs_queue=None, act_queue=None, infos_queue=None, 
                                  _agent_ids=set(), _thermal_zone_ids=set(), 
                                  observation_fn=None, action_fn=None)
        variables, var_handles = runner.set_site_variables()
        assert variables == {'var1': ('var1', 'Environment')}
        assert var_handles == {}

    def test_set_site_variables_with_incorrect_type_variables_env(self):
        """
        Test set_site_variables with incorrect type for variables_env in env_config.
        """
        runner = EnergyPlusRunner(env_config={'variables_env': 123}, episode=1, 
                                  obs_queue=None, act_queue=None, infos_queue=None, 
                                  _agent_ids=set(), _thermal_zone_ids=set(), 
                                  observation_fn=None, action_fn=None)
        with pytest.raises(AttributeError):
            runner.set_site_variables()

    def test_set_site_variables_with_invalid_variables_env(self):
        """
        Test set_site_variables with invalid variables_env in env_config.
        """
        runner = EnergyPlusRunner(env_config={'variables_env': 'invalid'}, episode=1, 
                                  obs_queue=None, act_queue=None, infos_queue=None, 
                                  _agent_ids=set(), _thermal_zone_ids=set(), 
                                  observation_fn=None, action_fn=None)
        with pytest.raises(AttributeError):
            runner.set_site_variables()

    def test_set_site_variables_with_valid_variables_env(self):
        """
        Test set_site_variables with valid variables_env in env_config.
        """
        runner = EnergyPlusRunner(env_config={'variables_env': ['var1', 'var2']}, episode=1, 
                                  obs_queue=None, act_queue=None, infos_queue=None, 
                                  _agent_ids=set(), _thermal_zone_ids={'zone1'}, 
                                  observation_fn=None, action_fn=None)
        variables, var_handles = runner.set_site_variables()
        assert variables == {'var1': ('var1', 'Environment'), 'var2': ('var2', 'Environment')}
        assert var_handles == {'zone1': {}}

    def test_set_static_variables_1(self):
        """
        Test that set_static_variables returns correct variables and var_handles when static_variables are defined.
        """
        # Mock the necessary attributes and config
        mock_env_config = {
            'static_variables': ['Zone Mean Air Temperature', 'Zone Air Relative Humidity']
        }
        mock_thermal_zone_ids = {'Zone1', 'Zone2'}

        # Create a partial EnergyPlusRunner instance with mocked attributes
        runner = EnergyPlusRunner.__new__(EnergyPlusRunner)
        runner.env_config = mock_env_config
        runner._thermal_zone_ids = mock_thermal_zone_ids

        # Call the method
        variables, var_handles = runner.set_static_variables()

        # Assert the correct structure and content of the returned values
        assert isinstance(variables, dict)
        assert isinstance(var_handles, dict)
        
        # Check that variables contains the correct structure
        for zone in mock_thermal_zone_ids:
            assert zone in variables
            assert isinstance(variables[zone], dict)
            for var in mock_env_config['static_variables']:
                assert var in variables[zone]
                assert variables[zone][var] == var

        # Check that var_handles contains the correct structure (empty dicts for each zone)
        for zone in mock_thermal_zone_ids:
            assert zone in var_handles
            assert isinstance(var_handles[zone], dict)
            assert len(var_handles[zone]) == 0

        # Check that the number of zones in variables and var_handles match
        assert len(variables) == len(var_handles) == len(mock_thermal_zone_ids)

    def test_set_static_variables_2(self):
        """
        Test set_static_variables when static_variables is not empty.
        """
        # Arrange
        env_config = {
            'static_variables': ['Variable1', 'Variable2'],
            'output_path': '/tmp',
            'epw_path': '/path/to/weather.epw',
            'epjson_path': '/path/to/model.epjson',
            'evaluation': False,
            'ep_terminal_output': False
        }
        thermal_zone_ids = {'Zone1', 'Zone2'}
        runner = EnergyPlusRunner(
            env_config=env_config,
            episode=1,
            obs_queue=None,
            act_queue=None,
            infos_queue=None,
            _agent_ids={'Agent1'},
            _thermal_zone_ids=thermal_zone_ids,
            observation_fn=None,
            action_fn=None
        )

        # Act
        variables, var_handles = runner.set_static_variables()

        # Assert
        expected_variables = {
            'Zone1': {'Variable1': 'Variable1', 'Variable2': 'Variable2'},
            'Zone2': {'Variable1': 'Variable1', 'Variable2': 'Variable2'}
        }
        expected_var_handles = {'Zone1': {}, 'Zone2': {}}

        assert variables == expected_variables
        assert var_handles == expected_var_handles
        assert isinstance(variables, dict)
        assert isinstance(var_handles, dict)
        assert all(isinstance(zone_vars, dict) for zone_vars in variables.values())
        assert all(isinstance(zone_handles, dict) for zone_handles in var_handles.values())

    def test_set_static_variables_empty_config(self):
        """
        Test set_static_variables with an empty environment configuration.
        """
        runner = EnergyPlusRunner({}, 1, None, None, None, set(), set(), None, None)
        variables, var_handles = runner.set_static_variables()
        assert variables == {}
        assert var_handles == {}

    def test_set_static_variables_empty_thermal_zone_ids(self):
        """
        Test set_static_variables with empty thermal_zone_ids.
        """
        runner = EnergyPlusRunner({'static_variables': ['var1']}, 1, None, None, None, set(), set(), None, None)
        variables, var_handles = runner.set_static_variables()
        assert variables == {}
        assert var_handles == {}

    def test_set_static_variables_incorrect_type(self):
        """
        Test set_static_variables with incorrect type for static_variables.
        """
        runner = EnergyPlusRunner({'static_variables': 'not_a_list'}, 1, None, None, None, set(), set(['zone1']), None, None)
        with pytest.raises(TypeError):
            runner.set_static_variables()

    def test_set_static_variables_invalid_thermal_zone_ids(self):
        """
        Test set_static_variables with invalid thermal_zone_ids.
        """
        runner = EnergyPlusRunner({'static_variables': ['var1', 'var2']}, 1, None, None, None, set(), set(['invalid_zone']), None, None)
        variables, var_handles = runner.set_static_variables()
        assert variables == {'invalid_zone': {'var1': 'var1', 'var2': 'var2'}}
        assert var_handles == {'invalid_zone': {}}

    def test_set_static_variables_missing_config(self):
        """
        Test set_static_variables with missing 'static_variables' in config.
        """
        runner = EnergyPlusRunner({}, 1, None, None, None, set(), set(['zone1']), None, None)
        variables, var_handles = runner.set_static_variables()
        assert variables == {'zone1': {}}
        assert var_handles == {'zone1': {}}

    def test_set_static_variables_none_thermal_zone_ids(self):
        """
        Test set_static_variables with None for thermal_zone_ids.
        """
        runner = EnergyPlusRunner({'static_variables': ['var1']}, 1, None, None, None, set(), None, None, None)
        with pytest.raises(TypeError):
            runner.set_static_variables()

    def test_set_thermalzone_variables_1(self):
        """
        Test set_thermalzone_variables method when variables_thz is not empty.
        """
        # Create a mock EnergyPlusRunner instance
        runner = EnergyPlusRunner.__new__(EnergyPlusRunner)
        
        # Set up the necessary attributes
        runner.env_config = {
            'variables_thz': ['Zone Mean Air Temperature', 'Zone Air Relative Humidity']
        }
        runner._thermal_zone_ids = {'Zone1', 'Zone2'}

        # Call the method
        variables, var_handles = runner.set_thermalzone_variables()

        # Assert the results
        assert isinstance(variables, dict)
        assert isinstance(var_handles, dict)
        
        # Check if variables contains the correct structure and data
        for zone in runner._thermal_zone_ids:
            assert zone in variables
            assert isinstance(variables[zone], dict)
            for var in runner.env_config['variables_thz']:
                assert var in variables[zone]
                assert variables[zone][var] == (var, zone)

        # Check if var_handles is correctly initialized
        for zone in runner._thermal_zone_ids:
            assert zone in var_handles
            assert isinstance(var_handles[zone], dict)
            assert len(var_handles[zone]) == 0  # Should be empty

        # Check the total number of variables
        total_variables = sum(len(zone_vars) for zone_vars in variables.values())
        expected_total = len(runner._thermal_zone_ids) * len(runner.env_config['variables_thz'])
        assert total_variables == expected_total

    def test_set_thermalzone_variables_1_2(self):
        """
        Test set_thermalzone_variables when variables_thz is not empty
        """
        # Create a mock EnergyPlusRunner instance
        runner = EnergyPlusRunner(
            env_config={'variables_thz': ['Temperature', 'Humidity']},
            episode=1,
            obs_queue=None,
            act_queue=None,
            infos_queue=None,
            _agent_ids=set(),
            _thermal_zone_ids={'Zone1', 'Zone2'},
            observation_fn=None,
            action_fn=None
        )

        # Call the method
        variables, var_handles = runner.set_thermalzone_variables()

        # Assert that the returned dictionaries contain the expected values
        expected_variables = {
            'Zone1': {
                'Temperature': ('Temperature', 'Zone1'),
                'Humidity': ('Humidity', 'Zone1')
            },
            'Zone2': {
                'Temperature': ('Temperature', 'Zone2'),
                'Humidity': ('Humidity', 'Zone2')
            }
        }
        assert variables == expected_variables
        assert var_handles == {'Zone1': {}, 'Zone2': {}}

    def test_set_thermalzone_variables_2(self):
        """
        Test set_thermalzone_variables when variables_thz is empty
        """
        # Create a mock EnergyPlusRunner instance
        runner = EnergyPlusRunner(
            env_config={'variables_thz': []},
            episode=1,
            obs_queue=None,
            act_queue=None,
            infos_queue=None,
            _agent_ids=set(),
            _thermal_zone_ids={'Zone1', 'Zone2'},
            observation_fn=None,
            action_fn=None
        )

        # Call the method
        variables, var_handles = runner.set_thermalzone_variables()

        # Assert that the returned dictionaries are empty
        assert variables == {'Zone1': {}, 'Zone2': {}}
        assert var_handles == {'Zone1': {}, 'Zone2': {}}

    def test_set_thermalzone_variables_empty_input(self):
        """
        Test set_thermalzone_variables with empty input.
        """
        runner = EnergyPlusRunner({}, 1, None, None, None, set(), set(), None, None)
        runner.env_config = {'variables_thz': []}
        runner._thermal_zone_ids = set()
        
        variables, var_handles = runner.set_thermalzone_variables()
        
        assert isinstance(variables, dict)
        assert isinstance(var_handles, dict)
        assert len(variables) == 0
        assert len(var_handles) == 0

    def test_set_thermalzone_variables_exception_handling(self):
        """
        Test set_thermalzone_variables exception handling.
        """
        runner = EnergyPlusRunner({}, 1, None, None, None, set(), set(), None, None)
        runner.env_config = None  # This will cause an AttributeError
        runner._thermal_zone_ids = set(['zone1'])
        
        with pytest.raises(AttributeError):
            runner.set_thermalzone_variables()

    def test_set_thermalzone_variables_incorrect_type(self):
        """
        Test set_thermalzone_variables with incorrect input type.
        """
        runner = EnergyPlusRunner({}, 1, None, None, None, set(), set(), None, None)
        runner.env_config = {'variables_thz': [1, 2, 3]}  # Should be strings
        runner._thermal_zone_ids = set(['zone1'])
        
        variables, var_handles = runner.set_thermalzone_variables()
        
        assert isinstance(variables, dict)
        assert isinstance(var_handles, dict)
        assert len(variables['zone1']) == 3
        assert all(isinstance(key, int) for key in variables['zone1'].keys())

    def test_set_thermalzone_variables_invalid_input(self):
        """
        Test set_thermalzone_variables with invalid input.
        """
        runner = EnergyPlusRunner({}, 1, None, None, None, set(), set(), None, None)
        runner.env_config = {'variables_thz': 'not_a_list'}
        runner._thermal_zone_ids = set(['zone1'])
        
        with pytest.raises(TypeError):
            runner.set_thermalzone_variables()

    def test_set_thermalzone_variables_multiple_zones(self):
        """
        Test set_thermalzone_variables with multiple thermal zones.
        """
        runner = EnergyPlusRunner({}, 1, None, None, None, set(), set(), None, None)
        runner.env_config = {'variables_thz': ['var1', 'var2']}
        runner._thermal_zone_ids = set(['zone1', 'zone2'])
        
        variables, var_handles = runner.set_thermalzone_variables()
        
        assert isinstance(variables, dict)
        assert isinstance(var_handles, dict)
        assert len(variables) == 2
        assert 'zone1' in variables and 'zone2' in variables
        assert len(variables['zone1']) == 2 and len(variables['zone2']) == 2
        assert len(var_handles) == 2
        assert 'zone1' in var_handles and 'zone2' in var_handles

    def test_set_thermalzone_variables_no_thermal_zones(self):
        """
        Test set_thermalzone_variables with no thermal zones.
        """
        runner = EnergyPlusRunner({}, 1, None, None, None, set(), set(), None, None)
        runner.env_config = {'variables_thz': ['var1', 'var2']}
        runner._thermal_zone_ids = set()
        
        variables, var_handles = runner.set_thermalzone_variables()
        
        assert isinstance(variables, dict)
        assert isinstance(var_handles, dict)
        assert len(variables) == 0
        assert len(var_handles) == 0

    def test_start_1(self):
        """
        Test that the start method initializes EnergyPlus correctly
        """
        # Mock the necessary dependencies
        mock_env_config = {
            'ep_terminal_output': False,
            'output_path': '/tmp',
            'epw_path': '/path/to/weather.epw',
            'epjson_path': '/path/to/model.epjson',
            'evaluation': False
        }
        mock_obs_queue = MagicMock()
        mock_act_queue = MagicMock()
        mock_infos_queue = MagicMock()
        mock_agent_ids = {'agent1', 'agent2'}
        mock_thermal_zone_ids = {'zone1', 'zone2'}
        mock_observation_fn = MagicMock()
        mock_action_fn = MagicMock()

        # Create an instance of EnergyPlusRunner
        runner = EnergyPlusRunner(
            env_config=mock_env_config,
            episode=1,
            obs_queue=mock_obs_queue,
            act_queue=mock_act_queue,
            infos_queue=mock_infos_queue,
            _agent_ids=mock_agent_ids,
            _thermal_zone_ids=mock_thermal_zone_ids,
            observation_fn=mock_observation_fn,
            action_fn=mock_action_fn
        )

        # Mock the api calls
        with patch.object(api.state_manager, 'new_state') as mock_new_state, \
             patch.object(api.runtime, 'callback_begin_zone_timestep_after_init_heat_balance') as mock_callback_begin, \
             patch.object(api.runtime, 'callback_end_zone_timestep_after_zone_reporting') as mock_callback_end, \
             patch.object(api.runtime, 'set_console_output_status') as mock_set_console_output, \
             patch.object(api.runtime, 'run_energyplus') as mock_run_energyplus:

            mock_new_state.return_value = MagicMock()
            
            # Call the start method
            runner.start()

            # Assert that the necessary methods were called
            mock_new_state.assert_called_once()
            mock_callback_begin.assert_called_once()
            mock_callback_end.assert_called_once()
            mock_set_console_output.assert_called_once_with(runner.energyplus_state, False)

            # Assert that the thread was started
            assert runner.energyplus_exec_thread is not None
            assert runner.energyplus_exec_thread.is_alive()

            # Clean up the thread
            runner.simulation_complete = True
            runner.energyplus_exec_thread.join()

    def test_start_with_callback_failure(self, mock_runner):
        """
        Test start method when setting callbacks fails
        """
        mock_state = c_void_p(1234)
        with patch.object(api.state_manager, 'new_state', return_value=mock_state):
            with patch.object(api.runtime, 'callback_begin_zone_timestep_after_init_heat_balance', side_effect=Exception("Callback error")):
                with pytest.raises(Exception, match="Callback error"):
                    mock_runner.start()

    def test_start_with_console_output_failure(self, mock_runner):
        """
        Test start method when setting console output status fails
        """
        mock_state = c_void_p(1234)
        with patch.object(api.state_manager, 'new_state', return_value=mock_state):
            with patch.object(api.runtime, 'callback_begin_zone_timestep_after_init_heat_balance', return_value=None):
                with patch.object(api.runtime, 'callback_end_zone_timestep_after_zone_reporting', return_value=None):
                    with patch.object(api.runtime, 'set_console_output_status', side_effect=Exception("Console output error")):
                        with pytest.raises(Exception, match="Console output error"):
                            mock_runner.start()

    def test_start_with_invalid_episode(self, mock_runner):
        """
        Test start method with an invalid episode number
        """
        mock_runner.episode = -1
        with pytest.raises(ValueError):
            mock_runner.start()

    def test_start_with_invalid_state(self, mock_runner):
        """
        Test start method with an invalid EnergyPlus state
        """
        with patch.object(api.state_manager, 'new_state', return_value=None):
            with pytest.raises(AttributeError):
                mock_runner.start()

    def test_start_with_missing_config(self, mock_runner):
        """
        Test start method with missing configuration
        """
        del mock_runner.env_config['epw_path']
        with pytest.raises(KeyError):
            mock_runner.start()

    def test_start_with_run_energyplus_failure(self, mock_runner):
        """
        Test start method when run_energyplus fails
        """
        mock_state = c_void_p(1234)
        with patch.object(api.state_manager, 'new_state', return_value=mock_state):
            with patch.object(api.runtime, 'callback_begin_zone_timestep_after_init_heat_balance', return_value=None):
                with patch.object(api.runtime, 'callback_end_zone_timestep_after_zone_reporting', return_value=None):
                    with patch.object(api.runtime, 'set_console_output_status', return_value=None):
                        with patch.object(api.runtime, 'run_energyplus', side_effect=Exception("EnergyPlus run error")):
                            mock_runner.start()
                            assert mock_runner.sim_results != 0
                            assert mock_runner.simulation_complete

    def test_stop_clears_callbacks(self, mock_runner_11):
        """Test that stop method clears callbacks"""
        mock_runner_11.stop()
        api.runtime.clear_callbacks.assert_called_once()

    def test_stop_deletes_energyplus_state(self, mock_runner_11):
        """Test that stop method deletes EnergyPlus state"""
        mock_runner_11.stop()
        api.state_manager.delete_state.assert_called_once_with(mock_runner_11.energyplus_state)

    def test_stop_flushes_queues(self, mock_runner_11):
        """Test that stop method flushes all queues"""
        with patch.object(mock_runner_11, '_flush_queues') as mock_flush:
            mock_runner_11.stop()
            mock_flush.assert_called_once()

    def test_stop_joins_exec_thread(self, mock_runner_11):
        """Test that stop method joins the execution thread"""
        mock_runner_11.stop()
        mock_runner_11.energyplus_exec_thread.join.assert_called_once()

    def test_stop_resets_first_observation(self, mock_runner_11):
        """Test that stop method resets first_observation flag"""
        mock_runner_11.first_observation = False
        mock_runner_11.stop()
        assert mock_runner_11.first_observation is True

    def test_stop_sets_exec_thread_to_none(self, mock_runner_11):
        """Test that stop method sets execution thread to None"""
        mock_runner_11.stop()
        assert mock_runner_11.energyplus_exec_thread is None

    def test_stop_simulation_already_complete(self):
        """
        Test stop method when simulation is already complete.
        """
        # Mock the necessary dependencies
        mock_env_config = {
            'ep_terminal_output': False,
            'output_path': '/tmp',
            'epw_path': '/path/to/weather.epw',
            'epjson_path': '/path/to/model.epjson',
            'evaluation': False
        }
        mock_obs_queue = Mock()
        mock_act_queue = Mock()
        mock_infos_queue = Mock()
        mock_agent_ids = set(['agent1'])
        mock_thermal_zone_ids = set(['zone1'])
        mock_observation_fn = Mock()
        mock_action_fn = Mock()

        # Create an instance of EnergyPlusRunner
        runner = EnergyPlusRunner(
            mock_env_config, 1, mock_obs_queue, mock_act_queue, mock_infos_queue,
            mock_agent_ids, mock_thermal_zone_ids, mock_observation_fn, mock_action_fn
        )

        # Set simulation_complete to True
        runner.simulation_complete = True
        runner.energyplus_state = Mock()
        runner.energyplus_exec_thread = Mock()

        # Mock the necessary methods and attributes
        with patch('time.sleep') as mock_sleep, \
             patch('eprllib.Env.MultiAgent.EnergyPlusRunner.api') as mock_api:
            
            # Call the stop method
            runner.stop()

            # Assertions
            assert runner.simulation_complete is True
            mock_sleep.assert_called_once_with(0.5)
            mock_api.runtime.stop_simulation.assert_called_once_with(runner.energyplus_state)
            runner.energyplus_exec_thread.join.assert_called_once()
            assert runner.energyplus_exec_thread is None
            assert runner.first_observation is True
            mock_api.runtime.clear_callbacks.assert_called_once()
            mock_api.state_manager.delete_state.assert_called_once_with(runner.energyplus_state)

    def test_stop_simulation_not_complete(self):
        """
        Test that stop() method correctly handles when simulation is not complete
        """
        # Mock the necessary objects and methods
        mock_env_config = {
            'ep_terminal_output': False,
            'output_path': '/tmp',
            'epw_path': '/path/to/weather.epw',
            'epjson_path': '/path/to/model.epjson',
            'evaluation': False
        }
        mock_obs_queue = Mock()
        mock_act_queue = Mock()
        mock_infos_queue = Mock()
        mock_agent_ids = set(['agent1'])
        mock_thermal_zone_ids = set(['zone1'])
        mock_observation_fn = Mock()
        mock_action_fn = Mock()

        # Create EnergyPlusRunner instance
        runner = EnergyPlusRunner(
            mock_env_config, 1, mock_obs_queue, mock_act_queue, mock_infos_queue,
            mock_agent_ids, mock_thermal_zone_ids, mock_observation_fn, mock_action_fn
        )

        # Set up the mocks
        runner.simulation_complete = False
        runner.energyplus_state = Mock()
        runner.energyplus_exec_thread = Mock()

        # Mock the api methods
        with patch.object(time, 'sleep') as mock_sleep, \
             patch.object(api.runtime, 'stop_simulation') as mock_stop_simulation, \
             patch.object(api.runtime, 'clear_callbacks') as mock_clear_callbacks, \
             patch.object(api.state_manager, 'delete_state') as mock_delete_state:

            # Call the stop method
            runner.stop()

            # Assertions
            assert runner.simulation_complete is True
            mock_sleep.assert_called_once_with(0.5)
            mock_stop_simulation.assert_called_once_with(runner.energyplus_state)
            runner.energyplus_exec_thread.join.assert_called_once()
            assert runner.energyplus_exec_thread is None
            assert runner.first_observation == True
            mock_clear_callbacks.assert_called_once()
            mock_delete_state.assert_called_once_with(runner.energyplus_state)

        # Check if queues were flushed
        mock_obs_queue.get.assert_called()
        mock_act_queue.get.assert_called()
        mock_infos_queue.get.assert_called()

    @patch('time.sleep')
    def test_stop_sleeps(self, mock_sleep, mock_runner_11):
        """Test that stop method calls time.sleep"""
        mock_runner_11.stop()
        mock_sleep.assert_called_once_with(0.5)

    def test_stop_when_simulation_not_complete(self, mock_runner_11):
        """Test stop method when simulation is not complete"""
        mock_runner_11.simulation_complete = False
        mock_runner_11.stop()
        assert mock_runner_11.simulation_complete is True

    def test_successful_simulation(self):
        """
        Test that failed() returns False when sim_results is zero
        """
        # Create a mock EnergyPlusRunner instance
        runner = EnergyPlusRunner(
            env_config={},
            episode=1,
            obs_queue=None,
            act_queue=None,
            infos_queue=None,
            _agent_ids=set(),
            _thermal_zone_ids=set(),
            observation_fn=None,
            action_fn=None
        )
        
        # Set sim_results to zero
        runner.sim_results = 0
        
        # Check that failed() returns False
        assert runner.failed() is False

    def test_update_infos_2(self):
        """
        Test update_infos method when belong_to is 'variables_env'
        """
        # Create a mock EnergyPlusRunner instance
        runner = EnergyPlusRunner(
            env_config={
                'infos_variables': {
                    'variables_env': ['temp', 'humidity']
                }
            },
            episode=1,
            obs_queue=None,
            act_queue=None,
            infos_queue=None,
            _agent_ids=set(),
            _thermal_zone_ids=set(),
            observation_fn=None,
            action_fn=None
        )

        # Test data
        dict_parameters: Dict[str, Any] = {
            'temp': 25.5,
            'humidity': 60,
            'pressure': 1013  # This should not be included in the result
        }
        belong_to = 'variables_env'

        # Call the method
        result = runner.update_infos(dict_parameters, belong_to)

        # Assert the result
        expected_result = {
            'temp': 25.5,
            'humidity': 60
        }
        assert result == expected_result, f"Expected {expected_result}, but got {result}"

        # Test that non-listed variables are not included
        assert 'pressure' not in result, "The 'pressure' variable should not be in the result"

        # Test with empty dict_parameters
        empty_result = runner.update_infos({}, belong_to)
        assert empty_result == {}, "Result should be an empty dictionary when input is empty"

        # Test with missing variables
        incomplete_dict = {'temp': 25.5}  # Missing 'humidity'
        incomplete_result = runner.update_infos(incomplete_dict, belong_to)
        assert 'humidity' not in incomplete_result, "Missing variables should not cause an error"

    def test_update_infos_3(self):
        """
        Test update_infos method when belong_to is 'simulation_parameters'
        """
        # Create a mock EnergyPlusRunner instance
        runner = EnergyPlusRunner(
            env_config={
                'infos_variables': {
                    'simulation_parameters': ['param1', 'param2', 'param3']
                }
            },
            episode=1,
            obs_queue=None,
            act_queue=None,
            infos_queue=None,
            _agent_ids=set(),
            _thermal_zone_ids=set(),
            observation_fn=None,
            action_fn=None
        )

        # Test input
        dict_parameters: Dict[str, Any] = {
            'param1': 10,
            'param2': 'test',
            'param3': 3.14,
            'extra_param': 'ignore_me'
        }
        belong_to = 'simulation_parameters'

        # Expected output
        expected_infos_dict: Dict[str, Any] = {
            'param1': 10,
            'param2': 'test',
            'param3': 3.14
        }

        # Call the method
        result = runner.update_infos(dict_parameters, belong_to)

        # Assert the result
        assert result == expected_infos_dict, f"Expected {expected_infos_dict}, but got {result}"

        # Test that it doesn't include parameters not in the config
        assert 'extra_param' not in result, "Result should not include parameters not in the config"

        # Test with empty dict_parameters
        empty_result = runner.update_infos({}, belong_to)
        assert empty_result == {}, "Result should be an empty dictionary when dict_parameters is empty"

        # Test with missing parameters
        partial_dict = {'param1': 5}
        partial_result = runner.update_infos(partial_dict, belong_to)
        assert partial_result == {'param1': 5}, "Result should only include available parameters"

    def test_update_infos_4(self):
        """
        Test update_infos method when belong_to is 'variables_obj' and reference is None.
        This should raise a ValueError.
        """
        # Create a mock EnergyPlusRunner instance
        runner = EnergyPlusRunner(
            env_config={
                'infos_variables': {
                    'variables_obj': {
                        'agent1': ['var1', 'var2']
                    }
                }
            },
            episode=1,
            obs_queue=None,
            act_queue=None,
            infos_queue=None,
            _agent_ids=set(['agent1']),
            _thermal_zone_ids=set(['zone1']),
            observation_fn=None,
            action_fn=None
        )

        # Prepare test input
        dict_parameters: Dict[str, Any] = {
            'var1': 10,
            'var2': 20
        }

        # Test the method
        with pytest.raises(ValueError, match="The 'reference' argument must be specified."):
            runner.update_infos(dict_parameters, belong_to='variables_obj', reference=None)

    def test_update_infos_5(self):
        """
        Test update_infos method with an invalid belong_to parameter.
        """
        # Create a mock EnergyPlusRunner instance
        runner = EnergyPlusRunner({
            'infos_variables': {
                'variables_env': [],
                'simulation_parameters': [],
                'variables_obj': {},
                'meters': {},
                'static_variables': {},
                'variables_thz': {},
                'zone_simulation_parameters': {}
            }
        }, 1, None, None, None, set(), set(), None, None)

        # Test data
        dict_parameters: Dict[str, Any] = {"test_param": 42}
        belong_to = "invalid_category"
        reference = "test_reference"

        # Assert that ValueError is raised with an invalid belong_to parameter
        with pytest.raises(ValueError) as exc_info:
            runner.update_infos(dict_parameters, belong_to, reference)

        # Check if the error message is correct
        assert str(exc_info.value) == f"The 'belong_to' argument must be one of the following: {runner.env_config['infos_variables'].keys()}"

    def test_update_infos_empty_config_category(self, energy_plus_runner):
        """Test update_infos with an empty category in the config"""
        energy_plus_runner.env_config['infos_variables']['empty_category'] = []
        result = energy_plus_runner.update_infos({}, belong_to='empty_category')
        assert result == {}

    def test_update_infos_empty_input(self, energy_plus_runner):
        """Test update_infos with empty input"""
        with pytest.raises(ValueError, match="The 'belong_to' argument must be specified."):
            energy_plus_runner.update_infos({})

    def test_update_infos_extra_variable(self, energy_plus_runner):
        """Test update_infos with an extra variable not in the config"""
        result = energy_plus_runner.update_infos({'var1': 1, 'var2': 2, 'extra': 3}, belong_to='variables_env')
        assert 'extra' not in result
        assert len(result) == 2

    def test_update_infos_incorrect_reference(self, energy_plus_runner):
        """Test update_infos with an incorrect reference"""
        result = energy_plus_runner.update_infos({'obj_var1': 1}, belong_to='variables_obj', reference='non_existent_agent')
        assert result == {}

    def test_update_infos_incorrect_type(self, energy_plus_runner):
        """Test update_infos with incorrect input types"""
        with pytest.raises(TypeError):
            energy_plus_runner.update_infos("not_a_dict", belong_to='variables_env')

    def test_update_infos_invalid_belong_to(self, energy_plus_runner):
        """Test update_infos with invalid belong_to parameter"""
        with pytest.raises(ValueError, match="The 'belong_to' argument must be one of the following:"):
            energy_plus_runner.update_infos({}, belong_to='invalid_category')

    def test_update_infos_missing_reference(self, energy_plus_runner):
        """Test update_infos with missing reference for categories that require it"""
        with pytest.raises(ValueError, match="The 'reference' argument must be specified."):
            energy_plus_runner.update_infos({}, belong_to='variables_obj')

    def test_update_infos_missing_variable(self, energy_plus_runner):
        """Test update_infos with a missing variable in the input dictionary"""
        result = energy_plus_runner.update_infos({'var1': 1}, belong_to='variables_env')
        assert 'var2' not in result

    def test_update_infos_raises_value_error_when_belong_to_is_none(self):
        """
        Test that update_infos raises a ValueError when belong_to is None
        """
        # Arrange
        runner = EnergyPlusRunner({}, 1, None, None, None, set(), set(), None, None)
        dict_parameters = {"test_param": "test_value"}

        # Act & Assert
        with pytest.raises(ValueError, match="The 'belong_to' argument must be specified."):
            runner.update_infos(dict_parameters, belong_to=None)