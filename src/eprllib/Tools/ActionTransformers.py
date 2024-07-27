"""This script will be contain some action transformer methods to implement in
eprllib. Most of them are applied in the test section where examples to test the 
library are developed.
"""

def thermostat_dual(agent_id, action):
    """This method take a discret action in the range of [0,4) and transforms it
    into a temperature of cooling or heating setpoints, depending the agent id 
    involve.
    """
    heating_agents = [
        'PB BUFETE Thermal Zone Heating Setpoint',
        'PB SALA DE REUNION N Thermal Zone Heating Setpoint',
        'PB SALA DE REUNION NW Thermal Zone Heating Setpoint',
        '1P CALL CENTER Thermal Zone Heating Setpoint',
        '1P PLANTA LIBRE Thermal Zone Heating Setpoint',
        '1P BOX B W Thermal Zone Heating Setpoint',
        '1P BOX A NW Thermal Zone Heating Setpoint',
        '2P BOX A NW Thermal Zone Heating Setpoint',
        '2P PLANTA LIBRE Thermal Zone Heating Setpoint',
        '3P LIVING Thermal Zone Heating Setpoint',
        '3P ESPARCIMIENTO Thermal Zone Heating Setpoint',
    ]
    
    cooling_agents = [
        'PB BUFETE Thermal Zone Cooling Setpoint',
        'PB SALA DE REUNION N Thermal Zone Cooling Setpoint',
        'PB SALA DE REUNION NW Thermal Zone Cooling Setpoint',
        '1P CALL CENTER Thermal Zone Cooling Setpoint',
        '1P PLANTA LIBRE Thermal Zone Cooling Setpoint',
        '1P BOX B W Thermal Zone Cooling Setpoint',
        '1P BOX A NW Thermal Zone Cooling Setpoint',
        '2P BOX A NW Thermal Zone Cooling Setpoint',
        '2P PLANTA LIBRE Thermal Zone Cooling Setpoint',
        '3P LIVING Thermal Zone Cooling Setpoint',
        '3P ESPARCIMIENTO Thermal Zone Cooling Setpoint',
    ]
    
    if agent_id in cooling_agents:
        transform_action = 23 + action
    elif agent_id in heating_agents:
        transform_action = 21 - action
    else:
        ValueError('Agent id not valid.')
    
    return transform_action

def thermostat_dual_mass_flow_rate(agent_id, action):
    """This method take a discret action in the range of [0,6) and transforms it
    into a temperature of cooling or heating setpoints, depending the agent id 
    involve.
    """
    if agent_id == 'cooling_setpoint':
        transform_action = 23 + action
    elif agent_id == 'heating_setpoint':
        transform_action = 21 - action
    elif agent_id == 'AirMassFlowRate':
        transform_action = action/10
    else:
        return action
    
    return transform_action

def thermostat_dual_mass_flow_rate_VN(agent_id, action):
    """This method take a discret action in the range of [0,6) and transforms it
    into a temperature of cooling or heating setpoints, depending the agent id 
    involve.
    """
    if agent_id == 'cooling_setpoint':
        transform_action = 23 + action
    elif agent_id == 'heating_setpoint':
        transform_action = 21 - action
    elif agent_id == 'AirMassFlowRate':
        transform_action = action/10
    elif agent_id == 'opening_window_1' or agent_id == 'opening_window_2':
        transform_action = action/5
    else:
        return action
    
    return transform_action