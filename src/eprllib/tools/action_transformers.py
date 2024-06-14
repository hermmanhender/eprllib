"""This script will be contain some action transformer methods to implement in
eprllib. Most of them are applied in the test section where examples to test the 
library are developed.
"""

def thermostat_dual(agent_id, action):
    """This method take a discret action in the range of [0,4) and transforms it
    into a temperature of cooling or heating setpoints, depending the agent id 
    involve.
    """
    if agent_id == 'cooling_setpoint':
        transform_action = 23 + action
    elif agent_id == 'heating_setpoint':
        transform_action = 21 - action
    elif agent_id == 'opening_window_1' or agent_id == 'opening_window_2':
        transform_action = action/3
    else:
        return action
    
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