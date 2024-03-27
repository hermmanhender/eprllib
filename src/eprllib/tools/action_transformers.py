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
        transform_action = 21 - 
    else:
        transform_action = -1
        print('The agent id it is not in the list of agents for this action_transform_method. The agent allowed are cooling_setpoint and heating_setpoint. Please notice that.')
    return transform_action