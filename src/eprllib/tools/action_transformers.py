"""This script will be contain some action transformer methods to implement in
eprllib. Most of them are applied in the test section where examples to test the 
library are developed.
"""

def thermostat_dual(action):
    """This method take a discret action in the range of [0,xx) and transform it
    into a temperature tuple of cooling and heating setpoints, respectively.
    """
    return cooling_action, heating_action