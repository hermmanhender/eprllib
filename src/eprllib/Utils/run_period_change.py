"""
Run period change
=================

This function is used to modify the RunPeriod longitude of an epJSON file.
"""
import numpy as np
import pandas as pd
from typing import Dict
from eprllib.Utils.from_julian_day import from_julian_day

def run_period_change(
    env_config:Dict,
    init_julian_day:int|str = None,
    end_julian_day:int = None,
    init_month: int = None,
    init_day: int = None,
    end_month:int = None,
    end_day:int = None,
    simulation_duration:int = None
    ) -> Dict:
    """
    This function is used to modify the RunPeriod longitude of a epJSON file.
    
    Args:
        epjson_file(str): path to the epJSON file.
        output_folder(str): path to the destination folder where the modified file will be saved.
        episode_len(int)[Optional]: longitude of the RunPeriod, or episode in the context of eprllib. Default is 7.
        init_julian_day(int): The initial julian day to determine the RunPeriod. Defaut is 0, that means a random choice.
        
    Return:
        str: path to the modified epJSON file.
    """
    env_configuration = env_config
    
    # Check that (init_julian_day or (init_month and init_day)) and (end_julian_day or (end_month and end_day)) are not both None
    if (init_julian_day is None and (init_month is None or init_day is None)) or (end_julian_day is None and (end_month is None or end_day is None)):
        raise ValueError("Both init_julian_day or (init_month and init_day) and end_julian_day or (end_month and end_day) must be provided.")
    # check that if end_julian_day is None and (end_month is None or end_day is None) -> simulation_duration must be provided
    if end_julian_day is None and (end_month is None or end_day is None) and simulation_duration is None:
        raise ValueError("If end_julian_day is None and (end_month is None or end_day is None), simulation_duration must be provided.")
    # check that if simulation_duration is provided, it must be an integer and lower that 364-init_julian_day
    if simulation_duration is not None and (not isinstance(simulation_duration, int) or simulation_duration > 364-init_julian_day):
        raise ValueError("simulation_duration must be an integer lower than 364-init_julian_day.")
    
    # Open the epjson file
    with open(env_configuration['epjson_path']) as epf:
        epjson_object = pd.read_json(epf)
        
    # Transform the julian day into day,month tuple
    if init_julian_day != None:
        if init_julian_day == 'random':
            if simulation_duration != None:
                init_julian_day = np.random.randint(1, 366-simulation_duration)
                init_day, init_month = from_julian_day(init_julian_day)
                end_day, end_month = from_julian_day(init_julian_day + simulation_duration)
            else:
                # raise an error when 'random' is used but simulation_duration is not provided
                raise ValueError("If init_julian_day is 'random', simulation_duration must be provided.")
        elif isinstance(init_julian_day, int):
            init_day, init_month = from_julian_day(init_julian_day)
            if end_julian_day != None:
                end_day, end_month = from_julian_day(end_julian_day)
            elif simulation_duration != None:
                end_julian_day = init_julian_day + simulation_duration
                end_day, end_month = from_julian_day(end_julian_day)
            else:
                # check that end_day and end_month are different that None
                if end_day is None or end_month is None:
                    raise ValueError("If end_julian_day is None, simulation_duration must be provided.")
                
        else:
            raise ValueError("init_julian_day must be an integer or 'random'.")
        
    # Change the values in the epjson file
    epjson_object['RunPeriod']['RunPeriod 1']['beging_month'] = init_month
    epjson_object['RunPeriod']['RunPeriod 1']['begin_day_of_month'] = init_day
    epjson_object['RunPeriod']['RunPeriod 1']['end_month'] = end_month
    epjson_object['RunPeriod']['RunPeriod 1']['end_day_of_month'] = end_day
    # Save the epjson file modified into the output folder
    df = pd.DataFrame(epjson_object)
    env_configuration['epjson_path'] = env_configuration['output_path'] + f'/epjson_file_{init_julian_day}.epjson'
    df.to_json(env_configuration['epjson_path'], orient='records')

    return env_configuration