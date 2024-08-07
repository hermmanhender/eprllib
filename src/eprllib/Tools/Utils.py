
from typing import Set, Dict
import pandas as pd
import numpy as np

def trial_str_creator(trial, name:str='eprllib'):
    """This method create a description for the folder where the outputs and checkpoints 
    will be save.

    Args:
        trial: A trial type of RLlib.
        name (str): Optional name for the trial. Default: eprllib

    Returns:
        str: Return a unique string for the folder of the trial.
    """
    return "{}_{}_{}".format(name, trial.trainable_name, trial.trial_id)

def len_episode(env_config:Dict) -> str:
    """This function is used to modify the RunPeriod longitude of a epJSON file.
    
    Args:
        epjson_file(str): path to the epJSON file.
        output_folder(str): path to the destination folder where the modified file will be saved.
        episode_len(int)[Optional]: longitude of the RunPeriod, or episode in the context of eprllib. Default is 7.
        init_julian_day(int): The initial julian day to determine the RunPeriod. Defaut is 0, that means a random choice.
        
    Return:
        str: path to the modified epJSON file.
    """
    epjson_file = env_config['epjson']
    output_folder = env_config['output']
    episode_len = env_config.get('episode_len',7)
    init_julian_day = env_config.get('init_julian_day', 0)
    # Open the epjson file
    with open(epjson_file) as epf:
        epjson_object = pd.read_json(epf)
    # Transform the julian day into day,month tuple
    if init_julian_day <= 0:
        init_julian_day = np.random.randint(1, 366-episode_len)
    init_day, init_month = from_julian_day(init_julian_day)
    # Calculate the final day and month
    end_julian_day = init_julian_day + episode_len
    end_day, end_month = from_julian_day(end_julian_day)
    # Change the values in the epjson file
    epjson_object['RunPeriod']['RunPeriod 1']['beging_month'] = init_month
    epjson_object['RunPeriod']['RunPeriod 1']['begin_day_of_month'] = init_day
    epjson_object['RunPeriod']['RunPeriod 1']['end_month'] = end_month
    epjson_object['RunPeriod']['RunPeriod 1']['end_day_of_month'] = end_day
    # Save the epjson file modified into the output folder
    df = pd.DataFrame(epjson_object)
    output_path = output_folder + f'/epjson_file_{init_julian_day}.epjson'
    df.to_json(output_path, orient='records')

    print(f"The epjson file with the RunPeriod modified was saved in: {output_path}.")

    return output_path

def from_julian_day(julian_day:int):
    """This funtion take a julian day and return the corresponding
    day and month for a tipical year of 365 days.
    
    Args:
        julian_day(int): Julian day to be transform
        
    Return:
        Tuple[int,int]: (day,month)
        
    Example:
    >>> from_julian_day(90)
    31,3
    """
    # Define the number of days in each month
    days_in_months = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    # Define the day variable as equal to julian day and discount it
    day = julian_day
    for month, days_in_month in enumerate(days_in_months):
        if day <= days_in_month:
            return (day, month + 1)
        day -= days_in_month
        
def variable_checking(
    epJSON_file:str,
) -> Set:
    """This function check if the epJSON file has the required variables.

    Args:
        epJSON_file(str): path to the epJSON file.

    Return:
        set: list of missing variables.
    """
    