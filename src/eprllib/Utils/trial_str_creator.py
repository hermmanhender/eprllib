"""
Trial Name
===========

This utility method create a name for the RLlib trial experiment.
"""

from ray.tune.experiment.trial import Trial
from eprllib import logger
from eprllib.Utils.annotations import trial_str_creator_for_tune

def trial_str_creator(trial: Trial, name:str='eprllib'):
    """
    This method create a description for the folder where the outputs and checkpoints 
    will be save.

    Args:
        trial: A trial type of RLlib.
        name (str): Optional name for the trial. Default: eprllib

    Returns:
        str: Return a unique string for the folder of the trial.
    """
    logger.warning("TrialStrCreator: This method is deprecated. Use trial_str_creator_for_tune instead.")
        
    trial_str_creator_for_tune(trial, name)
    
