"""
Trial Name
===========

This utility method create a name for the RLlib trial experiment.
"""

from ray.tune.experiment.trial import Trial
from eprllib import logger

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
    # Validar que trial tenga los atributos esperados
    if not (hasattr(trial, 'trainable_name') and hasattr(trial, 'trial_id')):
        msg = "TrialStrCreator: The 'trial' argument must have 'trainable_name' and 'trial_id' attributes."
        logger.error(msg)
        raise ValueError(msg)
    
    return "{}_{}_{}".format(name, trial.trainable_name, trial.trial_id)
