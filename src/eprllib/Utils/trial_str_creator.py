"""
Trial Name
===========

This utility method create a name for the RLlib trial experiment.
"""

from ray.tune.experiment.trial import Trial

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
    # Validar que name sea una cadena
    if not isinstance(name, str):
        raise ValueError(f"'name' must be a string, got {type(name).__name__}")

    # Validar que trial tenga los atributos esperados
    if not (hasattr(trial, 'trainable_name') and hasattr(trial, 'trial_id')):
        raise ValueError("The 'trial' argument must have 'trainable_name' and 'trial_id' attributes.")

    # Validar los tipos de los atributos de trial
    if not isinstance(trial.trainable_name, str):
        raise ValueError(f"'trial.trainable_name' must be a string, got {type(trial.trainable_name).__name__}")
    if not isinstance(trial.trial_id, str):
        raise ValueError(f"'trial.trial_id' must be a string, got {type(trial.trial_id).__name__}")
    
    return "{}_{}_{}".format(name, trial.trainable_name, trial.trial_id)
