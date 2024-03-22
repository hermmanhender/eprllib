
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