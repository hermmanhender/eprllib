

def natural_ventilation_action(central_action: int):
    """_summary_

    Args:
        central_action (int): _description_

    Returns:
        _type_: _description_
    """
    action_space = [
        [0,0],
        [0,1],
        [1,0],
        [1,1]
    ]
    return action_space[central_action]