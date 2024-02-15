

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

def natural_ventilation_central_action(action1: int, action2: int):
    """_summary_

    Args:
        action1 (int): _description_
        action2 (int): _description_

    Returns:
        _type_: _description_
    """
    action_space = [
        [0,0],
        [0,1],
        [1,0],
        [1,1]
    ]
    index = 0
    for a in action_space:
        if a == [action1, action2]:
            central_action = index
            break
        else:
            index += 1
    
    return central_action