"""Action spaces for diferent agents that operate devices in a centralized way.
"""

class DualSetPoint:
    def __init__(self):
        self.action_space = []
        for n in range(17,28,1):
            for m in range(18,29,1):
                if m > n:
                    self.action_space.append([n,m])
    
    def dual_action(self, central_action: int):
        return self.action_space[central_action]

class TwoWindowsCentralizedControl:
    def __init__(self):
        self.action_space = [
            [0,0],
            [0,1],
            [1,0],
            [1,1]
        ]
    
    def natural_ventilation_action(self, central_action: int):
        """_summary_

        Args:
            central_action (int): _description_

        Returns:
            _type_: _description_
        """
        return self.action_space[central_action]

    def natural_ventilation_central_action(self, action1: int, action2: int):
        """_summary_

        Args:
            action1 (int): _description_
            action2 (int): _description_

        Returns:
            _type_: _description_
        """
        index = 0
        for a in self.action_space:
            if a == [action1, action2]:
                central_action = index
                break
            else:
                index += 1
        
        return central_action