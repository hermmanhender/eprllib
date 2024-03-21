import gymnasium as gym

class Holon:
    def __init__(self, config: dict):
        self.observation_space: gym.Space = config['observation_space']
        self.action_space: gym.Space = config['action_space']
        self._holonparts: dict = config['holonparts']
        self._superholons: dict = config['superholons']
        self._authoritylist: dict = config['authoritylist']
        
    def holonparts(self) -> dict:
        return self._holonparts
    
    def superholons(self) -> dict:
        return self._superholons
    
    def authoritylist(self) -> dict:
        return self._authoritylist
    
    def addParts(self, **kargs: dict) -> None:
        for arg in kargs:
            self._holonparts.append(arg)
    
    def removeParts(self):
        return
    def requestStructure(self):
        return
    def changeAuthorities(self):
        return
    def requestAuthorities(self):
        return
    def requestStatus(self):
        return
    def close(self):
        return