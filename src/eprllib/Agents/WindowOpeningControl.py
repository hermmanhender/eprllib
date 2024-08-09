"""
Window Opening Control Agent
============================

This module contains the Window Opening Control agent that may 
involve in a dwelling.

The base class is :class:`~eprllib.Agents.ConventionalAgent.ConventionalAgent`.
"""

from typing import Dict, Any
from eprllib.Agents.ConventionalAgent import ConventionalAgent

class WindowOpeningControl(ConventionalAgent):
    def __init__(
        self,
        config: Dict[str, Any],
    ):
        """
        Control of the openings in windows.

        Args:
            config (Dict[str, Any]): This dictionary contains the configuration of the agent.
            It must contain the keys 'SP_temp', 'dT_up', 'dT_dn' and the keys that correspond to the
            indoor and outdoor temperatures ('Ti' and 'To' respectively) variables in the 
            EnergyPlus model.
        """
        super().__init__(config)
        
    def compute_single_action(self, infos:Dict, prev_action) -> int:
        """
        This method allows an binary operation of the opening of a window througt fixed rule
        based control.

        Args:
            infos (Dict): Dictionary that contains the observation of the environment needed to
            implement the control policy. In this case, the dictionary must contain the keys and 
            values corresponding to the variables of indoor and outdoor temperatures in the 
            EnergyPlus model.
            prev_action (float): Previous action applied by the agent in the environment.

        Returns:
            int: Return the action to be applied in the EnergyPlus model environment. (0 if the  
            window is close and 1 if the window is open. If Any Error apears, return -1. This 
            could be used as a flag.
        """
        # Se obtiene la configuración
        SP_temp = self.config['SP_temp']
        dT_up = self.config['dT_up']
        dT_dn = self.config['dT_dn']
        
        Ti = infos[self.config['Ti']]
        To = infos[self.config['To']]
        
        if Ti >= (SP_temp + dT_up):
            if Ti > To:
                action_v = 1
            else:
                action_v = 0

        elif Ti <= (SP_temp - dT_dn):
            if Ti > To:
                action_v = 0
            else:
                action_v = 1

        elif Ti < (SP_temp + dT_up) and Ti > (SP_temp - dT_dn):
            action_v = prev_action

        else:
            print(f"Window opening control fail. The policy apply was configured as:\n{self.config}\The infos dictionary was:\n{infos}\nand the previous action was:\n{prev_action}")
            action_v = -1
        
        return action_v