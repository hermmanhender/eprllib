"""
Window Shade Control Agent
==========================

This module contains the Window Shade Control agent that may 
involve in a dwelling.

The base class is :class:`~eprllib.Agents.ConventionalAgent.ConventionalAgent`.
"""

from typing import Dict, Any
from eprllib.Agents.ConventionalAgent import ConventionalAgent

class WindowShadeControl(ConventionalAgent):
    def __init__(
        self,
        config: Dict[str, Any],
    ):
        """
        Control of the shadows in windows.

        Args:
            config (Dict[str, Any]): This dictionary contains the configuration of the agent.
            It must contain the keys 'SP_temp', 'dT_up', 'dT_dn' and the keys that correspond to the
            temperature and solar radiation ('Ti' and 'Bw' respectively)variables in the 
            EnergyPlus model.
        """
        super().__init__(config)
        
    def compute_single_action(self, infos:Dict, prev_action:float) -> int:
        """
        This method allows an binary operation of a shadow (blind or shade) througt fixed rule
        based control.

        Args:
            infos (Dict): Dictionary that contains the observation of the environment needed to
            implement the control policy. In this case, the dictionary must contain the keys and 
            values corresponding to the variables of temperature and solar radiation in the EnergyPlus
            model.
            prev_action (float): Previous action applied by the agent in the environment.

        Returns:
            int: Return the action to be applied in the EnergyPlus model environment. (0 if must to not apply
            the shadow and 1 if the shadow must be apply. If Any Error apears, return -1. This could be used
            as a flag.
        """
        # Se obtiene la configuración
        SP_temp = self.config['SP_temp']
        dT_up = self.config['dT_up']
        dT_dn = self.config['dT_dn']
        
        Ti = infos[self.config['Ti']]
        Bw = infos[self.config['Bw']]
        
        #Control de la persiana
        if Ti >= (SP_temp + dT_up) and Bw == 0:
            action_p = 0 #Abrir la persiana
        elif Ti >= (SP_temp + dT_up) and Bw > 0:
            action_p = 1 #Cerrar la persiana
            
        elif Ti <= (SP_temp - dT_dn) and Bw == 0:
            action_p = 1
        elif Ti <= (SP_temp - dT_dn) and Bw > 0:
            action_p = 0
            
        elif Ti < (SP_temp + dT_up) and Ti > (SP_temp - dT_dn):
            action_p = prev_action

        else:
            print(f"Window shadow control fail. The policy apply was configured as:\n{self.config}\The infos dictionary was:\n{infos}\nand the previous action was:\n{prev_action}")
            action_p = -1
        
        return action_p