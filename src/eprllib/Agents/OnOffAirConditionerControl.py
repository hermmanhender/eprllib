"""
On-Off Air Conditioner Control Agent
====================================

This module contains the On-Off Air Conditioner Control agent that may 
involve in a dwelling.

The base class is :class:`~eprllib.Agents.ConventionalAgent.ConventionalAgent`.
"""

from typing import Dict, Any
from eprllib.Agents.ConventionalAgent import ConventionalAgent

class OnOffAirConditionerControl(ConventionalAgent):
    def __init__(
        self,
        config: Dict[str, Any],
    ):
        """
        Control of the Air Conditioner with an On-Off strategy.

        Args:
            config (Dict[str, Any]): This dictionary contains the configuration of the agent.
            It must contain the keys 'SP_temp', 'dT_up', 'dT_dn' and the keys that correspond to the
            temperature ('Ti') variable in the EnergyPlus model.
        """
        super().__init__(config)
        
    def compute_single_action(self, infos:Dict, prev_action) -> int:
        """
        This method allows an binary operation of an Air Conditioner througt fixed rule
        based control.

        Args:
            infos (Dict): Dictionary that contains the observation of the environment needed to
            implement the control policy. In this case, the dictionary must contain the keys and 
            values corresponding to the variables of zone temperature in the EnergyPlus model.
            prev_action (float): Previous action applied by the agent in the environment.

        Returns:
            int: Return the action to be applied in the EnergyPlus model environment. (0 if Off
            and 1 if On. If Any Error apears, return -1. This could be used as a flag.
        """
        
        # Se obtiene la configuración
        SP_temp = self.config['SP_temp']
        dT_up = self.config['dT_up']
        dT_dn = self.config['dT_dn']
        
        Ti = infos[self.config['Ti']]
        
        if Ti >= (SP_temp + dT_up):
            action_aa = 1

        elif Ti <= (SP_temp - dT_dn):
            action_aa = 0

        elif Ti < (SP_temp + dT_up) and Ti > (SP_temp - dT_dn):
            action_aa = prev_action

        else:
            print(f"On-Off Air Conditioner control fail. The policy apply was configured as:\n{self.config}\The infos dictionary was:\n{infos}\nand the previous action was:\n{prev_action}")
            action_aa = -1
        
        return action_aa