"""
Conventional Agent
==================

This module contains the base class of the conventional agents.
"""
from typing import Dict, Any

class ConventionalAgent:
    
    def __init__(
        self,
        config: Dict[str,Any],
        **kargs,
    ):
        """
        This agent perform conventional actions in an EnergyPlus model based on fixed rules
        that take into account the basics variables as temperature, radiation, humidity and others.
        
        Args:
            config (Dict[str,Any]): Here you can set the configuration of the agent.
        """
        self.config = config
    
    def compute_single_action(self, infos:Dict[str, int|float], **kargs) -> int|float:
        """Implement here your own function."""
        raise NotImplementedError
