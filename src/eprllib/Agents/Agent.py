"""
Agent
======

This module contains the Agent class, which is the base class for all agents in the library.
"""

from typing import Dict, Any
from eprllib.Agents.AgentSpec import AgentSpec

class Agent:
    def __init__(
        self,
        name: str,
        agent_config: Dict[str, Any] | AgentSpec = {},
        ):
        self.name = name
        self.agent_config = agent_config
    
    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)
    
    def get_name(self) -> str:
        return self.name
    
    def get_agent_config(self) -> Dict[str, Any]:
        if isinstance(self.agent_config, AgentSpec):
            return self.agent_config.build()
        else:
            return self.agent_config
    
    def build(self) -> Dict:            
        return vars(self)