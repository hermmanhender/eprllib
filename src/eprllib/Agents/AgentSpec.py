"""
Defining agents
================

This module implements the classes to define agents. Agents are defined by the ``AgentSpec`` class. This class
contains the observation, filter, action, trigger, and reward specifications. The observation is defined by the
``ObservationSpec`` class. The filter is defined by the ``FilterSpec`` class. The action is defined by the ``ActionSpec`` class.
The trigger is defined by the ``TriggerSpec`` class. The reward is defined by the ``RewardSpec`` class.

The ``AgentSpec`` class has a method called ``build`` that is used to build the ``AgentSpec`` object. This method is used to
validate the properties of the object and to return the object as a dictionary. It is used internally when you build
the environment to provide it to RLlib.
"""
from typing import Dict, Any, Optional
from eprllib.Agents.Rewards.RewardSpec import RewardSpec
from eprllib.Agents.Filters.FilterSpec import FilterSpec
from eprllib.Agents.ActionSpec import ActionSpec
from eprllib.Agents.Triggers.TriggerSpec import TriggerSpec
from eprllib.Agents.ObservationSpec import ObservationSpec
from eprllib import logger

class AgentSpec:
    """
    AgentSpec is the base class for an agent specification to safe configuration of the object.
    """
    def __init__(
        self,
        observation: Optional[ObservationSpec] = None,
        filter: Optional[FilterSpec] = None,
        action: Optional[ActionSpec] = None,
        trigger: Optional[TriggerSpec] = None,
        reward: Optional[RewardSpec] = None,
    **kwargs: Any) -> None:
        """
        Contruction method for the AgentSpec class.

        Args:
            observation (ObservationSpec, optional): Defines the observation of the agent using
            the ObservationSpec class or a Dict. Defaults to NotImplemented.
            filter (FilterSpec, optional): Defines the filter for this agent using FilterSpec or a Dict. Defaults to None.
            action (ActionSpec, optional): Defines the action characteristics of the agent using ActionSpec or a Dict. Defaults to NotImplemented.
            trigger (TriggerSpec, optional): Defines the trigger for the agent using TriggerSpec or a Dict. Defaults to None.
            reward (RewardSpec, optional): Defines the reward elements of the agent using RewardSpec or a Dict. Defaults to NotImplemented.

        Raises:
            NotImplementedError: _description_
            NotImplementedError: _description_
            NotImplementedError: _description_
        """
        if observation is None:
            logger.info("AgentSpec: No observation defined. Using default observation.")
            self.observation: ObservationSpec|Dict[str, Any] = ObservationSpec()
        else:
            self.observation = observation
        if filter is None:
            logger.info("AgentSpec: No filter defined. Using default filter.")
            self.filter: FilterSpec|Dict[str, Any] = FilterSpec()
        else:
            self.filter = filter
        if action is None:
            logger.info("AgentSpec: No action defined. Using default action.")
            self.action: ActionSpec|Dict[str, Any] = ActionSpec()
        else:
            self.action = action
        if trigger is None:
            logger.info("AgentSpec: No trigger defined. Using default trigger.")
            self.trigger: TriggerSpec|Dict[str, Any] = TriggerSpec()
        else:
            self.trigger = trigger
        if reward is None:
            logger.info("AgentSpec: No reward defined. Using default reward.")
            self.reward: RewardSpec|Dict[str, Any] = RewardSpec()
        else:
            self.reward = reward
        
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def __getitem__(self, key:str):
        return getattr(self, key)

    def __setitem__(self, key:str, value:Any):
        setattr(self, key, value)
        
        
    def build(self) -> Dict[str, Any]:
        """
        This method is used to build the AgentSpec object.
        """
        if isinstance(self.observation, ObservationSpec):
            self.observation = self.observation.build()
        else:
            msg = f"AgentSpec: The observation must be defined as an ObservationSpec object but {type(self.observation)} was given."
            logger.error(msg)
            raise ValueError(msg)
        if isinstance(self.filter, FilterSpec):
            self.filter = self.filter.build()
        else:
            msg = f"AgentSpec: The filter must be defined as a FilterSpec object but {type(self.filter)} was given."
            logger.error(msg)
            raise ValueError(msg)
        if isinstance(self.action, ActionSpec):
            self.action = self.action.build()
        else:
            msg = f"AgentSpec: The action must be defined as an ActionSpec object but {type(self.action)} was given."
            logger.error(msg)
            raise ValueError(msg)
        if isinstance(self.trigger, TriggerSpec):
            self.trigger = self.trigger.build()
        else:
            msg = f"AgentSpec: The trigger must be defined as a TriggerSpec object but {type(self.trigger)} was given."
            logger.error(msg)
            raise ValueError(msg)
        if isinstance(self.reward, RewardSpec):
            self.reward = self.reward.build()
        else:
            msg = f"AgentSpec: The reward must be defined as a RewardSpec object but {type(self.reward)} was given."
            logger.error(msg)
            raise ValueError(msg)

        return vars(self)
