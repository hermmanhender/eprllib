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
import logging
import sys
from typing import Dict

from eprllib.Agents.Rewards.RewardSpec import RewardSpec
from eprllib.Agents.Filters.FilterSpec import FilterSpec
from eprllib.Agents.ActionSpec import ActionSpec
from eprllib.Agents.Triggers.TriggerSpec import TriggerSpec
from eprllib.Agents.ObservationSpec import ObservationSpec

logger = logging.getLogger("ray.rllib")

class AgentSpec:
    """
    AgentSpec is the base class for an agent specification to safe configuration of the object.
    """
    def __init__(
        self,
        observation: ObservationSpec = None,
        filter: FilterSpec = None,
        action: ActionSpec = None,
        trigger: TriggerSpec = None,
        reward: RewardSpec = None,
        **kwargs):
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
            logger.info("No observation defined. Using default observation.")
            self.observation = ObservationSpec()
        else:
            self.observation = observation
        if filter is None:
            logger.info("No filter defined. Using default filter.")
            self.filter = FilterSpec()
        else:
            self.filter = filter
        if action is None:
            logger.info("No action defined. Using default action.")
            self.action = ActionSpec()
        else:
            self.action = action
        if trigger is None:
            logger.info("No trigger defined. Using default trigger.")
            self.trigger = TriggerSpec()
        else:
            self.trigger = trigger
        if reward is None:
            logger.info("No reward defined. Using default reward.")
            self.reward = RewardSpec()
        else:
            self.reward = reward
        
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)
        
        
    def build(self) -> Dict:
        """
        This method is used to build the AgentSpec object.
        """
        if isinstance(self.observation, ObservationSpec):
            self.observation = self.observation.build()
        else:
            logger.error(f"The observation must be defined as an ObservationSpec object but {type(self.observation)} was given.")
            raise ValueError(f"The observation must be defined as an ObservationSpec object but {type(self.observation)} was given.")
        if isinstance(self.filter, FilterSpec):
            self.filter = self.filter.build()
        else:
            logger.error(f"The filter must be defined as a FilterSpec object but {type(self.filter)} was given.")
            raise ValueError(f"The filter must be defined as a FilterSpec object but {type(self.filter)} was given.")
        if isinstance(self.action, ActionSpec):
            self.action = self.action.build()
        else:
            logger.error(f"The action must be defined as an ActionSpec object but {type(self.action)} was given.")
            raise ValueError(f"The action must be defined as an ActionSpec object but {type(self.action)} was given.")
        if isinstance(self.trigger, TriggerSpec):
            self.trigger = self.trigger.build()
        else:
            logger.error(f"The trigger must be defined as a TriggerSpec object but {type(self.trigger)} was given.")
            raise ValueError(f"The trigger must be defined as a TriggerSpec object but {type(self.trigger)} was given.")
        if isinstance(self.reward, RewardSpec):
            self.reward = self.reward.build()
        else:
            logger.error(f"The reward must be defined as a RewardSpec object but {type(self.reward)} was given.")
            raise ValueError(f"The reward must be defined as a RewardSpec object but {type(self.reward)} was given.")

        return vars(self)
