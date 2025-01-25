eprllib: EnergyPlus as RLlib multi-agent environment
=====================================================

The development of deep reinforcement learning (DRL) for optimal building control requires 
accurate models for correct learning of control policies for various devices.

eprllib has been developed to facilitate the implementation of DRL in models generated with 
EnergyPlus. For this purpose, RLlib is used as a framework for DRL.

Key Features
-------------

* Integration of EnergyPlus and RLlib: This package facilitates setting up a Reinforcement Learning environment 
  using EnergyPlus as the base, allowing for experimentation with energy control policies in buildings.
* Simplified Configuration: To use this environment, you simply need to provide a configuration in the form of a 
  dictionary that includes state variables, metrics, actuators (which will also serve as agents in the environment), 
  and other optional features.
* Flexibility and Compatibility: EnergyPlusRL easily integrates with RLlib, a popular framework for Reinforcement 
  Learning, enabling smooth setup and training of control policies for actionable elements in buildings.

.. toctree::
   :maxdepth: 3
   :caption: Contents:

   1-Overview
   2-GettingStarted
   3-KeyConcepts
   4-Environment
   5-Agents
   6-Actions
   7-Observations
   8-Rewards
   9-Episodes
   UserGuides/IntroductionGuides
   Examples/IntroductionExamples
   API Reference/eprllib
   