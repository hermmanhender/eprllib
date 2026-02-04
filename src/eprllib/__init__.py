"""
eprllib
========

``eprllib`` was born out of the need to bridge the gap between building modeling with 
**EnergyPlus** and Reinforcement Learning (**RL**). Traditionally, integrating these two 
disciplines has been complex and laborious. ``eprllib`` aims to simplify this process, 
offering an intuitive and flexible interface for developing intelligent agents that 
interact with building simulations.
"""
# Version management.
from .version import __version__, EP_VERSION

# Log configuration.
import logging
logger = logging.getLogger("ray.rllib") # See: https://docs.ray.io/en/latest/rllib/rllib-env.html#:~:text=(config))-,Tip,-When%20using%20logging