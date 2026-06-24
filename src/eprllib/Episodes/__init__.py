"""
Episode Functions
=================

This module contains the methods necesary to modify during runtime the environment for
each episode. This allow to program different training steps like requiered by
curriculum learning.
"""

from eprllib.Episodes.BaseEpisode import BaseEpisode
from eprllib.Episodes.DefaultEpisode import DefaultEpisode

__all__ = ["BaseEpisode", "DefaultEpisode"]