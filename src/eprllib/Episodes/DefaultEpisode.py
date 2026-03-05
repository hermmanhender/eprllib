"""
Default Episode
================

This module contains the default implementation of the episode functions for the EnergyPlus environment.
"""
from eprllib.Episodes.BaseEpisode import BaseEpisode
from eprllib.Utils.annotations import override

class DefaultEpisode(BaseEpisode):
    """
    This class provides the default implementation of the episode functions for the EnergyPlus environment.
    It inherits from the BaseEpisode class.
    """
    @override(BaseEpisode)
    def setup(self) -> None:
        pass
