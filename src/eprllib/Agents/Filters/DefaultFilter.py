"""
Default Filter
===============

This module contains the default filter class for preprocessing observations before they are fed to the agent.
The `DefaultFilter` class extends the `BaseFilter` class and provides a basic implementation that can be used
as-is or extended to create custom filters.
"""

from typing import Any, Dict
from eprllib.Agents.Filters.BaseFilter import BaseFilter

class DefaultFilter(BaseFilter):
    def __init__(
        self,
        filter_fn_config: Dict[str, Any]
    ):
        """
        Default filter class for preprocessing observations.

        Args:
            filter_fn_config (Dict[str, Any]): Configuration dictionary for the filter function.
        """
        super().__init__(filter_fn_config)