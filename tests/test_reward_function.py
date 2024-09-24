"""

"""
import sys
import os
from typing import Dict, Any
# Add the scr directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from eprllib.RewardFunctions import RewardFunctions, henderson_2024, EnergyTemperature

def test_RewardFunctionBase():
    """
    Test the RewardFunctionBase class.
    """
    reward_fn = RewardFunctions.RewardFunction()
    assert isinstance(reward_fn, RewardFunctions.RewardFunction)

def test_henderson_2024():
    """
    Test the RewardFunctionBase class.
    """
    reward_fn = henderson_2024.henderson_2024()
    assert isinstance(reward_fn, henderson_2024.henderson_2024)

def test_EnergyTemperature():
    """
    Test the RewardFunctionBase class.
    """
    reward_fn = EnergyTemperature.EnergyTemperatureReward()
    assert isinstance(reward_fn, EnergyTemperature.EnergyTemperatureReward)
