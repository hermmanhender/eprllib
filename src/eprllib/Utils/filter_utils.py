"""
Filter Utilities
===================

Work in progress...
"""

import numpy as np

def to_sin_transformation(
    value: float,
    min: float,
    max: float
    ) -> float:
    """
    Transform a value to a sin function.
    
    Args:
        value (float): The value to transform.
        min (float): The minimum value of the range.
        max (float): The maximum value of the range.
        
    Returns:
        float: The transformed value.
    """
    return np.sin(2 * np.pi * (value - min) / (max - min))

def normalization_minmax(
    value: float, 
    min: float, 
    max: float,
    a: float = -1.0,
    b: float = 1.0
    ) -> float:
    """
    Normalize a value between a and b.
    
    Args:
        value (float): The value to normalize.
        min (float): The minimum value of the range.
        max (float): The maximum value of the range.
        a (float, optional): The lower bound of the normalization range. Defaults to -1.0.
        b (float, optional): The upper bound of the normalization range. Defaults to 1.0.
        
    Returns:
        float: The normalized value.
    """
    return a + (b - a) * (value - min) / (max - min)

def desnormalization_minmax(
    value: float, 
    min: float, 
    max: float,
    a: float = -1.0,
    b: float = 1.0
    ) -> float:
    """
    Desnormalize a value between a and b.
    
    Args:
        value (float): The value to normalize.
        min (float): The minimum value of the range.
        max (float): The maximum value of the range.
        a (float, optional): The lower bound of the normalization range. Defaults to -1.0.
        b (float, optional): The upper bound of the normalization range. Defaults to 1.0.
        
    Returns:
        float: The desnormalized value.
    """
    return min + (max - min) * (value - a) / (b - a)