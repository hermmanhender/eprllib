"""
Filter Utilities
===================

Filters often involve normalization works. Here you can find the most 
common used normalization functions: sin and cos for Cyclical Feature Encoding, and
min_max for linear variables. Also here are provided the functions to get 
the original values previously normalized.

"""

import numpy as np

def to_sin_transformation(
    value: float,
    min_val: float,
    max_val: float
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
    return np.sin(2 * np.pi * (value - min_val) / (max_val - min_val))

def to_cos_transformation(
    value: float,
    min_val: float,
    max_val: float
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
    return np.cos(2 * np.pi * (value - min_val) / (max_val - min_val))

import numpy as np

def from_sin_cos_normalization(
    value_sin: float, 
    value_cos: float, 
    min_val: float|int, 
    max_val: float|int
    ) -> float:
    """
    Get the original value normalized with sin and cos.
    
    Args:
        value_sin (float): The sin transformed value.
        value_cos (float): The cos transformed value.
        min_val (float|int): The minimum value of the range.
        max_val (float|int): The maximum value of the range.
    
    Returns:
        float: The original value.
    """
    # 1. Calcular el rango original
    range = max_val - min_val
    
    # 2. Obtener el ángulo en radianes usando el arco tangente de dos parámetros
    # arctan2 gestiona automáticamente los signos para darnos el cuadrante correcto
    angulo = np.arctan2(value_sin, value_cos)
    
    # 3. Normalizar el ángulo al rango [0, 1]
    # np.mod(..., 2*pi) asegura que siempre sea positivo entre 0 y 2pi
    v_norm = np.mod(angulo, 2 * np.pi) / (2 * np.pi)
    
    # 4. Escalar al rango original y sumar el desplazamiento (offset)
    value = (v_norm * range) + min_val
    
    return value


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