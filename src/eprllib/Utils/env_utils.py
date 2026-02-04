
from typing import List, Dict
from eprllib.Agents import OCCUPATION_PROFILES
from numpy import random, exp
from datetime import datetime, timedelta

def _calculate_occupancy_once(
    current_hour: int, 
    current_day_type: int,
    current_holiday: bool, 
    user_type: str, 
    zone_type: str
) -> float:
    """
    Internal function that calculates the number of occupants for a single stochastic simulation.
    """
    profile = OCCUPATION_PROFILES[user_type]
    # Selecting the correct zone profile
    profile_zone: Dict[str, List[float]] = profile[f"zone_{zone_type}"]
    # Selecting the correct type of day
    if current_holiday or current_day_type in [5, 6]:
        base_schedule = profile_zone["weekends"]
    else:
        base_schedule = profile_zone["weekdays"]
    # Selecting the occupation based on the current time
    base_occupation = base_schedule[current_hour]
    
    return base_occupation


def calculate_occupancy(
    current_hour: int,
    current_day: int,
    current_month: int,
    current_year: int,
    current_holiday: bool,
    user_type: str,
    zone_type: str,
    confidence_level: float = 0.95
) -> float:
    """
    Calculates current occupancy and a probability forecast for a specific area type.

    Args:
        current_hour (int): Current time (0-23).
        current_day (int): Day of the month (1-31).
        current_month (int): Month of the year (1-12).
        current_year (int): Current year (e.g., 2023).
        current_holiday (bool): True if the current day is a holiday.
        user_type (str): The occupancy profile to use.
        zone_type (str): The type of zone to simulate ('day' or 'night').
        confidence_level (float): Confidence that the occupancy is the same that the deterministic.

    Returns:
        tuple[int, list[float]]: Una tupla conteniendo:
            - int: El número estimado de personas presentes en la hora actual.
            - list[float]: Una lista con 24 valores de probabilidad (0-1) de ocupación.
    """
    current_time_obj = datetime(current_year, current_month, current_day, current_hour)
    current_day_type = current_time_obj.weekday()

    current_occupation = _calculate_occupancy_once(
        current_hour = current_hour, 
        current_day_type = current_day_type,
        current_holiday = current_holiday, 
        user_type = user_type, 
        zone_type = zone_type
    )
    if current_occupation > 0:
        if random.random() > confidence_level:
            current_occupation = 0.
    else:
        if random.random() > confidence_level:
            current_occupation = 1.
    
    return current_occupation

def calculate_occupancy_forecast(
    current_hour: int,
    current_day: int,
    current_month: int,
    current_year: int,
    user_type: str,
    zone_type: str,
    occupation_prediction_hours: int = 24,
    confidence_level: float = 0.95,
    lambdaa: float = 0.05
) -> List[float]:
    """
    Calculates current occupancy and a probability forecast for a specific area type.

    Args:
        current_hour (int): Current time (0-23).
        current_day (int): Day of the month (1-31).
        current_month (int): Month of the year (1-12).
        current_year (int): Current year (e.g., 2023).
        current_holiday (bool): True if the current day is a holiday.
        user_type (str): The occupancy profile to use.
        zone_type (str): The type of zone to simulate ('day' or 'night').
        confidence_level (float): Confidence that the occupancy is the same that the deterministic.
        lambdaa (float): Decay value of the confidence level.

    Returns:
        tuple[int, list[float]]: Una tupla conteniendo:
            - int: El número estimado de personas presentes en la hora actual.
            - list[float]: Una lista con 24 valores de probabilidad (0-1) de ocupación.
    """
    current_time_obj = datetime(current_year, current_month, current_day, current_hour)
    
    forecast_probabilities: List[float] = []
    
    for h in range(1, occupation_prediction_hours+1):
        confidence_level_h = 0.5+(confidence_level-0.5)*exp(-lambdaa*h)
        future_time = current_time_obj + timedelta(hours=h)
        is_a_future_holiday = False  # Simplification for forecasting
        
        simulated_occupants = _calculate_occupancy_once(
            current_hour = future_time.hour, 
            current_day_type = future_time.weekday(),
            current_holiday = is_a_future_holiday, 
            user_type = user_type, 
            zone_type = zone_type
        )
        if simulated_occupants > 0:
            simulated_occupants = confidence_level_h
        else:
            simulated_occupants = 1 - confidence_level_h
            
        forecast_probabilities.append(simulated_occupants)
    
    return forecast_probabilities
