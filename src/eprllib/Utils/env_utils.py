"""
Environment Utilities
=====================

"""
from typing import List, Dict
from .constants import OCCUPATION_PROFILES
from numpy import random, exp
from datetime import datetime, timedelta

def _calculate_occupancy_once(
    current_time_step_in_hour: int,
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
    profile_zone: Dict[str, Dict[int,List[float]]] = profile[f"zone_{zone_type}"]
    # Selecting the correct type of day
    if current_holiday or current_day_type in [5, 6]:
        base_schedule = profile_zone["weekends"]
    else:
        base_schedule = profile_zone["weekdays"]
    # Selecting the occupation based on the current time
    base_occupation = base_schedule[current_hour][current_time_step_in_hour-1]

    return base_occupation


def calculate_occupancy(
    current_time_step_in_hour: int,
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
        current_time_step_in_hour (int): From 1 to the maximal time steps per hour configured in the thermal zone.
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
        current_time_step_in_hour = current_time_step_in_hour,
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
    lambdaa: float = 0.05,
    num_time_steps_in_hour: int = 6
) -> Dict[int, Dict[int, float]]:
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
            - dict[int, listfloat]]: Un diccionario con claves para la hora y una lista con valores de probabilidad (0-1) de ocupación para cada paso de tiempo.
    """
    current_time_obj = datetime(current_year, current_month, current_day, current_hour)

    forecast_probabilities: Dict[int,Dict[int, float]] = {h: {} for h in range(24)}

    for h in range(24):
        confidence_level_h = 0.5+(confidence_level-0.5)*exp(-lambdaa*h)
        future_time = current_time_obj + timedelta(hours=h)
        is_a_future_holiday = False  # Simplification for forecasting

        for timestep in range(1, num_time_steps_in_hour+1):

            simulated_occupants = _calculate_occupancy_once(
                current_time_step_in_hour = timestep,
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

            forecast_probabilities[h].update({
                timestep: simulated_occupants
            })

    return forecast_probabilities
