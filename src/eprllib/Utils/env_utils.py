
from typing import List
from eprllib.Utils import VALID_USER_TYPES, OCCUPATION_PROFILES, VALID_ZONE_TYPES
from numpy import random
from datetime import datetime, timedelta


def _calculate_occupancy_once(
    current_time: int, 
    current_day_type: int, 
    current_month: int, 
    current_holiday: bool, 
    user_type: str, 
    zone_type: str,
    probability_variation: float,
    probability_variation_evening_night_hours: float,
    summer_months: List[int]
) -> int:
    """
    Internal function that calculates the number of occupants for a single stochastic simulation.
    """
    profile = OCCUPATION_PROFILES[user_type]
    total_people_profile = profile["total_people"]
    
    # Selecting the correct zone profile
    profile_zone = profile[f"zone_{zone_type}"]

    if current_holiday or current_day_type in [6, 7]:
        base_schedule = profile_zone["weekends"]
    else:
        base_schedule = profile_zone["weekdays"]
    
    base_occupation = base_schedule[current_time]

    if random.random() < probability_variation:
        # Random variation cannot exceed the total number of people in the profile.
        if base_occupation < total_people_profile and random.random() < 0.5:
            base_occupation += 1
        elif base_occupation > 0:
            base_occupation -= 1

    late_evening_hours = range(18, 23)
    if current_month in summer_months and current_time in late_evening_hours and base_occupation > 0:
        if random.random() < probability_variation_evening_night_hours:
            base_occupation -= 1

    return max(0, min(base_occupation, total_people_profile))


def calculate_occupancy(
    current_time: int,
    current_day: int,
    current_month: int,
    current_year: int,
    current_holiday: bool,
    user_type: str,
    zone_type: str,
    num_simulations: int = 100,
    probability_variation: float = 0.15,
    probability_variation_evening_night_hours: float = 0.20,
    summer_months: List[int] = [6, 7, 8, 9]
) -> int:
    """
    Calculates current occupancy and a probability forecast for a specific area type.

    Args:
        current_time (int): Current time (0-23).
        current_day (int): Day of the month (1-31).
        current_month (int): Month of the year (1-12).
        current_year (int): Current year (e.g., 2023).
        current_holiday (bool): True if the current day is a holiday.
        user_type (str): The occupancy profile to use.
        zone_type (str): The type of zone to simulate ('day' or 'night').
        num_simulations (int): Number of iterations for the probability calculation.
        probability_variation (float): Probability that occupancy will vary within an hour.
        probability_variation_evening_night_hours (float): Probability that occupancy will vary in the evening/evening hours.
        summer_months (List[int]): List of months considered summer months (1-12).

    Returns:
        tuple[int, list[float]]: Una tupla conteniendo:
            - int: El número estimado de personas presentes en la hora actual.
            - list[float]: Una lista con 24 valores de probabilidad (0-1) de ocupación.
    """
    # --- INPUTS VALIDATION ---
    if user_type not in VALID_USER_TYPES:
        raise ValueError(f"User type '{user_type}' is not valid. Options: {VALID_USER_TYPES}")
    if zone_type not in VALID_ZONE_TYPES:
        raise ValueError(f"Zone type '{zone_type}' is not valid. Options: {VALID_ZONE_TYPES}")

    current_time_obj = datetime(current_year, current_month, current_day, current_time)
    current_day_type = current_time_obj.weekday() + 1

    current_occupation = _calculate_occupancy_once(
        current_time, current_day_type, current_month, current_holiday, user_type, zone_type,
        probability_variation, probability_variation_evening_night_hours, summer_months
    )

    return current_occupation

def calculate_occupancy_forecast(
    current_time: int,
    current_day: int,
    current_month: int,
    current_year: int,
    current_holiday: bool,
    user_type: str,
    zone_type: str,
    num_simulations: int = 100,
    probability_variation: float = 0.15,
    probability_variation_evening_night_hours: float = 0.20,
    summer_months: List[int] = [6, 7, 8, 9]
) -> list[float]:
    """
    Calculates current occupancy and a probability forecast for a specific area type.

    Args:
        current_time (int): Current time (0-23).
        current_day (int): Day of the month (1-31).
        current_month (int): Month of the year (1-12).
        current_year (int): Current year (e.g., 2023).
        current_holiday (bool): True if the current day is a holiday.
        user_type (str): The occupancy profile to use.
        zone_type (str): The type of zone to simulate ('day' or 'night').
        num_simulations (int): Number of iterations for the probability calculation.
        probability_variation (float): Probability that occupancy will vary within an hour.
        probability_variation_evening_night_hours (float): Probability that occupancy will vary in the evening/evening hours.
        summer_months (List[int]): List of months considered summer months (1-12).

    Returns:
        tuple[int, list[float]]: Una tupla conteniendo:
            - int: El número estimado de personas presentes en la hora actual.
            - list[float]: Una lista con 24 valores de probabilidad (0-1) de ocupación.
    """
    # --- INPUTS VALIDATION ---
    if user_type not in VALID_USER_TYPES:
        raise ValueError(f"User type '{user_type}' is not valid. Options: {VALID_USER_TYPES}")
    if zone_type not in VALID_ZONE_TYPES:
        raise ValueError(f"Zone type '{zone_type}' is not valid. Options: {VALID_ZONE_TYPES}")

    current_time_obj = datetime(current_year, current_month, current_day, current_time)
    
    forecast_probabilities: list[float] = []
    for h in range(1, 25):
        future_time = current_time_obj + timedelta(hours=h)
        is_a_future_holiday = False  # Simplification for forecasting

        times_busy = 0
        for _ in range(num_simulations):
            simulated_occupants = _calculate_occupancy_once(
                future_time.hour, future_time.weekday() + 1, future_time.month, 
                is_a_future_holiday, user_type, zone_type,
                probability_variation, probability_variation_evening_night_hours, summer_months
            )
            if simulated_occupants > 0:
                times_busy += 1
        
        forecast_probabilities.append(times_busy / num_simulations)

    return forecast_probabilities
