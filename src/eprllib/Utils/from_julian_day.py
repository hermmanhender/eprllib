

def from_julian_day(julian_day:int):
    """
    This funtion take a julian day and return the corresponding
    day and month for a tipical year of 365 days.
    
    Args:
        julian_day(int): Julian day to be transform
        
    Return:
        Tuple[int,int]: (day,month)
    """
    # Define the number of days in each month
    days_in_months = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    # Define the day variable as equal to julian day and discount it
    day = julian_day
    for month, days_in_month in enumerate(days_in_months):
        if day <= days_in_month:
            return (day, month + 1)
        day -= days_in_month