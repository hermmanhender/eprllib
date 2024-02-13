
import pandas as pd
from tools.tools import plus_day

class weather_function():
    """_summary_
    """
    def climatic_stads(epw_file_path, month):
        """_summary_

        Args:
            epw_file_path (_type_): _description_
            month (_type_): _description_

        Returns:
            _type_: _description_
        """
        print('Calculando los datos climáticos estadísticos para el archivo de clima EPW.')
        epw_file = pd.read_csv(epw_file_path,
                            header = None, #['Year', 'Month', 'Day', 'Hour', 'Minutes', 'Data Source and Uncertainty Flags', 'Dry Bulb Temperature', 'Dew Point Temperature', 'Relative Humidity', 'Atmospheric Station Pressure', 'Extraterrestrial Horizontal Radiation', 'Extraterrestrial Direct Normal Radiation', 'Horizontal Infrared Radiation Intensity', 'Global Horizontal Radiation', 'Direct Normal Radiation', 'Diffuse Horizontal Radiation', 'Global Horizontal Illuminance', 'Direct Normal Illuminance', 'Diffuse Horizontal Illuminance', 'Zenith Luminance', 'Wind Direction', 'Wind Speed', 'Total Sky Cover', 'Opaque Sky Cover', 'Visibility', 'Ceiling Height', 'Present Weather Observation', 'Present Weather Codes', 'Precipitable Water', 'Aerosol Optical Depth', 'Snow Depth', 'Days Since Last Snowfall', 'Albedo', 'Liquid Precipitation Depth', 'Liquid Precipitation Quantity'],
                            skiprows = 8
                            )
        output = {}
        if month == 2:
            days = range(1,29,1)
        elif month in [4,6,9,11]:
            days = range(1,31,1)
        else:
            days = range(1,32,1)
        for day in days:
            dict = {
                    str(day):
                        {
                        'T_max_0': weather_function.tmax(epw_file, day, month),
                        'T_min_0': weather_function.tmin(epw_file, day, month),
                        'RH_0': weather_function.rh_avg(epw_file, day, month),
                        'raining_total_0': weather_function.rain_tot(epw_file, day, month),
                        'wind_avg_0': weather_function.wind_avg(epw_file, day, month),
                        'wind_max_0': weather_function.wind_max(epw_file, day, month),
                        'total_sky_cover_0': weather_function.total_sky_cover(epw_file, day, month),
                        }
                    }
            output.update(dict)
        print('Se finalizó la creación de los estadísticos.')
        return output
        
    def tmax(epw_file, day, month, day_p=0):
        """_summary_

        Args:
            epw_file (_type_): _description_
            day (_type_): _description_
            month (_type_): _description_
            day_p (int, optional): _description_. Defaults to 0.

        Returns:
            _type_: _description_
        """
        day, month = plus_day(day, month, day_p)
        array = []
        for _ in range(0, 8760,1):
            if epw_file[1][_] == month and epw_file[2][_] == day:
                array.append(epw_file[6][_])
        tmax = max(array)
        return tmax

    def tmin(epw_file, day, month, day_p=0):
        """_summary_

        Args:
            epw_file (_type_): _description_
            day (_type_): _description_
            month (_type_): _description_
            day_p (int, optional): _description_. Defaults to 0.

        Returns:
            _type_: _description_
        """
        day, month = plus_day(day, month, day_p)
        array = []
        for _ in range(0, 8760,1):
            if epw_file[1][_] == month and epw_file[2][_] == day:
                array.append(epw_file[6][_])
        tmin = min(array)
        return tmin

    def rh_avg(epw_file, day, month, day_p=0):
        """_summary_

        Args:
            epw_file (_type_): _description_
            day (_type_): _description_
            month (_type_): _description_
            day_p (int, optional): _description_. Defaults to 0.

        Returns:
            _type_: _description_
        """
        day, month = plus_day(day, month, day_p)
        array = []
        for _ in range(0, 8760,1):
            if epw_file[1][_] == month and epw_file[2][_] == day:
                array.append(epw_file[8][_])
        rh_avg = sum(array)/len(array)
        return rh_avg

    def rain_tot(epw_file, day, month, day_p=0):
        """_summary_

        Args:
            epw_file (_type_): _description_
            day (_type_): _description_
            month (_type_): _description_
            day_p (int, optional): _description_. Defaults to 0.

        Returns:
            _type_: _description_
        """
        day, month = plus_day(day, month, day_p)
        array = []
        for _ in range(0, 8760,1):
            if epw_file[1][_] == month and epw_file[2][_] == day:
                array.append(epw_file[33][_])
        rain_tot = sum(array)
        return rain_tot

    def wind_avg(epw_file, day, month, day_p=0):
        """_summary_

        Args:
            epw_file (_type_): _description_
            day (_type_): _description_
            month (_type_): _description_
            day_p (int, optional): _description_. Defaults to 0.

        Returns:
            _type_: _description_
        """
        day, month = plus_day(day, month, day_p)
        array = []
        for _ in range(0, 8760,1):
            if epw_file[1][_] == month and epw_file[2][_] == day:
                array.append(epw_file[21][_])
        wind_avg = sum(array)/len(array)
        return wind_avg

    def wind_max(epw_file, day, month, day_p=0):
        """_summary_

        Args:
            epw_file (_type_): _description_
            day (_type_): _description_
            month (_type_): _description_
            day_p (int, optional): _description_. Defaults to 0.

        Returns:
            _type_: _description_
        """
        day, month = plus_day(day, month, day_p)
        array = []
        for _ in range(0, 8760,1):
            if epw_file[1][_] == month and epw_file[2][_] == day:
                array.append(epw_file[21][_])
        wind_max = max(array)
        return wind_max
    
    def total_sky_cover(epw_file, day, month, day_p=0):
        """_summary_

        Args:
            epw_file (_type_): _description_
            day (_type_): _description_
            month (_type_): _description_
            day_p (int, optional): _description_. Defaults to 0.

        Returns:
            _type_: _description_
        """
        day, month = plus_day(day, month, day_p)
        array = []
        for _ in range(0, 8760,1):
            if epw_file[1][_] == month and epw_file[2][_] == day:
                array.append(epw_file[22][_])
        total_sky_cover = sum(array)/len(array)
        return total_sky_cover