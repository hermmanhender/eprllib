
import pandas as pd
from pandas.core.frame import DataFrame
import numpy as np
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
    
class Probabilities():
    def __init__(
        self,
        env_config:dict
    ):
        self.env_config = env_config
        
        with open(self.env_config["epw"]) as file:
            self.weather_file: DataFrame = pd.read_csv(
                file,
                header = None,
                skiprows = 8
            )
            # Reading the weather epw file.
        self.ten_rows_added = False
        # Flag to be sure about the run of the next line.
        self.complement_10_days()
        
    def complement_10_days(self):
        """This method add rows to complement the predictions of the entire year of then days after the December 31th using the first 
        ten days of the year. For that, 240 rows are added because each day has 24 hours.
        """
        primeras_10_filas = self.weather_file.head(240)
        # Obtain the first 240 rows of the weather file.
        self.weather_file = pd.concat([self.weather_file, primeras_10_filas], ignore_index=True)
        # Add the rows to the same weather file.
        self.ten_rows_added = True
        # Put this flag in True mode.


    # Paso 1: Filtrar los datos para el día juliano dado y los próximos 9 días
    def julian_day_filter(self, dia_juliano: int):
        """This method implement a filter of the weather data based on the julian day `n` and create a NDarray with booleans with
        True values in the data filtered from `[n, n+10]` bouth inclusive.

        Args:
            dia_juliano (int): First julian day of the range filtered.

        Returns:
            np_ndarray_bool
        """
        if self.ten_rows_added:
            # The julian day of each row is calculated for a extended list with 10 days more.
            dias_julianos = ((self.weather_file.index % 9240) // 24 + 1)
        else:
            # The julian day of each row is calculated for a not extended.
            dias_julianos = (self.weather_file.index % 8760) // 24 + 1
        # Check if the Julian day is within the desired range and return
        return dias_julianos.isin(range(dia_juliano, dia_juliano + 10))

    def ten_days_predictions(self, julian_day: int):
        """This method calculate the probabilies of six variables list bellow with a normal probability based on the desviation 
        of the variable.
        
            Dry Bulb Temperature in °C with desviation of 1 °C, 
            Relative Humidity in % with desviation of 10%, 
            Wind Direction in degree with desviation of 20°, 
            Wind Speed in m/s with desviation of 0.5 m/s, 
            Total Sky in % Cover with desviation of 10%, 
            Liquid Precipitation Depth in mm with desviation of 0.2 mm.

        Args:
            julian_day (int): First julian day of the range of ten days predictions.

        Returns:
            NDArray: Array with the ten days predictions. The size of the array is a sigle shape with 1440 values.
        """
        interest_variables = [6, 8, 20, 21, 22, 33]
        # This corresponds with the epw file order.
        filtered_data: DataFrame = self.weather_file[self.julian_day_filter(julian_day)][interest_variables]
        # Filter the data whith the julian day of interes and ten days ahead.
        data_list: list = filtered_data.values.tolist()
        # Transform the DataFrame into a list. This list contain a list for each hour, but as an observation of a single shape in
        # the RLlib configuration, the list is transform into a new one with only a shape.
        single_shape_list = []
        for e in range(len(data_list)):
            for v in data_list[e]:
                single_shape_list.append(v)
                # append each value of each day and hour in a consecutive way in the empty list.
        desviation = [1, 10, 20, 0.5, 10, 0.2]
        # Assignation of the desviation for each variable, in order with the epw variables consulted.
        prob_index = 0
        for e in range(len(single_shape_list)):
            single_shape_list[e] = np.random.normal(single_shape_list[e], desviation[prob_index])
            if prob_index == (len(desviation)-1):
                prob_index = 0 
            else:
                prob_index += 1
        
        predictions = np.array(single_shape_list)
        # The prediction list is transformed in a Numpy Array to concatenate after with the rest of the observation variables.
        return predictions