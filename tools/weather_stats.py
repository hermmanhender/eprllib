
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
        self.ten_rows_added = False
        self.agregar_primeras_10_filas()
        
    def agregar_primeras_10_filas(self):
        # Obtener las primeras 10 filas del DataFrame
        primeras_10_filas = self.weather_file.head(240)
        # Agregar las primeras 10 filas al DataFrame original
        self.weather_file = pd.concat([self.weather_file, primeras_10_filas], ignore_index=True)
        self.ten_rows_added = True


    # Paso 1: Filtrar los datos para el día juliano dado y los próximos 9 días
    def filtrar_por_dia_juliano(self, dia_juliano):
        
        # Calcular el día juliano para cada fila del DataFrame
        if self.ten_rows_added:
            dias_julianos = ((self.weather_file.index % 9240) // 24 + 1)
        else:
            dias_julianos = (self.weather_file.index % 8760) // 24 + 1
        # Verificar si el día juliano está dentro del rango deseado
        return dias_julianos.isin(range(dia_juliano, dia_juliano + 10))

    def ten_days_predictions(self, julian_day:int):
        interest_variables = [6, 8, 20, 21, 22, 33]
        # 0'Year', 1'Month', 2'Day', 3'Hour', 4'Minutes', 5'Data Source and Uncertainty Flags', 
        # 6'Dry Bulb Temperature', 7'Dew Point Temperature', 8'Relative Humidity', 
        # 9'Atmospheric Station Pressure', 10'Extraterrestrial Horizontal Radiation', 
        # 11'Extraterrestrial Direct Normal Radiation', 12'Horizontal Infrared Radiation Intensity', 
        # 13'Global Horizontal Radiation', 14'Direct Normal Radiation', 15'Diffuse Horizontal Radiation', 
        # 16'Global Horizontal Illuminance', 17'Direct Normal Illuminance', 18'Diffuse Horizontal Illuminance', 
        # 19'Zenith Luminance', 20'Wind Direction', 21'Wind Speed', 22'Total Sky Cover', 23'Opaque Sky Cover', 
        # 24'Visibility', 25'Ceiling Height', 26'Present Weather Observation', 27'Present Weather Codes', 
        # 28'Precipitable Water', 29'Aerosol Optical Depth', 30'Snow Depth', 31'Days Since Last Snowfall', 
        # 32'Albedo', 33'Liquid Precipitation Depth', 34'Liquid Precipitation Quantity',
        datos_filtrados: DataFrame = self.weather_file[self.filtrar_por_dia_juliano(julian_day)][interest_variables]
        lista_de_datos: list = datos_filtrados.values.tolist()
        lista_final = []
        for e in range(len(lista_de_datos)):
            for v in lista_de_datos[e]:
                lista_final.append(v)
        
        desviacion = [1, 10, 20, 0.5, 10, 0.2]
        prob_index = 0
        for e in range(len(lista_final)):
            lista_final[e] = np.random.normal(lista_final[e], desviacion[prob_index])
            if prob_index == (len(desviacion)-1):
                prob_index = 0 
            else:
                prob_index += 1
        
        array_final = np.array(lista_final)
        return array_final