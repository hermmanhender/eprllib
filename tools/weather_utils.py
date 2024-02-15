
import pandas as pd
from pandas.core.frame import DataFrame
import numpy as np

def weather_file(env_config: dict, weather_choice:int = np.random.randint(0,24)):
    """_summary_

    Args:
        env_config (dict): _description_
        weather_choice (int, optional): _description_. Defaults to np.random.randint(0,24).

    Returns:
        _type_: _description_
    """
    folder_path = env_config['weather_folder']
    # clima aleatorio
    if not env_config['is_test']:
        weather_path = [
            ['GEF_Formosa-hour-H1',-26.0,-58.2,64],
            ['GEF_Lujan_de_cuyo-hour-H1',-32.985,-68.93,1043],
            ['GEF_Rawson-hour-H1',-43.300,-65.075,6],
            ['GEF_Salta-hour-H1',-24.770,-65.470,1280],
            ['GEF_San_Miguel_de_Tucuman-hour-H1',-26.839,-65.209,435],
            ['GEF_San_Nicolas_de_los_Arroyos-hour-H1',-33.362,-60.245,29],
            ['GEF_Ushuaia-hour-H1',-54.800,-68.317,14],
            ['GEF_Zapalla-hour-H1',-38.871,-70.112,1051],
            
            ['GEF_Formosa-hour-H2',-26.0,-58.2,64],
            ['GEF_Lujan_de_cuyo-hour-H2',-32.985,-68.93,1043],
            ['GEF_Rawson-hour-H2',-43.300,-65.075,6],
            ['GEF_Salta-hour-H2',-24.770,-65.470,1280],
            ['GEF_San_Miguel_de_Tucuman-hour-H2',-26.839,-65.209,435],
            ['GEF_San_Nicolas_de_los_Arroyos-hour-H2',-33.362,-60.245,29],
            ['GEF_Ushuaia-hour-H2',-54.800,-68.317,14],
            ['GEF_Zapalla-hour-H2',-38.871,-70.112,1051],
            
            ['GEF_Formosa-hour-H3',-26.0,-58.2,64],
            ['GEF_Lujan_de_cuyo-hour-H3',-32.985,-68.93,1043],
            ['GEF_Rawson-hour-H3',-43.300,-65.075,6],
            ['GEF_Salta-hour-H3',-24.770,-65.470,1280],
            ['GEF_San_Miguel_de_Tucuman-hour-H3',-26.839,-65.209,435],
            ['GEF_San_Nicolas_de_los_Arroyos-hour-H3',-33.362,-60.245,29],
            ['GEF_Ushuaia-hour-H3',-54.800,-68.317,14],
            ['GEF_Zapalla-hour-H3',-38.871,-70.112,1051],
        ]
        
        latitud = weather_path[weather_choice][1]
        longitud = weather_path[weather_choice][2]
        altitud = weather_path[weather_choice][3]
        return folder_path+'/'+weather_path[weather_choice][0]+'.epw', latitud, longitud, altitud
    
    else:
        return folder_path+'/GEF_Lujan_de_cuyo-hour-H4.epw', -32.985,-68.93,1043

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