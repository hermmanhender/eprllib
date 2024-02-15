
import json

class WindowOpenings:
    
    def __init__(
        self
    ):
        pass
        
    def set_actuators(self, env_config: dict):
        """This method read the epJSON_object and identify the surface names of the AirflowNetwork:MultiZone:Surface
        object in the epJSON file configuration.
        
        Airflow Network Openings (EnergyPlus documentation):
        
        An actuator called “AirFlow Network Window/Door Opening” is available with a control type 
        called “Venting Opening Factor.” It is available in models that have operable openings in the Airflow
        Network model and that are entered by using either AirflowNetwork:MultiZone:Component:DetailedOpening,
        AirflowNetwork:MultiZone:Component:SimpleOpening, or AirflowNetwork:MultiZone:Component:HorizontalOpening
        input objects. This control allows you to use EMS to vary the size of the opening during the
        airflow model calculations, such as for natural and hybrid ventilation.
        The unique identifier is the name of the surface (window, door or air boundary), not the name of
        the associated airflow network input objects. The actuator control involves setting the value of the
        opening factor between 0.0 and 1.0. Use of this actuator with an air boundary surface is allowed,
        but will generate a warning since air boundaries are typically always open.

        Args:
            epJSON_object (dict): EnergyPlus epJSON input file.

        Returns:
            dict: Dictionary that contains the actuators to use in EnergyPlus Runner.
        """
        with open(env_config['idf']) as file:
            epJSON_object: dict = json.load(file)
            
        # indentufy the keys for every element in the AirflowNetwork Multizone Surface element
        openings = [key for key in epJSON_object['AirflowNetwork:MultiZone:Surface']]
        # define a emptly list to save there the names of the surfaces that are opened with the actuator
        names = []
        for opening in openings:
            names.append(epJSON_object['AirflowNetwork:MultiZone:Surface'][opening]['surface_name'])
        # define a dictionary that append the actuators elements to run in the EnergyPlus Runner implementation
        actuators = {}
        i = 1
        for name in names:
            actuators['opening_window_'+str(i)] = ("AirFlow Network Window/Door Opening", "Venting Opening Factor", name)
            i =+ 1
            
        return actuators
    
    def action(self):
        return
    
    def precense_schedule(self):
        """Perform and precense schedule for the user in the house.
        """
        return
    
    def temperature_schedule(self, hora:int):
        """Esta función establece las temperaturas del termostato de una vivienda según la hora del
        día.

        Args:
            obs (dict): El diccionario debe contener en su observación al menos los siguientes
            elementos:
                'hora' es la hora del día de 0 a 23

        Returns:
            int: Temperatura de calefacción
            int: Temperatura de refrigeración
        """
        if hora <= 7 or hora >= 23:
            Theat = 17
            Tchill = 28
        
        elif hora > 7 and hora < 23:
            Theat = 20
            Tchill = 25

        return Theat, Tchill