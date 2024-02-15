

class User:
    
    def __init__(
        self,
        config: dict,
    ):
        self.config = config
        
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