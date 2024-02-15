"""Here are contained all the conventional agents that are present in a dwelling. Each of them
presents actions to do in different devices.
"""
from typing import Dict, Any

class Conventional:
    
    def __init__(
        self,
        config: Dict[str,Any] = {'SP_temp': 24, 'dT_up': 2, 'dT_dn':2}
    ):
        """This agent perform conventional actions in an EnergyPlus model based on fixed rules
        that take into account the basics variables as temperature, radiation, humidity and others.
        
        Args:
            config (Dict[str,Any]): as minimum, this config dictionary must to have:
                'SP_temp': float, # set point temperature of comfort
                'dT_up': float, # upper limit for the comfort range relative with respect to the SP_temp. Must be always a possitive number.
                'dT_dn': float, # lower limit for the comfort range relative with respect to the SP_temp. Must be always a possitive number.
        
        Example:
        ```
        >>> from conventional import Conventional
        >>> agent = Conventional({'SP_temp': 24, 'dT_up': 2, 'dT_dn':2})
        >>> shade_action = agent.window_shade(Ti=32, Bw=450, action_p=1)
        ```
        """
        self.config = config
        
    def window_shade(self, Ti: float, Bw:float,action_p:int):
        """Esta función permite la operación binaria (completamente cerrada [On] o completamente
        abierta [Off]) de una persiana a partir de reglas fijas.

        Args:
            observacion (dict): El diccionario debe contener en su observación al menos los siguientes
            elementos:
                'Ti' es la temperatura interior
                'Bw' es la radiación solar directa que existe en el plano de la ventana
                'action_p' es el estado actual de la persiana

        Returns:
            int: Regresa la acción a ser aplicada al elemento en EnergyPlus (0 si abre y 1 si cierra). 
            Devuelve -1 si hay un error.
        """
        # Se obtiene la configuración
        SP_temp = self.config['SP_temp']
        dT_up = self.config['dT_up']
        dT_dn = self.config['dT_dn']
        
        #Control de la persiana
        if Ti >= (SP_temp + dT_up) and Bw == 0:
            action_p = 0 #Abrir la persiana
        elif Ti >= (SP_temp + dT_up) and Bw > 0:
            action_p = 1 #Cerrar la persiana
            
        elif Ti <= (SP_temp - dT_dn) and Bw == 0:
            action_p = 1
        elif Ti <= (SP_temp - dT_dn) and Bw > 0:
            action_p = 0
            
        elif Ti < (SP_temp + dT_up) and Ti > (SP_temp - dT_dn):
            action_p = action_p

        else:
            print("Control de la persiana fallido")
            action_p = -1
        
        return action_p

    def air_conditioner(self, Ti:float, action_aa:int):
        """Esta función permite la operación binaria (encendido [On] o apagado [Off]) de un equipo
        de aire acondicionado a partir de reglas fijas.

        Args:
            observacion (dict): El diccionario debe contener en su observación al menos los siguientes
            elementos:
                'Ti' es la temperatura interior
                'action_aa' es el estado actual de operación del aire acondicionado

        Returns:
            int: Regresa la acción a ser aplicada al elemento en EnergyPlus (0 si apaga y 1 si prende). 
            Devuelve -1 si hay un error.
        """
        # Se obtiene la configuración
        SP_temp = self.config['SP_temp']
        dT_up = self.config['dT_up']
        dT_dn = self.config['dT_dn']
        
        if Ti >= (SP_temp + dT_up):
            action_aa = 1

        elif Ti <= (SP_temp - dT_dn):
            action_aa = 0

        elif Ti < (SP_temp + dT_up) and Ti > (SP_temp - dT_dn):
            action_aa = action_aa

        else:
            print("Control de Aire Acondicionado Fallido")
            action_aa = -1
        
        return action_aa

    def window_opening(self, Ti: float, To: float, action_v: int):
        """Esta función permite la operación binaria (encendido [On] o apagado [Off]) de 
        una ventana a partir de reglas fijas.

        Args:
            observacion (dict): El diccionario debe contener en su observación al menos los siguientes
            elementos:
                'Ti' es la temperatura interior
                'To' es la temperatura exterior
                'action_v' es el estado actual de la ventana

        Returns:
            int: Regresa la acción a ser aplicada al elemento en EnergyPlus (0 si cierra y 1 si abre). 
            Devuelve -1 si hay un error.
        """
        # Se obtiene la configuración
        SP_temp = self.config['SP_temp']
        dT_up = self.config['dT_up']
        dT_dn = self.config['dT_dn']
        
        if Ti >= (SP_temp + dT_up):
            if Ti > To:
                action_v = 1
            else:
                action_v = 0

        elif Ti <= (SP_temp - dT_dn):
            if Ti > To:
                action_v = 0
            else:
                action_v = 1

        elif Ti < (SP_temp + dT_up) and Ti > (SP_temp - dT_dn):
            action_v = action_v

        else:
            print("Control de Ventana Fallido")
            action_v = -1
        
        return action_v