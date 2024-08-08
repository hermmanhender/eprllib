"""This module is used to load an IDF file and recognize the content
of the file related with the agents that would be used in the eprllib
library.
"""

# Se debe cargar el path al archivo IDF y en el identificar los posibles
# actuadores, variables y metricas.

# Para esto se debe ejecutar un paso de tiempo la siulacion del IDF para
# generar los diferetnets tipos de archivos que EnergyPlus produce para
# verificar la configuracion de la simulacion.

# A partir de estas identificaciones, se podran establecer los espacios
# de observacion y accion, como asi tambien los agentes del entorno.

class EnergyPlusInputFile:

    def __initi__(self, idf_path:str, output_folder:str):
        # Se definen las variables de entrada
        self.path = idf_path
        self.output_folder = output_folder
        
        # Se establecen variables que almacenaran las propiedades del IDF
        self.variables = None
        self.meters = None
        self.actuators = None

        # Se inicia un paso de tiempo de simulacion en E+ para determinar las propiedades del IDF
        self.start()

    def start(self):
        # Se establece un estado para E+ API
        # self.state_argument = api.state_managment.init()
        # Se establece un punto de llamado para detener la simulacion en el primer paso de tiempo

        # Se establece la salida de la consola como False

        # Se genera el string de comando para iniiar la simulacion
        # Aqui se establece una propiedad de la clase que almacena la ubicacion de los archivos generados por E+
        
        # se inicia la simulacion
        pass