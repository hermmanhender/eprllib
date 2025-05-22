"""
Test script para verificar la simulación de EnergyPlus usando la copia temporal
creada por EP_API_add_path de env_config_utils.py.
"""
import sys
import os
import shutil
import tempfile

# Asegúrate de que el directorio 'src' de 'eprllib' esté en PYTHONPATH
# para que la siguiente importación funcione.
# Por ejemplo, si este script está en eprllib/tests/utils/
# y env_config_utils.py está en eprllib/src/eprllib/Utils/,
# necesitarás añadir eprllib/src al PYTHONPATH.
# Alternativamente, si eprllib está instalado, esto debería funcionar directamente.

# Añadimos el directorio src de eprllib al sys.path
# para que podamos importar el módulo env_config_utils.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from eprllib.Utils.env_config_utils import EP_API_add_path

# Path to the test weather file
WEATHER_FILE_PATH = "C:/Users/grhen/Documents/GitHub2/eprllib/tests/data/USA_CA_San.Francisco.Intl.AP.724940_TMY3.epw"
# Contenido mínimo para un archivo IDF
IDF_FILE_PATH = "C:/Users/grhen/Documents/GitHub2/eprllib/tests/data/1ZoneDataCenterCRAC_wApproachTemp.idf"

def test_ep_api_path_simulation():
    print("Iniciando test_ep_api_path_simulation...")
    # Directorio temporal para los archivos de esta prueba (IDF, salida de E+)
    temp_test_dir = tempfile.mkdtemp(prefix="ep_sim_test_")
    print(f"Directorio temporal de prueba creado: {temp_test_dir}")

    # Opcional: Especifica una ruta conocida a una instalación de EnergyPlus.
    # Si es None, EP_API_add_path intentará la autodetección.
    # Para entornos de prueba específicos o CI/CD, proporcionar esta ruta es más robusto.
    # Ejemplo: known_ep_path = "C:/EnergyPlusV23-2-0"
    known_ep_path = None 

    temp_ep_api_path = None # Para almacenar la ruta de la copia temporal de E+

    try:
        # --- 1. Configurar el entorno de EnergyPlus usando EP_API_add_path ---
        print(f"Llamando a EP_API_add_path con path: {known_ep_path}")
        # Esta llamada es crucial: modifica sys.path y crea la copia temporal.
        temp_ep_api_path = EP_API_add_path(path=known_ep_path)
        print(f"EP_API_add_path devolvió la ruta temporal de E+: {temp_ep_api_path}")
        
        if not os.path.isdir(temp_ep_api_path):
            raise AssertionError("La ruta temporal de E+ no existe o no es un directorio.")
        if temp_ep_api_path not in sys.path:
            raise AssertionError("La ruta temporal de E+ no está en sys.path.")

        # --- 2. Intentar importar la API de pyenergyplus ---
        # Esta importación debe ocurrir *después* de que EP_API_add_path haya modificado sys.path.
        print("Intentando importar EnergyPlusAPI...")
        from pyenergyplus.api import EnergyPlusAPI # type: ignore
        print("EnergyPlusAPI importada correctamente.")

        # --- 4. Ejecutar una simulación ---
        simulation_output_dir = os.path.join(temp_test_dir, "eplus_output")
        os.makedirs(simulation_output_dir, exist_ok=True)
        print(f"Directorio de salida de la simulación: {simulation_output_dir}")

        api = EnergyPlusAPI()
        state = api.state_manager.new_state() # Obtener una nueva instancia de estado

        # Argumentos para run_energyplus. El primer elemento es un placeholder.
        # EnergyPlus escribirá los archivos de salida en simulation_output_dir.
        # El IDD (Energy+.idd) debería ser encontrado automáticamente en la copia temporal.
        run_args = [
            "-w",
            WEATHER_FILE_PATH,
            "-d",
            simulation_output_dir,
            IDF_FILE_PATH
        ]
        print(f"Intentando ejecutar EnergyPlus con args: {run_args}")

        status = -1 # Por defecto, error
        try:
            # Llamada real a la simulación
            status = api.runtime.run_energyplus(state, run_args)
            print(f"run_energyplus completado con estado: {status}")
        except Exception as e:
            print(f"Excepción durante run_energyplus: {e}")
        finally:
            api.state_manager.delete_state(state) # Limpiar el estado de E+
            print("Estado de EnergyPlus eliminado.")

        # --- 5. Verificar simulación exitosa ---
        # Un estado de 0 usualmente significa éxito.
        # También, verificar la existencia de 'eplusout.end'.
        success_file = os.path.join(simulation_output_dir, "eplusout.end")
        simulation_succeeded = (status == 0 and os.path.exists(success_file))

        if simulation_succeeded:
            print(f"¡Simulación exitosa! Archivo de salida encontrado: {success_file}")
        else:
            print(f"Simulación fallida. Estado: {status}, Archivo de éxito existe: {os.path.exists(success_file)}")
            err_file_path = os.path.join(simulation_output_dir, "eplusout.err")
            if os.path.exists(err_file_path):
                print(f"--- Contenido de {err_file_path} ---")
                with open(err_file_path, "r") as f_err:
                    print(f_err.read())
                print("--- Fin del archivo de error ---")
            else:
                print(f"Archivo de error {err_file_path} no encontrado.")
        
        assert simulation_succeeded, "La simulación de EnergyPlus falló usando la ruta de API temporal."
        print("test_ep_api_path_simulation completado exitosamente.")
        return True

    except Exception as e:
        print(f"Error durante la prueba test_ep_api_path_simulation: {e}")
        import traceback
        traceback.print_exc()
        # Si EP_API_add_path llama a sys.exit(), el script terminará antes de llegar aquí
        # para ciertos errores.
        raise
    finally:
        # --- 6. Limpieza ---
        print(f"Limpiando directorio temporal de prueba: {temp_test_dir}")
        shutil.rmtree(temp_test_dir, ignore_errors=True)
        # La limpieza de la copia temporal de EnergyPlus (temp_ep_api_path)
        # es manejada por la función _cleanup_all_temp_ep_dirs registrada con atexit
        # en EP_API_add_path.

if __name__ == "__main__":
    # Para ejecutar este script:
    # 1. Asegúrate de que el directorio 'src' de tu proyecto 'eprllib' esté en PYTHONPATH.
    #    Por ejemplo, si tu proyecto está en 'c:\Users\grhen\Documents\GitHub2\eprllib\',
    #    añade 'c:\Users\grhen\Documents\GitHub2\eprllib\src' a PYTHONPATH.
    # 2. Ejecuta este script: python test_env_config_ep_simulation.py
    test_ep_api_path_simulation()