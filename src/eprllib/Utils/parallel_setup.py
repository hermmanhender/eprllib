import shutil
import json
import platform
import multiprocessing
from pathlib import Path
from typing import Dict, Optional, Any
from eprllib import logger

def parallel_energyplus_setup(ep_base_path: str, num_workers: Optional[int] = None) -> Dict[int, str]:
    """
    Verifica y crea copias del directorio de EnergyPlus para ejecución en paralelo por cada Worker.
    
    :param ep_base_path: Ruta original de EnergyPlus (ej. '/usr/local/EnergyPlus-24-2-0').
    :param num_workers: Cantidad de Rollout Workers configurados en RLlib.
    :return: Diccionario mapeando {worker_id: ruta_al_directorio_copia}.
    """
    if num_workers is None:
        num_workers = multiprocessing.cpu_count()
        
    ep_path = Path(ep_base_path)
    if not ep_path.exists():
        msg = f"El directorio base de EnergyPlus no existe: {ep_path}"
        logger.error(msg)
        raise FileNotFoundError(msg)
        
    # Archivo de control para evitar recrear los directorios innecesariamente
    archivo_json = ep_path / "parallel_setup_workers.json"
    rutas_workers: Dict[int, str] = {}
    
    # 1. Comprobar si las copias ya están creadas
    if archivo_json.exists():
        try:
            with open(archivo_json, "r") as f:
                datos = json.load(f)
                if datos.get("num_workers", 0) >= num_workers:
                    print("Las copias de EnergyPlus ya existen. Cargando rutas...")
                    # Convertimos de nuevo las llaves de string a int (JSON guarda las llaves como string)
                    return {int(k): v for k, v in datos.get("rutas_workers", {}).items()}
        except Exception as e:
            msg = f"Error leyendo el JSON de configuración. Recreando directorios. Detalles: {e}"
            logger.warning(msg)

    logger.info(f"Generando directorios de EnergyPlus para {num_workers} workers...")
    
    # 2. Identificar archivos necesarios considerando el SO
    sistema = platform.system()
    
    # Archivos comunes a todos los sistemas
    archivos_base = ["Energy+.idd", "pyenergyplus"]
    
    if sistema == "Windows":
        archivos_os = [
            "EnergyPlus.exe", "EPMacro.exe", "ExpandObjects.exe",
            "DElight2.dll", "libexpat.dll", "bcvtb.dll"
        ]
    elif sistema == "Linux":
        archivos_os = [
            "energyplus", "EPMacro", "ExpandObjects",
            "libDElight2.so", "libbcvtb.so",
            "libexpat.so", "libexpat.so.1" # A veces Linux incluye la versión en el nombre
        ]
    elif sistema == "Darwin":  # MacOS
        archivos_os = [
            "energyplus", "EPMacro", "ExpandObjects",
            "libDElight2.dylib", "libbcvtb.dylib",
            "libexpat.dylib", "libexpat.1.dylib" 
        ]
    else:
        raise OSError(f"Sistema operativo no soportado: {sistema}")
    
    archivos_requeridos = archivos_base + archivos_os
    
    # 3. Crear directorios (sumamos 1 porque RLlib tiene un Local Worker con índice 0 y N Remote Workers)
    for worker_id in range(num_workers + 1):
        nombre_dir = f"{ep_path.name}_worker_{worker_id}"
        worker_dir = ep_path.parent / nombre_dir
        worker_dir.mkdir(parents=True, exist_ok=True)
        rutas_workers[worker_id] = str(worker_dir)
        
        for archivo in archivos_requeridos:
            origen = ep_path / archivo
            destino = worker_dir / archivo
            
            # Usamos exist() porque pusimos variaciones de libexpat. Copiará la que encuentre.
            if origen.exists():
                if origen.is_dir() and not destino.exists():
                    shutil.copytree(origen, destino)
                elif not origen.is_dir():
                    shutil.copy2(origen, destino)
            else:
                # Opcional: Avisar si falta un archivo crítico, aunque con las variaciones de libexpat es normal que falte alguna
                if archivo in ["energyplus", "EnergyPlus.exe", "Energy+.idd", "pyenergyplus"]:
                    logger.warning(f"ADVERTENCIA: No se encontró el archivo crítico '{archivo}' en {ep_path}")
                    
        # 4. Asegurar que Energy+.ini esté vacío
        with open(worker_dir / "Energy+.ini", "w") as f:
            f.write("")
            
    # 5. Registrar el estado en el JSON
    config: Dict[str, Any]= {
        "num_workers": num_workers,
        "base_path": str(ep_path),
        "rutas_workers": rutas_workers
    }
    with open(archivo_json, "w") as f:
        json.dump(config, f, indent=4)
        
    logger.info("Directorios generados exitosamente.")
    
    return rutas_workers
