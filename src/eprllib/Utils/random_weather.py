"""
Weather utilities
==================

Work in progress...
"""
import os
import numpy as np

def get_random_weather(epw_files_folder_path:str) -> str:
    """
    This method select a random epw file from the folder and return the path.

    Args:
        epw_files_folder_path (str): Folder that contain the epw files.

    Return:
        str: The path to the epw file selected
    """
    # Check that the folder has at least one file of type .epw
    if not any(file.endswith('.epw') for file in os.listdir(epw_files_folder_path)):
        raise ValueError("The folder does not contain any .epw files.")
    
    # select a random epw file that has the extension .epw
    id_epw_file = np.random.randint(0, len(os.listdir(epw_files_folder_path)))
    while not os.listdir(epw_files_folder_path)[id_epw_file].endswith('.epw'):
        id_epw_file = np.random.randint(0, len(os.listdir(epw_files_folder_path)))
    
    # The path to the epjson file is defined and returned
    return os.path.join(epw_files_folder_path, os.listdir(epw_files_folder_path)[id_epw_file])