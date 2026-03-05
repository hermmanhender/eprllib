"""
Add EnergyPlus API path to sys.path for eprllib
================================================

This function manages the EnergyPlus API path for eprllib. It attempts to auto-detect the 
EnergyPlus installation based on the operating system and a provided version string. If a 
valid path is found, it adds it to `sys.path` for use by eprllib.
"""

from typing import Optional
import sys
from eprllib import logger
from eprllib.version import ep_version_list


def EP_API_add_path(path: Optional[str] = None) -> str:
    """
    Manages EnergyPlus paths for eprllib.

    If a 'path' argument is provided, it's used directly as the EnergyPlus installation path
    after validation.
    Otherwise, this method auto-detects installed EnergyPlus versions based on `LIST_OF_VERSIONS`.
    - If multiple versions are found, the latest one (as per `LIST_OF_VERSIONS`) is chosen,
      and a message is printed.
    - If no version is found, an error message is printed, and the program exits.

    The selected EnergyPlus installation (either user-provided or auto-detected) is then
    copied to a unique temporary directory. This temporary copy's path is added to `sys.path`,
    allowing isolated EnergyPlus environments for parallel execution.
    Temporary directories are registered for cleanup on program exit using `atexit`.

    Args:
        path (Optional[str], optional): Full path to an EnergyPlus installation directory.
            If None, auto-detection is performed. Defaults to None.

    Returns:
        str: The path to the temporary EnergyPlus copy that was added to `sys.path`.

    Raises:
        FileNotFoundError: If 'path' is provided but does not exist or is not a directory.
        RuntimeError: If auto-detection fails to find any suitable EnergyPlus installation,
                      or if copying the installation to a temporary directory fails.
    """
    logger.debug("EnvConfigUtils: Attempting to auto-detect EnergyPlus installation...")
    os_platform = sys.platform
    original_ep_path: Optional[str] = None
    
    # Check that the provided path is valid.
    if path not in ep_version_list:
        msg = f"EnvConfigUtils: Invalid EnergyPlus version provided: {path}. " \
              f"Valid versions are: {ep_version_list}"
        logger.error(msg)
        raise ValueError(msg)
    
    if os_platform.startswith("linux"):  # Covers "linux" and "linux2"
        original_ep_path = f"/usr/local/EnergyPlus-{path}"
    elif os_platform == "win32":
        original_ep_path = f"C:/EnergyPlusV{path}"
    elif os_platform == "darwin":
        original_ep_path = f"/Applications/EnergyPlus-{path}"

    if original_ep_path is not None:
        if original_ep_path not in sys.path:
            sys.path.insert(0, original_ep_path)
            logger.debug(f"EnvConfigUtils: EnergyPlus API path added to sys.path: {original_ep_path}")
        else:
            logger.debug(f"EnvConfigUtils: EnergyPlus API path already in sys.path: {original_ep_path}")
        
        return original_ep_path
    
    else:
        logger.error(f"EnvConfigUtils: Warning: EnergyPlus auto-detection is not configured for this OS: {os_platform}. "
                "Please provide the path manually if detection fails.")
        raise RuntimeError(f"EnergyPlus auto-detection failed for OS: {os_platform}. "
                           "Please provide the path manually or ensure EnergyPlus is installed correctly.")
        
