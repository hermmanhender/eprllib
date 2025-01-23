"""
Episode Functions Utils
========================

Work in progress...
"""

from typing import Dict, List, Tuple

def load_ep_model(model_path):
    """
    Load the episode model from the given path.

    Args:
        model_path (str): The path to the episode model.

    Returns:
        Dict: The loaded episode model.
    """
    import json
    
    # Check that the file exists and finish with .epJSON
    if not model_path.exists():
        raise FileNotFoundError(f'The file {model_path} does not exist')
    
    if not model_path.endswith('.epJSON'):
        raise ValueError('The file must be a .epJSON file')
    
    # Open the file
    with open(model_path, 'rb') as f:
        model: Dict = json.load(f)
    return model
