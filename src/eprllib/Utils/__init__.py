"""
Utilities
==========

Work in progress...
"""
from typing import Dict, Any

# --- CONFIGURING USER PROFILES BY ZONE ---
# Each profile now distinguishes between day and night zones.
# The sum of occupants in both zones at a given hour reflects the distribution of people.

OCCUPATION_PROFILES: Dict[str, Dict[str, Any]] = {
    "Single with an office job": {
        "total_people": 1,
        "zonas_daytime": {
            "weekdays":        [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0],
            "weekends": [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        },
        "zonas_nightly": {
            "semana":        [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
            "weekends": [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
        },
    },
    "Typical family, office job": { # 2 adults, 2 children
        "total_people": 4,
        "zone_daytime": {
            "weekdays":        [0, 0, 0, 0, 0, 0, 4, 4, 0, 0, 0, 0, 0, 0, 2, 2, 4, 4, 4, 4, 2, 0, 0, 0],
            "weekends": [0, 0, 0, 0, 0, 0, 1, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2, 0, 0, 0],
        },
        "zone_nightly": {
            "weekdays":        [4, 4, 4, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 4, 4, 4, 4],
            "weekends": [4, 4, 4, 4, 4, 4, 3, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 4, 4, 4, 4],
        },
    },
    # More profiles can be added here with the same structure.
}

# Validation list for allowed user types.
VALID_USER_TYPES = list(OCCUPATION_PROFILES.keys())
VALID_ZONE_TYPES = ["daytime", "nightly"]
