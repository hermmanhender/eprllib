"""Logging configuration for EPRL Library."""

import logging
from logging.handlers import RotatingFileHandler
import os
from typing import Optional

def setup_logging(
    log_path: str = "logs",
    log_level: int = logging.INFO,
    module_name: Optional[str] = None
) -> logging.Logger:
    """Setup logging configuration for the EPRL library.
    
    Args:
        log_path: Directory path where log files will be stored
        log_level: The logging level to use (e.g., logging.INFO)
        module_name: Optional name for the logger, uses root logger if None
    
    Returns:
        Configured logger instance
    """
    os.makedirs(log_path, exist_ok=True)
    
    # Get or create logger
    logger = logging.getLogger(module_name) if module_name else logging.getLogger()
    logger.setLevel(log_level)
    
    # Avoid duplicate handlers
    if not logger.handlers:
        # Create formatters
        detail_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
        )
        
        # File handler with rotation
        file_handler = RotatingFileHandler(
            os.path.join(log_path, f"{'eprllib' if not module_name else module_name}.log"),
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setFormatter(detail_formatter)
        file_handler.setLevel(log_level)
        logger.addHandler(file_handler)
        
        # Console handler with less verbose format
        console_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(console_formatter)
        console_handler.setLevel(max(log_level, logging.INFO))  # Console shows INFO and above
        logger.addHandler(console_handler)
    
    return logger