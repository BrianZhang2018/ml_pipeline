"""
Logging System

This module provides a simple logging system for tracking
experiment progress and debugging issues.
"""

import logging
import os
from typing import Optional
from datetime import datetime

from .config import get_config


class LogManager:
    """
    LogManager for setting up and managing logging
    
    What Is This? (Explain Like I'm 5)
    ================================
    This is like a smart diary system for our AI project. It
    automatically writes down what the computer is doing,
    like "Started working on data" or "Finished training model".
    This helps us understand what happened if something goes wrong.
    """
    
    def __init__(self, name: str = "ml_pipeline", environment: str = None):
        """
        Initialize the log manager
        
        Args:
            name (str): Name for the logger
            environment (str): Environment to configure logging for
        """
        self.name = name
        self.environment = environment or get_config().get_environment()
        self.logger = None
        self._setup_logger()
    
    def _setup_logger(self):
        """
        Set up the logger with appropriate configuration
        """
        # Create logger
        self.logger = logging.getLogger(self.name)
        
        # Avoid adding handlers multiple times
        if self.logger.handlers:
            return
        
        # Set level based on environment
        if self.environment == 'prod':
            self.logger.setLevel(logging.INFO)
        else:
            self.logger.setLevel(logging.DEBUG)
        
        # Create logs directory if it doesn't exist
        log_dir = get_config().get('LOG_DIR', 'logs')
        os.makedirs(log_dir, exist_ok=True)
        
        # Create file handler
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"{self.name}_{timestamp}.log")
        file_handler = logging.FileHandler(log_file)
        
        # Create console handler
        console_handler = logging.StreamHandler()
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers to logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # Prevent propagation to root logger
        self.logger.propagate = False
    
    def debug(self, message: str):
        """
        Log a debug message
        
        Args:
            message (str): Message to log
        """
        self.logger.debug(message)
    
    def info(self, message: str):
        """
        Log an info message
        
        Args:
            message (str): Message to log
        """
        self.logger.info(message)
    
    def warning(self, message: str):
        """
        Log a warning message
        
        Args:
            message (str): Message to log
        """
        self.logger.warning(message)
    
    def error(self, message: str):
        """
        Log an error message
        
        Args:
            message (str): Message to log
        """
        self.logger.error(message)
    
    def critical(self, message: str):
        """
        Log a critical message
        
        Args:
            message (str): Message to log
        """
        self.logger.critical(message)
    
    def get_logger(self) -> logging.Logger:
        """
        Get the underlying logger instance
        
        Returns:
            logging.Logger: Logger instance
        """
        return self.logger


# Global log manager instance
_log_manager = None


def get_logger(name: str = "ml_pipeline", environment: str = None) -> LogManager:
    """
    Get the global log manager instance
    
    Args:
        name (str): Name for the logger
        environment (str): Environment to configure logging for
        
    Returns:
        LogManager: Log manager instance
    """
    global _log_manager
    if _log_manager is None:
        _log_manager = LogManager(name, environment)
    return _log_manager


def log_debug(message: str, name: str = "ml_pipeline"):
    """
    Log a debug message
    
    Args:
        message (str): Message to log
        name (str): Name for the logger
    """
    logger = get_logger(name)
    logger.debug(message)


def log_info(message: str, name: str = "ml_pipeline"):
    """
    Log an info message
    
    Args:
        message (str): Message to log
        name (str): Name for the logger
    """
    logger = get_logger(name)
    logger.info(message)


def log_warning(message: str, name: str = "ml_pipeline"):
    """
    Log a warning message
    
    Args:
        message (str): Message to log
        name (str): Name for the logger
    """
    logger = get_logger(name)
    logger.warning(message)


def log_error(message: str, name: str = "ml_pipeline"):
    """
    Log an error message
    
    Args:
        message (str): Message to log
        name (str): Name for the logger
    """
    logger = get_logger(name)
    logger.error(message)


def log_critical(message: str, name: str = "ml_pipeline"):
    """
    Log a critical message
    
    Args:
        message (str): Message to log
        name (str): Name for the logger
    """
    logger = get_logger(name)
    logger.critical(message)