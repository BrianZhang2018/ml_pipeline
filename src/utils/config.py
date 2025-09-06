"""
Configuration Management System

This module provides a simple configuration management system
that loads settings based on the current environment.
"""

import os
import yaml
from typing import Dict, Any


class ConfigManager:
    """
    Configuration Manager for loading environment-specific settings
    
    What Is This? (Explain Like I'm 5)
    ================================
    This is like a smart toy box that knows which toys to give you
    based on whether you're playing at home, at a friend's house,
    or at school. It looks at where you are and gives you the right
    toys (settings) for that place.
    """
    
    def __init__(self, environment: str = None):
        """
        Initialize the configuration manager
        
        Args:
            environment (str): The environment to load config for (dev, test, prod)
                              If not provided, will use ENVIRONMENT env var or default to 'dev'
        """
        # If no environment specified, check environment variable or default to 'dev'
        self.environment = environment or os.getenv('ENVIRONMENT', 'dev')
        
        # Validate environment
        if self.environment not in ['dev', 'test', 'prod']:
            raise ValueError(f"Invalid environment: {self.environment}. Must be one of: dev, test, prod")
        
        # Load configuration
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """
        Load configuration from YAML file
        
        Returns:
            Dict[str, Any]: Configuration dictionary
        """
        config_path = f"configs/{self.environment}/config.yaml"
        
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            return config or {}
        except FileNotFoundError:
            print(f"Warning: Config file not found at {config_path}. Using empty config.")
            return {}
        except yaml.YAMLError as e:
            print(f"Error loading config file {config_path}: {e}")
            return {}
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value
        
        Args:
            key (str): Configuration key (e.g., 'MODEL_NAME')
            default (Any): Default value if key not found
            
        Returns:
            Any: Configuration value or default
        """
        return self.config.get(key, default)
    
    def get_nested(self, *keys: str, default: Any = None) -> Any:
        """
        Get a nested configuration value
        
        Args:
            *keys (str): Nested keys (e.g., 'DATABASE', 'HOST')
            default (Any): Default value if key not found
            
        Returns:
            Any: Configuration value or default
        """
        value = self.config
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def __getitem__(self, key: str) -> Any:
        """
        Get a configuration value using bracket notation
        
        Args:
            key (str): Configuration key
            
        Returns:
            Any: Configuration value
            
        Raises:
            KeyError: If key not found
        """
        if key not in self.config:
            raise KeyError(f"Configuration key '{key}' not found")
        return self.config[key]
    
    def __contains__(self, key: str) -> bool:
        """
        Check if a configuration key exists
        
        Args:
            key (str): Configuration key
            
        Returns:
            bool: True if key exists, False otherwise
        """
        return key in self.config
    
    def get_environment(self) -> str:
        """
        Get the current environment
        
        Returns:
            str: Current environment
        """
        return self.environment
    
    def get_all(self) -> Dict[str, Any]:
        """
        Get all configuration values
        
        Returns:
            Dict[str, Any]: All configuration values
        """
        return self.config.copy()


# Global configuration manager instance
_config_manager = None


def get_config(environment: str = None) -> ConfigManager:
    """
    Get the global configuration manager instance
    
    Args:
        environment (str): Environment to load config for
        
    Returns:
        ConfigManager: Configuration manager instance
    """
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager(environment)
    return _config_manager


def get_config_value(key: str, default: Any = None, environment: str = None) -> Any:
    """
    Get a configuration value from the global configuration manager
    
    Args:
        key (str): Configuration key
        default (Any): Default value if key not found
        environment (str): Environment to load config for (if None, uses existing config)
        
    Returns:
        Any: Configuration value or default
    """
    config = get_config(environment)
    return config.get(key, default)