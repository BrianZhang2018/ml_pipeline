"""
Model Loader Module

This module handles loading and caching machine learning models for our API.

What Is This? (Explain Like I'm 5)
===============================
This is like a toy box that keeps all our AI brains ready to use. Just like
you keep your favorite toys easily accessible so you can play with them
quickly, we keep our AI models loaded and ready so we can use them without
waiting for them to get ready each time.
"""

import sys
import os
from typing import Dict, Any, Optional, List
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.utils.logger import get_logger

# Get logger instance
logger = get_logger("model_loader")

# Global cache for loaded models
_model_cache: Dict[str, Dict[str, Any]] = {}


def load_model(model_name: str, max_length: int = 512, try_local_first: bool = True) -> Dict[str, Any]:
    """
    Load a pre-trained model and tokenizer.
    
    This function loads a model and tokenizer, trying local cache first to avoid
    network dependencies, with fallback to online download if needed.
    
    Args:
        model_name (str): Name of the model to load
        max_length (int): Maximum sequence length for the tokenizer
        try_local_first (bool): Whether to try loading from local cache first
        
    Returns:
        Dict[str, Any]: Dictionary containing the model, tokenizer, and metadata
    """
    cache_key = f"{model_name}_{max_length}"
    
    # Check if model is already cached in memory
    if cache_key in _model_cache:
        logger.info(f"Model '{model_name}' found in memory cache")
        return _model_cache[cache_key]
    
    logger.info(f"Loading model '{model_name}' with max_length={max_length}")
    
    model = None
    tokenizer = None
    
    # Try loading from local cache first (offline-first approach)
    if try_local_first:
        try:
            logger.info(f"Attempting to load '{model_name}' from local cache (offline mode)")
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                local_files_only=True  # Force offline mode
            )
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                local_files_only=True  # Force offline mode
            )
            logger.info(f"Successfully loaded '{model_name}' from local cache")
        except Exception as local_error:
            logger.warning(f"Failed to load '{model_name}' from local cache: {str(local_error)}")
            logger.info(f"Falling back to online download for '{model_name}'")
    
    # Fallback to online download if local loading failed or was skipped
    if model is None or tokenizer is None:
        try:
            logger.info(f"Downloading '{model_name}' from Hugging Face Hub")
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            logger.info(f"Successfully downloaded '{model_name}' from online")
        except Exception as online_error:
            logger.error(f"Failed to load model '{model_name}' both locally and online: {str(online_error)}")
            raise
    
    # Set max length
    tokenizer.model_max_length = max_length
    
    # Get model information
    num_labels = model.config.num_labels
    model_type = model.config.model_type
    
    # Store in memory cache
    _model_cache[cache_key] = {
        "model": model,
        "tokenizer": tokenizer,
        "model_name": model_name,
        "model_type": model_type,
        "num_labels": num_labels,
        "max_length": max_length,
        "loaded": True
    }
    
    logger.info(f"Model '{model_name}' cached successfully")
    return _model_cache[cache_key]


def get_model_info(model_name: str, max_length: int = 512) -> Dict[str, Any]:
    """
    Get information about a loaded model.
    
    Args:
        model_name (str): Name of the model
        max_length (int): Maximum sequence length
        
    Returns:
        Dict[str, Any]: Model information
    """
    cache_key = f"{model_name}_{max_length}"
    
    if cache_key in _model_cache:
        model_data = _model_cache[cache_key]
        return {
            "model_name": model_data["model_name"],
            "model_type": model_data["model_type"],
            "num_labels": model_data["num_labels"],
            "max_length": model_data["max_length"],
            "loaded": model_data["loaded"]
        }
    else:
        return {
            "model_name": model_name,
            "model_type": "unknown",
            "num_labels": 0,
            "max_length": max_length,
            "loaded": False
        }


def clear_model_cache() -> None:
    """
    Clear the model cache.
    
    This function removes all loaded models from the cache.
    """
    global _model_cache
    logger.info("Clearing model cache")
    _model_cache.clear()
    logger.info("Model cache cleared successfully")


def get_cached_models() -> List[str]:
    """
    Get a list of cached models.
    
    Returns:
        List[str]: List of cached model names
    """
    return list(_model_cache.keys())


# Example usage
if __name__ == "__main__":
    # This is for testing purposes only
    print("Model Loader Module")
    print("Available functions:")
    print("- load_model")
    print("- get_model_info")
    print("- clear_model_cache")
    print("- get_cached_models")