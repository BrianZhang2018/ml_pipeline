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


def load_model(model_name: str, max_length: int = 512) -> Dict[str, Any]:
    """
    Load a pre-trained model and tokenizer.
    
    This function loads a model and tokenizer from the Hugging Face model hub
    and caches them for future use.
    
    Args:
        model_name (str): Name of the model to load
        max_length (int): Maximum sequence length for the tokenizer
        
    Returns:
        Dict[str, Any]: Dictionary containing the model, tokenizer, and metadata
    """
    cache_key = f"{model_name}_{max_length}"
    
    # Check if model is already cached
    if cache_key in _model_cache:
        logger.info(f"Model '{model_name}' found in cache")
        return _model_cache[cache_key]
    
    logger.info(f"Loading model '{model_name}' with max_length={max_length}")
    
    try:
        # Load model and tokenizer
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Set max length
        tokenizer.model_max_length = max_length
        
        # Get model information
        num_labels = model.config.num_labels
        model_type = model.config.model_type
        
        # Store in cache
        _model_cache[cache_key] = {
            "model": model,
            "tokenizer": tokenizer,
            "model_name": model_name,
            "model_type": model_type,
            "num_labels": num_labels,
            "max_length": max_length,
            "loaded": True
        }
        
        logger.info(f"Model '{model_name}' loaded successfully")
        return _model_cache[cache_key]
        
    except Exception as e:
        logger.error(f"Failed to load model '{model_name}': {str(e)}")
        raise


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