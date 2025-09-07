"""
Model Builder Module

This module handles building transformer-based models for text classification
using Hugging Face Transformers.

What Is This? (Explain Like I'm 5)
===============================
This is like a LEGO set instruction manual. Just like you follow instructions
to build different LEGO models, this module has instructions to build different
AI brain models. We can build simple ones or more complex ones depending on
what we need.
"""

import sys
import os
from typing import Dict, Any, Optional
import logging

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.utils.logger import get_logger
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    AutoConfig
)

# Get logger instance
logger = get_logger("model_builder")


def build_model(
    model_name: str = "bert-base-uncased",
    num_labels: int = 2,
    cache_dir: Optional[str] = None,
    try_local_first: bool = True
) -> Dict[str, Any]:
    """
    Build a transformer-based model for text classification.
    
    Args:
        model_name (str): Name of the pre-trained model to use
        num_labels (int): Number of classification labels
        cache_dir (str, optional): Directory to cache models
        try_local_first (bool): Whether to try loading from local cache first
        
    Returns:
        Dict[str, Any]: Dictionary containing model, tokenizer, and config
    """
    logger.info(f"Building model '{model_name}' with {num_labels} labels")
    
    config = None
    tokenizer = None
    model = None
    
    # Try loading from local cache first (offline-first approach)
    if try_local_first:
        try:
            logger.info(f"Attempting to load '{model_name}' from local cache (offline mode)")
            
            # Load model configuration
            config = AutoConfig.from_pretrained(
                model_name,
                num_labels=num_labels,
                cache_dir=cache_dir,
                local_files_only=True  # Force offline mode
            )
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                use_fast=True,
                local_files_only=True  # Force offline mode
            )
            
            # Load model
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                config=config,
                cache_dir=cache_dir,
                local_files_only=True  # Force offline mode
            )
            
            logger.info(f"Successfully loaded '{model_name}' from local cache")
            
        except Exception as local_error:
            logger.warning(f"Failed to load '{model_name}' from local cache: {str(local_error)}")
            logger.info(f"Falling back to online download for '{model_name}'")
    
    # Fallback to online download if local loading failed or was skipped
    if config is None or tokenizer is None or model is None:
        try:
            logger.info(f"Downloading '{model_name}' from Hugging Face Hub")
            
            # Load model configuration
            config = AutoConfig.from_pretrained(
                model_name,
                num_labels=num_labels,
                cache_dir=cache_dir
            )
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                use_fast=True
            )
            
            # Load model
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                config=config,
                cache_dir=cache_dir
            )
            
            logger.info(f"Successfully downloaded '{model_name}' from online")
            
        except Exception as online_error:
            logger.error(f"Failed to build model '{model_name}' both locally and online: {str(online_error)}")
            raise
    
    logger.info(f"Successfully built model '{model_name}'")
    
    return {
        "model": model,
        "tokenizer": tokenizer,
        "config": config,
        "model_name": model_name
    }


def get_supported_models() -> Dict[str, str]:
    """
    Get a list of supported pre-trained models.
    
    Returns:
        Dict[str, str]: Dictionary of model names and descriptions
    """
    return {
        "bert-base-uncased": "BERT base model (uncased)",
        "bert-large-uncased": "BERT large model (uncased)",
        "roberta-base": "RoBERTa base model",
        "roberta-large": "RoBERTa large model",
        "distilbert-base-uncased": "DistilBERT base model (uncased)",
        "albert-base-v2": "ALBERT base model",
        "facebook/bart-base": "BART base model"
    }


def customize_model_config(
    base_config: AutoConfig,
    **kwargs
) -> AutoConfig:
    """
    Customize model configuration with additional parameters.
    
    Args:
        base_config (AutoConfig): Base configuration to customize
        **kwargs: Additional configuration parameters
        
    Returns:
        AutoConfig: Customized configuration
    """
    logger.info("Customizing model configuration")
    
    # Update configuration with provided parameters
    for key, value in kwargs.items():
        if hasattr(base_config, key):
            setattr(base_config, key, value)
            logger.debug(f"Set {key} = {value}")
        else:
            logger.warning(f"Configuration parameter '{key}' not found in base config")
    
    return base_config


# Example usage
if __name__ == "__main__":
    # This is for testing purposes only
    print("Model Builder Module")
    print("Available functions:")
    print("- build_model()")
    print("- get_supported_models()")
    print("- customize_model_config()")