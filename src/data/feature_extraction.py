"""
Feature Extraction Module

This module handles tokenization and feature extraction using Hugging Face tokenizers.

What Is This? (Explain Like I'm 5)
===============================
This is like a translator that converts our text puzzle pieces into a special
language that our AI robot can understand. Just like you might translate
English words into numbers for a computer game, this module converts text
into numbers that our AI can work with.
"""

import sys
import os
from typing import List, Dict, Any, Union
import numpy as np
import logging

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.utils.logger import get_logger
from src.utils.config import get_config

# Try to import transformers, but handle if not available
try:
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers library not available. Feature extraction will be limited.")

# Get logger instance
logger = get_logger("feature_extraction")

# Get configuration
config = get_config()


class FeatureExtractor:
    """
    Feature extractor for text data using Hugging Face tokenizers.
    """
    
    def __init__(self, model_name: str = None):
        """
        Initialize the feature extractor.
        
        Args:
            model_name (str): Name of the pre-trained model to use for tokenization
        """
        logger.info("Initializing FeatureExtractor")
        
        if not TRANSFORMERS_AVAILABLE:
            logger.error("Transformers library not available")
            raise ImportError("Transformers library is required for feature extraction")
        
        # Use model name from config if not provided
        self.model_name = model_name or config.get("MODEL_NAME", "distilbert-base-uncased")
        self.max_length = config.get("MAX_SEQ_LENGTH", 512)
        self.tokenizer = None
        
        # Load tokenizer
        self._load_tokenizer()
    
    def _load_tokenizer(self):
        """
        Load the tokenizer for the specified model.
        """
        logger.info(f"Loading tokenizer for model: {self.model_name}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            logger.info("Tokenizer loaded successfully")
        except Exception as e:
            logger.error(f"Error loading tokenizer: {e}")
            raise
    
    def tokenize_text(self, text: str) -> Dict[str, Any]:
        """
        Tokenize a single text.
        
        Args:
            text (str): Text to tokenize
            
        Returns:
            Dict[str, Any]: Tokenized text with input_ids, attention_mask, etc.
        """
        if not isinstance(text, str):
            text = str(text)
        
        logger.debug(f"Tokenizing text: {text[:50]}...")
        
        try:
            # Tokenize the text
            encoded = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors="np"
            )
            
            logger.debug("Text tokenized successfully")
            return {
                "input_ids": encoded["input_ids"][0],
                "attention_mask": encoded["attention_mask"][0]
            }
        except Exception as e:
            logger.error(f"Error tokenizing text: {e}")
            raise
    
    def tokenize_texts(self, texts: List[str]) -> Dict[str, np.ndarray]:
        """
        Tokenize a list of texts.
        
        Args:
            texts (List[str]): List of texts to tokenize
            
        Returns:
            Dict[str, np.ndarray]: Tokenized texts with input_ids, attention_mask, etc.
        """
        logger.info(f"Tokenizing {len(texts)} texts")
        
        try:
            # Tokenize all texts
            encoded = self.tokenizer(
                texts,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors="np"
            )
            
            logger.info("All texts tokenized successfully")
            return {
                "input_ids": encoded["input_ids"],
                "attention_mask": encoded["attention_mask"]
            }
        except Exception as e:
            logger.error(f"Error tokenizing texts: {e}")
            raise
    
    def extract_features(self, texts: List[str]) -> Dict[str, np.ndarray]:
        """
        Extract features from texts (alias for tokenize_texts).
        
        Args:
            texts (List[str]): List of texts to extract features from
            
        Returns:
            Dict[str, np.ndarray]: Extracted features
        """
        logger.info("Extracting features from texts")
        return self.tokenize_texts(texts)
    
    def get_vocabulary_size(self) -> int:
        """
        Get the size of the tokenizer vocabulary.
        
        Returns:
            int: Vocabulary size
        """
        if self.tokenizer:
            return len(self.tokenizer)
        return 0
    
    def get_model_name(self) -> str:
        """
        Get the model name being used.
        
        Returns:
            str: Model name
        """
        return self.model_name


# Convenience functions
def create_feature_extractor(model_name: str = None) -> FeatureExtractor:
    """
    Create a feature extractor instance.
    
    Args:
        model_name (str): Name of the pre-trained model to use
        
    Returns:
        FeatureExtractor: Feature extractor instance
    """
    return FeatureExtractor(model_name)


def extract_features_from_texts(texts: List[str], model_name: str = None) -> Dict[str, np.ndarray]:
    """
    Extract features from texts using a feature extractor.
    
    Args:
        texts (List[str]): List of texts to extract features from
        model_name (str): Name of the pre-trained model to use
        
    Returns:
        Dict[str, np.ndarray]: Extracted features
    """
    logger.info("Extracting features from texts (convenience function)")
    
    extractor = FeatureExtractor(model_name)
    return extractor.extract_features(texts)


# Example usage
if __name__ == "__main__":
    # This is for testing purposes only
    print("Feature Extraction Module")
    print("Available classes and functions:")
    print("- FeatureExtractor class")
    print("- create_feature_extractor()")
    print("- extract_features_from_texts()")