"""
Data Preprocessing Module

This module handles text cleaning and preprocessing for the text classification pipeline.

What Is This? (Explain Like I'm 5)
===============================
This is like a cleaning station for our puzzle pieces. Just like you might wipe
dirt off puzzle pieces or organize them by color before putting them together,
this module cleans and organizes our text data so our AI can work with it better.
"""

import re
import string
import sys
import os
from typing import List, Union
import logging

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.utils.logger import get_logger

# Get logger instance
logger = get_logger("data_preprocessing")


def clean_text(text: str) -> str:
    """
    Clean text by removing unnecessary characters and formatting.
    
    Args:
        text (str): Text to clean
        
    Returns:
        str: Cleaned text
    """
    logger.debug(f"Cleaning text: {text[:50]}...")
    
    if not isinstance(text, str):
        text = str(text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    logger.debug(f"Cleaned text: {text[:50]}...")
    return text


def remove_html_tags(text: str) -> str:
    """
    Remove HTML tags from text.
    
    Args:
        text (str): Text to process
        
    Returns:
        str: Text with HTML tags removed
    """
    logger.debug("Removing HTML tags")
    
    # Remove HTML tags
    clean = re.compile('<.*?>')
    text = re.sub(clean, '', text)
    
    return text


def remove_urls(text: str) -> str:
    """
    Remove URLs from text.
    
    Args:
        text (str): Text to process
        
    Returns:
        str: Text with URLs removed
    """
    logger.debug("Removing URLs")
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    return text


def remove_punctuation(text: str) -> str:
    """
    Remove punctuation from text.
    
    Args:
        text (str): Text to process
        
    Returns:
        str: Text with punctuation removed
    """
    logger.debug("Removing punctuation")
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    return text


def normalize_text(text: str) -> str:
    """
    Apply all normalization steps to text.
    
    Args:
        text (str): Text to normalize
        
    Returns:
        str: Normalized text
    """
    logger.info("Normalizing text")
    
    # Apply all cleaning steps
    text = remove_html_tags(text)
    text = remove_urls(text)
    text = clean_text(text)
    
    logger.info("Text normalization complete")
    return text


def preprocess_texts(texts: List[str]) -> List[str]:
    """
    Preprocess a list of texts.
    
    Args:
        texts (List[str]): List of texts to preprocess
        
    Returns:
        List[str]: List of preprocessed texts
    """
    logger.info(f"Preprocessing {len(texts)} texts")
    
    preprocessed_texts = []
    for i, text in enumerate(texts):
        if i % 1000 == 0 and i > 0:
            logger.info(f"Preprocessed {i}/{len(texts)} texts")
        
        preprocessed_text = normalize_text(text)
        preprocessed_texts.append(preprocessed_text)
    
    logger.info("Text preprocessing complete")
    return preprocessed_texts


# Example usage
if __name__ == "__main__":
    # This is for testing purposes only
    print("Data Preprocessing Module")
    print("Available functions:")
    print("- clean_text()")
    print("- remove_html_tags()")
    print("- remove_urls()")
    print("- remove_punctuation()")
    print("- normalize_text()")
    print("- preprocess_texts()")