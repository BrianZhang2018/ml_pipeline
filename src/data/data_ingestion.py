"""
Data Ingestion Module

This module handles loading and downloading datasets for the text classification pipeline.

What Is This? (Explain Like I'm 5)
===============================
This is like a smart toy collector that knows where to find the best puzzle pieces
for our AI project. It can go to different toy stores (datasets) and bring back
the exact puzzle pieces (text data) we need for our text-sorting machine.
"""

import os
import sys
import pandas as pd
from datasets import load_dataset
from typing import Union, Dict, Any
import logging

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.utils.logger import get_logger

# Get logger instance
logger = get_logger("data_ingestion")


def load_imdb_dataset(split: str = "train") -> pd.DataFrame:
    """
    Load the IMDB movie reviews dataset.
    
    Args:
        split (str): Dataset split to load ('train', 'test', or 'all')
        
    Returns:
        pd.DataFrame: DataFrame containing the dataset
        
    Example:
        >>> df = load_imdb_dataset('train')
        >>> print(df.head())
    """
    logger.info(f"Loading IMDB dataset with split: {split}")
    
    try:
        # Load dataset using Hugging Face datasets
        dataset = load_dataset("imdb", split=split)
        
        # Convert to pandas DataFrame
        df = dataset.to_pandas()
        
        logger.info(f"Successfully loaded IMDB dataset with {len(df)} samples")
        return df
        
    except Exception as e:
        logger.error(f"Error loading IMDB dataset: {e}")
        raise


def load_ag_news_dataset(split: str = "train") -> pd.DataFrame:
    """
    Load the AG News dataset.
    
    Args:
        split (str): Dataset split to load ('train', 'test', or 'all')
        
    Returns:
        pd.DataFrame: DataFrame containing the dataset
    """
    logger.info(f"Loading AG News dataset with split: {split}")
    
    try:
        # Load dataset using Hugging Face datasets
        dataset = load_dataset("ag_news", split=split)
        
        # Convert to pandas DataFrame
        df = dataset.to_pandas()
        
        logger.info(f"Successfully loaded AG News dataset with {len(df)} samples")
        return df
        
    except Exception as e:
        logger.error(f"Error loading AG News dataset: {e}")
        raise


def load_custom_dataset(file_path: str) -> pd.DataFrame:
    """
    Load a custom dataset from a file.
    
    Args:
        file_path (str): Path to the dataset file
        
    Returns:
        pd.DataFrame: DataFrame containing the dataset
    """
    logger.info(f"Loading custom dataset from: {file_path}")
    
    try:
        # Determine file type from extension
        _, ext = os.path.splitext(file_path)
        
        if ext.lower() == '.csv':
            df = pd.read_csv(file_path)
        elif ext.lower() == '.json':
            df = pd.read_json(file_path)
        elif ext.lower() == '.tsv':
            df = pd.read_csv(file_path, sep='\t')
        else:
            raise ValueError(f"Unsupported file format: {ext}")
        
        logger.info(f"Successfully loaded custom dataset with {len(df)} samples")
        return df
        
    except Exception as e:
        logger.error(f"Error loading custom dataset from {file_path}: {e}")
        raise


def save_dataset(df: pd.DataFrame, file_path: str) -> None:
    """
    Save a dataset to a file.
    
    Args:
        df (pd.DataFrame): DataFrame to save
        file_path (str): Path to save the dataset
    """
    logger.info(f"Saving dataset to: {file_path}")
    
    try:
        # Determine file type from extension
        _, ext = os.path.splitext(file_path)
        
        if ext.lower() == '.csv':
            df.to_csv(file_path, index=False)
        elif ext.lower() == '.json':
            df.to_json(file_path, orient='records')
        elif ext.lower() == '.tsv':
            df.to_csv(file_path, sep='\t', index=False)
        else:
            raise ValueError(f"Unsupported file format: {ext}")
        
        logger.info(f"Successfully saved dataset with {len(df)} samples")
        
    except Exception as e:
        logger.error(f"Error saving dataset to {file_path}: {e}")
        raise


def get_dataset_info(dataset_name: str) -> Dict[str, Any]:
    """
    Get information about a dataset.
    
    Args:
        dataset_name (str): Name of the dataset
        
    Returns:
        Dict[str, Any]: Dictionary containing dataset information
    """
    logger.info(f"Getting info for dataset: {dataset_name}")
    
    info = {
        "imdb": {
            "description": "IMDB Movie Reviews Dataset",
            "classes": ["negative", "positive"],
            "num_classes": 2,
            "default_split": "train"
        },
        "ag_news": {
            "description": "AG News Dataset",
            "classes": ["World", "Sports", "Business", "Sci/Tech"],
            "num_classes": 4,
            "default_split": "train"
        }
    }
    
    return info.get(dataset_name, {"description": "Unknown dataset"})


# Example usage
if __name__ == "__main__":
    # This is for testing purposes only
    print("Data Ingestion Module")
    print("Available functions:")
    print("- load_imdb_dataset()")
    print("- load_ag_news_dataset()")
    print("- load_custom_dataset()")
    print("- save_dataset()")
    print("- get_dataset_info()")