"""
Model Evaluation Metrics Module

This module provides functions to calculate various metrics for evaluating 
text classification models.

What Is This? (Explain Like I'm 5)
===============================
This is like a report card for our AI brain. Just like you get grades for
different subjects in school, we give our AI grades for how well it does
at sorting text into the right categories. We check things like:
- How many times it got the answer right (accuracy)
- How good it is at finding all the right answers (recall)
- How good it is at only picking the right answers (precision)
"""

import sys
import os
from typing import Dict, List, Tuple, Union
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.utils.logger import get_logger

# Get logger instance
logger = get_logger("metrics")


def calculate_accuracy(y_true: List[int], y_pred: List[int]) -> float:
    """
    Calculate accuracy score.
    
    Accuracy is the fraction of predictions our model got right.
    
    Args:
        y_true (List[int]): True labels
        y_pred (List[int]): Predicted labels
        
    Returns:
        float: Accuracy score between 0 and 1
    """
    logger.debug("Calculating accuracy score")
    accuracy = accuracy_score(y_true, y_pred)
    logger.info(f"Accuracy: {accuracy:.4f}")
    return accuracy


def calculate_precision(y_true: List[int], y_pred: List[int], average: str = 'weighted') -> float:
    """
    Calculate precision score.
    
    Precision is the ability of a model to identify only the relevant data points.
    It's the ratio of correctly predicted positive observations to the total 
    predicted positive observations.
    
    Args:
        y_true (List[int]): True labels
        y_pred (List[int]): Predicted labels
        average (str): Type of averaging to use ('micro', 'macro', 'weighted', 'samples', None)
        
    Returns:
        float: Precision score
    """
    logger.debug(f"Calculating precision score with average='{average}'")
    precision = precision_score(y_true, y_pred, average=average, zero_division=0)
    logger.info(f"Precision ({average}): {precision:.4f}")
    return precision


def calculate_recall(y_true: List[int], y_pred: List[int], average: str = 'weighted') -> float:
    """
    Calculate recall score.
    
    Recall is the ability of a model to find all the relevant cases.
    It's the ratio of correctly predicted positive observations to all 
    observations in the actual class.
    
    Args:
        y_true (List[int]): True labels
        y_pred (List[int]): Predicted labels
        average (str): Type of averaging to use ('micro', 'macro', 'weighted', 'samples', None)
        
    Returns:
        float: Recall score
    """
    logger.debug(f"Calculating recall score with average='{average}'")
    recall = recall_score(y_true, y_pred, average=average, zero_division=0)
    logger.info(f"Recall ({average}): {recall:.4f}")
    return recall


def calculate_f1_score(y_true: List[int], y_pred: List[int], average: str = 'weighted') -> float:
    """
    Calculate F1 score.
    
    F1 Score is the weighted average of Precision and Recall.
    It tries to find the balance between precision and recall.
    
    Args:
        y_true (List[int]): True labels
        y_pred (List[int]): Predicted labels
        average (str): Type of averaging to use ('micro', 'macro', 'weighted', 'samples', None)
        
    Returns:
        float: F1 score
    """
    logger.debug(f"Calculating F1 score with average='{average}'")
    f1 = f1_score(y_true, y_pred, average=average, zero_division=0)
    logger.info(f"F1 Score ({average}): {f1:.4f}")
    return f1


def calculate_confusion_matrix(y_true: List[int], y_pred: List[int]) -> np.ndarray:
    """
    Calculate confusion matrix.
    
    A confusion matrix is a table that is often used to describe the 
    performance of a classification model on a set of test data.
    
    Args:
        y_true (List[int]): True labels
        y_pred (List[int]): Predicted labels
        
    Returns:
        np.ndarray: Confusion matrix
    """
    logger.debug("Calculating confusion matrix")
    cm = confusion_matrix(y_true, y_pred)
    logger.info(f"Confusion matrix shape: {cm.shape}")
    return cm


def generate_classification_report(y_true: List[int], y_pred: List[int], 
                                 target_names: List[str] = None) -> str:
    """
    Generate a detailed classification report.
    
    This function provides a text report showing the main classification metrics.
    
    Args:
        y_true (List[int]): True labels
        y_pred (List[int]): Predicted labels
        target_names (List[str], optional): List of class names
        
    Returns:
        str: Classification report
    """
    logger.debug("Generating classification report")
    report = classification_report(y_true, y_pred, target_names=target_names, zero_division=0)
    logger.info("Classification report generated")
    return report


def calculate_all_metrics(y_true: List[int], y_pred: List[int], 
                         target_names: List[str] = None) -> Dict[str, Union[float, np.ndarray, str]]:
    """
    Calculate all evaluation metrics.
    
    This function calculates all the important metrics for evaluating a 
    text classification model and returns them in a dictionary.
    
    Args:
        y_true (List[int]): True labels
        y_pred (List[int]): Predicted labels
        target_names (List[str], optional): List of class names
        
    Returns:
        Dict[str, Union[float, np.ndarray, str]]: Dictionary containing all metrics
    """
    logger.info("Calculating all evaluation metrics")
    
    metrics = {
        'accuracy': calculate_accuracy(y_true, y_pred),
        'precision': calculate_precision(y_true, y_pred),
        'recall': calculate_recall(y_true, y_pred),
        'f1_score': calculate_f1_score(y_true, y_pred),
        'confusion_matrix': calculate_confusion_matrix(y_true, y_pred),
        'classification_report': generate_classification_report(y_true, y_pred, target_names)
    }
    
    logger.info("All metrics calculated successfully")
    return metrics


# Example usage
if __name__ == "__main__":
    # This is for testing purposes only
    print("Model Evaluation Metrics Module")
    print("Available functions:")
    print("- calculate_accuracy")
    print("- calculate_precision")
    print("- calculate_recall")
    print("- calculate_f1_score")
    print("- calculate_confusion_matrix")
    print("- generate_classification_report")
    print("- calculate_all_metrics")