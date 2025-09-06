"""
Cross-Validation Module

This module provides functions to perform cross-validation for evaluating 
text classification models.

What Is This? (Explain Like I'm 5)
===============================
This is like giving our AI brain multiple tests to make sure it's really smart.
Instead of just one test, we give it several different tests to see if it 
can do well each time. It's like when you practice math problems from 
different worksheets to make sure you understand the concept, not just 
memorize one set of problems.
"""

import sys
import os
from typing import Dict, List, Tuple, Any, Callable
import numpy as np
from sklearn.model_selection import StratifiedKFold
import torch

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.utils.logger import get_logger
from src.evaluation.metrics import calculate_all_metrics

# Get logger instance
logger = get_logger("cross_validation")


class CrossValidator:
    """
    A class to perform cross-validation for text classification models.
    """
    
    def __init__(self, n_folds: int = 5, random_state: int = 42):
        """
        Initialize the CrossValidator.
        
        Args:
            n_folds (int): Number of folds for cross-validation
            random_state (int): Random state for reproducibility
        """
        logger.info(f"Initializing CrossValidator with {n_folds} folds")
        
        self.n_folds = n_folds
        self.random_state = random_state
        self.skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        
        logger.info("CrossValidator initialized successfully")
    
    def validate_model(self, 
                      model: Any,
                      texts: List[str],
                      labels: List[int],
                      train_function: Callable,
                      predict_function: Callable,
                      target_names: List[str] = None) -> Dict[str, Any]:
        """
        Perform cross-validation on a model.
        
        Args:
            model (Any): The model to validate
            texts (List[str]): Text data
            labels (List[int]): Labels for the text data
            train_function (Callable): Function to train the model
            predict_function (Callable): Function to make predictions with the model
            target_names (List[str], optional): Names of the target classes
            
        Returns:
            Dict[str, Any]: Cross-validation results
        """
        logger.info(f"Starting cross-validation with {self.n_folds} folds")
        
        # Convert to numpy arrays for easier handling
        texts = np.array(texts)
        labels = np.array(labels)
        
        # Store results for each fold
        fold_results = []
        all_metrics = []
        
        # Perform cross-validation
        for fold, (train_idx, val_idx) in enumerate(self.skf.split(texts, labels)):
            logger.info(f"Processing fold {fold + 1}/{self.n_folds}")
            
            # Split data
            train_texts, val_texts = texts[train_idx], texts[val_idx]
            train_labels, val_labels = labels[train_idx], labels[val_idx]
            
            # Train model
            logger.debug("Training model for fold")
            trained_model = train_function(model, train_texts.tolist(), train_labels.tolist())
            
            # Make predictions
            logger.debug("Making predictions for validation set")
            val_predictions = predict_function(trained_model, val_texts.tolist())
            
            # Calculate metrics
            logger.debug("Calculating metrics for fold")
            metrics = calculate_all_metrics(val_labels.tolist(), val_predictions, target_names)
            
            # Store results
            fold_result = {
                'fold': fold + 1,
                'train_size': len(train_idx),
                'val_size': len(val_idx),
                'metrics': metrics
            }
            
            fold_results.append(fold_result)
            all_metrics.append(metrics)
            
            logger.info(f"Fold {fold + 1} completed - Accuracy: {metrics['accuracy']:.4f}")
        
        # Calculate aggregate statistics
        logger.info("Calculating aggregate statistics")
        aggregate_results = self._calculate_aggregate_stats(all_metrics)
        
        # Prepare final results
        results = {
            'fold_results': fold_results,
            'aggregate_results': aggregate_results,
            'n_folds': self.n_folds
        }
        
        logger.info("Cross-validation completed successfully")
        return results
    
    def _calculate_aggregate_stats(self, all_metrics: List[Dict]) -> Dict[str, Any]:
        """
        Calculate aggregate statistics across all folds.
        
        Args:
            all_metrics (List[Dict]): List of metrics from each fold
            
        Returns:
            Dict[str, Any]: Aggregate statistics
        """
        logger.debug("Calculating aggregate statistics")
        
        # Extract metric values
        accuracies = [m['accuracy'] for m in all_metrics]
        precisions = [m['precision'] for m in all_metrics]
        recalls = [m['recall'] for m in all_metrics]
        f1_scores = [m['f1_score'] for m in all_metrics]
        
        # Calculate statistics
        aggregate_stats = {
            'accuracy': {
                'mean': float(np.mean(accuracies)),
                'std': float(np.std(accuracies)),
                'min': float(np.min(accuracies)),
                'max': float(np.max(accuracies))
            },
            'precision': {
                'mean': float(np.mean(precisions)),
                'std': float(np.std(precisions)),
                'min': float(np.min(precisions)),
                'max': float(np.max(precisions))
            },
            'recall': {
                'mean': float(np.mean(recalls)),
                'std': float(np.std(recalls)),
                'min': float(np.min(recalls)),
                'max': float(np.max(recalls))
            },
            'f1_score': {
                'mean': float(np.mean(f1_scores)),
                'std': float(np.std(f1_scores)),
                'min': float(np.min(f1_scores)),
                'max': float(np.max(f1_scores))
            }
        }
        
        logger.debug("Aggregate statistics calculated successfully")
        return aggregate_stats


def simple_kfold_cv(texts: List[str], 
                   labels: List[int], 
                   model_fn: Callable,
                   n_folds: int = 5) -> Dict[str, Any]:
    """
    Simple k-fold cross-validation function.
    
    This is a simplified version of cross-validation for basic use cases.
    
    Args:
        texts (List[str]): Text data
        labels (List[int]): Labels for the text data
        model_fn (Callable): Function that returns a trained model
        n_folds (int): Number of folds
        
    Returns:
        Dict[str, Any]: Cross-validation results
    """
    logger.info(f"Starting simple k-fold cross-validation with {n_folds} folds")
    
    # Initialize cross-validator
    cv = CrossValidator(n_folds=n_folds)
    
    # Simple train and predict functions
    def train_fn(model, train_texts, train_labels):
        # This would normally train the model
        # For now, we just return the model
        return model_fn()
    
    def predict_fn(model, test_texts):
        # This would normally make predictions
        # For now, we return random predictions as an example
        return [np.random.randint(0, len(set(labels))) for _ in test_texts]
    
    # Perform cross-validation
    results = cv.validate_model(
        model=None,  # Not used in this simple example
        texts=texts,
        labels=labels,
        train_function=train_fn,
        predict_function=predict_fn
    )
    
    logger.info("Simple k-fold cross-validation completed")
    return results


# Example usage
if __name__ == "__main__":
    # This is for testing purposes only
    print("Cross-Validation Module")
    print("Available classes and functions:")
    print("- CrossValidator class")
    print("- simple_kfold_cv function")