"""
Model Trainer Module

This module handles training transformer-based models for text classification
using TensorFlow/PyTorch.

What Is This? (Explain Like I'm 5)
===============================
This is like a teacher for our AI brain. Just like a teacher helps you learn
new things by showing you examples and testing your knowledge, this module
teaches our AI by showing it lots of examples and testing how well it learns.
"""

import sys
import os
from typing import Dict, Any, Optional, List
import logging
import torch
from torch.utils.data import DataLoader
from transformers import (
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.utils.logger import get_logger

# Get logger instance
logger = get_logger("model_trainer")


def compute_metrics(eval_pred):
    """
    Compute metrics for model evaluation.
    
    Args:
        eval_pred: Evaluation predictions from the model
        
    Returns:
        Dict[str, float]: Dictionary of metrics
    """
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    accuracy = accuracy_score(labels, predictions)
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


class ModelTrainer:
    """
    A class to handle model training with TensorFlow/PyTorch.
    """
    
    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        train_dataset: Any,
        eval_dataset: Any = None,
        output_dir: str = "./results",
        num_train_epochs: int = 3,
        per_device_train_batch_size: int = 16,
        per_device_eval_batch_size: int = 16,
        warmup_steps: int = 500,
        weight_decay: float = 0.01,
        logging_dir: str = "./logs",
        evaluation_strategy: str = "epoch",
        save_strategy: str = "epoch",
        load_best_model_at_end: bool = True,
        metric_for_best_model: str = "f1"
    ):
        """
        Initialize the ModelTrainer.
        
        Args:
            model (Any): The model to train
            tokenizer (Any): The tokenizer for the model
            train_dataset (Any): Training dataset
            eval_dataset (Any, optional): Evaluation dataset
            output_dir (str): Directory to save model outputs
            num_train_epochs (int): Number of training epochs
            per_device_train_batch_size (int): Batch size for training
            per_device_eval_batch_size (int): Batch size for evaluation
            warmup_steps (int): Number of warmup steps
            weight_decay (float): Weight decay for optimization
            logging_dir (str): Directory for logging
            evaluation_strategy (str): When to evaluate ("no", "steps", "epoch")
            save_strategy (str): When to save ("no", "steps", "epoch")
            load_best_model_at_end (bool): Whether to load the best model at end
            metric_for_best_model (str): Metric to use for best model selection
        """
        logger.info("Initializing ModelTrainer")
        
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        
        # Training arguments
        self.training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            warmup_steps=warmup_steps,
            weight_decay=weight_decay,
            logging_dir=logging_dir,
            evaluation_strategy=evaluation_strategy,
            save_strategy=save_strategy,
            load_best_model_at_end=load_best_model_at_end,
            metric_for_best_model=metric_for_best_model,
            logging_steps=10,
            save_total_limit=2,
            seed=42
        )
        
        # Initialize trainer
        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        
        logger.info("ModelTrainer initialized successfully")
    
    def train(self) -> Dict[str, Any]:
        """
        Train the model.
        
        Returns:
            Dict[str, Any]: Training results
        """
        logger.info("Starting model training")
        
        try:
            # Train the model
            train_result = self.trainer.train()
            
            # Log training results
            logger.info("Model training completed successfully")
            logger.info(f"Training loss: {train_result.training_loss}")
            
            return {
                "train_result": train_result,
                "training_args": self.training_args
            }
            
        except Exception as e:
            logger.error(f"Model training failed: {str(e)}")
            raise
    
    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate the model.
        
        Returns:
            Dict[str, float]: Evaluation metrics
        """
        logger.info("Starting model evaluation")
        
        try:
            # Evaluate the model
            eval_result = self.trainer.evaluate()
            
            # Log evaluation results
            logger.info("Model evaluation completed successfully")
            logger.info(f"Evaluation results: {eval_result}")
            
            return eval_result
            
        except Exception as e:
            logger.error(f"Model evaluation failed: {str(e)}")
            raise
    
    def save_model(self, output_dir: str):
        """
        Save the trained model and tokenizer.
        
        Args:
            output_dir (str): Directory to save the model
        """
        logger.info(f"Saving model to {output_dir}")
        
        try:
            # Save model
            self.trainer.save_model(output_dir)
            
            # Save tokenizer
            self.tokenizer.save_pretrained(output_dir)
            
            logger.info("Model saved successfully")
            
        except Exception as e:
            logger.error(f"Failed to save model: {str(e)}")
            raise


# Example usage
if __name__ == "__main__":
    # This is for testing purposes only
    print("Model Trainer Module")
    print("Available classes and functions:")
    print("- ModelTrainer class")
    print("- compute_metrics()")