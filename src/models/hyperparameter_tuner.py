"""
Hyperparameter Tuner Module

This module handles hyperparameter tuning using Optuna for our text classification pipeline.

What Is This? (Explain Like I'm 5)
===============================
This is like a smart helper that finds the best settings for our AI brain.
Just like you might try different amounts of sugar and flour to make the
yummies cookies, this module tries different settings to make our AI work
its best.
"""

import sys
import os
from typing import Dict, Any, Callable, Optional
import logging
import optuna
from optuna.study import Study
from transformers import TrainingArguments

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.utils.logger import get_logger

# Get logger instance
logger = get_logger("hyperparameter_tuner")


class HyperparameterTuner:
    """
    A class to handle hyperparameter tuning with Optuna.
    """
    
    def __init__(
        self,
        study_name: str = "text_classification_study",
        storage: Optional[str] = None,
        direction: str = "maximize",
        load_if_exists: bool = True
    ):
        """
        Initialize the HyperparameterTuner.
        
        Args:
            study_name (str): Name of the Optuna study
            storage (str, optional): Storage URL for Optuna
            direction (str): Direction for optimization ("maximize" or "minimize")
            load_if_exists (bool): Whether to load existing study if it exists
        """
        logger.info(f"Initializing HyperparameterTuner for study '{study_name}'")
        
        self.study_name = study_name
        self.direction = direction
        
        # Create or load study
        self.study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            direction=direction,
            load_if_exists=load_if_exists
        )
        
        logger.info("HyperparameterTuner initialized successfully")
    
    def suggest_hyperparameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Suggest hyperparameters for a trial.
        
        Args:
            trial (optuna.Trial): Optuna trial object
            
        Returns:
            Dict[str, Any]: Suggested hyperparameters
        """
        logger.debug("Suggesting hyperparameters for trial")
        
        # Suggest hyperparameters
        hyperparameters = {
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True),
            "num_train_epochs": trial.suggest_int("num_train_epochs", 1, 5),
            "per_device_train_batch_size": trial.suggest_categorical(
                "per_device_train_batch_size", [8, 16, 32]
            ),
            "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.3),
            "warmup_steps": trial.suggest_int("warmup_steps", 0, 500),
            "adam_epsilon": trial.suggest_float("adam_epsilon", 1e-8, 1e-6, log=True)
        }
        
        logger.debug(f"Suggested hyperparameters: {hyperparameters}")
        return hyperparameters
    
    def create_training_args(
        self,
        hyperparameters: Dict[str, Any],
        output_dir: str = "./results"
    ) -> TrainingArguments:
        """
        Create TrainingArguments from hyperparameters.
        
        Args:
            hyperparameters (Dict[str, Any]): Hyperparameters
            output_dir (str): Output directory for training results
            
        Returns:
            TrainingArguments: Training arguments for Hugging Face Trainer
        """
        logger.debug("Creating TrainingArguments from hyperparameters")
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=hyperparameters["num_train_epochs"],
            per_device_train_batch_size=hyperparameters["per_device_train_batch_size"],
            per_device_eval_batch_size=hyperparameters["per_device_train_batch_size"],
            learning_rate=hyperparameters["learning_rate"],
            weight_decay=hyperparameters["weight_decay"],
            warmup_steps=hyperparameters["warmup_steps"],
            adam_epsilon=hyperparameters["adam_epsilon"],
            logging_dir="./logs",
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            logging_steps=10,
            save_total_limit=2,
            seed=42
        )
        
        logger.debug("TrainingArguments created successfully")
        return training_args
    
    def optimize(
        self,
        objective_function: Callable[[optuna.Trial], float],
        n_trials: int = 20
    ) -> Dict[str, Any]:
        """
        Run hyperparameter optimization.
        
        Args:
            objective_function (Callable): Function to optimize
            n_trials (int): Number of trials to run
            
        Returns:
            Dict[str, Any]: Optimization results
        """
        logger.info(f"Starting hyperparameter optimization with {n_trials} trials")
        
        try:
            # Run optimization
            self.study.optimize(objective_function, n_trials=n_trials)
            
            # Get best parameters and value
            best_params = self.study.best_params
            best_value = self.study.best_value
            
            logger.info("Hyperparameter optimization completed successfully")
            logger.info(f"Best parameters: {best_params}")
            logger.info(f"Best value: {best_value}")
            
            return {
                "best_params": best_params,
                "best_value": best_value,
                "study": self.study
            }
            
        except Exception as e:
            logger.error(f"Hyperparameter optimization failed: {str(e)}")
            raise
    
    def get_study_results(self) -> Dict[str, Any]:
        """
        Get study results.
        
        Returns:
            Dict[str, Any]: Study results
        """
        logger.info("Getting study results")
        
        try:
            results = {
                "study_name": self.study_name,
                "best_params": self.study.best_params,
                "best_value": self.study.best_value,
                "trials": len(self.study.trials)
            }
            
            logger.info("Study results retrieved successfully")
            return results
            
        except Exception as e:
            logger.error(f"Failed to get study results: {str(e)}")
            raise


# Example usage
if __name__ == "__main__":
    # This is for testing purposes only
    print("Hyperparameter Tuner Module")
    print("Available classes and functions:")
    print("- HyperparameterTuner class")