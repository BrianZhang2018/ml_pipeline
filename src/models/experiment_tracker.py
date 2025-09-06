"""
Experiment Tracker Module

This module handles experiment tracking using MLflow for our text classification pipeline.

What Is This? (Explain Like I'm 5)
===============================
This is like a science lab notebook for our AI experiments. Just like scientists
write down what they did, what happened, and what they learned in their
notebooks, this module keeps track of all our AI experiments so we can see
what worked best.
"""

import sys
import os
from typing import Dict, Any, Optional, List
import logging
import mlflow
from mlflow.tracking import MlflowClient
import tempfile

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.utils.logger import get_logger

# Get logger instance
logger = get_logger("experiment_tracker")


class ExperimentTracker:
    """
    A class to handle experiment tracking with MLflow.
    """
    
    def __init__(
        self,
        experiment_name: str = "text_classification_experiment",
        tracking_uri: Optional[str] = None
    ):
        """
        Initialize the ExperimentTracker.
        
        Args:
            experiment_name (str): Name of the experiment
            tracking_uri (str, optional): URI for MLflow tracking server
        """
        logger.info(f"Initializing ExperimentTracker for experiment '{experiment_name}'")
        
        self.experiment_name = experiment_name
        
        # Set tracking URI if provided
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        
        # Set experiment
        mlflow.set_experiment(experiment_name)
        
        # Get MLflow client
        self.client = MlflowClient()
        
        logger.info("ExperimentTracker initialized successfully")
    
    def start_run(self, run_name: Optional[str] = None) -> mlflow.ActiveRun:
        """
        Start a new MLflow run.
        
        Args:
            run_name (str, optional): Name of the run
            
        Returns:
            mlflow.ActiveRun: Active MLflow run
        """
        logger.info(f"Starting new MLflow run with name '{run_name}'")
        
        try:
            run = mlflow.start_run(run_name=run_name)
            logger.info(f"Started MLflow run with ID: {run.info.run_id}")
            return run
            
        except Exception as e:
            logger.error(f"Failed to start MLflow run: {str(e)}")
            raise
    
    def end_run(self):
        """
        End the current MLflow run.
        """
        logger.info("Ending current MLflow run")
        
        try:
            mlflow.end_run()
            logger.info("Ended MLflow run successfully")
            
        except Exception as e:
            logger.error(f"Failed to end MLflow run: {str(e)}")
            raise
    
    def log_params(self, params: Dict[str, Any]):
        """
        Log parameters to the current run.
        
        Args:
            params (Dict[str, Any]): Parameters to log
        """
        logger.info(f"Logging parameters: {params}")
        
        try:
            mlflow.log_params(params)
            logger.info("Parameters logged successfully")
            
        except Exception as e:
            logger.error(f"Failed to log parameters: {str(e)}")
            raise
    
    def log_metrics(self, metrics: Dict[str, float]):
        """
        Log metrics to the current run.
        
        Args:
            metrics (Dict[str, float]): Metrics to log
        """
        logger.info(f"Logging metrics: {metrics}")
        
        try:
            mlflow.log_metrics(metrics)
            logger.info("Metrics logged successfully")
            
        except Exception as e:
            logger.error(f"Failed to log metrics: {str(e)}")
            raise
    
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """
        Log an artifact to the current run.
        
        Args:
            local_path (str): Local path to the artifact
            artifact_path (str, optional): Artifact path in MLflow
        """
        logger.info(f"Logging artifact from {local_path}")
        
        try:
            mlflow.log_artifact(local_path, artifact_path)
            logger.info("Artifact logged successfully")
            
        except Exception as e:
            logger.error(f"Failed to log artifact: {str(e)}")
            raise
    
    def log_model(self, model: Any, artifact_path: str = "model"):
        """
        Log a model to the current run.
        
        Args:
            model (Any): Model to log
            artifact_path (str): Artifact path for the model
        """
        logger.info(f"Logging model to artifact path '{artifact_path}'")
        
        try:
            # For Hugging Face models, we need to save them first
            with tempfile.TemporaryDirectory() as temp_dir:
                # Save model to temporary directory
                model.save_pretrained(temp_dir)
                
                # Log the saved model as an artifact
                mlflow.log_artifact(temp_dir, artifact_path)
            
            logger.info("Model logged successfully")
            
        except Exception as e:
            logger.error(f"Failed to log model: {str(e)}")
            raise
    
    def get_best_run(self, metric: str = "f1", mode: str = "max") -> Optional[Dict[str, Any]]:
        """
        Get the best run based on a metric.
        
        Args:
            metric (str): Metric to use for comparison
            mode (str): Direction for optimization ("max" or "min")
            
        Returns:
            Optional[Dict[str, Any]]: Best run information or None
        """
        logger.info(f"Finding best run based on metric '{metric}' with mode '{mode}'")
        
        try:
            # Get experiment ID
            experiment = self.client.get_experiment_by_name(self.experiment_name)
            if not experiment:
                logger.warning(f"Experiment '{self.experiment_name}' not found")
                return None
            
            # Get all runs for the experiment
            runs = self.client.search_runs(experiment.experiment_id)
            
            if not runs:
                logger.warning("No runs found for experiment")
                return None
            
            # Find best run
            best_run = None
            best_value = None
            
            for run in runs:
                if metric in run.data.metrics:
                    value = run.data.metrics[metric]
                    if best_value is None or (
                        (mode == "max" and value > best_value) or 
                        (mode == "min" and value < best_value)
                    ):
                        best_value = value
                        best_run = run
            
            if best_run:
                logger.info(f"Best run found with {metric} = {best_value}")
                return {
                    "run_id": best_run.info.run_id,
                    "metrics": best_run.data.metrics,
                    "params": best_run.data.params,
                    "tags": best_run.data.tags
                }
            else:
                logger.warning("No best run found")
                return None
                
        except Exception as e:
            logger.error(f"Failed to find best run: {str(e)}")
            raise


# Example usage
if __name__ == "__main__":
    # This is for testing purposes only
    print("Experiment Tracker Module")
    print("Available classes and functions:")
    print("- ExperimentTracker class")