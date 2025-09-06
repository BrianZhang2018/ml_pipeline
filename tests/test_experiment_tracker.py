"""
Unit tests for the experiment tracker module.
"""

import sys
import os
import pytest
from unittest.mock import Mock, patch

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.experiment_tracker import ExperimentTracker


@patch('src.models.experiment_tracker.mlflow')
@patch('src.models.experiment_tracker.MlflowClient')
def test_experiment_tracker_initialization(mock_mlflow_client, mock_mlflow):
    """Test that we can initialize the ExperimentTracker."""
    # Create mock client
    mock_client = Mock()
    mock_mlflow_client.return_value = mock_client
    
    # Initialize ExperimentTracker
    tracker = ExperimentTracker(experiment_name="test_experiment")
    
    # Check that the tracker was initialized
    assert tracker.experiment_name == "test_experiment"
    assert tracker.client == mock_client


@patch('src.models.experiment_tracker.mlflow')
@patch('src.models.experiment_tracker.MlflowClient')
def test_experiment_tracker_start_run(mock_mlflow_client, mock_mlflow):
    """Test that we can start a run."""
    # Create mock client and run
    mock_client = Mock()
    mock_mlflow_client.return_value = mock_client
    mock_run = Mock()
    mock_run.info.run_id = "test_run_id"
    mock_mlflow.start_run.return_value = mock_run
    
    # Initialize ExperimentTracker
    tracker = ExperimentTracker(experiment_name="test_experiment")
    
    # Call start_run
    run = tracker.start_run(run_name="test_run")
    
    # Check that start_run was called and returned expected result
    mock_mlflow.start_run.assert_called_once_with(run_name="test_run")
    assert run == mock_run


@patch('src.models.experiment_tracker.mlflow')
@patch('src.models.experiment_tracker.MlflowClient')
def test_experiment_tracker_log_params(mock_mlflow_client, mock_mlflow):
    """Test that we can log parameters."""
    # Create mock client
    mock_client = Mock()
    mock_mlflow_client.return_value = mock_client
    
    # Initialize ExperimentTracker
    tracker = ExperimentTracker(experiment_name="test_experiment")
    
    # Call log_params
    params = {"learning_rate": 0.001, "batch_size": 32}
    tracker.log_params(params)
    
    # Check that log_params was called with expected arguments
    mock_mlflow.log_params.assert_called_once_with(params)


@patch('src.models.experiment_tracker.mlflow')
@patch('src.models.experiment_tracker.MlflowClient')
def test_experiment_tracker_log_metrics(mock_mlflow_client, mock_mlflow):
    """Test that we can log metrics."""
    # Create mock client
    mock_client = Mock()
    mock_mlflow_client.return_value = mock_client
    
    # Initialize ExperimentTracker
    tracker = ExperimentTracker(experiment_name="test_experiment")
    
    # Call log_metrics
    metrics = {"accuracy": 0.95, "f1": 0.92}
    tracker.log_metrics(metrics)
    
    # Check that log_metrics was called with expected arguments
    mock_mlflow.log_metrics.assert_called_once_with(metrics)


if __name__ == "__main__":
    pytest.main([__file__])