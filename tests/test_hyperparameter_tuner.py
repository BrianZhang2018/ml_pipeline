"""
Unit tests for the hyperparameter tuner module.
"""

import sys
import os
import pytest
from unittest.mock import Mock, patch

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.hyperparameter_tuner import HyperparameterTuner


@patch('src.models.hyperparameter_tuner.optuna.create_study')
def test_hyperparameter_tuner_initialization(mock_create_study):
    """Test that we can initialize the HyperparameterTuner."""
    # Create mock study
    mock_study = Mock()
    mock_create_study.return_value = mock_study
    
    # Initialize HyperparameterTuner
    tuner = HyperparameterTuner(study_name="test_study")
    
    # Check that the tuner was initialized
    assert tuner.study_name == "test_study"
    assert tuner.direction == "maximize"
    assert tuner.study == mock_study


@patch('src.models.hyperparameter_tuner.optuna.create_study')
def test_hyperparameter_tuner_suggest_hyperparameters(mock_create_study):
    """Test that we can suggest hyperparameters."""
    # Create mock study and trial
    mock_study = Mock()
    mock_create_study.return_value = mock_study
    mock_trial = Mock()
    
    # Set up mock trial to return specific values
    mock_trial.suggest_float.return_value = 0.001
    mock_trial.suggest_int.return_value = 3
    mock_trial.suggest_categorical.return_value = 16
    
    # Initialize HyperparameterTuner
    tuner = HyperparameterTuner(study_name="test_study")
    
    # Call suggest_hyperparameters
    hyperparameters = tuner.suggest_hyperparameters(mock_trial)
    
    # Check that we got expected hyperparameters
    assert "learning_rate" in hyperparameters
    assert "num_train_epochs" in hyperparameters
    assert "per_device_train_batch_size" in hyperparameters
    assert "weight_decay" in hyperparameters
    assert "warmup_steps" in hyperparameters
    assert "adam_epsilon" in hyperparameters


@patch('src.models.hyperparameter_tuner.optuna.create_study')
def test_hyperparameter_tuner_create_training_args(mock_create_study):
    """Test that we can create training arguments."""
    # Create mock study
    mock_study = Mock()
    mock_create_study.return_value = mock_study
    
    # Initialize HyperparameterTuner
    tuner = HyperparameterTuner(study_name="test_study")
    
    # Create hyperparameters
    hyperparameters = {
        "learning_rate": 0.001,
        "num_train_epochs": 3,
        "per_device_train_batch_size": 16,
        "weight_decay": 0.01,
        "warmup_steps": 100,
        "adam_epsilon": 1e-8
    }
    
    # Call create_training_args
    training_args = tuner.create_training_args(hyperparameters)
    
    # Check that we got expected training arguments
    assert training_args.learning_rate == 0.001
    assert training_args.num_train_epochs == 3
    assert training_args.per_device_train_batch_size == 16
    assert training_args.weight_decay == 0.01
    assert training_args.warmup_steps == 100


if __name__ == "__main__":
    pytest.main([__file__])