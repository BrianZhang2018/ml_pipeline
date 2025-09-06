"""
Unit tests for the model trainer module.
"""

import sys
import os
import pytest
import numpy as np
from unittest.mock import Mock, patch

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_compute_metrics():
    """Test that we can compute metrics."""
    # Import here to avoid environment issues
    from src.models.model_trainer import compute_metrics
    
    # Create mock predictions and labels
    predictions = np.array([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7], [0.6, 0.4]])
    labels = np.array([1, 0, 1, 0])
    
    # Call compute_metrics
    result = compute_metrics((predictions, labels))
    
    # Check that we got expected metrics
    assert "accuracy" in result
    assert "f1" in result
    assert "precision" in result
    assert "recall" in result
    assert isinstance(result["accuracy"], float)
    assert 0 <= result["accuracy"] <= 1


@patch('src.models.model_trainer.Trainer')
@patch('src.models.model_trainer.TrainingArguments')
def test_model_trainer_initialization(mock_training_args, mock_trainer):
    """Test that we can initialize the ModelTrainer."""
    # Import here to avoid environment issues
    from src.models.model_trainer import ModelTrainer
    
    # Create mock model, tokenizer, and datasets
    mock_model = Mock()
    mock_tokenizer = Mock()
    mock_train_dataset = Mock()
    mock_eval_dataset = Mock()
    
    # Initialize ModelTrainer
    trainer = ModelTrainer(
        model=mock_model,
        tokenizer=mock_tokenizer,
        train_dataset=mock_train_dataset,
        eval_dataset=mock_eval_dataset,
        output_dir="/tmp/test_output",
        num_train_epochs=1
    )
    
    # Check that the trainer was initialized
    assert trainer.model == mock_model
    assert trainer.tokenizer == mock_tokenizer
    assert trainer.train_dataset == mock_train_dataset
    assert trainer.eval_dataset == mock_eval_dataset


@patch('src.models.model_trainer.Trainer')
@patch('src.models.model_trainer.TrainingArguments')
def test_model_trainer_train(mock_training_args, mock_trainer):
    """Test that we can train a model."""
    # Import here to avoid environment issues
    from src.models.model_trainer import ModelTrainer
    
    # Create mock model, tokenizer, and datasets
    mock_model = Mock()
    mock_tokenizer = Mock()
    mock_train_dataset = Mock()
    mock_eval_dataset = Mock()
    
    # Create mock train result
    mock_train_result = Mock()
    mock_train_result.training_loss = 0.5
    
    # Configure mock trainer
    mock_trainer_instance = Mock()
    mock_trainer_instance.train.return_value = mock_train_result
    mock_trainer.return_value = mock_trainer_instance
    
    # Initialize ModelTrainer
    trainer = ModelTrainer(
        model=mock_model,
        tokenizer=mock_tokenizer,
        train_dataset=mock_train_dataset,
        eval_dataset=mock_eval_dataset,
        output_dir="/tmp/test_output",
        num_train_epochs=1
    )
    
    # Call train
    result = trainer.train()
    
    # Check that train was called and returned expected result
    mock_trainer_instance.train.assert_called_once()
    assert "train_result" in result
    assert result["train_result"].training_loss == 0.5


if __name__ == "__main__":
    pytest.main([__file__])