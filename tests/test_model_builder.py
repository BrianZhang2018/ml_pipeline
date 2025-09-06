"""
Unit tests for the model builder module.
"""

import sys
import os
import pytest
from unittest.mock import Mock, patch

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.model_builder import (
    build_model,
    get_supported_models,
    customize_model_config
)
from transformers import AutoConfig


def test_get_supported_models():
    """Test that we can get supported models."""
    models = get_supported_models()
    assert isinstance(models, dict)
    assert len(models) > 0
    assert "bert-base-uncased" in models


@patch('src.models.model_builder.AutoModelForSequenceClassification')
@patch('src.models.model_builder.AutoTokenizer')
@patch('src.models.model_builder.AutoConfig')
def test_build_model(mock_auto_config, mock_auto_tokenizer, mock_auto_model):
    """Test that we can build a model."""
    # Create mock objects
    mock_config = Mock()
    mock_tokenizer = Mock()
    mock_model = Mock()
    
    # Configure mocks
    mock_auto_config.from_pretrained.return_value = mock_config
    mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer
    mock_auto_model.from_pretrained.return_value = mock_model
    
    # Call build_model
    result = build_model(
        model_name="distilbert-base-uncased",
        num_labels=2
    )
    
    # Check that we got all expected components
    assert "model" in result
    assert "tokenizer" in result
    assert "config" in result
    assert "model_name" in result
    
    # Check types
    assert result["model_name"] == "distilbert-base-uncased"
    assert result["model"] == mock_model
    assert result["tokenizer"] == mock_tokenizer
    assert result["config"] == mock_config


def test_customize_model_config():
    """Test that we can customize model configuration."""
    # Create a base config
    base_config = AutoConfig.from_pretrained("bert-base-uncased", num_labels=2)
    
    # Customize it
    custom_config = customize_model_config(
        base_config,
        hidden_dropout_prob=0.3,
        attention_probs_dropout_prob=0.3
    )
    
    # Check that customization worked
    assert custom_config.hidden_dropout_prob == 0.3
    assert custom_config.attention_probs_dropout_prob == 0.3


@patch('src.models.model_builder.AutoModelForSequenceClassification')
@patch('src.models.model_builder.AutoTokenizer')
@patch('src.models.model_builder.AutoConfig')
def test_build_model_invalid(mock_auto_config, mock_auto_tokenizer, mock_auto_model):
    """Test that building an invalid model raises an exception."""
    # Configure mock to raise an exception
    mock_auto_config.from_pretrained.side_effect = Exception("Invalid model")
    
    # Check that building an invalid model raises an exception
    with pytest.raises(Exception):
        build_model(model_name="invalid-model-name-12345", num_labels=2)


if __name__ == "__main__":
    pytest.main([__file__])