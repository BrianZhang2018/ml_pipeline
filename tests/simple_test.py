#!/usr/bin/env python3
"""
Simple test script to verify our implementation without using pytest.
"""

import sys
import os
import numpy as np

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

def test_compute_metrics():
    """Test that we can compute metrics."""
    # Import the function directly
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
    
    print("✓ compute_metrics test passed")
    return True

def test_get_supported_models():
    """Test that we can get supported models."""
    from src.models.model_builder import get_supported_models
    
    models = get_supported_models()
    assert isinstance(models, dict)
    assert len(models) > 0
    assert "bert-base-uncased" in models
    
    print("✓ get_supported_models test passed")
    return True

def test_customize_model_config():
    """Test that we can customize model configuration."""
    from src.models.model_builder import customize_model_config
    from transformers import AutoConfig
    
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
    
    print("✓ customize_model_config test passed")
    return True

if __name__ == "__main__":
    print("Running simple tests...")
    
    try:
        test_compute_metrics()
        test_get_supported_models()
        test_customize_model_config()
        print("\n✓ All tests passed!")
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        sys.exit(1)