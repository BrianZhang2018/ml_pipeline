"""
Test script for API models
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from api.models import (
    TextClassificationRequest,
    Prediction,
    TextClassificationResponse,
    HealthCheckResponse,
    ModelInfoResponse
)


def test_text_classification_request():
    """Test TextClassificationRequest model"""
    print("Testing TextClassificationRequest...")
    
    # Test case 1: Basic request
    request_data = {
        "texts": ["This is a test text", "Another test text"],
        "model_name": "distilbert-base-uncased",
        "max_length": 512
    }
    
    request = TextClassificationRequest(**request_data)
    
    assert request.texts == request_data["texts"]
    assert request.model_name == request_data["model_name"]
    assert request.max_length == request_data["max_length"]
    print("✅ Basic request test passed")
    
    # Test case 2: Request with default values
    request_data_minimal = {
        "texts": ["Test text"]
    }
    
    request = TextClassificationRequest(**request_data_minimal)
    
    assert request.texts == request_data_minimal["texts"]
    assert request.model_name == "distilbert-base-uncased"  # Default value
    assert request.max_length == 512  # Default value
    print("✅ Request with defaults test passed")
    
    print("✅ All TextClassificationRequest tests passed\n")


def test_prediction_model():
    """Test Prediction model"""
    print("Testing Prediction model...")
    
    # Test case: Basic prediction
    prediction_data = {
        "label": "positive",
        "confidence": 0.95,
        "probabilities": {
            "positive": 0.95,
            "negative": 0.05
        }
    }
    
    prediction = Prediction(**prediction_data)
    
    assert prediction.label == prediction_data["label"]
    assert prediction.confidence == prediction_data["confidence"]
    assert prediction.probabilities == prediction_data["probabilities"]
    print("✅ Basic prediction test passed")
    
    print("✅ All Prediction tests passed\n")


def test_text_classification_response():
    """Test TextClassificationResponse model"""
    print("Testing TextClassificationResponse model...")
    
    # Test case: Basic response
    response_data = {
        "predictions": [
            {
                "label": "positive",
                "confidence": 0.95,
                "probabilities": {
                    "positive": 0.95,
                    "negative": 0.05
                }
            }
        ],
        "model_name": "distilbert-base-uncased",
        "processing_time": 0.123
    }
    
    response = TextClassificationResponse(**response_data)
    
    assert len(response.predictions) == 1
    assert response.model_name == response_data["model_name"]
    assert response.processing_time == response_data["processing_time"]
    print("✅ Basic response test passed")
    
    print("✅ All TextClassificationResponse tests passed\n")


def test_health_check_response():
    """Test HealthCheckResponse model"""
    print("Testing HealthCheckResponse model...")
    
    # Test case: Basic health check
    health_data = {
        "status": "healthy",
        "timestamp": "2023-01-01 12:00:00"
    }
    
    health = HealthCheckResponse(**health_data)
    
    assert health.status == health_data["status"]
    assert health.timestamp == health_data["timestamp"]
    assert health.service == "LLM Text Classification API"  # Default value
    print("✅ Basic health check test passed")
    
    print("✅ All HealthCheckResponse tests passed\n")


def test_model_info_response():
    """Test ModelInfoResponse model"""
    print("Testing ModelInfoResponse model...")
    
    # Test case: Basic model info
    info_data = {
        "model_name": "distilbert-base-uncased",
        "model_type": "distilbert",
        "num_labels": 2,
        "max_length": 512,
        "loaded": True
    }
    
    info = ModelInfoResponse(**info_data)
    
    assert info.model_name == info_data["model_name"]
    assert info.model_type == info_data["model_type"]
    assert info.num_labels == info_data["num_labels"]
    assert info.max_length == info_data["max_length"]
    assert info.loaded == info_data["loaded"]
    print("✅ Basic model info test passed")
    
    print("✅ All ModelInfoResponse tests passed\n")


def test_api_models():
    """Test all API models"""
    print("Testing API Models...")
    print("=" * 20)
    
    try:
        test_text_classification_request()
        test_prediction_model()
        test_text_classification_response()
        test_health_check_response()
        test_model_info_response()
        
        print("🎉 All API model tests passed!")
        return True
    except Exception as e:
        print(f"❌ Error in API model tests: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_api_models()
    sys.exit(0 if success else 1)