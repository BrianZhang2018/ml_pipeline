"""
API Models Module

This module defines the Pydantic models for request and response validation
in our FastAPI application.

What Is This? (Explain Like I'm 5)
===============================
This is like a checklist for what information our AI needs to work and what
it will give back to us. Just like when you fill out a form at school, you
need to write your name and grade in the right places. Our AI has a similar
form that makes sure it gets the right information and gives back the right
answers.
"""

from pydantic import BaseModel
from typing import List, Optional, Dict, Any


class TextClassificationRequest(BaseModel):
    """
    Model for text classification request.
    
    This model defines what information we need to classify text.
    """
    texts: List[str]
    model_name: str = "distilbert-base-uncased"
    max_length: int = 512


class Prediction(BaseModel):
    """
    Model for a single prediction.
    
    This model defines what a single prediction looks like.
    """
    label: str
    confidence: float
    probabilities: Dict[str, float]


class TextClassificationResponse(BaseModel):
    """
    Model for text classification response.
    
    This model defines what our API will return after classifying text.
    """
    predictions: List[Prediction]
    model_name: str
    processing_time: float


class HealthCheckResponse(BaseModel):
    """
    Model for health check response.
    
    This model defines what our health check endpoint returns.
    """
    status: str
    timestamp: str
    service: str = "LLM Text Classification API"


class ModelInfoResponse(BaseModel):
    """
    Model for model information response.
    
    This model defines what information about loaded models we provide.
    """
    model_name: str
    model_type: str
    num_labels: int
    max_length: int
    loaded: bool


# Example usage
if __name__ == "__main__":
    # This is for testing purposes only
    print("API Models Module")
    print("Available models:")
    print("- TextClassificationRequest")
    print("- Prediction")
    print("- TextClassificationResponse")
    print("- HealthCheckResponse")
    print("- ModelInfoResponse")