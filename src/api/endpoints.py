"""
API Endpoints Module

This module defines the API endpoints for our text classification service.

What Is This? (Explain Like I'm 5)
===============================
This is like the buttons on a toy that make different things happen. Each
button does something specific - one might make the toy walk, another might
make it talk. Our API has different buttons (endpoints) that do different
things like check if the AI is working, classify text, or tell us about
the AI brain we're using.
"""

import sys
import os
import time
from typing import Dict, List, Any
import torch
import torch.nn.functional as F
from fastapi import APIRouter, HTTPException
from transformers import AutoTokenizer

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.api.models import (
    TextClassificationRequest,
    TextClassificationResponse,
    Prediction,
    HealthCheckResponse,
    ModelInfoResponse
)
from src.api.model_loader import load_model, get_model_info
from src.utils.logger import get_logger

# Get logger instance
logger = get_logger("api_endpoints")

# Create router
router = APIRouter()


@router.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """
    Health check endpoint.
    
    This endpoint checks if the API is running and responding correctly.
    
    Returns:
        HealthCheckResponse: Health status information
    """
    logger.info("Health check requested")
    
    return HealthCheckResponse(
        status="healthy",
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
    )


@router.get("/models/{model_name}", response_model=ModelInfoResponse)
async def get_model_info_endpoint(model_name: str, max_length: int = 512):
    """
    Get model information endpoint.
    
    This endpoint provides information about a specific model.
    
    Args:
        model_name (str): Name of the model
        max_length (int): Maximum sequence length
        
    Returns:
        ModelInfoResponse: Model information
    """
    logger.info(f"Model info requested for '{model_name}'")
    
    try:
        model_info = get_model_info(model_name, max_length)
        return ModelInfoResponse(**model_info)
    except Exception as e:
        logger.error(f"Failed to get model info for '{model_name}': {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get model info: {str(e)}")


@router.post("/classify", response_model=TextClassificationResponse)
async def classify_text(request: TextClassificationRequest):
    """
    Text classification endpoint.
    
    This endpoint classifies text using a pre-trained model.
    
    Args:
        request (TextClassificationRequest): Classification request
        
    Returns:
        TextClassificationResponse: Classification results
    """
    logger.info(f"Classification requested for {len(request.texts)} texts using model '{request.model_name}'")
    
    start_time = time.time()
    
    try:
        # Load model and tokenizer
        model_data = load_model(request.model_name, request.max_length)
        model = model_data["model"]
        tokenizer = model_data["tokenizer"]
        num_labels = model_data["num_labels"]
        
        # Prepare predictions list
        predictions = []
        
        # Process each text
        for text in request.texts:
            # Tokenize text
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=request.max_length
            )
            
            # Get predictions
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                probabilities = F.softmax(logits, dim=-1)
                
                # Get predicted label
                predicted_class_id = logits.argmax().item()
                confidence = probabilities[0][predicted_class_id].item()
                
                # Convert probabilities to dictionary
                prob_dict = {}
                for i in range(num_labels):
                    prob_dict[f"label_{i}"] = probabilities[0][i].item()
                
                # Create prediction
                prediction = Prediction(
                    label=f"label_{predicted_class_id}",
                    confidence=confidence,
                    probabilities=prob_dict
                )
                
                predictions.append(prediction)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        logger.info(f"Classification completed in {processing_time:.4f} seconds")
        
        # Return response
        return TextClassificationResponse(
            predictions=predictions,
            model_name=request.model_name,
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Classification failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")


@router.post("/classify/batch", response_model=TextClassificationResponse)
async def classify_text_batch(request: TextClassificationRequest):
    """
    Batch text classification endpoint.
    
    This endpoint classifies multiple texts in a batch for better performance.
    
    Args:
        request (TextClassificationRequest): Classification request
        
    Returns:
        TextClassificationResponse: Classification results
    """
    logger.info(f"Batch classification requested for {len(request.texts)} texts using model '{request.model_name}'")
    
    start_time = time.time()
    
    try:
        # Load model and tokenizer
        model_data = load_model(request.model_name, request.max_length)
        model = model_data["model"]
        tokenizer = model_data["tokenizer"]
        num_labels = model_data["num_labels"]
        
        # Tokenize all texts at once
        inputs = tokenizer(
            request.texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=request.max_length
        )
        
        # Get predictions
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = F.softmax(logits, dim=-1)
            
            # Process each prediction
            predictions = []
            for i in range(len(request.texts)):
                # Get predicted label
                predicted_class_id = logits[i].argmax().item()
                confidence = probabilities[i][predicted_class_id].item()
                
                # Convert probabilities to dictionary
                prob_dict = {}
                for j in range(num_labels):
                    prob_dict[f"label_{j}"] = probabilities[i][j].item()
                
                # Create prediction
                prediction = Prediction(
                    label=f"label_{predicted_class_id}",
                    confidence=confidence,
                    probabilities=prob_dict
                )
                
                predictions.append(prediction)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        logger.info(f"Batch classification completed in {processing_time:.4f} seconds")
        
        # Return response
        return TextClassificationResponse(
            predictions=predictions,
            model_name=request.model_name,
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Batch classification failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch classification failed: {str(e)}")


# Example usage
if __name__ == "__main__":
    # This is for testing purposes only
    print("API Endpoints Module")
    print("Available endpoints:")
    print("- GET /health")
    print("- GET /models/{model_name}")
    print("- POST /classify")
    print("- POST /classify/batch")