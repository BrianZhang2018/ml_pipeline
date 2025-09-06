"""
Model Interpretability Module

This module provides functions to interpret and explain text classification models
using SHAP and LIME.

What Is This? (Explain Like I'm 5)
===============================
This is like having a teacher who can explain why our AI brain made a certain
decision. Just like when you ask your teacher "why did you mark this answer
wrong?", this module helps us understand why our AI thinks a movie review is
positive or negative. It shows us which words were most important in making
the decision.
"""

import sys
import os
from typing import Dict, List, Tuple, Any, Callable
import numpy as np
import warnings

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.utils.logger import get_logger

# Get logger instance
logger = get_logger("interpretability")

# Try to import SHAP and LIME, but handle cases where they're not available
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("SHAP not available. Install shap for model interpretability features.")

try:
    import lime
    import lime.lime_text
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    logger.warning("LIME not available. Install lime for model interpretability features.")


class ModelExplainer:
    """
    A class to explain model predictions using SHAP and LIME.
    """
    
    def __init__(self, model: Any = None, explainer_type: str = "shap"):
        """
        Initialize the ModelExplainer.
        
        Args:
            model (Any): The model to explain
            explainer_type (str): Type of explainer to use ("shap" or "lime")
        """
        logger.info(f"Initializing ModelExplainer with {explainer_type}")
        
        self.model = model
        self.explainer_type = explainer_type.lower()
        
        if self.explainer_type == "shap" and not SHAP_AVAILABLE:
            logger.warning("SHAP not available. Falling back to LIME if available.")
            if LIME_AVAILABLE:
                self.explainer_type = "lime"
            else:
                raise ImportError("Neither SHAP nor LIME is available. Please install one of them.")
        
        if self.explainer_type == "lime" and not LIME_AVAILABLE:
            logger.warning("LIME not available. Falling back to SHAP if available.")
            if SHAP_AVAILABLE:
                self.explainer_type = "shap"
            else:
                raise ImportError("Neither LIME nor SHAP is available. Please install one of them.")
        
        self.explainer = None
        logger.info("ModelExplainer initialized successfully")
    
    def create_explainer(self, 
                        training_texts: List[str] = None,
                        predict_function: Callable = None):
        """
        Create an explainer for the model.
        
        Args:
            training_texts (List[str], optional): Training texts for LIME
            predict_function (Callable, optional): Function to make predictions
        """
        logger.info(f"Creating {self.explainer_type} explainer")
        
        try:
            if self.explainer_type == "shap":
                self._create_shap_explainer()
            elif self.explainer_type == "lime":
                self._create_lime_explainer(training_texts, predict_function)
            
            logger.info(f"{self.explainer_type.upper()} explainer created successfully")
        except Exception as e:
            logger.error(f"Failed to create {self.explainer_type} explainer: {str(e)}")
            raise
    
    def _create_shap_explainer(self):
        """
        Create a SHAP explainer.
        """
        logger.debug("Creating SHAP explainer")
        
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP is not installed. Please install shap package.")
        
        # For transformer models, we would typically use shap.Explainer
        # This is a placeholder implementation
        logger.warning("SHAP explainer creation is not fully implemented in this example")
        self.explainer = "shap_placeholder"
    
    def _create_lime_explainer(self, 
                              training_texts: List[str],
                              predict_function: Callable):
        """
        Create a LIME explainer.
        
        Args:
            training_texts (List[str]): Training texts
            predict_function (Callable): Function to make predictions
        """
        logger.debug("Creating LIME explainer")
        
        if not LIME_AVAILABLE:
            raise ImportError("LIME is not installed. Please install lime package.")
        
        # Create LIME text explainer
        self.explainer = lime.lime_text.LimeTextExplainer(
            class_names=["Class_0", "Class_1"],  # Placeholder class names
            random_state=42
        )
        
        self.predict_function = predict_function
        self.training_texts = training_texts
        logger.debug("LIME explainer created successfully")
    
    def explain_prediction(self, 
                          text: str,
                          num_features: int = 10) -> Dict[str, Any]:
        """
        Explain a single prediction.
        
        Args:
            text (str): Text to explain
            num_features (int): Number of features to show in explanation
            
        Returns:
            Dict[str, Any]: Explanation results
        """
        logger.info(f"Explaining prediction for text: {text[:50]}...")
        
        if self.explainer is None:
            raise ValueError("Explainer not created. Call create_explainer() first.")
        
        try:
            if self.explainer_type == "shap":
                explanation = self._explain_with_shap(text, num_features)
            elif self.explainer_type == "lime":
                explanation = self._explain_with_lime(text, num_features)
            
            logger.info("Prediction explanation completed successfully")
            return explanation
        except Exception as e:
            logger.error(f"Failed to explain prediction: {str(e)}")
            raise
    
    def _explain_with_shap(self, text: str, num_features: int) -> Dict[str, Any]:
        """
        Explain prediction using SHAP.
        
        Args:
            text (str): Text to explain
            num_features (int): Number of features to show
            
        Returns:
            Dict[str, Any]: SHAP explanation
        """
        logger.debug("Explaining with SHAP")
        
        # This is a placeholder implementation
        # In a real implementation, we would use the SHAP explainer
        explanation = {
            "text": text,
            "explanation_type": "shap",
            "feature_importance": [],  # Placeholder
            "prediction": None  # Placeholder
        }
        
        return explanation
    
    def _explain_with_lime(self, text: str, num_features: int) -> Dict[str, Any]:
        """
        Explain prediction using LIME.
        
        Args:
            text (str): Text to explain
            num_features (int): Number of features to show
            
        Returns:
            Dict[str, Any]: LIME explanation
        """
        logger.debug("Explaining with LIME")
        
        # Create explanation using LIME
        explanation = self.explainer.explain_instance(
            text,
            self.predict_function,
            num_features=num_features
        )
        
        # Extract features and weights
        feature_weights = explanation.as_list()
        
        # Format the explanation
        result = {
            "text": text,
            "explanation_type": "lime",
            "feature_importance": feature_weights,
            "prediction": None  # Would need to get actual prediction
        }
        
        return result
    
    def get_feature_importance(self, 
                              texts: List[str],
                              labels: List[int] = None) -> Dict[str, Any]:
        """
        Get global feature importance.
        
        Args:
            texts (List[str]): Texts to analyze
            labels (List[int], optional): Labels for the texts
            
        Returns:
            Dict[str, Any]: Feature importance results
        """
        logger.info("Calculating global feature importance")
        
        # This is a placeholder implementation
        feature_importance = {
            "method": self.explainer_type,
            "texts_analyzed": len(texts),
            "feature_importance": []  # Placeholder
        }
        
        logger.info("Feature importance calculation completed")
        return feature_importance


def simple_explanation(text: str, 
                      model: Any = None,
                      explainer_type: str = "shap") -> Dict[str, Any]:
    """
    Simple function to explain a prediction.
    
    Args:
        text (str): Text to explain
        model (Any): Model to explain
        explainer_type (str): Type of explainer to use
        
    Returns:
        Dict[str, Any]: Explanation results
    """
    logger.info(f"Creating simple explanation with {explainer_type}")
    
    try:
        explainer = ModelExplainer(model, explainer_type)
        # In a real implementation, we would create the explainer properly
        # For now, we return a placeholder explanation
        explanation = {
            "text": text,
            "explanation_type": explainer_type,
            "feature_importance": [("word1", 0.5), ("word2", 0.3), ("word3", 0.2)],
            "prediction": "positive"  # Placeholder
        }
        
        logger.info("Simple explanation created successfully")
        return explanation
    except Exception as e:
        logger.error(f"Failed to create simple explanation: {str(e)}")
        raise


# Example usage
if __name__ == "__main__":
    # This is for testing purposes only
    print("Model Interpretability Module")
    print("Available classes and functions:")
    print("- ModelExplainer class")
    print("- simple_explanation function")
    
    # Check if SHAP and LIME are available
    print(f"\nSHAP available: {SHAP_AVAILABLE}")
    print(f"LIME available: {LIME_AVAILABLE}")