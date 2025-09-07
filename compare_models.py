#!/usr/bin/env python3
"""
Model Evaluation and Comparison Script

This script compares the performance of our trained model against the pre-trained baseline
to demonstrate the improvement from domain-specific training.

What Does This Do? (Explain Like I'm 5)
=======================================
This is like a test to see how much smarter our AI became after training!
We compare the old AI (before training) with the new AI (after training)
to see which one is better at understanding movie reviews.
"""

import sys
import os
from typing import Dict, List, Tuple
import torch
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import pandas as pd

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from data.data_ingestion import load_imdb_dataset
from data.data_preprocessing import preprocess_texts
from utils.logger import get_logger

# Get logger instance
logger = get_logger("model_comparison")


def load_trained_model(model_path: str) -> Tuple[AutoModelForSequenceClassification, AutoTokenizer]:
    """
    Load a trained model and tokenizer.
    
    Args:
        model_path (str): Path to the trained model
        
    Returns:
        Tuple[AutoModelForSequenceClassification, AutoTokenizer]: Model and tokenizer
    """
    logger.info(f"Loading trained model from {model_path}")
    
    try:
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        logger.info("Trained model loaded successfully")
        return model, tokenizer
    except Exception as e:
        logger.error(f"Failed to load trained model: {str(e)}")
        raise


def load_pretrained_model(model_name: str) -> Tuple[AutoModelForSequenceClassification, AutoTokenizer]:
    """
    Load a pre-trained model (baseline).
    
    Args:
        model_name (str): Name of the pre-trained model
        
    Returns:
        Tuple[AutoModelForSequenceClassification, AutoTokenizer]: Model and tokenizer
    """
    logger.info(f"Loading pre-trained model {model_name}")
    
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=2,
            local_files_only=True  # Use our offline-first approach
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        logger.info("Pre-trained model loaded successfully")
        return model, tokenizer
    except Exception as e:
        logger.info("Local model not found, downloading...")
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        logger.info("Pre-trained model downloaded and loaded")
        return model, tokenizer


def predict_batch(
    texts: List[str], 
    model: AutoModelForSequenceClassification, 
    tokenizer: AutoTokenizer,
    max_length: int = 512
) -> List[Dict[str, float]]:
    """
    Make predictions on a batch of texts.
    
    Args:
        texts (List[str]): List of texts to predict
        model: The model to use for prediction
        tokenizer: The tokenizer to use
        max_length (int): Maximum sequence length
        
    Returns:
        List[Dict[str, float]]: List of predictions with probabilities
    """
    predictions = []
    
    # Set model to evaluation mode
    model.eval()
    
    with torch.no_grad():
        for text in texts:
            # Tokenize
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=max_length
            )
            
            # Get predictions
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            
            # Extract probabilities
            negative_prob = probabilities[0][0].item()
            positive_prob = probabilities[0][1].item()
            
            # Determine prediction
            predicted_label = "positive" if positive_prob > negative_prob else "negative"
            confidence = max(positive_prob, negative_prob)
            
            predictions.append({
                "predicted_label": predicted_label,
                "confidence": confidence,
                "negative_prob": negative_prob,
                "positive_prob": positive_prob
            })
    
    return predictions


def evaluate_model(
    texts: List[str], 
    true_labels: List[str], 
    model: AutoModelForSequenceClassification, 
    tokenizer: AutoTokenizer,
    model_name: str
) -> Dict[str, float]:
    """
    Evaluate a model on the given texts and labels.
    
    Args:
        texts (List[str]): List of texts
        true_labels (List[str]): List of true labels
        model: Model to evaluate
        tokenizer: Tokenizer to use
        model_name (str): Name of the model (for logging)
        
    Returns:
        Dict[str, float]: Evaluation metrics
    """
    logger.info(f"Evaluating {model_name} on {len(texts)} samples")
    
    # Get predictions
    predictions = predict_batch(texts, model, tokenizer)
    
    # Calculate metrics
    correct = 0
    total_confidence = 0
    
    for pred, true_label in zip(predictions, true_labels):
        if pred["predicted_label"] == true_label:
            correct += 1
        total_confidence += pred["confidence"]
    
    accuracy = correct / len(texts)
    avg_confidence = total_confidence / len(texts)
    
    metrics = {
        "accuracy": accuracy,
        "avg_confidence": avg_confidence,
        "total_samples": len(texts)
    }
    
    logger.info(f"{model_name} Results:")
    logger.info(f"  Accuracy: {accuracy:.4f}")
    logger.info(f"  Average Confidence: {avg_confidence:.4f}")
    
    return metrics


def compare_models(
    trained_model_path: str,
    pretrained_model_name: str = "distilbert-base-uncased",
    num_samples: int = 1000
) -> Dict[str, Dict[str, float]]:
    """
    Compare trained model vs pre-trained baseline.
    
    Args:
        trained_model_path (str): Path to the trained model
        pretrained_model_name (str): Name of the pre-trained model
        num_samples (int): Number of samples to evaluate on
        
    Returns:
        Dict[str, Dict[str, float]]: Comparison results
    """
    logger.info("="*60)
    logger.info("MODEL COMPARISON: TRAINED vs PRE-TRAINED")
    logger.info("="*60)
    
    # Load test data
    logger.info("Loading test data...")
    test_df = load_imdb_dataset("test").head(num_samples)
    
    # Preprocess texts
    texts = preprocess_texts(test_df['text'].tolist())
    labels = ["positive" if label == 1 else "negative" for label in test_df['label']]
    
    logger.info(f"Loaded {len(texts)} test samples")
    
    # Load models
    logger.info("\nLoading models...")
    trained_model, trained_tokenizer = load_trained_model(trained_model_path)
    pretrained_model, pretrained_tokenizer = load_pretrained_model(pretrained_model_name)
    
    # Evaluate trained model
    logger.info("\n📊 Evaluating TRAINED model...")
    trained_results = evaluate_model(
        texts, labels, trained_model, trained_tokenizer, "TRAINED MODEL"
    )
    
    # Evaluate pre-trained model
    logger.info("\n📊 Evaluating PRE-TRAINED baseline...")
    pretrained_results = evaluate_model(
        texts, labels, pretrained_model, pretrained_tokenizer, "PRE-TRAINED BASELINE"
    )
    
    # Calculate improvements
    accuracy_improvement = trained_results["accuracy"] - pretrained_results["accuracy"]
    confidence_improvement = trained_results["avg_confidence"] - pretrained_results["avg_confidence"]
    
    logger.info("\n" + "="*60)
    logger.info("🏆 COMPARISON RESULTS")
    logger.info("="*60)
    logger.info(f"📈 ACCURACY:")
    logger.info(f"   Pre-trained: {pretrained_results['accuracy']:.4f}")
    logger.info(f"   Trained:     {trained_results['accuracy']:.4f}")
    logger.info(f"   Improvement: +{accuracy_improvement:.4f} ({accuracy_improvement*100:.2f}%)")
    
    logger.info(f"\n🎯 CONFIDENCE:")
    logger.info(f"   Pre-trained: {pretrained_results['avg_confidence']:.4f}")
    logger.info(f"   Trained:     {trained_results['avg_confidence']:.4f}")
    logger.info(f"   Improvement: +{confidence_improvement:.4f} ({confidence_improvement*100:.2f}%)")
    
    if accuracy_improvement > 0 and confidence_improvement > 0:
        logger.info(f"\n🎉 SUCCESS: Training improved both accuracy AND confidence!")
    elif accuracy_improvement > 0:
        logger.info(f"\n✅ Training improved accuracy!")
    elif confidence_improvement > 0:
        logger.info(f"\n✅ Training improved confidence!")
    else:
        logger.info(f"\n⚠️  Training may need adjustment - no clear improvement detected")
    
    logger.info("="*60)
    
    return {
        "trained": trained_results,
        "pretrained": pretrained_results,
        "improvements": {
            "accuracy": accuracy_improvement,
            "confidence": confidence_improvement
        }
    }


def test_individual_examples(
    trained_model_path: str,
    pretrained_model_name: str = "distilbert-base-uncased"
) -> None:
    """
    Test individual examples to see the difference.
    
    Args:
        trained_model_path (str): Path to the trained model
        pretrained_model_name (str): Name of the pre-trained model
    """
    logger.info("\n🧪 TESTING INDIVIDUAL EXAMPLES")
    logger.info("="*40)
    
    # Load models
    trained_model, trained_tokenizer = load_trained_model(trained_model_path)
    pretrained_model, pretrained_tokenizer = load_pretrained_model(pretrained_model_name)
    
    # Test examples
    test_examples = [
        "This movie was absolutely amazing! I loved every minute of it.",
        "Terrible film. Complete waste of time. Very disappointing.",
        "The movie was okay, nothing special but not bad either.",
        "Outstanding performance by the actors. Highly recommended!",
        "Boring and predictable plot. Fell asleep halfway through."
    ]
    
    for i, text in enumerate(test_examples, 1):
        logger.info(f"\n--- Example {i} ---")
        logger.info(f"Text: {text[:80]}...")
        
        # Get predictions from both models
        trained_pred = predict_batch([text], trained_model, trained_tokenizer)[0]
        pretrained_pred = predict_batch([text], pretrained_model, pretrained_tokenizer)[0]
        
        logger.info(f"Trained:     {trained_pred['predicted_label']} (conf: {trained_pred['confidence']:.3f})")
        logger.info(f"Pre-trained: {pretrained_pred['predicted_label']} (conf: {pretrained_pred['confidence']:.3f})")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare trained vs pre-trained models")
    parser.add_argument("--trained_model", required=True, help="Path to trained model")
    parser.add_argument("--pretrained_model", default="distilbert-base-uncased", help="Pre-trained model name")
    parser.add_argument("--samples", type=int, default=1000, help="Number of test samples")
    parser.add_argument("--examples", action="store_true", help="Show individual examples")
    
    args = parser.parse_args()
    
    try:
        # Compare models
        results = compare_models(
            args.trained_model,
            args.pretrained_model,
            args.samples
        )
        
        # Show individual examples if requested
        if args.examples:
            test_individual_examples(args.trained_model, args.pretrained_model)
        
        print("\n✅ Model comparison completed successfully!")
        
    except Exception as e:
        logger.error(f"Comparison failed: {str(e)}")
        print(f"\n❌ ERROR: Comparison failed - {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()