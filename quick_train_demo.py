#!/usr/bin/env python3
"""
Quick Training Demo

This script demonstrates model training with synthetic data to avoid large downloads.
Perfect for testing our training pipeline quickly!

What Does This Do? (Explain Like I'm 5)
=======================================
Instead of waiting for huge movie review files to download, we create our own
tiny fake movie reviews to teach our AI. It's like practicing with toy cars
before driving a real car - faster and gets the same learning done!
"""

import sys
import os
from typing import Dict, List
import torch
from datasets import Dataset
import pandas as pd

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from models.model_builder import build_model
from models.model_trainer import ModelTrainer
from utils.logger import get_logger

# Get logger instance
logger = get_logger("quick_train_demo")


def create_synthetic_data() -> Dict[str, Dataset]:
    """
    Create synthetic movie review data for quick training.
    
    Returns:
        Dict[str, Dataset]: Train and eval datasets
    """
    logger.info("Creating synthetic movie review data...")
    
    # Positive movie reviews
    positive_reviews = [
        "This movie was absolutely amazing! I loved every minute of it.",
        "Outstanding performance by the actors. Highly recommended!",
        "Brilliant storytelling and excellent cinematography. A masterpiece!",
        "One of the best films I've ever seen. Incredible and moving.",
        "Fantastic movie with great acting and beautiful visuals.",
        "Wonderful experience! The plot was engaging and well-developed.",
        "Superb direction and amazing soundtrack. Truly exceptional!",
        "Loved this film! Great characters and compelling story.",
        "An absolute gem! Perfect entertainment for the whole family.",
        "Excellent movie with outstanding special effects and acting."
    ]
    
    # Negative movie reviews  
    negative_reviews = [
        "Terrible film. Complete waste of time. Very disappointing.",
        "Boring and predictable plot. Fell asleep halfway through.",
        "Poor acting and weak storyline. Not worth watching.",
        "Awful movie with terrible dialogue and bad direction.",
        "Completely boring. Nothing interesting happens at all.",
        "Bad script and unconvincing performances. Skip this one.",
        "Disappointing sequel that ruins the original. Very bad.",
        "Waste of money. Poorly made with terrible special effects.",
        "Boring characters and uninteresting plot. Very dull.",
        "Bad acting and confusing story. Not recommended at all."
    ]
    
    # Create training data (more samples)
    train_texts = positive_reviews * 5 + negative_reviews * 5  # 100 samples total
    train_labels = [1] * 50 + [0] * 50  # 1=positive, 0=negative
    
    # Create eval data (smaller)
    eval_texts = positive_reviews[:5] + negative_reviews[:5]  # 10 samples
    eval_labels = [1] * 5 + [0] * 5
    
    # Create DataFrames
    train_df = pd.DataFrame({"text": train_texts, "label": train_labels})
    eval_df = pd.DataFrame({"text": eval_texts, "label": eval_labels})
    
    # Convert to Datasets
    train_dataset = Dataset.from_pandas(train_df)
    eval_dataset = Dataset.from_pandas(eval_df)
    
    logger.info(f"Created synthetic data - Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")
    
    return {
        "train": train_dataset,
        "eval": eval_dataset
    }


def tokenize_datasets(datasets: Dict[str, Dataset], model_name: str, max_length: int = 128) -> Dict[str, Dataset]:
    """
    Tokenize the datasets for training.
    
    Args:
        datasets (Dict[str, Dataset]): Raw datasets
        model_name (str): Name of the model/tokenizer to use
        max_length (int): Maximum sequence length (shorter for speed)
        
    Returns:
        Dict[str, Dataset]: Tokenized datasets
    """
    logger.info(f"Tokenizing datasets with {model_name}")
    
    # Build model to get tokenizer
    model_components = build_model(model_name, num_labels=2)
    tokenizer = model_components["tokenizer"]
    
    def tokenize_function(examples):
        # Tokenize the texts
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors=None  # Return Python lists for datasets compatibility
        )
        
        # Add labels
        tokenized["labels"] = examples["label"]
        return tokenized
    
    # Tokenize both datasets
    tokenized_datasets = {}
    for split, dataset in datasets.items():
        logger.info(f"Tokenizing {split} dataset...")
        tokenized_datasets[split] = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )
    
    logger.info("Tokenization completed")
    return tokenized_datasets


def quick_train_demo(
    model_name: str = "distilbert-base-uncased",
    output_dir: str = "./quick_trained_model",
    num_epochs: int = 3,
    batch_size: int = 8
) -> Dict[str, any]:
    """
    Run a quick training demonstration.
    
    Args:
        model_name (str): Name of the pre-trained model to use
        output_dir (str): Directory to save the trained model
        num_epochs (int): Number of training epochs
        batch_size (int): Training batch size
        
    Returns:
        Dict[str, any]: Training results
    """
    logger.info("🚀 STARTING QUICK TRAINING DEMO")
    logger.info("="*50)
    logger.info(f"Model: {model_name}")
    logger.info(f"Epochs: {num_epochs}")
    logger.info(f"Batch size: {batch_size}")
    
    # Step 1: Create synthetic data
    logger.info("\n📊 STEP 1: Creating synthetic movie review data...")
    datasets = create_synthetic_data()
    
    # Step 2: Tokenize datasets
    logger.info("\n🔤 STEP 2: Tokenizing datasets...")
    tokenized_datasets = tokenize_datasets(datasets, model_name, max_length=128)
    
    # Step 3: Build model
    logger.info("\n🧠 STEP 3: Building model...")
    model_components = build_model(model_name, num_labels=2)
    model = model_components["model"]
    tokenizer = model_components["tokenizer"]
    
    # Step 4: Initialize trainer
    logger.info("\n🎯 STEP 4: Initializing trainer...")
    trainer = ModelTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["eval"],
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=2e-5,
        warmup_steps=10,  # Less warmup for small dataset
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1"
    )
    
    # Step 5: Train model
    logger.info("\n🚀 STEP 5: Starting training...")
    training_results = trainer.train()
    
    # Step 6: Evaluate model
    logger.info("\n📈 STEP 6: Evaluating model...")
    eval_results = trainer.evaluate()
    
    # Step 7: Save model
    logger.info("\n💾 STEP 7: Saving trained model...")
    trainer.save_model(output_dir)
    
    # Test a prediction
    logger.info("\n🧪 STEP 8: Testing prediction...")
    test_text = "This movie was fantastic and amazing!"
    
    # Tokenize test text
    inputs = tokenizer(test_text, return_tensors="pt", truncation=True, padding=True)
    
    # Get prediction
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_class = torch.argmax(probabilities, dim=-1).item()
        confidence = probabilities[0][predicted_class].item()
    
    prediction = "positive" if predicted_class == 1 else "negative"
    
    # Log final results
    logger.info("\n🎉 QUICK TRAINING DEMO COMPLETED!")
    logger.info("="*50)
    logger.info("FINAL RESULTS:")
    logger.info(f"Training Loss: {training_results['train_result'].training_loss:.4f}")
    logger.info(f"Evaluation Accuracy: {eval_results['eval_accuracy']:.4f}")
    logger.info(f"Evaluation F1: {eval_results['eval_f1']:.4f}")
    logger.info(f"Test Prediction: '{test_text}' → {prediction} (confidence: {confidence:.3f})")
    logger.info(f"Model saved to: {output_dir}")
    logger.info("="*50)
    
    return {
        "training_results": training_results,
        "eval_results": eval_results,
        "test_prediction": {
            "text": test_text,
            "prediction": prediction,
            "confidence": confidence
        },
        "model_path": output_dir
    }


def main():
    """Main function."""
    try:
        # Run quick training demo
        results = quick_train_demo()
        
        print("\n🎊 SUCCESS! Quick training demo completed!")
        print(f"📍 Model saved to: {results['model_path']}")
        print(f"📊 Final F1 Score: {results['eval_results']['eval_f1']:.4f}")
        print(f"🧪 Test: '{results['test_prediction']['text']}' → {results['test_prediction']['prediction']} ({results['test_prediction']['confidence']:.3f})")
        
    except Exception as e:
        logger.error(f"Quick training demo failed: {str(e)}")
        print(f"\n❌ ERROR: Training failed - {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()