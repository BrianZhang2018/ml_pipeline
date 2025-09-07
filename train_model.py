#!/usr/bin/env python3
"""
End-to-End Model Training Script

This script demonstrates how to train a model using our complete data pipeline.
It trains on domain-specific data to achieve meaningful and accurate predictions.

What Does This Do? (Explain Like I'm 5)
=======================================
This is like a complete recipe for teaching our AI brain to be really smart
at understanding text. It gets the training data, cleans it up, teaches the
AI with examples, and then tests how well it learned.
"""

import sys
import os
import argparse
from typing import Dict, Any
import torch
from datasets import Dataset
import pandas as pd

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from data.data_ingestion import load_imdb_dataset, get_dataset_info
from data.data_preprocessing import preprocess_texts
from data.feature_extraction import FeatureExtractor
from models.model_builder import build_model
from models.model_trainer import ModelTrainer
from utils.logger import get_logger
from utils.config import get_config

# Get logger instance
logger = get_logger("train_model")


def prepare_dataset(dataset_name: str = "imdb", max_samples: int = None) -> Dict[str, Dataset]:
    """
    Prepare training and evaluation datasets.
    
    Args:
        dataset_name (str): Name of the dataset to use
        max_samples (int, optional): Maximum number of samples to use (for testing)
        
    Returns:
        Dict[str, Dataset]: Dictionary containing train and eval datasets
    """
    logger.info(f"Preparing {dataset_name} dataset")
    
    # Load dataset
    if dataset_name == "imdb":
        train_df = load_imdb_dataset("train")
        test_df = load_imdb_dataset("test")
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    # Limit samples for testing if specified
    if max_samples:
        train_df = train_df.head(max_samples)
        test_df = test_df.head(max_samples // 4)  # Smaller eval set
        logger.info(f"Limited to {len(train_df)} train and {len(test_df)} eval samples")
    
    # Preprocess texts
    logger.info("Preprocessing training texts...")
    train_df['text'] = preprocess_texts(train_df['text'].tolist())
    
    logger.info("Preprocessing evaluation texts...")
    test_df['text'] = preprocess_texts(test_df['text'].tolist())
    
    # Convert to Hugging Face datasets
    train_dataset = Dataset.from_pandas(train_df)
    eval_dataset = Dataset.from_pandas(test_df)
    
    logger.info(f"Prepared datasets - Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")
    
    return {
        "train": train_dataset,
        "eval": eval_dataset
    }


def tokenize_datasets(datasets: Dict[str, Dataset], model_name: str, max_length: int = 512) -> Dict[str, Dataset]:
    """
    Tokenize the datasets for training.
    
    Args:
        datasets (Dict[str, Dataset]): Raw datasets
        model_name (str): Name of the model/tokenizer to use
        max_length (int): Maximum sequence length
        
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


def train_model(
    model_name: str = "distilbert-base-uncased",
    dataset_name: str = "imdb",
    output_dir: str = "./trained_models",
    num_epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    max_samples: int = None
) -> Dict[str, Any]:
    """
    Train a model end-to-end.
    
    Args:
        model_name (str): Name of the pre-trained model to use
        dataset_name (str): Name of the dataset to train on
        output_dir (str): Directory to save the trained model
        num_epochs (int): Number of training epochs
        batch_size (int): Training batch size
        learning_rate (float): Learning rate for training
        max_samples (int, optional): Maximum number of samples (for testing)
        
    Returns:
        Dict[str, Any]: Training results and metrics
    """
    logger.info("="*50)
    logger.info("STARTING END-TO-END MODEL TRAINING")
    logger.info("="*50)
    logger.info(f"Model: {model_name}")
    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"Epochs: {num_epochs}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Learning rate: {learning_rate}")
    
    # Step 1: Prepare datasets
    logger.info("\n📊 STEP 1: Preparing datasets...")
    datasets = prepare_dataset(dataset_name, max_samples)
    
    # Step 2: Tokenize datasets
    logger.info("\n🔤 STEP 2: Tokenizing datasets...")
    tokenized_datasets = tokenize_datasets(datasets, model_name)
    
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
        learning_rate=learning_rate,
        warmup_steps=500,
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
    final_output_dir = os.path.join(output_dir, f"{model_name}_{dataset_name}_final")
    trainer.save_model(final_output_dir)
    
    # Log final results
    logger.info("\n🎉 TRAINING COMPLETED SUCCESSFULLY!")
    logger.info("="*50)
    logger.info("FINAL RESULTS:")
    logger.info(f"Training Loss: {training_results['train_result'].training_loss:.4f}")
    logger.info(f"Evaluation Accuracy: {eval_results['eval_accuracy']:.4f}")
    logger.info(f"Evaluation F1: {eval_results['eval_f1']:.4f}")
    logger.info(f"Evaluation Precision: {eval_results['eval_precision']:.4f}")
    logger.info(f"Evaluation Recall: {eval_results['eval_recall']:.4f}")
    logger.info(f"Model saved to: {final_output_dir}")
    logger.info("="*50)
    
    return {
        "training_results": training_results,
        "eval_results": eval_results,
        "model_path": final_output_dir,
        "model_name": model_name,
        "dataset_name": dataset_name
    }


def main():
    """Main function to run training with command line arguments."""
    parser = argparse.ArgumentParser(description="Train a text classification model")
    parser.add_argument("--model", default="distilbert-base-uncased", help="Model name")
    parser.add_argument("--dataset", default="imdb", help="Dataset name")
    parser.add_argument("--output_dir", default="./trained_models", help="Output directory")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--max_samples", type=int, help="Maximum samples for testing")
    parser.add_argument("--test_run", action="store_true", help="Run with small dataset for testing")
    
    args = parser.parse_args()
    
    # Set max_samples for test run
    if args.test_run:
        args.max_samples = 1000
        args.epochs = 1
        logger.info("Running in test mode with limited samples")
    
    try:
        # Run training
        results = train_model(
            model_name=args.model,
            dataset_name=args.dataset,
            output_dir=args.output_dir,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            max_samples=args.max_samples
        )
        
        print("\n🎊 SUCCESS! Model training completed successfully!")
        print(f"📍 Model saved to: {results['model_path']}")
        print(f"📊 Final F1 Score: {results['eval_results']['eval_f1']:.4f}")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        print(f"\n❌ ERROR: Training failed - {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()