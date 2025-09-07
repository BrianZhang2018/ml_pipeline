# Training Process Guide

This document explains the complete model training process in our LLM-based text classification pipeline, including detailed workflows and system diagrams.

## Overview

Our training pipeline transforms pre-trained models into domain-specific classifiers through fine-tuning on task-specific data. This addresses the "Training Necessity for Confidence" requirement where pre-trained models without fine-tuning produce low prediction confidence.

## Training Scripts

### 1. Full Training Script (`train_model.py`)
- **Purpose**: End-to-end training on real datasets (IMDB, AG News)
- **Dataset Size**: 25K training samples, 25K test samples
- **Duration**: 30-60 minutes depending on epochs
- **Use Case**: Production model training

### 2. Quick Training Demo (`quick_train_demo.py`)
- **Purpose**: Fast validation of training pipeline
- **Dataset Size**: 100 synthetic samples
- **Duration**: ~4 minutes with perfect evaluation metrics
- **Use Case**: Pipeline validation and debugging

## Complete Training Workflow

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Start         │    │ Load             │    │ Dataset         │
│   Training      │───▶│ Configuration    │───▶│ Selection       │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                           │
                        ┌──────────────────┐              ▼
                        │ Generate Sample  │         ┌─────────────┐
                        │ Reviews         │◀────────│ Dataset     │
                        └──────────────────┘         │ Type?       │
                                  │                  └─────────────┘
                                  │                         │
                                  ▼                         ▼
                        ┌──────────────────┐    ┌──────────────────┐
                        │ Data             │    │ Download         │
                        │ Preprocessing    │◀───│ IMDB/AG News     │
                        └──────────────────┘    └──────────────────┘
                                  │
                                  ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Text Cleaning   │───▶│ Tokenization     │───▶│ Model Loading   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                           │
                        ┌──────────────────┐              ▼
                        │ Load from Cache  │         ┌─────────────┐
                        └──────────────────┘◀────────│ Model       │
                                  │                  │ Source?     │
                                  │                  └─────────────┘
                                  ▼                         │
                        ┌──────────────────┐              ▼
                        │ Fine-tuning      │    ┌──────────────────┐
                        │ Setup           │◀───│ Download from    │
                        └──────────────────┘    │ Hub             │
                                  │              └──────────────────┘
                                  ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Training Loop   │───▶│ Evaluation       │───▶│ Model Saving    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                        │                        │
         │                        ▼                        ▼
         │              ┌──────────────────┐    ┌─────────────────┐
         └──────────────│ Performance      │───▶│ Training        │
                        │ Testing          │    │ Complete        │
                        └──────────────────┘    └─────────────────┘
```

## Detailed Step-by-Step Process

## Detailed Step-by-Step Process

### Step 1: Dataset Preparation

```
Raw Dataset
     │
     ▼
┌─────────────────────────────────────┐
│         Text Preprocessing          │
│  ┌─────────────┐ ┌─────────────────┐ │
│  │ Remove HTML │ │ Remove URLs     │ │
│  │ Tags        │ │                 │ │
│  └─────────────┘ └─────────────────┘ │
│  ┌─────────────┐ ┌─────────────────┐ │
│  │ Normalize   │ │ Clean           │ │
│  │ Text        │ │ Punctuation     │ │
│  └─────────────┘ └─────────────────┘ │
└─────────────────────────────────────┘
     │
     ▼
┌─────────────────┐    ┌─────────────────┐
│ Label Encoding  │───▶│ Train/Eval Split │
└─────────────────┘    └─────────────────┘
     │                         │
     ▼                         ▼
┌─────────────────────────────────────────┐
│            Dataset Objects              │
└─────────────────────────────────────────┘
```

**Data Pipeline Components:**
- **Data Ingestion**: [`data_ingestion.py`](../src/data/data_ingestion.py) - Loads IMDB, AG News, or custom datasets
- **Data Preprocessing**: [`data_preprocessing.py`](../src/data/data_preprocessing.py) - Cleans and normalizes text
- **Feature Extraction**: [`feature_extraction.py`](../src/data/feature_extraction.py) - Tokenizes text for model input

### Step 2: Model Architecture Setup

```
┌─────────────────────────────────────────────────────────────────┐
│                      Pre-trained Model                         │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │ DistilBERT Base │  │ General Language│  │ Vocabulary:     │ │
│  │                 │  │ Understanding   │  │ 30K tokens      │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
│                              │                                  │
│                              ▼                                  │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                Parameters: 66M                              │ │
│  └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Model Loading Strategy                        │
│                              │                                  │
│                              ▼                                  │
│                     ┌─────────────────┐                        │
│                     │ Cache Available?│                        │
│                     └─────────────────┘                        │
│                        YES │   │ NO                            │
│            ┌───────────────┘   └──────────────┐                │
│            ▼                                   ▼                │
│  ┌─────────────────┐                ┌─────────────────┐        │
│  │ Load from       │                │ Download from   │        │
│  │ Local Cache     │                │ Hugging Face    │        │
│  └─────────────────┘                └─────────────────┘        │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Model Adaptation                            │
│                              │                                  │
│                              ▼                                  │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │ Add             │  │ Configure for   │  │ DistilBERT +    │ │
│  │ Classification  │──│ Fine-tuning     │──│ Classifier      │ │
│  │ Head            │  │                 │  │                 │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
│                              │                     │            │
│                              ▼                     ▼            │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │ Pre-classifier  │  │ Classifier      │  │ 2 Labels:       │ │
│  │ Layer           │──│ Layer           │──│ Positive/       │ │
│  │                 │  │                 │  │ Negative        │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

**Key Components:**
- **Model Builder**: [`model_builder.py`](../src/models/model_builder.py) - Loads pre-trained models with offline-first strategy
- **Model Loader**: [`model_loader.py`](../src/api/model_loader.py) - Handles model caching and loading

### Step 3: Training Configuration

```
┌──────────────────────────────────────────────────────────────────────┐
│                        Training Arguments                            │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│ ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐    │
│ │ Hyperparameters  │  │ Optimization     │  │ Evaluation       │    │
│ │                  │  │ Settings         │  │ Strategy         │    │
│ │ • Learning Rate: │  │ • Weight Decay:  │  │ • Strategy:      │    │
│ │   2e-5           │  │   0.01           │  │   Epoch          │    │
│ │ • Batch Size: 16 │  │ • Warmup Steps:  │  │ • Metric:        │    │
│ │ • Epochs: 1-3    │  │   500            │  │   F1 Score       │    │
│ │                  │  │ • Optimizer:     │  │ • Early Stopping:│    │
│ │                  │  │   AdamW          │  │   3 epochs       │    │
│ └──────────────────┘  └──────────────────┘  └──────────────────┘    │
│                                                                      │
│ ┌──────────────────────────────────────────────────────────────────┐ │
│ │                      Checkpointing                              │ │
│ │                                                                  │ │
│ │ • Save Strategy: Epoch                                          │ │
│ │ • Best Model Loading: Yes                                       │ │
│ │ • Checkpoint Limit: 2                                           │ │
│ └──────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────┘
```

### Step 4: Training Loop Execution

```
Trainer          Model           Dataset         Evaluator       Saver
   │               │                 │                │             │
   │ Initialize    │                 │                │             │
   │ with pre-     │                 │                │             │
   │ trained       │                 │                │             │
   │ weights       │                 │                │             │
   ├──────────────▶│                 │                │             │
   │               │                 │                │             │
   │               │     FOR EACH EPOCH                │             │
   │               │     ┌─────────────────────┐       │             │
   │ Get training  │     │                     │       │             │
   │ batch         │     │                     │       │             │
   ├──────────────────────▶                   │       │             │
   │               │     │                     │       │             │
   │ Forward pass  │     │                     │       │             │
   ├──────────────▶│     │                     │       │             │
   │               │     │                     │       │             │
   │ Return        │     │                     │       │             │
   │ predictions   │     │                     │       │             │
   │ & loss        │     │                     │       │             │
   │◀──────────────┤     │                     │       │             │
   │               │     │                     │       │             │
   │ Backward pass │     │                     │       │             │
   │ & update      │     │                     │       │             │
   │ weights       │     │                     │       │             │
   ├──────────────▶│     │                     │       │             │
   │               │     │                     │       │             │
   │               │     │ Log training        │       │             │
   │               │     │ metrics             │       │             │
   │               │     │                     │       │             │
   │ Evaluate on   │     │                     │       │             │
   │ validation    │     │                     │       │             │
   │ set           │     │                     │       │             │
   ├─────────────────────────────────────────────────▶│             │
   │               │     │                     │       │             │
   │ Return        │     │                     │       │             │
   │ evaluation    │     │                     │       │             │
   │ metrics       │     │                     │       │             │
   │◀─────────────────────────────────────────────────┤             │
   │               │     │                     │       │             │
   │               │     │ IF best model so far│       │             │
   │ Save          │     │                     │       │             │
   │ checkpoint    │     │                     │       │             │
   ├─────────────────────────────────────────────────────────────────▶
   │               │     │                     │       │             │
   │               │     │ IF early stopping  │       │             │
   │               │     │ triggered: BREAK    │       │             │
   │               │     └─────────────────────┘       │             │
   │               │                 │                │             │
   │ Save final    │                 │                │             │
   │ model         │                 │                │             │
   ├─────────────────────────────────────────────────────────────────▶
   │               │                 │                │             │
   │ Load best     │                 │                │             │
   │ checkpoint    │                 │                │             │
   │◀─────────────────────────────────────────────────────────────────┤
```

### Step 5: Model Evaluation and Comparison

```
┌─────────────────┐           ┌─────────────────┐
│  Trained Model  │           │  Pre-trained    │
│                 │           │  Baseline       │
└─────────┬───────┘           └─────────┬───────┘
          │                             │
          └─────────────┐   ┐───────────┘
                        │   │
                        ▼   ▼
               ┌─────────────────────┐
               │  Evaluation Metrics │
               └─────────────────────┘
                        │
          ┌─────────────┼─────────────┐
          │             │             │
          ▼             ▼             ▼
┌─────────────┐ ┌─────────────┐ ┌─────────────┐
│  Accuracy   │ │ Precision   │ │   Recall    │
└─────────────┘ └─────────────┘ └─────────────┘
          │             │             │
          └─────────────┼─────────────┘
                        │
          ┌─────────────┼─────────────┐
          │             │             │
          ▼             ▼             ▼
┌─────────────┐ ┌─────────────┐ ┌─────────────┐
│  F1 Score   │ │ Confidence  │ │ Performance │
│             │ │   Scores    │ │ Comparison  │
└─────────────┘ └─────────────┘ └─────────────┘
                                      │
                                      ▼
                             ┌─────────────────┐
                             │ Improvement     │
                             │ Detected?       │
                             └─────────────────┘
                                  YES │   │ NO
                      ┌──────────────┘   └──────────────┐
                      ▼                                 ▼
              ┌─────────────────┐                ┌─────────────────┐
              │ Training        │                │ Review          │
              │ Successful      │                │ Hyperparameters│
              └─────────────────┘                └─────────────────┘
```

## Training Performance Expectations

### Quick Training Demo Results
- **Dataset**: 100 synthetic movie reviews
- **Training Time**: ~4 minutes
- **Expected Metrics**: Perfect scores (accuracy: 1.0, F1: 1.0)
- **Confidence**: High (>0.9) due to simple synthetic data

### Full Training Results
- **Dataset**: 25K IMDB movie reviews
- **Training Time**: 30-60 minutes
- **Expected Metrics**: 
  - Accuracy: 85-90%
  - F1 Score: 0.85-0.90
  - Confidence improvement: 0.53 → 0.85+

## Architecture Dependencies

```
                     Training Infrastructure
    ┌────────────────────────────────────────────────────────────┐
    │  ┌─────────────────┐  ┌────────────────────┐  ┌─────────────────┐  │
    │  │ train_model.py  │  │ quick_train_demo.py │  │ compare_models.py│  │
    │  └─────────────────┘  └────────────────────┘  └─────────────────┘  │
    └────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
                          Data Pipeline
    ┌────────────────────────────────────────────────────────────┐
    │  ┌─────────────────┐  ┌────────────────────┐  ┌─────────────────┐  │
    │  │ data_ingestion  │  │ data_preprocessing │  │ feature_extract │  │
    │  │     .py         │  │       .py          │  │     ion.py      │  │
    │  └─────────────────┘  └────────────────────┘  └─────────────────┘  │
    └────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
                       Model Components
    ┌────────────────────────────────────────────────────────────┐
    │  ┌─────────────────┐  ┌────────────────────┐  ┌─────────────────┐  │
    │  │ model_builder   │  │ model_trainer.py   │  │ model_loader.py │  │
    │  │     .py         │  │                    │  │                 │  │
    │  └─────────────────┘  └────────────────────┘  └─────────────────┘  │
    └────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
                         Utilities
    ┌────────────────────────────────────────────────────────────┐
    │                      ┌─────────────────┐  ┌─────────────────┐                      │
    │                      │ logger.py       │  │ config.py       │                      │
    │                      └─────────────────┘  └─────────────────┘                      │
    └────────────────────────────────────────────────────────────┘

Dependency Flow:
train_model.py ─────────────────▶ All Data Pipeline + Model Components + Utilities
quick_train_demo.py ─────────────▶ Model Components + Utilities
compare_models.py ───────────────▶ Model Components + Utilities
```

## Docker Environment Requirements

### Build-Time Dependencies
Based on the "Docker Build-Time Dependency Requirement for Trainer" memory:

```dockerfile
# Core ML libraries
RUN pip install torch>=1.12.0
RUN pip install transformers>=4.20.0
RUN pip install datasets>=2.0.0

# CRITICAL: Required for Hugging Face Trainer
RUN pip install accelerate>=0.26.0

# Additional training dependencies
RUN pip install scikit-learn pandas numpy
```

### Model Pre-downloading Strategy
Following the "Model Loading Strategy" memory:

```dockerfile
# Pre-download models during build to avoid runtime network issues
RUN python -c "from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification; \
    AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased', cache_dir='/app/.cache/huggingface'); \
    AutoTokenizer.from_pretrained('distilbert-base-uncased', cache_dir='/app/.cache/huggingface')"
```

## Training Command Examples

### Quick Validation (Recommended First)
```bash
# Fast pipeline validation with synthetic data
docker run -it --rm -v $(pwd):/workspace ml_pipeline \
    bash -c "cd /workspace && python quick_train_demo.py"
```

### Full Training
```bash
# Test run with limited samples
docker run -it --rm -v $(pwd):/workspace ml_pipeline \
    bash -c "cd /workspace && python train_model.py --test_run"

# Full production training
docker run -it --rm -v $(pwd):/workspace ml_pipeline \
    bash -c "cd /workspace && python train_model.py --epochs 3 --batch_size 16"
```

### Model Comparison
```bash
# Compare trained vs pre-trained models
docker run -it --rm -v $(pwd):/workspace ml_pipeline \
    bash -c "cd /workspace && python compare_models.py --trained_model ./trained_models/distilbert-base-uncased_imdb_final --examples"
```

## Key Benefits of Our Training Pipeline

### 1. Confidence Improvement
- **Before Training**: Low confidence (~0.53) from pre-trained models
- **After Training**: High confidence (~0.85+) from domain-specific fine-tuning

### 2. Network Resilience
- **Offline-First**: Models pre-downloaded during build
- **Fast Iteration**: Synthetic data option for quick validation
- **Robust Deployment**: No runtime network dependencies

### 3. Production Ready
- **Comprehensive Metrics**: Accuracy, Precision, Recall, F1 Score
- **Model Persistence**: Automatic saving and loading
- **Comparison Tools**: Built-in performance benchmarking

## Troubleshooting Common Issues

### Missing Dependencies
```bash
# Error: accelerate library missing
# Solution: Rebuild Docker image with updated requirements.txt
```

### Network Timeouts
```bash
# Error: Dataset download hanging
# Solution: Use quick_train_demo.py for validation
```

### Low Performance
```bash
# Issue: Poor evaluation metrics
# Solution: Adjust hyperparameters, increase epochs, check data quality
```

## Next Steps

1. **Start with Quick Demo**: Validate pipeline with synthetic data
2. **Full Training**: Run complete training on real datasets
3. **Performance Analysis**: Use comparison tools to measure improvements
4. **Production Deployment**: Deploy trained models with confidence

This training process transforms the system from using "pre-trained models without fine-tuning" (low confidence) to "domain-specific trained models" (high confidence), directly addressing the core requirement for meaningful and accurate predictions.