# Training Architecture Diagrams

This document contains detailed architectural diagrams for the training process.

## System Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                            SYSTEM ARCHITECTURE OVERVIEW                     │
└──────────────────────────────────────────────────────────────────────────────┘

┌─────────────────┐    ┌─────────────────┐
│   INPUT LAYER   │    │                 │
├─────────────────┤    │ ┌─────────────┐ │
│ Raw Text Data   │────┼→│ Data        │ │
│                 │    │ │ Ingestion   │ │
│ Configuration   │────┼→│             │ │
│ Files           │    │ └─────────────┘ │
└─────────────────┘    │       │         │
                       │       ▼         │
┌─────────────────┐    │ ┌─────────────┐ │
│ DATA PROCESSING │    │ │ Text        │ │
│     LAYER       │    │ │ Preprocessing│ │
├─────────────────┤    │ │             │ │
│ • Data Ingestion│◄───┘ └─────────────┘ │
│ • Preprocessing │            │         │
│ • Tokenization  │            ▼         │
│ • Dataset       │      ┌─────────────┐ │
│   Creation      │      │ Tokenization│ │
└─────────────────┘      │             │ │
         │                └─────────────┘ │
         ▼                      │         │
┌─────────────────┐            ▼         │
│   MODEL LAYER   │      ┌─────────────┐ │
├─────────────────┤      │ Dataset     │ │
│ Pre-trained     │◄─────│ Creation    │ │
│ Model Loading   │      │             │ │
│                 │      └─────────────┘ │
│ Model Arch      │                      │
│ Setup           │                      │
│                 │                      │
│ Fine-tuning     │                      │
│ Configuration   │                      │
└─────────────────┘                      │
         │                                │
         ▼                                │
┌─────────────────┐                      │
│ TRAINING LAYER  │                      │
├─────────────────┤                      │
│ Training Loop ◄─┼─────────────────────┘
│      │          │
│      ▼          │
│ Loss           │
│ Calculation    │
│      │          │
│      ▼          │
│ Gradient       │
│ Updates        │
│      │          │
│      ▼          │
│ Evaluation ────┼─┐
└─────────────────┘ │
         │           │
         ▼           │
┌─────────────────┐  │
│  OUTPUT LAYER   │  │
├─────────────────┤  │
│ Trained Model   │◄─┘
│                 │
│ Evaluation      │
│ Metrics         │
│                 │
│ Model           │
│ Artifacts       │
└─────────────────┘
```

## Training Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                             TRAINING DATA FLOW                              │
└─────────────────────────────────────────────────────────────────────────────┘

                           ┌─────────┐
                           │  START  │
                           └────┬────┘
                                │
                                ▼
                      ┌─────────────────┐
                      │ Training Type?  │
                      └─────┬───────┬───┘
                            │       │
                  Quick Demo│       │Full Training
                            ▼       ▼
               ┌─────────────────┐ ┌─────────────────┐
               │   Generate      │ │   Load Real     │
               │  Synthetic Data │ │    Dataset      │
               └─────────┬───────┘ └─────────┬───────┘
                         │                   │
                         ▼                   ▼
               ┌─────────────────┐ ┌─────────────────┐
               │  100 Movie      │ │  25K IMDB       │
               │   Reviews       │ │   Reviews       │
               └─────────┬───────┘ └─────────┬───────┘
                         │                   │
                         └─────────┬─────────┘
                                   │
                                   ▼
                         ┌─────────────────┐
                         │ Text            │
                         │ Preprocessing   │
                         └─────────┬───────┘
                                   │
                                   ▼
                         ┌─────────────────┐
                         │ Tokenization    │
                         └─────────┬───────┘
                                   │
                     ┌──────────┼──────────┐
                     │           │           │
                     ▼           ▼           ▼
           ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
           │ Input IDs   │ │ Attention   │ │   Labels    │
           │             │ │   Masks     │ │             │
           └──────┬──────┘ └──────┬──────┘ └──────┬──────┘
                  │               │               │
                  └─────────────┼──────────────┘
                                  │
                                  ▼
                        ┌─────────────────┐
                        │ Training Dataset│
                        └─────────┬───────┘
                                  │
                                  ▼
                        ┌─────────────────┐
                        │ Model Training  │
                        └─────────┬───────┘
                                  │
                                  ▼
                        ┌─────────────────┐
                        │   Evaluation?   │
                        └─────┬───────┬───┘
                              │       │
                  Good Performance   Poor Performance
                              │       │
                              ▼       ▼
                  ┌─────────────────┐ ┌─────────────────┐
                  │   Save Model    │ │ Adjust          │
                  │                 │ │ Parameters      │
                  └─────────┬───────┘ └─────────┬───────┘
                            │                   │
                            │                   │
                            ▼                   │
                  ┌─────────────────┐           │
                  │ Training        │           │
                  │ Complete        │           │
                  └─────────────────┘           │
                                                │
                            ┌───────────────────┘
                            │
                            ▼
                  ┌─────────────────┐
                  │ Model Training  │ (Back to training)
                  └─────────────────┘
```

## Model Transformation Process

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         MODEL TRANSFORMATION PROCESS                        │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────┐         ┌─────────────────┐         ┌─────────────────┐
│  PRE-TRAINED    │         │  FINE-TUNING    │         │  TRAINED MODEL  │
│     MODEL       │         │    PROCESS      │         │                 │
├─────────────────┤         ├─────────────────┤         ├─────────────────┤
│                 │         │                 │         │                 │
│ DistilBERT Base │────────►│ Add             │────────►│ DistilBERT +    │
│                 │         │ Classification  │         │ Classifier      │
│                 │         │ Head            │         │                 │
│                 │         │                 │         │                 │
│ General         │────────►│ Task-Specific   │────────►│ Movie Review    │
│ Language        │         │ Training        │         │ Understanding   │
│ Understanding   │         │                 │         │                 │
│                 │         │                 │         │                 │
│ Vocabulary:     │────────►│ Freeze/Unfreeze│────────►│ Sentiment       │
│ 30K tokens      │         │ Layers          │         │ Classification  │
│                 │         │                 │         │                 │
│                 │         │                 │         │                 │
│ Parameters:     │────────►│ Domain          │────────►│ High Confidence │
│ 66M             │         │ Adaptation      │         │ Predictions     │
│                 │         │                 │         │                 │
└─────────────────┘         └─────────────────┘         └─────────────────┘
                                                                 ▲
                                    ┌────────────────────────────┘
                                    │
                          All components contribute to:
                          • Enhanced accuracy
                          • Domain-specific understanding
                          • Reliable predictions
```

## Training Performance Metrics Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      TRAINING PERFORMANCE METRICS FLOW                      │
└─────────────────────────────────────────────────────────────────────────────┘

                        ┌─────────────────┐
                        │ Training Epoch  │ ◄─────────────┐
                        └─────────┬───────┘               │
                                  │                       │
                                  ▼                       │
                        ┌─────────────────┐               │
                        │ Forward Pass    │               │
                        └─────────┬───────┘               │
                                  │                       │
                                  ▼                       │
                        ┌─────────────────┐               │
                        │ Calculate Loss  │               │
                        └─────────┬───────┘               │
                                  │                       │
                                  ▼                       │
                        ┌─────────────────┐               │
                        │ Backward Pass   │               │
                        └─────────┬───────┘               │
                                  │                       │
                                  ▼                       │
                        ┌─────────────────┐               │
                        │ Update Weights  │               │
                        └─────────┬───────┘               │
                                  │                       │
                                  ▼                       │
                        ┌─────────────────┐               │
                        │ Evaluation      │               │
                        │ Phase           │               │
                        └─────────┬───────┘               │
                                  │                       │
                                  ▼                       │
                        ┌─────────────────┐               │
                        │ Calculate       │               │
                        │ Metrics         │               │
                        └─────────┬───────┘               │
                                  │                       │
                  ┌───────────────┼───────────────┐       │
                  │               │               │       │
                  ▼               ▼               ▼       │
        ┌─────────────┐ ┌─────────────┐ ┌─────────────┐   │
        │  Accuracy   │ │ Precision   │ │   Recall    │   │
        └──────┬──────┘ └──────┬──────┘ └──────┬──────┘   │
               │               │               │          │
               └───────────────┼───────────────┘          │
                               │                          │
                               ▼                          │
                     ┌─────────────────┐                 │
                     │   F1 Score      │                 │
                     └─────────┬───────┘                 │
                               │                          │
                               ▼                          │
                     ┌─────────────────┐                 │
                     │ Best Model?     │                 │
                     └─────┬───────┬───┘                 │
                           │       │                     │
                       Yes │       │ No                  │
                           │       │                     │
                           ▼       ▼                     │
                ┌─────────────────┐ ┌─────────────────┐   │
                │ Save            │ │ Continue        │   │
                │ Checkpoint      │ │ Training        │   │
                └─────────┬───────┘ └─────────┬───────┘   │
                          │                   │           │
                          └─────────┬─────────┘           │
                                    │                     │
                                    ▼                     │
                          ┌─────────────────┐             │
                          │ Early Stopping? │             │
                          └─────┬───────┬───┘             │
                                │       │                 │
                            Yes │       │ No              │
                                │       │                 │
                                ▼       ▼                 │
                    ┌─────────────────┐ ┌─────────────────┐ │
                    │ Training        │ │ Next Epoch      │─┘
                    │ Complete        │ │                 │
                    └─────────────────┘ └─────────────────┘
```

## Component Interaction Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         COMPONENT INTERACTION SEQUENCE                      │
└─────────────────────────────────────────────────────────────────────────────┘

User    train_model.py  Data Pipeline  Model Builder  Model Trainer  Evaluator
 │           │               │              │              │            │
 │ Execute   │               │              │              │            │
 │ training──┼───────────────►│              │              │            │
 │ command   │               │              │              │            │
 │           │               │              │              │            │
 │           │ Load and      │              │              │            │
 │           │ preprocess────┼──────────────►│              │            │
 │           │ data          │              │              │            │
 │           │               │              │              │            │
 │           │               │ Clean and    │              │            │
 │           │               │ tokenize─────┼►(internal)  │            │
 │           │               │ text         │              │            │
 │           │               │              │              │            │
 │           │ Return        │              │              │            │
 │           │◄──────────────┼──processed   │              │            │
 │           │ datasets      │              │              │            │
 │           │               │              │              │            │
 │           │ Build model   │              │              │            │
 │           │ architecture──┼──────────────┼──────────────►│            │
 │           │               │              │              │            │
 │           │               │              │ Load         │            │
 │           │               │              │ pre-trained──┼►(internal)│
 │           │               │              │ weights      │            │
 │           │               │              │              │            │
 │           │ Return        │              │              │            │
 │           │◄──────────────┼──────────────┼──configured  │            │
 │           │ model         │              │ model        │            │
 │           │               │              │              │            │
 │           │ Initialize    │              │              │            │
 │           │ trainer───────┼──────────────┼──────────────┼────────────►│
 │           │               │              │              │            │
 │           │               │              │              │ Setup      │
 │           │               │              │              │ training───┼►(internal)
 │           │               │              │              │ arguments  │
 │           │               │              │              │            │
 │           │               │              │              │            │
 │           │               │     ╔══════════════════════════════════════════╗
 │           │               │     ║            TRAINING LOOP                 ║
 │           │               │     ╠══════════════════════════════════════════╣
 │           │               │     ║              │ Forward pass              ║
 │           │               │     ║              │ ─────────────►(internal) ║
 │           │               │     ║              │                          ║
 │           │               │     ║              │ Calculate loss           ║
 │           │               │     ║              │ ─────────────►(internal) ║
 │           │               │     ║              │                          ║
 │           │               │     ║              │ Backward pass            ║
 │           │               │     ║              │ ─────────────►(internal) ║
 │           │               │     ║              │                          ║
 │           │               │     ║              │ Update parameters        ║
 │           │               │     ║              │ ─────────────►(internal) ║
 │           │               │     ║              │                          ║
 │           │               │     ║              │ Evaluate model           ║
 │           │               │     ║              │ ─────────────────────────┼───►│
 │           │               │     ║              │                          ║    │
 │           │               │     ║              │ Return metrics           ║    │
 │           │               │     ║              │◄─────────────────────────┼────│
 │           │               │     ║              │                          ║
 │           │               │     ║              │ [If best model found]    ║
 │           │               │     ║              │ Save checkpoint          ║
 │           │               │     ║              │ ─────────────►(internal) ║
 │           │               │     ╚══════════════════════════════════════════╝
 │           │               │              │              │            │
 │           │ Return        │              │              │            │
 │           │◄──────────────┼──────────────┼──training────┼────────────│
 │           │ results       │              │ results      │            │
 │           │               │              │              │            │
 │ Display   │               │              │              │            │
 │◄──────────┼──final        │              │              │            │
 │ metrics   │               │              │              │            │
 │           │               │              │              │            │
```

## Quick vs Full Training Comparison

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      QUICK VS FULL TRAINING COMPARISON                      │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────┐                        ┌─────────────────────┐
│ QUICK TRAINING DEMO │                        │   FULL TRAINING     │
├─────────────────────┤                        ├─────────────────────┤
│                     │                        │                     │
│ • Synthetic Data:   │                        │ • Real Data:        │
│   100 samples       │                        │   25K samples       │
│                     │                        │                     │
│ • Training Time:    │                        │ • Training Time:    │
│   ~4 minutes        │                        │   30-60 minutes     │
│                     │                        │                     │
│ • Perfect Metrics:  │                        │ • Realistic Metrics│
│   1.0 accuracy      │                        │   85-90% accuracy   │
│                     │                        │                     │
│ • Use Case:         │                        │ • Use Case:         │
│   Pipeline          │                        │   Production        │
│   validation        │                        │   models            │
│                     │                        │                     │
└─────────┬───────────┘                        └─────────┬───────────┘
          │                                              │
          │              ┌─────────────────┐             │
          │              │    BENEFITS     │             │
          │              │   COMPARISON    │             │
          │              ├─────────────────┤             │
          │              │                 │             │
          └─────────────►│ • Fast          │◄────────────┘
                         │   Iteration     │
                         │                 │
                         │ • Debug         │
                         │   Friendly      │             ┌─────────────┘
                         │                 │             │
                         │ • Production    │◄────────────┘
                         │   Ready         │
                         │                 │
                         │ • High Quality  │
                         │                 │
                         └─────────────────┘
```
    
    B1 --> C2
    B2 --> C2
    B3 --> C4
    B4 --> C4
    
    style A1 fill:#f3e5f5
    style B1 fill:#e3f2fd
    style C1 fill:#fff3e0
    style C4 fill:#c8e6c9
```

## Docker Environment Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       DOCKER ENVIRONMENT ARCHITECTURE                      │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────┐
│  DOCKER BUILD STAGE  │
├─────────────────────┤
│                     │
│ Base Python Image   │
│          │          │
│          ▼          │
│ Install             │
│ Dependencies        │
│          │          │
│          ▼          │
│ Pre-download        │
│ Models              │
│          │          │
│          ▼          │
│ Copy Application    │
│ Code                │
│                     │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  RUNTIME ENVIRONMENT │
├─────────────────────┤
│                     │
│ • Cached Models     │
│                     │
│ • Training Scripts  │
│                     │
│ • Data Pipeline    │
│                     │
│ • Utilities        │
│                     │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ TRAINING EXECUTION  │
├─────────────────────┤
│                     │
│ Load Cached         │
│ Models              │
│          │          │
│          ▼          │
│ Process Data        │
│          │          │
│          ▼          │
│ Execute Training    │
│          │          │
│          ▼          │
│ Save Results        │
│                     │
└─────────────────────┘

Key Benefits:
• Offline-first model loading (no network dependencies)
• Fast startup (models pre-cached)
• Consistent environment across deployments
• Self-contained training pipeline
```

## Error Handling and Recovery Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      ERROR HANDLING AND RECOVERY FLOW                      │
└─────────────────────────────────────────────────────────────────────────────┘

                        ┌─────────────────┐
                        │ Start Training  │
                        └─────────┬───────┘
                                  │
                                  ▼
                        ┌─────────────────┐
                        │ Dependencies     │
                        │ Available?       │
                        └─────┬───────┬───┘
                              │       │
                          No  │       │ Yes
                              │       │
                              ▼       ▼
                    ┌─────────────────┐ ┌─────────────────┐
                    │ Error: Missing   │ │ Load Model       │
                    │ accelerate       │ │                  │
                    └─────────┬───────┘ └─────────┬───────┘
                              │                   │
                              ▼                   ▼
                    ┌─────────────────┐ ┌─────────────────┐
                    │ Rebuild Docker   │ │ Model           │
                    │ Image            │ │ Available?      │
                    └─────────┬───────┘ └─────┬───────┬───┘
                              │                 │       │
                              │             Local Cache  Download Required
                              │                 │       │
                              │                 ▼       ▼
                              │       ┌─────────────────┐ ┌─────────────────┐
                              │       │ Load from Cache │ │ Download from   │
                              │       │                 │ │ Hub             │
                              │       └─────────┬───────┘ └─────────┬───────┘
                              │                 │                   │
                              │                 │                   ▼
                              │                 │         ┌─────────────────┐
                              │                 │         │ Network Issues? │
                              │                 │         └─────┬───────┬───┘
                              │                 │               │       │
                              │                 │           Yes │       │ No
                              │                 │               │       │
                              │                 │               ▼       ▼
                              │                 │     ┌─────────────────┐ ┌─────────────────┐
                              │                 │     │ Use Quick       │ │ Continue with   │
                              │                 │     │ Demo Mode       │ │ Real Data       │
                              │                 │     └─────────┬───────┘ └─────────┬───────┘
                              │                 │               │                   │
                              │                 └───────────────┼───────────────┘
                              │                               │
                              │                               ▼
                              │                     ┌─────────────────┐
                              │                     │ Synthetic Data  │
                              │                     │ Training        │
                              │                     └─────────┬───────┘
                              │                               │
                              │                               ▼
                              │                     ┌─────────────────┐
                              │                     │ Validate        │
                              │                     │ Pipeline        │
                              │                     └─────────┬───────┘
                              │                               │
                              │                               ▼
                              │                     ┌─────────────────┐
                              │                     │ Pipeline        │
                              │                     │ Working?        │
                              │                     └─────┬───────┬───┘
                              │                           │       │
                              │                       Yes │       │ No
                              │                           │       │
                              │                           ▼       ▼
                              │               ┌─────────────────┐ ┌─────────────────┐
                              │               │ Switch to Full  │ │ Debug Issues    │
                              │               │ Training        │ │                 │
                              │               └─────────┬───────┘ └─────────┬───────┘
                              │                         │                   │
                              │                         │                   ▼
                              │                         │         ┌─────────────────┐
                              │                         │         │ Fix             │
                              │                         │         │ Configuration   │
                              │                         │         └─────────┬───────┘
                              │                         │                   │
                              │                         │                   │
                              └─────────────────────────┼─────────────────────────┘
                                                          │
                                                          ▼
                                                ┌─────────────────┐
                                                │ Real Data       │
                                                │ Training        │
                                                └─────────┬───────┘
                                                          │
                                                          ▼
                                                ┌─────────────────┐
                                                │ Production      │
                                                │ Training        │
                                                └─────────┬───────┘
                                                          │
                                                          ▼
                                                ┌─────────────────┐
                                                │ Training        │
                                                │ Complete        │
                                                └─────────────────┘
```

These diagrams provide a comprehensive visual guide to understanding the training process, from data preparation through model deployment, including error handling and performance optimization strategies.