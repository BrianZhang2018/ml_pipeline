# Training Quick Start Guide

A fast-track guide to understanding and running model training in our pipeline.

## 🚀 Quick Overview

Our training pipeline transforms pre-trained models into domain-specific classifiers, improving prediction confidence from ~0.53 to ~0.85+ through fine-tuning on movie review data.

## 📋 Prerequisites

1. Docker installed and running
2. ML Pipeline Docker image built:
   ```bash
   cd /Users/bz/ai/ml_pipeline
   docker build -t ml_pipeline .
   ```

## ⚡ Quick Start (4 Minutes)

### Step 1: Validate Pipeline
```bash
# Run quick training demo with synthetic data
docker run -it --rm -v $(pwd):/workspace ml_pipeline \
    bash -c "cd /workspace && python quick_train_demo.py"
```

**Expected Output:**
```
🚀 STARTING QUICK TRAINING DEMO
📊 STEP 1: Creating synthetic movie review data...
🔤 STEP 2: Tokenizing datasets...
🧠 STEP 3: Building model...
🎯 STEP 4: Initializing trainer...
🚀 STEP 5: Starting training...
📈 STEP 6: Evaluating model...
💾 STEP 7: Saving trained model...
🧪 STEP 8: Testing prediction...

🎉 QUICK TRAINING DEMO COMPLETED!
Final F1 Score: 1.0000
Test: 'This movie was fantastic and amazing!' → positive (0.999)
```

### Step 2: Compare Performance
```bash
# Compare trained vs pre-trained models
docker run -it --rm -v $(pwd):/workspace ml_pipeline \
    bash -c "cd /workspace && python compare_models.py --trained_model ./quick_trained_model --examples"
```

## 🎯 Training Options

### Option A: Quick Demo (Recommended for validation)
- **Data**: 100 synthetic movie reviews
- **Time**: ~4 minutes
- **Purpose**: Validate training pipeline
- **Metrics**: Perfect scores (accuracy: 1.0, F1: 1.0)

### Option B: Test Run (Limited real data)
```bash
docker run -it --rm -v $(pwd):/workspace ml_pipeline \
    bash -c "cd /workspace && python train_model.py --test_run"
```
- **Data**: 1000 real IMDB reviews
- **Time**: ~15 minutes
- **Purpose**: Realistic testing

### Option C: Full Production Training
```bash
docker run -it --rm -v $(pwd):/workspace ml_pipeline \
    bash -c "cd /workspace && python train_model.py --epochs 3 --batch_size 16"
```
- **Data**: 25K IMDB reviews
- **Time**: 30-60 minutes
- **Purpose**: Production model

## 📊 Understanding Results

### Before Training (Pre-trained Model)
```
Text: "This movie was amazing!"
Prediction: positive (confidence: 0.536)  # Low confidence
```

### After Training (Fine-tuned Model)
```
Text: "This movie was amazing!"
Prediction: positive (confidence: 0.987)  # High confidence
```

## 🔧 Training Process Flow

```
Raw Text → Preprocessing → Tokenization → Model Loading → Fine-tuning → Evaluation → Saving
```

1. **Data Preparation**: Clean and tokenize movie reviews
2. **Model Setup**: Load pre-trained DistilBERT with classification head
3. **Training**: Fine-tune on sentiment classification task
4. **Evaluation**: Measure accuracy, F1, precision, recall
5. **Saving**: Persist trained model for deployment

## 📁 Output Files

After training, you'll find:
```
./quick_trained_model/          # Quick demo output
├── config.json                 # Model configuration
├── pytorch_model.bin           # Trained weights
├── tokenizer_config.json       # Tokenizer settings
├── tokenizer.json              # Tokenizer files
└── vocab.txt                   # Vocabulary

./trained_models/               # Full training output
└── distilbert-base-uncased_imdb_final/
    ├── config.json
    ├── pytorch_model.bin
    └── ...
```

## 🚨 Troubleshooting

### Error: "accelerate library missing"
**Solution**: Rebuild Docker image (dependency now included in requirements.txt)

### Error: "Network timeout during dataset download"
**Solution**: Use quick demo mode with synthetic data

### Error: "CUDA out of memory"
**Solution**: Reduce batch size or use CPU-only training

## 📚 Detailed Documentation

- [📖 Complete Training Guide](training_process_guide.md)
- [🏗️ Architecture Diagrams](training_architecture_diagrams.md)
- [⚙️ Configuration Options](../src/models/model_trainer.py)

## 🎯 Next Steps

1. **Start Here**: Run quick demo to validate pipeline
2. **Test Real Data**: Use `--test_run` for realistic testing  
3. **Production Training**: Run full training for deployment
4. **Deploy Model**: Use trained model in API endpoints
5. **Compare Performance**: Measure improvement vs baseline

## 💡 Pro Tips

- Always start with quick demo for fast validation
- Use synthetic data to avoid network bottlenecks
- Monitor Docker build logs for dependency issues
- Save training outputs for later comparison
- Check evaluation metrics to ensure training success

**Ready to train? Start with the quick demo above! 🚀**