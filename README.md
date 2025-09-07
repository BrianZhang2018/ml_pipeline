# LLM-Based Text Classification Pipeline

This project demonstrates a complete machine learning pipeline for text classification using Large Language Models (LLMs). The pipeline includes data preprocessing, model training, evaluation, hyperparameter tuning, and model interpretability.

## What Are We Building? (Explain Like I'm 5)
We're building a smart computer program that can read text (like movie reviews or news articles) and figure out what category they belong to. Think of it like teaching a computer to sort your mail into different boxes based on what's written on the envelope.

## Project Overview

This repository contains a modular, production-ready implementation of an LLM-based text classification system. The project showcases key machine learning engineering skills including:

- Data pipeline implementation (getting and organizing data)
- Model selection and evaluation (choosing and testing the best approach)
- Hyperparameter tuning (fine-tuning for best performance)
- Model interpretability (understanding why the computer made certain decisions)
- Production-ready code structure (organized like real-world software)
- Experiment tracking (keeping track of what works and what doesn't)
- Model deployment (making the system available for others to use)

## Phases

See our detailed [Project Phase Structure](docs/reference/project_phases.md) document for the complete breakdown of phases and objectives.

### Phase 1: Project Setup and Configuration
1. Create project directory structure ✅ Completed
2. Set up virtual environment with required dependencies ✅ Completed
3. Create configuration management system ✅ Completed
4. Implement logging system for tracking experiment progress ✅ Completed

### Phase 2: Data Pipeline
1. Create data ingestion module to load datasets (IMDB reviews, AG News, etc.) ✅ Completed
2. Implement data preprocessing module for text cleaning and tokenization ✅ Completed
3. Create data validation module to ensure data quality ✅ Completed
4. Implement feature extraction using Hugging Face tokenizers ✅ Completed

### Phase 3: Model Development
1. Create model builder module using Hugging Face Transformers ✅ Completed
2. Implement model trainer with TensorFlow/PyTorch ✅ Completed
3. Add MLflow integration for experiment tracking ✅ Completed
4. Implement hyperparameter tuning with Optuna ✅ Completed

### Phase 4: Model Evaluation and Interpretability
1. Create model evaluation module with multiple metrics ✅ Completed
2. Implement cross-validation for robust evaluation ✅ Completed
3. Add model interpretability using SHAP or LIME ✅ Completed
4. Create visualization tools for model analysis ✅ Completed

### Phase 5: Deployment and Testing
1. Build FastAPI endpoint for model serving ✅ Completed
2. Create Dockerfile for containerization ✅ Completed
3. Write unit tests for all modules ✅ Completed
4. Create integration tests for the complete pipeline ✅ Completed
5. Successfully deploy Docker container ✅ Completed
6. Verify basic API functionality ✅ Completed
7. Identify model loading network issues ⚠️ In Progress

### Phase 6: Documentation and Finalization
1. Create comprehensive README with project overview
2. Document all modules and their usage
3. Create example notebooks for demonstration
4. Run final end-to-end validation

## Technology Stack

- Python 3.8+ (the programming language we're using)
- TensorFlow/PyTorch (tools for building AI systems)
- Hugging Face Transformers (pre-built AI models we can use)
- MLflow (tracking our experiments)
- Optuna (helping us find the best settings)
- SHAP/LIME (helping us understand the AI's decisions)
- FastAPI (making our system available over the internet)
- Docker (packaging our system so it runs the same everywhere)

## Getting Started

### Prerequisites
- Python 3.8 or higher
- Docker (for containerized deployment)

### Installation
```bash
# Clone the repository
git clone <repository-url>

# Navigate to the project directory
cd ml_pipeline

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Running the API with Docker
```bash
# Build the Docker image
docker build -t ml-pipeline-api .

# Run the container
docker run -p 8000:8000 ml-pipeline-api

# The API will be available at http://localhost:8000
```

### Model Training

Our pipeline supports both quick validation and full production training:

### Quick Training Demo (Recommended First)
```bash
# Fast pipeline validation with synthetic data (~4 minutes)
docker run -it --rm -v $(pwd):/workspace ml_pipeline \
    bash -c "cd /workspace && python quick_train_demo.py"
```

### Full Model Training
```bash
# Test run with limited samples
docker run -it --rm -v $(pwd):/workspace ml_pipeline \
    bash -c "cd /workspace && python train_model.py --test_run"

# Full production training
docker run -it --rm -v $(pwd):/workspace ml_pipeline \
    bash -c "cd /workspace && python train_model.py --epochs 3 --batch_size 16"
```

### Training Documentation
- [📚 Training Process Guide](docs/training_process_guide.md) - Complete training workflow with diagrams
- [🏗️ Training Architecture Diagrams](docs/training_architecture_diagrams.md) - Detailed system architecture
- [🔄 Model Comparison Guide](compare_models.py) - Compare trained vs pre-trained models

**Key Benefits:**
- ✅ **Improved Confidence**: Training increases prediction confidence from ~0.53 to ~0.85+
- ✅ **Domain-Specific**: Models learn movie review patterns for better sentiment analysis
- ✅ **Offline-First**: Pre-downloaded models eliminate network dependencies
- ✅ **Fast Validation**: Quick demo completes in ~4 minutes

## API Usage
Once the Docker container is running, you can access the following endpoints:

- `GET /` - Root endpoint with API information
- `GET /api/v1/health` - Health check endpoint
- `GET /api/v1/models/{model_name}` - Get information about a specific model
- `POST /api/v1/classify` - Classify text using a pre-trained model
- `POST /api/v1/classify/batch` - Classify multiple texts in batch

For detailed API documentation, visit `http://localhost:8000/docs` when the container is running.

## Project Structure

See our detailed [File Structure](docs/reference/file_structure.md) document for the complete breakdown of directories and files.

```
ml_pipeline/
├── data/
│   ├── raw/          # Original data files
│   ├── processed/    # Cleaned and prepared data
│   └── external/     # Data from external sources
├── models/
│   ├── trained/      # Finished AI models
│   └── configs/      # Model settings
├── src/
│   ├── data/         # Code for handling data
│   ├── features/     # Code for preparing data for AI
│   ├── models/       # Code for building AI models
│   ├── evaluation/   # Code for evaluating and interpreting models
│   ├── api/          # Code for making our system available online
│   └── utils/        # Helper code
├── tests/            # Code to check our system works correctly
├── notebooks/        # Interactive experiments
├── configs/          # System settings
├── experiments/      # Experiment results
├── docs/             # Documentation
├── README.md         # This file
└── requirements.txt  # List of tools we need
```

## Documentation

See our detailed [File Structure](docs/reference/file_structure.md) document for the complete breakdown of directories and files.

Phase documentation can be found in the `docs/` directory:
- [Documentation README](docs/README.md) - Main documentation index
- [Setup Documentation](docs/setup/) - Project setup and configuration
- [Reference Materials](docs/reference/) - General reference materials
- [Clean Documentation Structure](docs/clean/) - Organized documentation

See specific documentation files:
- [Project Phase Structure](docs/reference/project_phases.md)
- [Progress Overview](docs/reference/progress_overview.md)
- [Glossary](docs/reference/glossary.md)
- [Phase 5 Summary](docs/setup/phase5_summary.md)
- [Phase 6 Implementation Plan](docs/setup/phase6_implementation_plan.md)
- [Project Status Summary](docs/setup/project_status_summary.md)

## Current Status

✅ **All phases implemented**
✅ **Docker container running**
✅ **Basic API endpoints working**
⚠️ **Model loading experiencing network issues**
🚧 **Phase 6 documentation in progress**

For a detailed current status, see our [Project Status Summary](docs/setup/project_status_summary.md).

## Glossary

For simple explanations of technical terms, see our [Glossary](docs/glossary.md)