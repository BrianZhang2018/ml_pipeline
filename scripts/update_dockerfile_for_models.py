#!/bin/bash
# Script to update Dockerfile with model pre-download steps

# This script shows how we could modify our Dockerfile to pre-download models
# during the build process to avoid network issues at runtime

cat << 'EOF'
# Add these lines to your Dockerfile after the requirements installation:

# Pre-download commonly used models to avoid network issues at runtime
RUN python -c "from transformers import AutoModel, AutoTokenizer; \
    model_name = 'distilbert-base-uncased'; \
    print(f'Downloading {model_name}...'); \
    AutoModel.from_pretrained(model_name); \
    AutoTokenizer.from_pretrained(model_name); \
    print(f'{model_name} downloaded successfully')"

# If you want to include additional models, add them here:
RUN python -c "from transformers import AutoModel, AutoTokenizer; \
    models = ['bert-base-uncased', 'roberta-base']; \
    for model_name in models: \
        print(f'Downloading {model_name}...'); \
        AutoModel.from_pretrained(model_name); \
        AutoTokenizer.from_pretrained(model_name); \
        print(f'{model_name} downloaded successfully')"

EOF