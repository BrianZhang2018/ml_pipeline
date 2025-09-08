#!/bin/bash
# Virtual Environment Activation Script for ML Pipeline
# Usage: source activate_env.sh

# Activate the virtual environment
source /Users/bz/ai/ml_pipeline/venv/bin/activate

# Verify activation
echo "✓ Virtual environment activated:"
echo "  Python: $(which python)"
echo "  Pip: $(which pip)"

# Show key package versions
echo ""
echo "✓ Key package versions:"
python -c "
import sys
packages = ['datasets', 'scikit-learn', 'torch', 'transformers', 'pytest']
for pkg in packages:
    try:
        exec(f'import {pkg.replace(\"-\", \"_\")}')
        version = eval(f'{pkg.replace(\"-\", \"_\")}.__version__')
        print(f'  {pkg}: {version}')
    except:
        print(f'  {pkg}: not available')
"

echo ""
echo "✓ Environment ready for development!"
echo "  IDE Python interpreter should be set to:"
echo "  /Users/bz/ai/ml_pipeline/venv/bin/python"