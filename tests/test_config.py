"""
Test script for configuration management system
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.config import ConfigManager, get_config, get_config_value


def test_config_manager():
    """Test the configuration manager"""
    print("Testing Configuration Manager...")
    print("=" * 40)
    
    # Test 1: Create config manager for dev environment
    print("Test 1: Creating config manager for dev environment")
    try:
        config = ConfigManager('dev')
        print(f"✅ Environment: {config.get_environment()}")
        print(f"✅ Model name: {config.get('MODEL_NAME')}")
        print(f"✅ Batch size: {config.get('BATCH_SIZE')}")
        print("✅ Dev environment config loaded successfully\n")
    except Exception as e:
        print(f"❌ Error loading dev config: {e}\n")
        return False
    
    # Test 2: Create config manager for test environment
    print("Test 2: Creating config manager for test environment")
    try:
        config = ConfigManager('test')
        print(f"✅ Environment: {config.get_environment()}")
        print(f"✅ Model name: {config.get('MODEL_NAME')}")
        print(f"✅ Epochs: {config.get('EPOCHS')}")
        print("✅ Test environment config loaded successfully\n")
    except Exception as e:
        print(f"❌ Error loading test config: {e}\n")
        return False
    
    # Test 3: Test global config manager
    print("Test 3: Testing global config manager")
    try:
        # Get config for prod environment
        config = get_config('prod')
        print(f"✅ Global config environment: {config.get_environment()}")
        print(f"✅ MLflow tracking URI: {config.get('MLFLOW_TRACKING_URI')}")
        print("✅ Global config manager works\n")
    except Exception as e:
        print(f"❌ Error with global config manager: {e}\n")
        return False
    
    # Test 4: Test config value getter
    print("Test 4: Testing config value getter")
    try:
        model_name = get_config_value('MODEL_NAME', 'default-model')
        print(f"✅ Model name from getter: {model_name}")
        
        # Test non-existent key with default
        non_existent = get_config_value('NON_EXISTENT_KEY', 'default-value')
        print(f"✅ Non-existent key with default: {non_existent}")
        print("✅ Config value getter works\n")
    except Exception as e:
        print(f"❌ Error with config value getter: {e}\n")
        return False
    
    print("🎉 All configuration manager tests passed!")
    return True


if __name__ == "__main__":
    success = test_config_manager()
    sys.exit(0 if success else 1)