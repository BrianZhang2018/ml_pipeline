"""
Test script for feature extraction module
"""

import sys
import os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.feature_extraction import (
    FeatureExtractor,
    create_feature_extractor,
    extract_features_from_texts
)


def test_feature_extraction():
    """Test the feature extraction functions"""
    print("Testing Feature Extraction Module...")
    print("=" * 40)
    
    # Test texts
    test_texts = [
        "This is a sample text for testing.",
        "Another example text to tokenize.",
        "Short text"
    ]
    
    # Test 1: Create feature extractor
    print("Test 1: Creating feature extractor")
    try:
        extractor = FeatureExtractor("distilbert-base-uncased")
        print(f"✅ Feature extractor created with model: {extractor.get_model_name()}")
        print(f"✅ Vocabulary size: {extractor.get_vocabulary_size()}")
        print("✅ Feature extractor creation works\n")
    except Exception as e:
        print(f"❌ Error creating feature extractor: {e}\n")
        return False
    
    # Test 2: Tokenize single text
    print("Test 2: Tokenizing single text")
    try:
        result = extractor.tokenize_text(test_texts[0])
        print(f"✅ Tokenized text shape: {result['input_ids'].shape}")
        print(f"✅ Input IDs type: {type(result['input_ids'])}")
        print(f"✅ Attention mask type: {type(result['attention_mask'])}")
        print("✅ Single text tokenization works\n")
    except Exception as e:
        print(f"❌ Error tokenizing single text: {e}\n")
        return False
    
    # Test 3: Tokenize multiple texts
    print("Test 3: Tokenizing multiple texts")
    try:
        result = extractor.tokenize_texts(test_texts)
        print(f"✅ Tokenized texts shape: {result['input_ids'].shape}")
        print(f"✅ Batch size: {result['input_ids'].shape[0]}")
        print(f"✅ Sequence length: {result['input_ids'].shape[1]}")
        print("✅ Multiple text tokenization works\n")
    except Exception as e:
        print(f"❌ Error tokenizing multiple texts: {e}\n")
        return False
    
    # Test 4: Extract features (alias)
    print("Test 4: Extracting features (alias)")
    try:
        result = extractor.extract_features(test_texts)
        print(f"✅ Extracted features shape: {result['input_ids'].shape}")
        print("✅ Feature extraction works\n")
    except Exception as e:
        print(f"❌ Error extracting features: {e}\n")
        return False
    
    # Test 5: Convenience functions
    print("Test 5: Testing convenience functions")
    try:
        # Test create_feature_extractor
        extractor2 = create_feature_extractor("distilbert-base-uncased")
        print("✅ create_feature_extractor works")
        
        # Test extract_features_from_texts
        result = extract_features_from_texts(test_texts)
        print(f"✅ extract_features_from_texts shape: {result['input_ids'].shape}")
        print("✅ Convenience functions work\n")
    except Exception as e:
        print(f"❌ Error with convenience functions: {e}\n")
        return False
    
    print("🎉 All feature extraction tests passed!")
    return True


if __name__ == "__main__":
    success = test_feature_extraction()
    sys.exit(0 if success else 1)