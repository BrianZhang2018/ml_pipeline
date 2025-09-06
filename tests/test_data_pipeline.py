"""
Integration test for the complete data pipeline
"""

import sys
import os
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.data_ingestion import load_imdb_dataset
from data.data_preprocessing import preprocess_texts
from data.data_validation import generate_validation_report
from data.feature_extraction import FeatureExtractor


def test_complete_data_pipeline():
    """Test the complete data pipeline"""
    print("Testing Complete Data Pipeline...")
    print("=" * 40)
    
    # Test 1: Data Ingestion
    print("Step 1: Data Ingestion")
    try:
        # Load a small sample of the IMDB dataset
        df = load_imdb_dataset("train[:5]")
        print(f"✅ Loaded dataset with {len(df)} samples")
        print(f"✅ Columns: {list(df.columns)}")
        print("✅ Data ingestion works\n")
    except Exception as e:
        print(f"❌ Error in data ingestion: {e}\n")
        return False
    
    # Test 2: Data Preprocessing
    print("Step 2: Data Preprocessing")
    try:
        # Preprocess the text data
        original_texts = df['text'].tolist()
        preprocessed_texts = preprocess_texts(original_texts)
        print(f"✅ Preprocessed {len(preprocessed_texts)} texts")
        print(f"✅ Sample processed text: {preprocessed_texts[0][:50]}...")
        print("✅ Data preprocessing works\n")
    except Exception as e:
        print(f"❌ Error in data preprocessing: {e}\n")
        return False
    
    # Test 3: Data Validation
    print("Step 3: Data Validation")
    try:
        # Update the dataframe with preprocessed texts
        df['processed_text'] = preprocessed_texts
        
        # Generate validation report
        report = generate_validation_report(df, 'processed_text', 'label')
        print(f"✅ Validation status: {report['overall_status']}")
        print(f"✅ Found {len(report['data_quality']['issues'])} data quality issues")
        print(f"✅ Found {len(report['text_validation']['issues'])} text validation issues")
        print(f"✅ Found {len(report['label_validation']['issues'])} label validation issues")
        print("✅ Data validation works\n")
    except Exception as e:
        print(f"❌ Error in data validation: {e}\n")
        return False
    
    # Test 4: Feature Extraction
    print("Step 4: Feature Extraction")
    try:
        # Create feature extractor
        extractor = FeatureExtractor("distilbert-base-uncased")
        print(f"✅ Feature extractor loaded with model: {extractor.get_model_name()}")
        
        # Extract features
        features = extractor.extract_features(preprocessed_texts)
        print(f"✅ Extracted features with shape: {features['input_ids'].shape}")
        print(f"✅ Input IDs range: {features['input_ids'].min()} to {features['input_ids'].max()}")
        print("✅ Feature extraction works\n")
    except Exception as e:
        print(f"❌ Error in feature extraction: {e}\n")
        return False
    
    print("🎉 Complete data pipeline test passed!")
    print("\nSummary of Pipeline Steps:")
    print("1. ✅ Data Ingestion: Loaded dataset successfully")
    print("2. ✅ Data Preprocessing: Cleaned and normalized text data")
    print("3. ✅ Data Validation: Checked data quality and identified issues")
    print("4. ✅ Feature Extraction: Converted text to model-ready format")
    
    return True


if __name__ == "__main__":
    success = test_complete_data_pipeline()
    sys.exit(0 if success else 1)