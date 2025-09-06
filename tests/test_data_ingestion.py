"""
Test script for data ingestion module
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
from data.data_ingestion import (
    load_imdb_dataset, 
    load_ag_news_dataset, 
    get_dataset_info,
    save_dataset,
    load_custom_dataset
)


def test_data_ingestion():
    """Test the data ingestion functions"""
    print("Testing Data Ingestion Module...")
    print("=" * 40)
    
    # Test 1: Get dataset info
    print("Test 1: Getting dataset info")
    try:
        imdb_info = get_dataset_info("imdb")
        print(f"✅ IMDB info: {imdb_info}")
        
        ag_news_info = get_dataset_info("ag_news")
        print(f"✅ AG News info: {ag_news_info}")
        print("✅ Dataset info retrieval works\n")
    except Exception as e:
        print(f"❌ Error getting dataset info: {e}\n")
        return False
    
    # Test 2: Load small sample of IMDB dataset
    print("Test 2: Loading small sample of IMDB dataset")
    try:
        # Load a small sample for testing
        df = load_imdb_dataset("train[:10]")
        print(f"✅ Loaded IMDB dataset with {len(df)} samples")
        print(f"✅ Columns: {list(df.columns)}")
        print(f"✅ Sample review: {df.iloc[0]['text'][:50]}...")
        print("✅ IMDB dataset loading works\n")
    except Exception as e:
        print(f"❌ Error loading IMDB dataset: {e}\n")
        return False
    
    # Test 3: Load small sample of AG News dataset
    print("Test 3: Loading small sample of AG News dataset")
    try:
        # Load a small sample for testing
        df = load_ag_news_dataset("train[:10]")
        print(f"✅ Loaded AG News dataset with {len(df)} samples")
        print(f"✅ Columns: {list(df.columns)}")
        print(f"✅ Sample text: {df.iloc[0]['text'][:50]}...")
        print("✅ AG News dataset loading works\n")
    except Exception as e:
        print(f"❌ Error loading AG News dataset: {e}\n")
        return False
    
    # Test 4: Test save and load custom dataset
    print("Test 4: Testing save and load custom dataset")
    try:
        # Create a simple test dataframe
        test_df = pd.DataFrame({
            'text': ['This is a test', 'Another test example'],
            'label': [0, 1]
        })
        
        # Save to CSV
        save_dataset(test_df, 'test_data.csv')
        print("✅ Saved test dataset to CSV")
        
        # Load from CSV
        loaded_df = load_custom_dataset('test_data.csv')
        print(f"✅ Loaded test dataset with {len(loaded_df)} samples")
        print(f"✅ Data matches: {test_df.equals(loaded_df)}")
        
        # Clean up
        os.remove('test_data.csv')
        print("✅ Save/load cycle works\n")
    except Exception as e:
        print(f"❌ Error in save/load test: {e}\n")
        return False
    
    print("🎉 All data ingestion tests passed!")
    return True


if __name__ == "__main__":
    success = test_data_ingestion()
    sys.exit(0 if success else 1)