"""
Test script for data validation module
"""

import sys
import os
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.data_validation import (
    check_data_quality,
    validate_text_data,
    validate_labels,
    generate_validation_report
)


def test_data_validation():
    """Test the data validation functions"""
    print("Testing Data Validation Module...")
    print("=" * 40)
    
    # Create test data
    test_df = pd.DataFrame({
        'text': [
            'This is a good example',
            'Another text example',
            '',  # Empty text
            'Short',
            'A' * 10001,  # Very long text
            'Normal text example'
        ],
        'label': [0, 1, 0, 1, 0, 1]
    })
    
    # Test 1: Check data quality
    print("Test 1: Checking data quality")
    try:
        quality_report = check_data_quality(test_df)
        print(f"✅ Total rows: {quality_report['total_rows']}")
        print(f"✅ Total columns: {quality_report['total_columns']}")
        print(f"✅ Issues found: {len(quality_report['issues'])}")
        print("✅ Data quality check works\n")
    except Exception as e:
        print(f"❌ Error checking data quality: {e}\n")
        return False
    
    # Test 2: Validate text data
    print("Test 2: Validating text data")
    try:
        text_report = validate_text_data(test_df, 'text')
        print(f"✅ Total texts: {text_report['total_texts']}")
        print(f"✅ Empty texts: {text_report['empty_texts']}")
        print(f"✅ Very short texts: {text_report['very_short_texts']}")
        print(f"✅ Very long texts: {text_report['very_long_texts']}")
        print(f"✅ Issues found: {len(text_report['issues'])}")
        print("✅ Text validation works\n")
    except Exception as e:
        print(f"❌ Error validating text data: {e}\n")
        return False
    
    # Test 3: Validate labels
    print("Test 3: Validating labels")
    try:
        label_report = validate_labels(test_df, 'label')
        print(f"✅ Total labels: {label_report['total_labels']}")
        print(f"✅ Unique labels: {label_report['unique_labels']}")
        print(f"✅ Missing labels: {label_report['missing_labels']}")
        print(f"✅ Issues found: {len(label_report['issues'])}")
        print("✅ Label validation works\n")
    except Exception as e:
        print(f"❌ Error validating labels: {e}\n")
        return False
    
    # Test 4: Generate complete validation report
    print("Test 4: Generating complete validation report")
    try:
        full_report = generate_validation_report(test_df, 'text', 'label')
        print(f"✅ Overall status: {full_report['overall_status']}")
        print(f"✅ Data quality issues: {len(full_report['data_quality']['issues'])}")
        print(f"✅ Text validation issues: {len(full_report['text_validation']['issues'])}")
        print(f"✅ Label validation issues: {len(full_report['label_validation']['issues'])}")
        print("✅ Complete validation report works\n")
    except Exception as e:
        print(f"❌ Error generating validation report: {e}\n")
        return False
    
    print("🎉 All data validation tests passed!")
    return True


if __name__ == "__main__":
    success = test_data_validation()
    sys.exit(0 if success else 1)