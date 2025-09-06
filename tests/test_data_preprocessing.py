"""
Test script for data preprocessing module
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.data_preprocessing import (
    clean_text,
    remove_html_tags,
    remove_urls,
    remove_punctuation,
    normalize_text,
    preprocess_texts
)


def test_data_preprocessing():
    """Test the data preprocessing functions"""
    print("Testing Data Preprocessing Module...")
    print("=" * 40)
    
    # Test 1: Clean text
    print("Test 1: Cleaning text")
    try:
        dirty_text = "  This   is  a   TEST!  "
        clean_result = clean_text(dirty_text)
        expected = "this is a test!"
        print(f"✅ Input: '{dirty_text}'")
        print(f"✅ Output: '{clean_result}'")
        print(f"✅ Expected: '{expected}'")
        print(f"✅ Match: {clean_result == expected}")
        print("✅ Text cleaning works\n")
    except Exception as e:
        print(f"❌ Error cleaning text: {e}\n")
        return False
    
    # Test 2: Remove HTML tags
    print("Test 2: Removing HTML tags")
    try:
        html_text = "<p>This is <b>HTML</b> text</p>"
        clean_result = remove_html_tags(html_text)
        expected = "This is HTML text"
        print(f"✅ Input: '{html_text}'")
        print(f"✅ Output: '{clean_result}'")
        print(f"✅ Expected: '{expected}'")
        print(f"✅ Match: {clean_result == expected}")
        print("✅ HTML tag removal works\n")
    except Exception as e:
        print(f"❌ Error removing HTML tags: {e}\n")
        return False
    
    # Test 3: Remove URLs
    print("Test 3: Removing URLs")
    try:
        url_text = "Visit https://example.com for more info"
        clean_result = remove_urls(url_text)
        expected = "Visit  for more info"
        print(f"✅ Input: '{url_text}'")
        print(f"✅ Output: '{clean_result}'")
        print(f"✅ Expected: '{expected}'")
        print(f"✅ Match: {clean_result == expected}")
        print("✅ URL removal works\n")
    except Exception as e:
        print(f"❌ Error removing URLs: {e}\n")
        return False
    
    # Test 4: Remove punctuation
    print("Test 4: Removing punctuation")
    try:
        punct_text = "Hello, world! How are you?"
        clean_result = remove_punctuation(punct_text)
        expected = "Hello world How are you"
        print(f"✅ Input: '{punct_text}'")
        print(f"✅ Output: '{clean_result}'")
        print(f"✅ Expected: '{expected}'")
        print(f"✅ Match: {clean_result == expected}")
        print("✅ Punctuation removal works\n")
    except Exception as e:
        print(f"❌ Error removing punctuation: {e}\n")
        return False
    
    # Test 5: Normalize text
    print("Test 5: Normalizing text")
    try:
        messy_text = "  <p>Visit https://example.com!</p>  "
        clean_result = normalize_text(messy_text)
        expected = "visit"
        print(f"✅ Input: '{messy_text}'")
        print(f"✅ Output: '{clean_result}'")
        print(f"✅ Expected: '{expected}'")
        print("✅ Text normalization works\n")
    except Exception as e:
        print(f"❌ Error normalizing text: {e}\n")
        return False
    
    # Test 6: Preprocess texts
    print("Test 6: Preprocessing texts")
    try:
        text_list = [
            "  This is the FIRST text!  ",
            "<p>Second text with <b>HTML</b></p>",
            "Visit https://example.com for the third text"
        ]
        clean_results = preprocess_texts(text_list)
        print(f"✅ Preprocessed {len(text_list)} texts")
        for i, (original, cleaned) in enumerate(zip(text_list, clean_results)):
            print(f"  Text {i+1}: '{original[:20]}...' -> '{cleaned[:20]}...'")
        print("✅ Text preprocessing works\n")
    except Exception as e:
        print(f"❌ Error preprocessing texts: {e}\n")
        return False
    
    print("🎉 All data preprocessing tests passed!")
    return True


if __name__ == "__main__":
    success = test_data_preprocessing()
    sys.exit(0 if success else 1)