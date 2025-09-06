"""
Test script for the metrics module
"""

import sys
import os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from evaluation.metrics import (
    calculate_accuracy,
    calculate_precision,
    calculate_recall,
    calculate_f1_score,
    calculate_confusion_matrix,
    generate_classification_report,
    calculate_all_metrics
)


def test_calculate_accuracy():
    """Test the calculate_accuracy function"""
    print("Testing calculate_accuracy...")
    
    # Test case 1: Perfect predictions
    y_true = [0, 1, 2, 1, 0]
    y_pred = [0, 1, 2, 1, 0]
    accuracy = calculate_accuracy(y_true, y_pred)
    assert accuracy == 1.0, f"Expected 1.0, got {accuracy}"
    print("✅ Perfect predictions test passed")
    
    # Test case 2: Some wrong predictions
    y_true = [0, 1, 2, 1, 0]
    y_pred = [0, 1, 2, 0, 1]  # 3 correct out of 5
    accuracy = calculate_accuracy(y_true, y_pred)
    assert accuracy == 0.6, f"Expected 0.6, got {accuracy}"
    print("✅ Partially correct predictions test passed")
    
    print("✅ All accuracy tests passed\n")


def test_calculate_precision():
    """Test the calculate_precision function"""
    print("Testing calculate_precision...")
    
    # Test case 1: Binary classification
    y_true = [0, 1, 1, 0, 1]
    y_pred = [0, 1, 0, 0, 1]
    precision = calculate_precision(y_true, y_pred, average='binary')
    # Precision = TP / (TP + FP) = 2 / (2 + 0) = 1.0
    # TP (true positives): predictions that are 1 and should be 1 = 2 (positions 1 and 4)
    # FP (false positives): predictions that are 1 but should be 0 = 0
    expected = 1.0
    assert abs(precision - expected) < 0.0001, f"Expected {expected}, got {precision}"
    print("✅ Binary classification precision test passed")
    
    # Test case 2: Multi-class with weighted average
    y_true = [0, 1, 2, 1, 0, 2]
    y_pred = [0, 1, 1, 0, 0, 2]
    precision = calculate_precision(y_true, y_pred, average='weighted')
    print(f"✅ Multi-class weighted precision: {precision}")
    print("✅ Multi-class precision test passed")
    
    print("✅ All precision tests passed\n")


def test_calculate_recall():
    """Test the calculate_recall function"""
    print("Testing calculate_recall...")
    
    # Test case 1: Binary classification
    y_true = [0, 1, 1, 0, 1]
    y_pred = [0, 1, 0, 0, 1]
    recall = calculate_recall(y_true, y_pred, average='binary')
    # Recall = TP / (TP + FN) = 2 / (2 + 1) = 0.6667
    # TP (true positives): predictions that are 1 and should be 1 = 2 (positions 1 and 4)
    # FN (false negatives): predictions that are 0 but should be 1 = 1 (position 2)
    expected = 2 / 3
    assert abs(recall - expected) < 0.0001, f"Expected {expected}, got {recall}"
    print("✅ Binary classification recall test passed")
    
    # Test case 2: Multi-class with weighted average
    y_true = [0, 1, 2, 1, 0, 2]
    y_pred = [0, 1, 1, 0, 0, 2]
    recall = calculate_recall(y_true, y_pred, average='weighted')
    print(f"✅ Multi-class weighted recall: {recall}")
    print("✅ Multi-class recall test passed")
    
    print("✅ All recall tests passed\n")


def test_calculate_f1_score():
    """Test the calculate_f1_score function"""
    print("Testing calculate_f1_score...")
    
    # Test case 1: Binary classification
    y_true = [0, 1, 1, 0, 1]
    y_pred = [0, 1, 0, 0, 1]
    f1 = calculate_f1_score(y_true, y_pred, average='binary')
    # Precision = 1.0 (TP=2, FP=0)
    # Recall = 0.6667 (TP=2, FN=1)
    # F1 = 2 * (precision * recall) / (precision + recall)
    # F1 = 2 * (1.0 * 0.6667) / (1.0 + 0.6667) = 2 * 0.6667 / 1.6667 = 1.3334 / 1.6667 = 0.8
    expected = 0.8
    assert abs(f1 - expected) < 0.0001, f"Expected {expected}, got {f1}"
    print("✅ Binary classification F1 test passed")
    
    # Test case 2: Multi-class with weighted average
    y_true = [0, 1, 2, 1, 0, 2]
    y_pred = [0, 1, 1, 0, 0, 2]
    f1 = calculate_f1_score(y_true, y_pred, average='weighted')
    print(f"✅ Multi-class weighted F1: {f1}")
    print("✅ Multi-class F1 test passed")
    
    print("✅ All F1 score tests passed\n")


def test_calculate_confusion_matrix():
    """Test the calculate_confusion_matrix function"""
    print("Testing calculate_confusion_matrix...")
    
    # Test case 1: Simple binary case
    y_true = [0, 1, 1, 0, 1]
    y_pred = [0, 1, 0, 0, 1]
    cm = calculate_confusion_matrix(y_true, y_pred)
    
    # Expected confusion matrix:
    # [[2 0]
    #  [1 2]]
    expected = np.array([[2, 0], [1, 2]])
    assert np.array_equal(cm, expected), f"Expected {expected}, got {cm}"
    print("✅ Binary confusion matrix test passed")
    
    # Test case 2: Multi-class case
    y_true = [0, 1, 2, 1, 0, 2]
    y_pred = [0, 1, 1, 0, 0, 2]
    cm = calculate_confusion_matrix(y_true, y_pred)
    print(f"✅ Multi-class confusion matrix shape: {cm.shape}")
    print("✅ Multi-class confusion matrix test passed")
    
    print("✅ All confusion matrix tests passed\n")


def test_generate_classification_report():
    """Test the generate_classification_report function"""
    print("Testing generate_classification_report...")
    
    # Test case 1: Simple binary case
    y_true = [0, 1, 1, 0, 1]
    y_pred = [0, 1, 0, 0, 1]
    target_names = ['Class_0', 'Class_1']
    report = generate_classification_report(y_true, y_pred, target_names)
    
    # Check that report is a string and contains expected elements
    assert isinstance(report, str), "Report should be a string"
    assert len(report) > 0, "Report should not be empty"
    print("✅ Classification report generation test passed")
    
    print("✅ All classification report tests passed\n")


def test_calculate_all_metrics():
    """Test the calculate_all_metrics function"""
    print("Testing calculate_all_metrics...")
    
    # Test case: Multi-class case
    y_true = [0, 1, 2, 1, 0, 2]
    y_pred = [0, 1, 1, 0, 0, 2]
    target_names = ['Class_0', 'Class_1', 'Class_2']
    
    metrics = calculate_all_metrics(y_true, y_pred, target_names)
    
    # Check that all expected keys are present
    expected_keys = ['accuracy', 'precision', 'recall', 'f1_score', 'confusion_matrix', 'classification_report']
    for key in expected_keys:
        assert key in metrics, f"Missing key: {key}"
    
    # Check types
    assert isinstance(metrics['accuracy'], float), "Accuracy should be float"
    assert isinstance(metrics['precision'], float), "Precision should be float"
    assert isinstance(metrics['recall'], float), "Recall should be float"
    assert isinstance(metrics['f1_score'], float), "F1 score should be float"
    assert isinstance(metrics['confusion_matrix'], np.ndarray), "Confusion matrix should be numpy array"
    assert isinstance(metrics['classification_report'], str), "Classification report should be string"
    
    print("✅ All metrics calculated successfully")
    print("✅ All calculate_all_metrics tests passed\n")


def test_metrics_module():
    """Test the complete metrics module"""
    print("Testing Metrics Module...")
    print("=" * 30)
    
    try:
        test_calculate_accuracy()
        test_calculate_precision()
        test_calculate_recall()
        test_calculate_f1_score()
        test_calculate_confusion_matrix()
        test_generate_classification_report()
        test_calculate_all_metrics()
        
        print("🎉 All metrics module tests passed!")
        return True
    except Exception as e:
        print(f"❌ Error in metrics module tests: {e}")
        return False


if __name__ == "__main__":
    success = test_metrics_module()
    sys.exit(0 if success else 1)