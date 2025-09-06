"""
Test script for the cross-validation module
"""

import sys
import os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from evaluation.cross_validation import CrossValidator, simple_kfold_cv


def mock_model_fn():
    """Mock model function for testing"""
    return "mock_model"


def mock_train_function(model, train_texts, train_labels):
    """Mock train function for testing"""
    return model


def mock_predict_function(model, test_texts):
    """Mock predict function for testing"""
    # Return random predictions for testing
    return [np.random.randint(0, 2) for _ in test_texts]


def test_cross_validator_initialization():
    """Test CrossValidator initialization"""
    print("Testing CrossValidator initialization...")
    
    # Test default initialization
    cv = CrossValidator()
    assert cv.n_folds == 5, f"Expected 5 folds, got {cv.n_folds}"
    assert cv.random_state == 42, f"Expected random_state 42, got {cv.random_state}"
    print("✅ Default initialization test passed")
    
    # Test custom initialization
    cv = CrossValidator(n_folds=3, random_state=123)
    assert cv.n_folds == 3, f"Expected 3 folds, got {cv.n_folds}"
    assert cv.random_state == 123, f"Expected random_state 123, got {cv.random_state}"
    print("✅ Custom initialization test passed")
    
    print("✅ All CrossValidator initialization tests passed\n")


def test_simple_kfold_cv():
    """Test simple_kfold_cv function"""
    print("Testing simple_kfold_cv...")
    
    # Create sample data
    texts = [f"text_{i}" for i in range(20)]
    labels = [i % 2 for i in range(20)]  # Binary labels
    
    # Test with default parameters
    results = simple_kfold_cv(texts, labels, mock_model_fn)
    
    # Check results structure
    assert 'fold_results' in results, "Missing fold_results key"
    assert 'aggregate_results' in results, "Missing aggregate_results key"
    assert 'n_folds' in results, "Missing n_folds key"
    
    # Check fold results
    fold_results = results['fold_results']
    assert len(fold_results) == 5, f"Expected 5 folds, got {len(fold_results)}"
    
    # Check aggregate results
    aggregate_results = results['aggregate_results']
    expected_metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    for metric in expected_metrics:
        assert metric in aggregate_results, f"Missing metric: {metric}"
        assert 'mean' in aggregate_results[metric], f"Missing mean for {metric}"
        assert 'std' in aggregate_results[metric], f"Missing std for {metric}"
    
    print("✅ Simple k-fold CV test passed")
    print("✅ All simple_kfold_cv tests passed\n")


def test_cross_validator_validate_model():
    """Test CrossValidator.validate_model method"""
    print("Testing CrossValidator.validate_model...")
    
    # Create sample data
    texts = [f"text_{i}" for i in range(20)]
    labels = [i % 2 for i in range(20)]  # Binary labels
    
    # Create cross-validator
    cv = CrossValidator(n_folds=3)
    
    # Test model validation
    results = cv.validate_model(
        model="mock_model",
        texts=texts,
        labels=labels,
        train_function=mock_train_function,
        predict_function=mock_predict_function
    )
    
    # Check results structure
    assert 'fold_results' in results, "Missing fold_results key"
    assert 'aggregate_results' in results, "Missing aggregate_results key"
    assert results['n_folds'] == 3, f"Expected 3 folds, got {results['n_folds']}"
    
    # Check fold results
    fold_results = results['fold_results']
    assert len(fold_results) == 3, f"Expected 3 folds, got {len(fold_results)}"
    
    for fold_result in fold_results:
        assert 'fold' in fold_result, "Missing fold key in fold result"
        assert 'train_size' in fold_result, "Missing train_size key in fold result"
        assert 'val_size' in fold_result, "Missing val_size key in fold result"
        assert 'metrics' in fold_result, "Missing metrics key in fold result"
        
        # Check metrics
        metrics = fold_result['metrics']
        expected_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'confusion_matrix', 'classification_report']
        for metric in expected_metrics:
            assert metric in metrics, f"Missing metric: {metric}"
    
    # Check aggregate results
    aggregate_results = results['aggregate_results']
    expected_metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    for metric in expected_metrics:
        assert metric in aggregate_results, f"Missing metric: {metric}"
        metric_data = aggregate_results[metric]
        expected_stats = ['mean', 'std', 'min', 'max']
        for stat in expected_stats:
            assert stat in metric_data, f"Missing {stat} for {metric}"
            assert isinstance(metric_data[stat], float), f"{stat} for {metric} should be float"
    
    print("✅ CrossValidator model validation test passed")
    print("✅ All CrossValidator.validate_model tests passed\n")


def test_cross_validation_module():
    """Test the complete cross-validation module"""
    print("Testing Cross-Validation Module...")
    print("=" * 35)
    
    try:
        test_cross_validator_initialization()
        test_simple_kfold_cv()
        test_cross_validator_validate_model()
        
        print("🎉 All cross-validation module tests passed!")
        return True
    except Exception as e:
        print(f"❌ Error in cross-validation module tests: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_cross_validation_module()
    sys.exit(0 if success else 1)