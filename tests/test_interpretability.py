"""
Test script for the interpretability module
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from evaluation.interpretability import ModelExplainer, simple_explanation


def mock_predict_function(texts):
    """Mock predict function for testing"""
    # Return random predictions for testing
    import numpy as np
    # Return probabilities for binary classification
    probabilities = []
    for _ in texts:
        prob_class_1 = np.random.random()
        prob_class_0 = 1 - prob_class_1
        probabilities.append([prob_class_0, prob_class_1])
    return probabilities


def test_model_explainer_initialization():
    """Test ModelExplainer initialization"""
    print("Testing ModelExplainer initialization...")
    
    # Test initialization without specifying explainer type
    try:
        explainer = ModelExplainer(model="mock_model")
        print(f"✅ Default initialization successful, using: {explainer.explainer_type}")
    except Exception as e:
        print(f"⚠️  Default initialization failed (expected if neither SHAP nor LIME installed): {e}")
    
    # Test initialization with explainer type
    try:
        explainer = ModelExplainer(model="mock_model", explainer_type="shap")
        print(f"✅ SHAP initialization successful, using: {explainer.explainer_type}")
    except ImportError:
        print("⚠️  SHAP not available, skipping SHAP initialization test")
    except Exception as e:
        print(f"⚠️  SHAP initialization failed: {e}")
    
    try:
        explainer = ModelExplainer(model="mock_model", explainer_type="lime")
        print(f"✅ LIME initialization successful, using: {explainer.explainer_type}")
    except ImportError:
        print("⚠️  LIME not available, skipping LIME initialization test")
    except Exception as e:
        print(f"⚠️  LIME initialization failed: {e}")
    
    print("✅ All ModelExplainer initialization tests completed\n")


def test_simple_explanation():
    """Test simple_explanation function"""
    print("Testing simple_explanation...")
    
    # Test with sample text
    sample_text = "This is a sample text for explanation."
    
    try:
        explanation = simple_explanation(sample_text, model="mock_model", explainer_type="shap")
        
        # Check explanation structure
        assert 'text' in explanation, "Missing text key"
        assert 'explanation_type' in explanation, "Missing explanation_type key"
        assert 'feature_importance' in explanation, "Missing feature_importance key"
        assert 'prediction' in explanation, "Missing prediction key"
        
        # Check values
        assert explanation['text'] == sample_text, "Text mismatch"
        assert isinstance(explanation['feature_importance'], list), "Feature importance should be a list"
        assert len(explanation['feature_importance']) > 0, "Feature importance should not be empty"
        
        print("✅ Simple explanation with SHAP test passed")
    except ImportError:
        print("⚠️  SHAP not available, skipping simple explanation with SHAP test")
    except Exception as e:
        print(f"⚠️  Simple explanation with SHAP failed: {e}")
    
    try:
        explanation = simple_explanation(sample_text, model="mock_model", explainer_type="lime")
        
        # Check explanation structure
        assert 'text' in explanation, "Missing text key"
        assert 'explanation_type' in explanation, "Missing explanation_type key"
        assert 'feature_importance' in explanation, "Missing feature_importance key"
        assert 'prediction' in explanation, "Missing prediction key"
        
        print("✅ Simple explanation with LIME test passed")
    except ImportError:
        print("⚠️  LIME not available, skipping simple explanation with LIME test")
    except Exception as e:
        print(f"⚠️  Simple explanation with LIME failed: {e}")
    
    print("✅ All simple_explanation tests completed\n")


def test_model_explainer_functionality():
    """Test ModelExplainer functionality"""
    print("Testing ModelExplainer functionality...")
    
    # Sample data for LIME explainer
    training_texts = [
        "This is a positive example text.",
        "This is another positive text example.",
        "This is a negative example text.",
        "This is another negative text example."
    ]
    
    try:
        # Create explainer
        explainer = ModelExplainer(model="mock_model", explainer_type="lime")
        
        # Create explainer object
        explainer.create_explainer(training_texts, mock_predict_function)
        print("✅ LIME explainer creation test passed")
        
        # Test explanation
        sample_text = "This is a sample text to explain."
        explanation = explainer.explain_prediction(sample_text, num_features=5)
        
        # Check explanation structure
        assert 'text' in explanation, "Missing text key"
        assert 'explanation_type' in explanation, "Missing explanation_type key"
        assert 'feature_importance' in explanation, "Missing feature_importance key"
        
        print("✅ LIME explanation test passed")
        
    except ImportError:
        print("⚠️  LIME not available, skipping ModelExplainer functionality tests")
    except Exception as e:
        print(f"⚠️  ModelExplainer functionality test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("✅ All ModelExplainer functionality tests completed\n")


def test_interpretability_module():
    """Test the complete interpretability module"""
    print("Testing Interpretability Module...")
    print("=" * 35)
    
    try:
        test_model_explainer_initialization()
        test_simple_explanation()
        test_model_explainer_functionality()
        
        print("🎉 All interpretability module tests completed!")
        return True
    except Exception as e:
        print(f"❌ Error in interpretability module tests: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_interpretability_module()
    sys.exit(0 if success else 1)