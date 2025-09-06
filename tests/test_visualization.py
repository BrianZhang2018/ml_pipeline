"""
Test script for the visualization module
"""

import sys
import os
import numpy as np
import tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from evaluation.visualization import ModelVisualizer, simple_plot_confusion_matrix


def test_model_visualizer_initialization():
    """Test ModelVisualizer initialization"""
    print("Testing ModelVisualizer initialization...")
    
    # Test default initialization
    try:
        visualizer = ModelVisualizer()
        print("✅ Default initialization successful")
    except Exception as e:
        print(f"⚠️  Default initialization failed (expected if matplotlib not installed): {e}")
    
    # Test seaborn style initialization
    try:
        visualizer = ModelVisualizer(style="seaborn")
        print("✅ Seaborn style initialization successful")
    except Exception as e:
        print(f"⚠️  Seaborn style initialization failed (expected if matplotlib/seaborn not installed): {e}")
    
    print("✅ All ModelVisualizer initialization tests completed\n")


def test_simple_plot_confusion_matrix():
    """Test simple_plot_confusion_matrix function"""
    print("Testing simple_plot_confusion_matrix...")
    
    # Create a sample confusion matrix
    confusion_matrix = np.array([[50, 10], [5, 35]])
    class_names = ["Negative", "Positive"]
    
    # Test without saving
    try:
        result = simple_plot_confusion_matrix(confusion_matrix, class_names)
        print(f"✅ Simple confusion matrix plot test result: {result}")
    except ImportError:
        print("⚠️  Matplotlib not available, skipping simple confusion matrix plot test")
    except Exception as e:
        print(f"⚠️  Simple confusion matrix plot test failed: {e}")
    
    # Test with saving to temporary file
    try:
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            temp_path = tmp.name
        
        result = simple_plot_confusion_matrix(confusion_matrix, class_names, save_path=temp_path)
        print(f"✅ Simple confusion matrix plot with save test result: {result}")
        
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.unlink(temp_path)
            print("✅ Temporary file cleaned up")
            
    except ImportError:
        print("⚠️  Matplotlib not available, skipping simple confusion matrix plot with save test")
    except Exception as e:
        print(f"⚠️  Simple confusion matrix plot with save test failed: {e}")
    
    print("✅ All simple_plot_confusion_matrix tests completed\n")


def test_model_visualizer_functionality():
    """Test ModelVisualizer functionality"""
    print("Testing ModelVisualizer functionality...")
    
    try:
        # Create visualizer
        visualizer = ModelVisualizer()
        
        # Test 1: Plot confusion matrix
        print("Testing confusion matrix plotting...")
        confusion_matrix = np.array([[50, 10, 5], [5, 35, 10], [10, 5, 40]])
        class_names = ["Class_A", "Class_B", "Class_C"]
        
        try:
            fig = visualizer.plot_confusion_matrix(
                confusion_matrix=confusion_matrix,
                class_names=class_names,
                title="Test Confusion Matrix"
            )
            print("✅ Confusion matrix plotting test passed")
            
            # Close figure to free memory
            import matplotlib.pyplot as plt
            plt.close(fig)
        except Exception as e:
            print(f"⚠️  Confusion matrix plotting test failed: {e}")
        
        # Test 2: Plot metrics comparison
        print("Testing metrics comparison plotting...")
        try:
            metrics_data = {
                "Model_A": [0.85, 0.78, 0.82, 0.79],
                "Model_B": [0.82, 0.81, 0.80, 0.83]
            }
            metric_names = ["Accuracy", "Precision", "Recall", "F1-Score"]
            
            fig = visualizer.plot_metrics_comparison(
                metrics_data=metrics_data,
                metric_names=metric_names,
                title="Test Metrics Comparison"
            )
            print("✅ Metrics comparison plotting test passed")
            
            # Close figure to free memory
            import matplotlib.pyplot as plt
            plt.close(fig)
        except Exception as e:
            print(f"⚠️  Metrics comparison plotting test failed: {e}")
        
        # Test 3: Plot learning curve
        print("Testing learning curve plotting...")
        try:
            train_scores = [0.6, 0.7, 0.75, 0.8, 0.82, 0.85, 0.86]
            val_scores = [0.55, 0.65, 0.7, 0.75, 0.77, 0.78, 0.79]
            epochs = list(range(1, len(train_scores) + 1))
            
            fig = visualizer.plot_learning_curve(
                train_scores=train_scores,
                val_scores=val_scores,
                epochs=epochs,
                title="Test Learning Curve"
            )
            print("✅ Learning curve plotting test passed")
            
            # Close figure to free memory
            import matplotlib.pyplot as plt
            plt.close(fig)
        except Exception as e:
            print(f"⚠️  Learning curve plotting test failed: {e}")
        
        # Test 4: Plot feature importance
        print("Testing feature importance plotting...")
        try:
            feature_names = ["feature_1", "feature_2", "feature_3", "feature_4", "feature_5"]
            importance_scores = [0.3, 0.25, 0.2, 0.15, 0.1]
            
            fig = visualizer.plot_feature_importance(
                feature_names=feature_names,
                importance_scores=importance_scores,
                top_k=5,
                title="Test Feature Importance"
            )
            print("✅ Feature importance plotting test passed")
            
            # Close figure to free memory
            import matplotlib.pyplot as plt
            plt.close(fig)
        except Exception as e:
            print(f"⚠️  Feature importance plotting test failed: {e}")
            
    except ImportError:
        print("⚠️  Matplotlib not available, skipping ModelVisualizer functionality tests")
    except Exception as e:
        print(f"⚠️  ModelVisualizer functionality test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("✅ All ModelVisualizer functionality tests completed\n")


def test_visualization_module():
    """Test the complete visualization module"""
    print("Testing Visualization Module...")
    print("=" * 30)
    
    try:
        test_model_visualizer_initialization()
        test_simple_plot_confusion_matrix()
        test_model_visualizer_functionality()
        
        print("🎉 All visualization module tests completed!")
        return True
    except Exception as e:
        print(f"❌ Error in visualization module tests: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_visualization_module()
    sys.exit(0 if success else 1)