"""
Visualization Module

This module provides functions to create visualizations for model evaluation results.

What Is This? (Explain Like I'm 5)
===============================
This is like drawing pictures to show how well our AI brain is doing. Just like
when you draw a chart to show your test scores in different subjects, we create
pictures to show how good our AI is at sorting text. We make charts that show:
- How many times it got the right answer (like a scoreboard)
- Which kinds of text it's good at sorting vs. which it struggles with
- Which words help it make the right decisions
"""

import sys
import os
from typing import Dict, List, Tuple, Any, Union
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.utils.logger import get_logger

# Get logger instance
logger = get_logger("visualization")

# Try to import visualization libraries, but handle cases where they're not available
try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("Matplotlib not available. Install matplotlib for visualization features.")

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False
    logger.warning("Seaborn not available. Install seaborn for enhanced visualization features.")


class ModelVisualizer:
    """
    A class to create visualizations for model evaluation.
    """
    
    def __init__(self, style: str = "seaborn"):
        """
        Initialize the ModelVisualizer.
        
        Args:
            style (str): Visualization style
        """
        logger.info("Initializing ModelVisualizer")
        
        self.style = style
        
        # Set up plotting style
        if MATPLOTLIB_AVAILABLE:
            if self.style == "seaborn" and SEABORN_AVAILABLE:
                sns.set_style("whitegrid")
            plt.style.use('seaborn-v0_8' if SEABORN_AVAILABLE else 'default')
        
        logger.info("ModelVisualizer initialized successfully")
    
    def plot_confusion_matrix(self, 
                             confusion_matrix: np.ndarray,
                             class_names: List[str] = None,
                             title: str = "Confusion Matrix",
                             save_path: str = None) -> plt.Figure:
        """
        Plot a confusion matrix.
        
        Args:
            confusion_matrix (np.ndarray): Confusion matrix
            class_names (List[str], optional): Names of the classes
            title (str): Title for the plot
            save_path (str, optional): Path to save the plot
            
        Returns:
            plt.Figure: The matplotlib figure
        """
        logger.info("Plotting confusion matrix")
        
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("Matplotlib is not installed. Please install matplotlib package.")
        
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Plot confusion matrix
        if SEABORN_AVAILABLE:
            sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=class_names or range(confusion_matrix.shape[1]),
                       yticklabels=class_names or range(confusion_matrix.shape[0]), ax=ax)
        else:
            im = ax.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
            ax.figure.colorbar(im, ax=ax)
            
            # Set ticks and labels
            ax.set(xticks=np.arange(confusion_matrix.shape[1]),
                   yticks=np.arange(confusion_matrix.shape[0]),
                   xticklabels=class_names or range(confusion_matrix.shape[1]),
                   yticklabels=class_names or range(confusion_matrix.shape[0]),
                   xlabel='Predicted Label',
                   ylabel='True Label')
            
            # Add text annotations
            thresh = confusion_matrix.max() / 2.
            for i in range(confusion_matrix.shape[0]):
                for j in range(confusion_matrix.shape[1]):
                    ax.text(j, i, format(confusion_matrix[i, j], 'd'),
                           ha="center", va="center",
                           color="white" if confusion_matrix[i, j] > thresh else "black")
        
        ax.set_title(title)
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {save_path}")
        
        logger.info("Confusion matrix plotted successfully")
        return fig
    
    def plot_metrics_comparison(self, 
                               metrics_data: Dict[str, List[float]],
                               metric_names: List[str] = None,
                               title: str = "Metrics Comparison",
                               save_path: str = None) -> plt.Figure:
        """
        Plot a comparison of different metrics.
        
        Args:
            metrics_data (Dict[str, List[float]]): Dictionary with metric names and values
            metric_names (List[str], optional): Names of the metrics to plot
            title (str): Title for the plot
            save_path (str, optional): Path to save the plot
            
        Returns:
            plt.Figure: The matplotlib figure
        """
        logger.info("Plotting metrics comparison")
        
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("Matplotlib is not installed. Please install matplotlib package.")
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Prepare data
        metric_names = metric_names or list(metrics_data.keys())
        x = np.arange(len(metric_names))
        width = 0.35
        
        # Plot bars for each method/model
        methods = list(metrics_data.keys())
        for i, method in enumerate(methods):
            values = metrics_data[method]
            ax.bar(x + i*width/len(methods), values, width/len(methods), 
                   label=method, alpha=0.8)
        
        # Customize plot
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Score')
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(metric_names)
        ax.legend()
        ax.set_ylim(0, 1.05)
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Metrics comparison saved to {save_path}")
        
        logger.info("Metrics comparison plotted successfully")
        return fig
    
    def plot_learning_curve(self, 
                           train_scores: List[float],
                           val_scores: List[float],
                           epochs: List[int] = None,
                           title: str = "Learning Curve",
                           save_path: str = None) -> plt.Figure:
        """
        Plot a learning curve.
        
        Args:
            train_scores (List[float]): Training scores for each epoch
            val_scores (List[float]): Validation scores for each epoch
            epochs (List[int], optional): Epoch numbers
            title (str): Title for the plot
            save_path (str, optional): Path to save the plot
            
        Returns:
            plt.Figure: The matplotlib figure
        """
        logger.info("Plotting learning curve")
        
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("Matplotlib is not installed. Please install matplotlib package.")
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Prepare x-axis
        if epochs is None:
            epochs = list(range(1, len(train_scores) + 1))
        
        # Plot curves
        ax.plot(epochs, train_scores, 'o-', label='Training Score', linewidth=2)
        ax.plot(epochs, val_scores, 'o-', label='Validation Score', linewidth=2)
        
        # Customize plot
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Score')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Learning curve saved to {save_path}")
        
        logger.info("Learning curve plotted successfully")
        return fig
    
    def plot_feature_importance(self, 
                               feature_names: List[str],
                               importance_scores: List[float],
                               top_k: int = 20,
                               title: str = "Feature Importance",
                               save_path: str = None) -> plt.Figure:
        """
        Plot feature importance.
        
        Args:
            feature_names (List[str]): Names of the features
            importance_scores (List[float]): Importance scores for each feature
            top_k (int): Number of top features to show
            title (str): Title for the plot
            save_path (str, optional): Path to save the plot
            
        Returns:
            plt.Figure: The matplotlib figure
        """
        logger.info("Plotting feature importance")
        
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("Matplotlib is not installed. Please install matplotlib package.")
        
        # Sort features by importance
        sorted_indices = np.argsort(importance_scores)[::-1]
        top_indices = sorted_indices[:top_k]
        
        top_features = [feature_names[i] for i in top_indices]
        top_importance = [importance_scores[i] for i in top_indices]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, max(6, top_k * 0.3)))
        
        # Plot horizontal bar chart
        y_pos = np.arange(len(top_features))
        ax.barh(y_pos, top_importance, align='center', alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_features)
        ax.invert_yaxis()  # Labels read top-to-bottom
        ax.set_xlabel('Importance')
        ax.set_title(title)
        
        # Add value labels on bars
        for i, v in enumerate(top_importance):
            ax.text(v + 0.01, i, f"{v:.3f}", va='center')
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Feature importance plot saved to {save_path}")
        
        logger.info("Feature importance plotted successfully")
        return fig


def simple_plot_confusion_matrix(confusion_matrix: np.ndarray,
                                class_names: List[str] = None,
                                save_path: str = None) -> bool:
    """
    Simple function to plot a confusion matrix.
    
    Args:
        confusion_matrix (np.ndarray): Confusion matrix
        class_names (List[str], optional): Names of the classes
        save_path (str, optional): Path to save the plot
        
    Returns:
        bool: True if successful
    """
    logger.info("Creating simple confusion matrix plot")
    
    try:
        visualizer = ModelVisualizer()
        fig = visualizer.plot_confusion_matrix(
            confusion_matrix=confusion_matrix,
            class_names=class_names,
            title="Confusion Matrix",
            save_path=save_path
        )
        
        if save_path is None:
            plt.show()
        else:
            plt.close(fig)
        
        logger.info("Simple confusion matrix plot created successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to create simple confusion matrix plot: {str(e)}")
        return False


# Example usage
if __name__ == "__main__":
    # This is for testing purposes only
    print("Visualization Module")
    print("Available classes and functions:")
    print("- ModelVisualizer class")
    print("- simple_plot_confusion_matrix function")
    
    # Check if visualization libraries are available
    print(f"\nMatplotlib available: {MATPLOTLIB_AVAILABLE}")
    print(f"Seaborn available: {SEABORN_AVAILABLE}")