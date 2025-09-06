"""
Data Validation Module

This module handles data quality checks and validation for the text classification pipeline.

What Is This? (Explain Like I'm 5)
===============================
This is like a quality control inspector for our puzzle pieces. Just like you'd
check puzzle pieces to make sure they're not broken or missing before putting
them together, this module checks our text data to make sure it's good quality
and ready for our AI to learn from.
"""

import sys
import os
import pandas as pd
from typing import Dict, List, Any
import logging

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.utils.logger import get_logger

# Get logger instance
logger = get_logger("data_validation")


def check_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Check overall data quality and return a report.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        
    Returns:
        Dict[str, Any]: Quality report
    """
    logger.info("Checking data quality")
    
    report = {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "missing_values": {},
        "duplicate_rows": 0,
        "data_types": {},
        "issues": []
    }
    
    # Check for missing values
    missing = df.isnull().sum()
    report["missing_values"] = missing[missing > 0].to_dict()
    
    # Check for duplicate rows
    report["duplicate_rows"] = df.duplicated().sum()
    
    # Check data types
    report["data_types"] = df.dtypes.to_dict()
    
    # Check for issues
    if report["duplicate_rows"] > 0:
        report["issues"].append(f"Found {report['duplicate_rows']} duplicate rows")
    
    for col, missing_count in report["missing_values"].items():
        if missing_count > 0:
            report["issues"].append(f"Column '{col}' has {missing_count} missing values")
    
    logger.info(f"Data quality check complete. Found {len(report['issues'])} issues")
    return report


def validate_text_data(df: pd.DataFrame, text_column: str = "text") -> Dict[str, Any]:
    """
    Validate text data specifically.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        text_column (str): Name of the text column
        
    Returns:
        Dict[str, Any]: Validation report
    """
    logger.info(f"Validating text data in column '{text_column}'")
    
    report = {
        "text_column": text_column,
        "total_texts": len(df),
        "empty_texts": 0,
        "very_short_texts": 0,
        "very_long_texts": 0,
        "encoding_issues": 0,
        "issues": []
    }
    
    if text_column not in df.columns:
        report["issues"].append(f"Text column '{text_column}' not found in DataFrame")
        return report
    
    # Check each text
    for i, text in enumerate(df[text_column]):
        if pd.isna(text) or text == "":
            report["empty_texts"] += 1
        elif len(str(text)) < 5:
            report["very_short_texts"] += 1
        elif len(str(text)) > 10000:
            report["very_long_texts"] += 1
        
        # Check for encoding issues
        try:
            str(text).encode('utf-8')
        except UnicodeEncodeError:
            report["encoding_issues"] += 1
    
    # Add issues to report
    if report["empty_texts"] > 0:
        report["issues"].append(f"Found {report['empty_texts']} empty texts")
    
    if report["very_short_texts"] > 0:
        report["issues"].append(f"Found {report['very_short_texts']} very short texts (< 5 characters)")
    
    if report["very_long_texts"] > 0:
        report["issues"].append(f"Found {report['very_long_texts']} very long texts (> 10000 characters)")
    
    if report["encoding_issues"] > 0:
        report["issues"].append(f"Found {report['encoding_issues']} texts with encoding issues")
    
    logger.info(f"Text validation complete. Found {len(report['issues'])} issues")
    return report


def validate_labels(df: pd.DataFrame, label_column: str = "label") -> Dict[str, Any]:
    """
    Validate label data.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        label_column (str): Name of the label column
        
    Returns:
        Dict[str, Any]: Validation report
    """
    logger.info(f"Validating labels in column '{label_column}'")
    
    report = {
        "label_column": label_column,
        "total_labels": len(df),
        "unique_labels": 0,
        "label_distribution": {},
        "missing_labels": 0,
        "issues": []
    }
    
    if label_column not in df.columns:
        report["issues"].append(f"Label column '{label_column}' not found in DataFrame")
        return report
    
    # Check for missing labels
    report["missing_labels"] = df[label_column].isnull().sum()
    
    # Get unique labels and distribution
    label_counts = df[label_column].value_counts()
    report["unique_labels"] = len(label_counts)
    report["label_distribution"] = label_counts.to_dict()
    
    # Check for issues
    if report["missing_labels"] > 0:
        report["issues"].append(f"Found {report['missing_labels']} missing labels")
    
    if report["unique_labels"] < 2:
        report["issues"].append("Need at least 2 unique labels for classification")
    
    logger.info(f"Label validation complete. Found {len(report['issues'])} issues")
    return report


def generate_validation_report(df: pd.DataFrame, text_column: str = "text", 
                             label_column: str = "label") -> Dict[str, Any]:
    """
    Generate a complete validation report.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        text_column (str): Name of the text column
        label_column (str): Name of the label column
        
    Returns:
        Dict[str, Any]: Complete validation report
    """
    logger.info("Generating complete validation report")
    
    report = {
        "data_quality": check_data_quality(df),
        "text_validation": validate_text_data(df, text_column),
        "label_validation": validate_labels(df, label_column),
        "overall_status": "PASS"
    }
    
    # Check if there are any critical issues
    total_issues = (len(report["data_quality"]["issues"]) + 
                   len(report["text_validation"]["issues"]) + 
                   len(report["label_validation"]["issues"]))
    
    if total_issues > 0:
        report["overall_status"] = "WARNING"
    
    # Check for critical issues
    critical_issues = []
    if report["label_validation"]["unique_labels"] < 2:
        critical_issues.append("Not enough unique labels")
    
    if report["text_validation"]["empty_texts"] == report["text_validation"]["total_texts"]:
        critical_issues.append("All texts are empty")
    
    if critical_issues:
        report["overall_status"] = "FAIL"
        report["critical_issues"] = critical_issues
    
    logger.info(f"Validation report complete. Status: {report['overall_status']}")
    return report


# Example usage
if __name__ == "__main__":
    # This is for testing purposes only
    print("Data Validation Module")
    print("Available functions:")
    print("- check_data_quality()")
    print("- validate_text_data()")
    print("- validate_labels()")
    print("- generate_validation_report()")