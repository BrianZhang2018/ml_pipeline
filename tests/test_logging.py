"""
Test script for logging system
"""

import sys
import os
import shutil
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.logger import LogManager, get_logger, log_info, log_debug, log_error


def test_logging_system():
    """Test the logging system"""
    print("Testing Logging System...")
    print("=" * 30)
    
    # Test 1: Create log manager
    print("Test 1: Creating log manager")
    try:
        logger = LogManager("test_logger")
        print("✅ Log manager created successfully")
        
        # Test logging messages
        logger.debug("This is a debug message")
        logger.info("This is an info message")
        logger.warning("This is a warning message")
        logger.error("This is an error message")
        print("✅ All log levels work\n")
    except Exception as e:
        print(f"❌ Error with log manager: {e}\n")
        return False
    
    # Test 2: Test global logger
    print("Test 2: Testing global logger")
    try:
        log_info("This is a global info message", "global_test")
        log_debug("This is a global debug message", "global_test")
        log_error("This is a global error message", "global_test")
        print("✅ Global logger works\n")
    except Exception as e:
        print(f"❌ Error with global logger: {e}\n")
        return False
    
    # Test 3: Test get_logger function
    print("Test 3: Testing get_logger function")
    try:
        logger = get_logger("function_test")
        logger.info("This message is from get_logger function")
        print("✅ get_logger function works\n")
    except Exception as e:
        print(f"❌ Error with get_logger function: {e}\n")
        return False
    
    print("🎉 All logging system tests passed!")
    return True


if __name__ == "__main__":
    # Clean up any existing test logs
    if os.path.exists("logs"):
        shutil.rmtree("logs")
    
    success = test_logging_system()
    
    # Clean up test logs
    if os.path.exists("logs"):
        shutil.rmtree("logs")
    
    sys.exit(0 if success else 1)