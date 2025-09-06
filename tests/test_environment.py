#!/usr/bin/env python3
"""
Simple test script to verify our environment setup
"""

def test_environment():
    """Test that all our key libraries can be imported"""
    libraries = [
        ("TensorFlow", "tensorflow"),
        ("PyTorch", "torch"),
        ("Transformers", "transformers"),
        ("Datasets", "datasets"),
        ("Pandas", "pandas"),
        ("NumPy", "numpy")
    ]
    
    success_count = 0
    
    for lib_name, lib_import in libraries:
        try:
            __import__(lib_import)
            print(f"✅ {lib_name} imported successfully")
            success_count += 1
        except ImportError as e:
            print(f"❌ {lib_name} import failed: {e}")
    
    print(f"\n🎉 {success_count}/{len(libraries)} libraries imported successfully!")
    print("Our environment is ready for development." if success_count == len(libraries) else "Some libraries failed to import.")
    return success_count == len(libraries)

def main():
    """Main function"""
    print("Testing environment setup...\n")
    success = test_environment()
    print("\nEnvironment test completed.")
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())