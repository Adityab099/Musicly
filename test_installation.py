#!/usr/bin/env python3
"""
Test script to verify OCR utility installation
Run this to check if all dependencies are properly installed
"""

import sys
import importlib

def test_import(module_name, package_name=None):
    """Test if a module can be imported"""
    try:
        if package_name:
            module = importlib.import_module(module_name, package_name)
        else:
            module = importlib.import_module(module_name)
        print(f"✓ {module_name} imported successfully")
        return True
    except ImportError as e:
        print(f"✗ {module_name} import failed: {e}")
        return False

def test_ocr_utility():
    """Test if our OCR utility can be imported"""
    try:
        from ocr_utils import OCRProcessor, create_sample_image
        print("✓ OCR utility imported successfully")
        return True
    except ImportError as e:
        print(f"✗ OCR utility import failed: {e}")
        return False

def main():
    """Main test function"""
    print("=== OCR Utility Installation Test ===\n")
    
    # Test core dependencies
    print("Testing core dependencies:")
    dependencies = [
        'paddlepaddle',
        'paddleocr', 
        'cv2',  # opencv-python
        'PIL',  # Pillow
        'numpy'
    ]
    
    all_deps_ok = True
    for dep in dependencies:
        if not test_import(dep):
            all_deps_ok = False
    
    print()
    
    # Test OCR utility
    print("Testing OCR utility:")
    ocr_ok = test_ocr_utility()
    
    print()
    
    # Summary
    if all_deps_ok and ocr_ok:
        print("🎉 All tests passed! Your OCR utility is ready to use.")
        print("\nNext steps:")
        print("1. Create a sample image: python main.py --create-sample")
        print("2. Test OCR: python main.py sample.jpg")
        print("3. See examples: python main.py --examples")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        print("\nTroubleshooting:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Check Python version (requires 3.7+)")
        print("3. Ensure you're in the correct directory")
        
        if not all_deps_ok:
            print("\nMissing dependencies detected. Run:")
            print("pip install -r requirements.txt")
    
    return all_deps_ok and ocr_ok

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 