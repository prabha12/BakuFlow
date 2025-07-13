#!/usr/bin/env python3
"""
YOLOE Installation Test Script for BakuFlow
Tests whether YOLOE is properly installed and configured.
"""

import sys
import os
import subprocess
from pathlib import Path

def print_status(message, status="INFO"):
    """Print colored status messages"""
    colors = {
        "INFO": "\033[94m",     # Blue
        "SUCCESS": "\033[92m",   # Green
        "WARNING": "\033[93m",   # Yellow
        "ERROR": "\033[91m",     # Red
        "RESET": "\033[0m"       # Reset
    }
    print(f"{colors.get(status, '')}{status}: {message}{colors['RESET']}")

def check_directory_structure():
    """Check if required directories exist"""
    print_status("Checking directory structure...", "INFO")
    
    required_dirs = [
        "labelimg/",
        "pretrain/",
    ]
    
    missing_dirs = []
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            missing_dirs.append(dir_path)
            print_status(f"Missing directory: {dir_path}", "ERROR")
        else:
            print_status(f"Found directory: {dir_path}", "SUCCESS")
    
    # Check if YOLOE directory exists
    yoloe_dir = "labelimg/yoloe"
    if not os.path.exists(yoloe_dir):
        print_status(f"YOLOE directory not found: {yoloe_dir}", "ERROR")
        print_status("You need to install YOLOE first!", "WARNING")
        print_status("Run: git clone https://github.com/THU-MIG/yoloe.git labelimg/yoloe", "INFO")
        return False
    else:
        print_status(f"Found YOLOE directory: {yoloe_dir}", "SUCCESS")
    
    return len(missing_dirs) == 0

def check_yoloe_installation():
    """Check if YOLOE is properly installed"""
    print_status("Checking YOLOE installation...", "INFO")
    
    # Add YOLOE to Python path
    yoloe_path = os.path.abspath('labelimg/yoloe')
    if yoloe_path not in sys.path:
        sys.path.insert(0, yoloe_path)
        print_status(f"Added YOLOE to Python path: {yoloe_path}", "INFO")
    
    try:
        # Test basic imports
        from ultralytics import YOLOE
        print_status("Successfully imported YOLOE", "SUCCESS")
        
        from ultralytics.models.yolo.yoloe.predict_vp import YOLOEVPSegPredictor
        print_status("Successfully imported YOLOEVPSegPredictor", "SUCCESS")
        
        return True
        
    except ImportError as e:
        print_status(f"Failed to import YOLOE: {e}", "ERROR")
        print_status("Please install YOLOE dependencies:", "WARNING")
        print_status("cd labelimg/yoloe && pip install -r requirements.txt && pip install -e .", "INFO")
        return False
    except Exception as e:
        print_status(f"Unexpected error: {e}", "ERROR")
        return False

def check_model_files():
    """Check if model files are available"""
    print_status("Checking model files...", "INFO")
    
    model_files = [
        "pretrain/yoloe-v8l-seg.pt",
        "pretrain/yoloe-11l-seg.pt",
        "pretrain/yoloe-v8s-seg.pt",
        "pretrain/yoloe-v8m-seg.pt"
    ]
    
    found_models = []
    missing_models = []
    
    for model_file in model_files:
        if os.path.exists(model_file):
            size_mb = os.path.getsize(model_file) / (1024 * 1024)
            print_status(f"Found model: {model_file} ({size_mb:.1f} MB)", "SUCCESS")
            found_models.append(model_file)
        else:
            print_status(f"Missing model: {model_file}", "WARNING")
            missing_models.append(model_file)
    
    if not found_models:
        print_status("No model files found!", "ERROR")
        print_status("Download models with:", "INFO")
        print_status("pip install huggingface-hub", "INFO")
        print_status("huggingface-cli download jameslahm/yoloe yoloe-v8l-seg.pt --local-dir pretrain", "INFO")
        return False
    
    return True

def test_model_loading():
    """Test loading a YOLOE model"""
    print_status("Testing model loading...", "INFO")
    
    # Find available model
    model_files = [
        "pretrain/yoloe-v8l-seg.pt",
        "pretrain/yoloe-11l-seg.pt",
        "pretrain/yoloe-v8s-seg.pt",
        "pretrain/yoloe-v8m-seg.pt"
    ]
    
    available_model = None
    for model_file in model_files:
        if os.path.exists(model_file):
            available_model = model_file
            break
    
    if not available_model:
        print_status("No models available for testing", "ERROR")
        return False
    
    try:
        # Add YOLOE to path again (in case this function is called separately)
        yoloe_path = os.path.abspath('labelimg/yoloe')
        if yoloe_path not in sys.path:
            sys.path.insert(0, yoloe_path)
        
        from ultralytics import YOLOE
        
        print_status(f"Loading model: {available_model}", "INFO")
        model = YOLOE(available_model)
        print_status(f"Successfully loaded model: {available_model}", "SUCCESS")
        print_status(f"Model type: {type(model)}", "INFO")
        
        return True
        
    except Exception as e:
        print_status(f"Failed to load model: {e}", "ERROR")
        return False

def check_dependencies():
    """Check if required dependencies are installed"""
    print_status("Checking Python dependencies...", "INFO")
    import sys
    print_status(f"Python executable: {sys.executable}", "INFO")
    print_status(f"sys.path: {sys.path}", "INFO")

    required_packages = [
        ("torch", "torch"),
        ("torchvision", "torchvision"),
        ("numpy", "numpy"),
        ("opencv-python", "cv2"),
        ("ultralytics", "ultralytics"),
        ("huggingface-hub", "huggingface_hub")
    ]

    missing_packages = []

    for pip_name, import_name in required_packages:
        try:
            __import__(import_name)
            print_status(f"Found package: {pip_name}", "SUCCESS")
        except ImportError:
            print_status(f"Missing package: {pip_name}", "ERROR")
            missing_packages.append(pip_name)

    if missing_packages:
        print_status(f"Install missing packages: pip install {' '.join(missing_packages)}", "INFO")
        return False

    return True

def main():
    """Main test function"""
    print_status("BakuFlow YOLOE Installation Test", "INFO")
    print_status("=" * 50, "INFO")
    
    tests = [
        ("Directory Structure", check_directory_structure),
        ("Python Dependencies", check_dependencies),
        ("YOLOE Installation", check_yoloe_installation),
        ("Model Files", check_model_files),
        ("Model Loading", test_model_loading),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print_status(f"\n--- Testing {test_name} ---", "INFO")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print_status(f"Test {test_name} failed with exception: {e}", "ERROR")
            results[test_name] = False
    
    # Summary
    print_status("\n" + "=" * 50, "INFO")
    print_status("TEST SUMMARY", "INFO")
    print_status("=" * 50, "INFO")
    
    all_passed = True
    for test_name, passed in results.items():
        status = "SUCCESS" if passed else "ERROR"
        symbol = "✅" if passed else "❌"
        print_status(f"{symbol} {test_name}: {'PASSED' if passed else 'FAILED'}", status)
        if not passed:
            all_passed = False
    
    print_status("\n" + "=" * 50, "INFO")
    if all_passed:
        print_status("🎉 All tests passed! YOLOE is ready to use in BakuFlow.", "SUCCESS")
        print_status("You can now use AI features in the application.", "SUCCESS")
    else:
        print_status("❌ Some tests failed. Please fix the issues above.", "ERROR")
        print_status("Refer to the README.md YOLOE Setup section for help.", "WARNING")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

# Patch: Ensure subprocess uses UTF-8 encoding if used in this script
import subprocess as _subprocess
_old_run = _subprocess.run
def _patched_run(*args, **kwargs):
    if 'encoding' not in kwargs:
        kwargs['encoding'] = 'utf-8'
    return _old_run(*args, **kwargs)
_subprocess.run = _patched_run