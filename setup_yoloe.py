#!/usr/bin/env python3
"""
BakuFlow YOLOE Setup Script
Automatically installs YOLOE and applies BakuFlow customizations.
"""

import os
import sys
import subprocess
import shutil
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

def run_command(cmd, description=""):
    """Run shell command with error handling"""
    if description:
        print_status(f"Running: {description}", "INFO")
    
    print_status(f"Command: {cmd}", "INFO")
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, 
                              capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print_status(f"Command failed: {e}", "ERROR")
        if e.stdout:
            print(f"STDOUT: {e.stdout}")
        if e.stderr:
            print(f"STDERR: {e.stderr}")
        return False

def clone_yoloe():
    """Clone YOLOE repository"""
    print_status("Step 1: Cloning YOLOE repository...", "INFO")
    
    yoloe_dir = "labelimg/yoloe"
    
    if os.path.exists(yoloe_dir):
        print_status(f"YOLOE directory already exists: {yoloe_dir}", "WARNING")
        response = input("Do you want to remove and re-clone? (y/N): ")
        if response.lower() == 'y':
            print_status("Removing existing YOLOE directory...", "INFO")
            shutil.rmtree(yoloe_dir)
        else:
            print_status("Skipping YOLOE clone.", "INFO")
            return True
    
    # Clone YOLOE
    success = run_command(
        "git clone https://github.com/THU-MIG/yoloe.git labelimg/yoloe",
        "Cloning YOLOE repository"
    )
    
    if success:
        print_status("YOLOE repository cloned successfully!", "SUCCESS")
    
    return success

def install_yoloe_dependencies():
    """Install YOLOE dependencies"""
    print_status("Step 2: Installing YOLOE dependencies...", "INFO")
    
    yoloe_dir = "labelimg/yoloe"
    if not os.path.exists(yoloe_dir):
        print_status("YOLOE directory not found. Please clone first.", "ERROR")
        return False
    
    # Change to YOLOE directory and install
    original_dir = os.getcwd()
    
    try:
        os.chdir(yoloe_dir)
        
        # Install requirements
        if os.path.exists("requirements.txt"):
            success = run_command(
                "pip install -r requirements.txt",
                "Installing YOLOE requirements"
            )
            if not success:
                return False
        
        # Install in development mode
        success = run_command(
            "pip install -e .",
            "Installing YOLOE in development mode"
        )
        
        if success:
            print_status("YOLOE dependencies installed successfully!", "SUCCESS")
        
        return success
        
    finally:
        os.chdir(original_dir)

def apply_bakuflow_patches():
    """Apply BakuFlow customizations to YOLOE"""
    print_status("Step 3: Applying BakuFlow customizations...", "INFO")
    
    patches_dir = "yoloe_patches"
    yoloe_dir = "labelimg/yoloe"
    
    if not os.path.exists(patches_dir):
        print_status(f"Patches directory not found: {patches_dir}", "ERROR")
        return False
    
    if not os.path.exists(yoloe_dir):
        print_status(f"YOLOE directory not found: {yoloe_dir}", "ERROR")
        return False
    
    # Apply predict_vp.py patch
    source_file = f"{patches_dir}/predict_vp.py"
    target_file = f"{yoloe_dir}/ultralytics/models/yolo/yoloe/predict_vp.py"
    
    if os.path.exists(source_file):
        # Create backup
        if os.path.exists(target_file):
            backup_file = f"{target_file}.backup"
            shutil.copy2(target_file, backup_file)
            print_status(f"Created backup: {backup_file}", "INFO")
        
        # Apply patch
        shutil.copy2(source_file, target_file)
        print_status(f"Applied patch: {source_file} -> {target_file}", "SUCCESS")
    else:
        print_status(f"Patch file not found: {source_file}", "ERROR")
        return False
    
    print_status("BakuFlow customizations applied successfully!", "SUCCESS")
    return True

def download_models():
    """Download pre-trained models"""
    print_status("Step 4: Downloading pre-trained models...", "INFO")
    
    # Create pretrain directory
    pretrain_dir = "pretrain"
    os.makedirs(pretrain_dir, exist_ok=True)
    
    # Check if huggingface-hub is installed
    try:
        import huggingface_hub
    except ImportError:
        print_status("Installing huggingface-hub...", "INFO")
        success = run_command(
            "pip install huggingface-hub==0.26.3",
            "Installing huggingface-hub"
        )
        if not success:
            print_status("Failed to install huggingface-hub", "ERROR")
            return False
    
    # Download models
    models = [
        "yoloe-v8l-seg.pt",
        "yoloe-11l-seg.pt"
    ]
    
    for model in models:
        model_path = f"{pretrain_dir}/{model}"
        if os.path.exists(model_path):
            print_status(f"Model already exists: {model}", "INFO")
            continue
        
        print_status(f"Downloading {model}...", "INFO")
        success = run_command(
            f"huggingface-cli download jameslahm/yoloe {model} --local-dir {pretrain_dir}",
            f"Downloading {model}"
        )
        
        if success:
            print_status(f"Downloaded: {model}", "SUCCESS")
        else:
            print_status(f"Failed to download: {model}", "WARNING")
    
    return True

def verify_installation():
    """Verify the installation"""
    print_status("Step 5: Verifying installation...", "INFO")
    
    # Run the test script
    if os.path.exists("test_yoloe_setup.py"):
        print_status("Running verification test...", "INFO")
        success = run_command(
            "python test_yoloe_setup.py",
            "Running YOLOE installation test"
        )
        return success
    else:
        print_status("Test script not found, skipping verification.", "WARNING")
        return True

def main():
    """Main setup function"""
    print_status("BakuFlow YOLOE Setup Script", "INFO")
    print_status("=" * 50, "INFO")
    
    # Check if we're in the right directory
    if not os.path.exists("labelimg") or not os.path.exists("bakuai-labelimg.py"):
        print_status("Error: Please run this script from the BakuFlow root directory", "ERROR")
        sys.exit(1)
    
    steps = [
        ("Clone YOLOE", clone_yoloe),
        ("Install Dependencies", install_yoloe_dependencies),
        ("Apply Patches", apply_bakuflow_patches),
        ("Download Models", download_models),
        ("Verify Installation", verify_installation),
    ]
    
    for step_name, step_func in steps:
        print_status(f"\n--- {step_name} ---", "INFO")
        try:
            success = step_func()
            if not success:
                print_status(f"Step '{step_name}' failed!", "ERROR")
                response = input("Continue anyway? (y/N): ")
                if response.lower() != 'y':
                    print_status("Setup aborted.", "ERROR")
                    sys.exit(1)
        except Exception as e:
            print_status(f"Step '{step_name}' failed with exception: {e}", "ERROR")
            response = input("Continue anyway? (y/N): ")
            if response.lower() != 'y':
                print_status("Setup aborted.", "ERROR")
                sys.exit(1)
    
    print_status("\n" + "=" * 50, "INFO")
    print_status("🎉 BakuFlow YOLOE setup completed!", "SUCCESS")
    print_status("You can now use AI features in BakuFlow.", "SUCCESS")
    print_status("Run 'python bakuai-labelimg.py' to start the application.", "INFO")

if __name__ == "__main__":
    main() 