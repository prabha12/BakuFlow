#!/usr/bin/env python3
"""
BakuFlow Cache Cleanup & Diagnostics Script
Removes Python caches and checks for hardcoded class counts in YOLOE codebase.
"""
import os
import sys
import fnmatch

CACHE_PATTERNS = ['__pycache__', '*.pyc', '*.pyo']
SEARCH_ROOTS = ['labelimg/yoloe', 'yoloe_patches']

def print_status(msg, status="INFO"):
    colors = {
        "INFO": "\033[94m", "SUCCESS": "\033[92m", "WARNING": "\033[93m", "ERROR": "\033[91m", "RESET": "\033[0m"
    }
    print(f"{colors.get(status, '')}{status}: {msg}{colors['RESET']}")

def remove_caches():
    print_status("Starting cache cleanup...", "INFO")
    removed = 0
    for root, dirs, files in os.walk('.'):
        for d in dirs:
            if d == '__pycache__':
                path = os.path.join(root, d)
                try:
                    import shutil
                    shutil.rmtree(path)
                    print_status(f"Removed cache dir: {path}", "SUCCESS")
                    removed += 1
                except Exception as e:
                    print_status(f"Failed to remove {path}: {e}", "ERROR")
        for pattern in ['*.pyc', '*.pyo']:
            for f in fnmatch.filter(files, pattern):
                path = os.path.join(root, f)
                try:
                    os.remove(path)
                    print_status(f"Removed cache file: {path}", "SUCCESS")
                    removed += 1
                except Exception as e:
                    print_status(f"Failed to remove {path}: {e}", "ERROR")
    print_status(f"Cache cleanup complete. {removed} items removed.", "INFO")

def search_hardcoded_class_count():
    print_status("Searching for hardcoded class counts (e.g., '74')...", "INFO")
    found = 0
    for search_root in SEARCH_ROOTS:
        for root, dirs, files in os.walk(search_root):
            for f in files:
                if f.endswith('.py'):
                    path = os.path.join(root, f)
                    try:
                        with open(path, 'r', encoding='utf-8') as file:
                            for i, line in enumerate(file):
                                if '74' in line:
                                    print_status(f"Found '74' in {path}:{i+1}: {line.strip()}", "WARNING")
                                    found += 1
                    except Exception as e:
                        print_status(f"Failed to read {path}: {e}", "ERROR")
    if found == 0:
        print_status("No hardcoded '74' found in codebase.", "SUCCESS")
    else:
        print_status(f"Total hardcoded '74' found: {found}", "WARNING")

def main():
    print_status("BakuFlow Cache Cleanup & Diagnostics", "INFO")
    print_status("=" * 50, "INFO")
    remove_caches()
    search_hardcoded_class_count()
    print_status("=" * 50, "INFO")
    print_status("Diagnostics complete.", "SUCCESS")

if __name__ == "__main__":
    main()
