#!/usr/bin/env python3
"""Test that mimics how the actual app imports modules"""
import sys
sys.path.insert(0, 'src')

try:
    print("Testing app import chain (RealSense mode)...")
    
    # This is what object_extraction.py does
    print("1. Importing CLI...", end=" ")
    from automata_detection.cli import main
    print("OK")
    
    # Verify the modules that get imported
    print("2. Verifying model imports...", end=" ")
    from automata_detection.model import load_model, extract_object
    from BiRefNet.models.birefnet import BiRefNet
    print("OK")
    
    print("3. Verifying camera imports (RealSense)...", end=" ")
    from automata_detection.camera import read_camera
    print("OK")
    
    print("\nSUCCESS: All app imports working!")
    print("Note: camera_orbbec will only be imported when --camera-type orbbec is used")
    
except Exception as e:
    print(f"FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
