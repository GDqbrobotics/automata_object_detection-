#!/usr/bin/env python3
import sys
sys.path.insert(0, 'src')

try:
    print("Testing import chain...")
    print(f"Python path: {sys.path[:3]}")
    
    print("1. Importing automata_detection...", end=" ")
    from automata_detection import ROOT_DIR, PROJECT_ROOT, BIREFNET_DIR
    print(f"OK (BIREFNET_DIR={BIREFNET_DIR})")
    
    print("2. Importing model...", end=" ")
    from automata_detection import model
    print("OK")
    
    print("3. Loading BiRefNet model...", end=" ")
    # Don't actually load the model (GPU not available), just test import
    from BiRefNet.models.birefnet import BiRefNet
    print("OK")
    
    print("\nSUCCESS: All critical imports working!")
    
except Exception as e:
    print(f"FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

