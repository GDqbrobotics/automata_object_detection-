from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
BIREFNET_DIR = PROJECT_ROOT / "BiRefNet"

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Add project root to path so BiRefNet (git submodule) can be imported
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Add BiRefNet directory to path so its absolute imports work
# (birefnet.py uses `from config import Config` instead of relative imports)
if str(BIREFNET_DIR) not in sys.path:
    sys.path.insert(0, str(BIREFNET_DIR))

__all__ = ["ROOT_DIR", "PROJECT_ROOT", "BIREFNET_DIR"]
