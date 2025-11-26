"""
Model deployment and inference utilities
"""
import sys
from pathlib import Path

# Add parent directory to path for core module imports
src_dir = Path(__file__).parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))
