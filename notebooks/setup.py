# notebooks/setup.py
import sys
from pathlib import Path

def setup_project_root():
    root = Path(__file__).resolve().parents[1]  # Points to project root
    src_path = root / "src"
    if str(src_path) not in sys.path:
        sys.path.append(str(src_path))