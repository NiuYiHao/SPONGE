"""Redraw all 10 FUS RMSD pairs and highlight target 7."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _rmsd import main

if __name__ == "__main__":
    main("FUS", selected_target=7)
